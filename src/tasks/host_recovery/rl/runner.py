import inspect
import os

import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from src.tasks.host_recovery.mdp.pull_force import PullForceState


def _onnx_export_kwargs_single_file() -> dict:
  """Build kwargs that request single-file ONNX export across torch versions."""
  try:
    params = inspect.signature(torch.onnx.export).parameters
  except (TypeError, ValueError):
    return {}

  if "external_data" in params:
    return {"external_data": False}
  if "use_external_data_format" in params:
    return {"use_external_data_format": False}
  return {}


def _inline_external_onnx_data(onnx_path: str) -> None:
  """Merge external tensor data back into a single ONNX file if needed."""
  data_path = f"{onnx_path}.data"
  if not os.path.exists(data_path):
    return

  try:
    import onnx

    model = onnx.load(onnx_path, load_external_data=True)
    onnx.save_model(model, onnx_path, save_as_external_data=False)
    if os.path.exists(data_path):
      os.remove(data_path)
    print(f"[INFO]: Inlined external ONNX data into single file: {onnx_path}")
  except Exception as exc:
    print(f"[WARN]: Failed to inline ONNX external data for {onnx_path}: {exc}")


class _OnnxPolicyWrapper(torch.nn.Module):
  """Expose ``act_inference`` as ``forward`` for ONNX export."""

  def __init__(self, actor_critic, obs_normalizer=None):
    super().__init__()
    self.actor_critic = actor_critic
    self.obs_normalizer = obs_normalizer

  def forward(self, obs):
    if self.obs_normalizer is not None:
      obs = self.obs_normalizer(obs)
    return self.actor_critic.act_inference(obs)


class HoSTOnPolicyRunner(MjlabOnPolicyRunner):
  """On-policy runner with ONNX export, same shape as the other tasks here."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    """Pin the HoST policy's final activation to ``tanh``.

    The upstream mjlab MLP leaves the actor output unbounded, while HoST's own
    ``ActorCritic`` ends with ``nn.Tanh()``.  The PPO distribution then samples
    around an unbounded mean, which is what previously pushed actions to 10-20
    even though the environment clipped them to [-1, 1].
    """
    train_cfg.setdefault("actor", {})["action_output_activation"] = "tanh"
    super().__init__(env, train_cfg, log_dir, device)
  def load(
    self,
    path: str,
    load_optimizer: bool = True,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Compatibility wrapper for ``scripts/play.py``.

    The mjlab runner currently loads policy weights only; ``load_optimizer`` is
    accepted so playback can request an inference-only load without overriding
    the whole checkpoint logic.
    """
    infos = super().load(
      path, load_cfg=load_cfg, strict=strict, map_location=map_location
    )
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    if load_optimizer and "optimizer_state_dict" in loaded:
      self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
    if getattr(self, "empirical_normalization", False) and "obs_norm_state_dict" in loaded:
      self.obs_normalizer.load_state_dict(loaded["obs_norm_state_dict"])
      priv = getattr(self, "privileged_obs_normalizer", None)
      if priv is not None and "privileged_obs_norm_state_dict" in loaded:
        priv.load_state_dict(loaded["privileged_obs_norm_state_dict"])

    # Restore the HoST curricula that live outside the model weights.
    env_state = (infos or {}).get("env_state", {})
    env = self.env.unwrapped
    # Playback uses a different env count and a fixed play action scale, and it
    # disables curricula entirely; restoring per-env curriculum tensors there
    # would either crash the step event or override the play scale.
    if getattr(env.cfg, "curriculum", {}):
      if "host_action_rescale" in env_state:
        rescale = env_state["host_action_rescale"].to(self.device)
        value = float(rescale.mean())
        if rescale.shape[0] != env.num_envs:
          rescale = rescale.new_full((env.num_envs, rescale.shape[-1]), value)
        env._host_action_rescale = rescale  # type: ignore[attr-defined]
        term = env.action_manager.get_term("joint_pos")
        scale_cfg = term.cfg.scale
        if isinstance(scale_cfg, dict):
          term.cfg.scale = {key: value for key in scale_cfg}
        else:
          term.cfg.scale = value
        if isinstance(term._scale, torch.Tensor):
          term._scale[:] = value
        else:
          term._scale = value
      if "pull_force" in env_state:
        force = env_state["pull_force"].to(self.device)
        if force.shape[0] != env.num_envs:
          force = force.new_full((env.num_envs, force.shape[-1]), float(force.mean()))
        PullForceState.get().force = force
    return infos

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export the actor to ONNX using the vendored rsl_rl policy."""
    policy = self.alg.get_policy()
    obs_normalizer = self.obs_normalizer if self.empirical_normalization else None
    wrapper = _OnnxPolicyWrapper(policy, obs_normalizer)
    wrapper.to("cpu")
    wrapper.eval()

    num_obs = policy.actor[0].in_features
    dummy_input = torch.zeros(1, num_obs)
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      wrapper,
      dummy_input,
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      input_names=["obs"],
      output_names=["actions"],
      dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
      **_onnx_export_kwargs_single_file(),
    )
    _inline_external_onnx_data(os.path.join(path, filename))

    policy.to(self.device)
    if obs_normalizer is not None:
      obs_normalizer.to(self.device)

  def save(self, path: str, infos=None):
    env = self.env.unwrapped
    env_state = {"common_step_counter": env.common_step_counter}
    rescale = getattr(env, "_host_action_rescale", None)
    if rescale is not None:
      env_state["host_action_rescale"] = rescale.detach().cpu()
    pull_state = PullForceState.get()
    if pull_state.force is not None:
      env_state["pull_force"] = pull_state.force.detach().cpu()
    infos = {**(infos or {}), "env_state": env_state}

    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    if getattr(self, "empirical_normalization", False):
      saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
      priv = getattr(self, "privileged_obs_normalizer", None)
      if priv is not None:
        saved_dict["privileged_obs_norm_state_dict"] = priv.state_dict()
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
