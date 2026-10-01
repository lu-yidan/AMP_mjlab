"""RL configuration for the Unitree G1 HoST standing-up task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_host_standup_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO hyper-parameters taken from HoST's ``G1CfgPPO``.

  NOTE: HoST also modifies the PPO objective itself (a value/policy smoothness
  regulariser with ``smoothness_lower_bound``/``smoothness_upper_bound`` and a
  multi-critic over its four reward groups). Neither exists in
  ``RslRlPpoAlgorithmCfg``; adding them means patching the vendored ``rsl_rl``
  (see ``rsl_rl/algorithms/ppo.py`` in HoST) and is out of scope for this port.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.8,
        "std_type": "per_dim",
        # Official HoST starts at 0.8 without an upper clamp.  The vendored
        # generic policy otherwise caps exploration at 0.5.
        "max_std": None,
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_host_standup",
    save_interval=500,
    num_steps_per_env=50,
    max_iterations=12000,
    upload_model=False,
    clip_actions=None,
  )
