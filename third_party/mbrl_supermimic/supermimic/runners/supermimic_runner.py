"""SuperMimicRunner: rsl_rl's OnPolicyRunner interface, with SuperMimic in the place of PPO.

    runner = SuperMimicRunner(task, train_cfg, log_dir, device)
    runner.learn(num_learning_iterations)
    runner.save("ckpt.pt")
    policy = runner.get_inference_policy()

train_cfg (dict):
    num_steps_per_env   vectorized sim steps collected per iteration (default 2.5: the paper's UTD 0.4)
    save_interval, log_interval, seed
    policy              {"class_name": "ActorCritic" | "pkg.module:Class", **kwargs}; built as cls(obs, num_actions, **kwargs)
    world_model         WorldModel kwargs (hidden_dims, activation, layer_norm)
    algorithm           SuperMimicCfg fields
    logger              "tensorboard" | "none" (a jsonl log is always written when log_dir is set)

Multi-GPU: launch with torchrun; each rank builds its own task on cuda:LOCAL_RANK.
"""

from __future__ import annotations

import importlib
import json
import os
import statistics
import time
from collections import deque

import torch
import torch.distributed as dist

from ..algorithms.supermimic import SuperMimic, SuperMimicCfg
from ..env.tracking_task import TrackingTask
from ..modules.actor_critic import ActorCritic
from ..modules.world_model import WorldModel
from ..utils.distributed import broadcast_module, is_main, world_size


def _resolve(name):
    if not isinstance(name, str):
        return name
    if name == "ActorCritic":
        return ActorCritic
    mod, _, cls = name.rpartition(":") if ":" in name else name.rpartition(".")
    return getattr(importlib.import_module(mod), cls)


def _scalar(x) -> float:
    return float(x.detach().float().mean()) if torch.is_tensor(x) else float(x)


class SuperMimicRunner:
    def __init__(self, task: TrackingTask, train_cfg: dict, log_dir: str | None = None, device="cpu", policy=None,
                 callbacks=()):
        """callbacks: fn(runner, it) called at the end of every iteration (the mainline's per-iteration duties:
        adaptive-sampling sync, curricula ...); returning True means it loaded other motions.
        train_cfg["motion_resample_interval"]: every that many iterations task.resample_motions() is called;
        required when task.motion_subset."""
        self.task, self.cfg, self.log_dir = task, dict(train_cfg), log_dir
        self.callbacks = list(callbacks)
        self.resample_interval = self.cfg.get("motion_resample_interval")
        self.device = device
        self._configure_multi_gpu()
        torch.manual_seed(self.cfg.get("seed", 1) + self.rank)
        obs = task.get_observations()
        if policy is None:
            pcfg = dict(self.cfg.get("policy", {}))
            policy = _resolve(pcfg.pop("class_name", "ActorCritic"))(obs, task.num_actions, **pcfg)
        policy = policy.to(device)
        wm = WorldModel(task.layout, task.num_actions, task.xi_dim, task.aux_dim, **self.cfg.get("world_model", {})).to(device)
        self.alg_cfg = SuperMimicCfg(**self.cfg.get("algorithm", {}))
        self.alg = SuperMimic(policy, wm, self.alg_cfg, task.step_dt, device)
        self.alg.init_storage(task.num_envs, task.state_dim, task.num_actions, task.xi_dim, task.aux_dim)
        broadcast_module(policy)
        broadcast_module(wm)
        self.num_steps_per_env = self.cfg.get("num_steps_per_env", 2.5)  # may be fractional (steps per update, 1 / UTD)
        self.save_interval = self.cfg.get("save_interval", 50)
        self.log_interval = self.cfg.get("log_interval", 10)
        self.current_learning_iteration = 0
        self.tot_time, self.tot_steps = 0.0, 0
        self.writer = None

    @property
    def policy(self):
        return self.alg.policy

    @property
    def world_model(self):
        return self.alg.wm

    def _configure_multi_gpu(self):
        n = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        if n > 1 and not dist.is_initialized():
            local = int(os.environ.get("LOCAL_RANK", "0"))
            self.device = f"cuda:{local}"
            torch.cuda.set_device(local)
            dist.init_process_group("nccl", rank=self.rank, world_size=n)

    def _init_logging(self):
        if self.log_dir is None or not is_main() or self.writer is not None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        self.jsonl = open(os.path.join(self.log_dir, "supermimic_log.jsonl"), "a")
        if self.cfg.get("logger", "tensorboard") == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(self.log_dir, flush_secs=10)
            except ImportError:
                self.writer = False
        else:
            self.writer = False

    # ------------------------------------------------------------------ training
    def learn(self, num_learning_iterations: int):
        task, alg, layout = self.task, self.alg, self.task.layout
        if getattr(task, "motion_subset", False) and not self.resample_interval:
            raise ValueError("the task holds a subset of its motion library: set train_cfg['motion_resample_interval'] "
                             "(the PPO runners resample periodically), or it trains on the same subset forever")  # fmt: skip
        self._init_logging()
        self.policy.train()
        obs = task.get_observations()
        N = task.num_envs
        cur_ret, cur_len = torch.zeros(N, device=self.device), torch.zeros(N, device=self.device)
        rewbuf, lenbuf = deque(maxlen=100), deque(maxlen=100)
        agg, ep_logs = {}, {}
        start = self.current_learning_iteration
        t_log = time.time()
        for it in range(start, start + num_learning_iterations):
            t0 = time.time()
            alg.set_learning_rate(it)
            self._step_acc = getattr(self, "_step_acc", 0.0) + self.num_steps_per_env
            n_steps = int(self._step_acc)
            self._step_acc -= n_steps
            with torch.no_grad():
                for _ in range(n_steps):
                    s, xi, side = task.get_state(), task.get_context(), task.get_side_state()
                    a = alg.act(obs)
                    obs, r, dones, extras = task.step(a)
                    alg.process_env_step(layout, s, a, task.get_state(), xi, task.get_wm_aux(), dones,
                                         extras.get("terminal_state"), extras.get("teleported"), side)
                    cur_ret += r
                    cur_len += 1
                    done = dones.bool()
                    if done.any():
                        rewbuf.extend(cur_ret[done].tolist())
                        lenbuf.extend(cur_len[done].tolist())
                        cur_ret[done], cur_len[done] = 0, 0
                    for k, v in (extras.get("log") or extras.get("episode") or {}).items():
                        ep_logs.setdefault(k, []).append(_scalar(v))
            alg.sync_statistics()
            t_collect = time.time() - t0
            stats = alg.update_world_model(layout)
            if it >= self.alg_cfg.wm_warmup_iters:
                stats.update(alg.update_policy(task, obs))
            for k, v in stats.items():
                agg.setdefault(k, []).append(v)
            agg.setdefault("time/collect", []).append(t_collect)
            self.tot_time += time.time() - t0
            self.tot_steps += n_steps * N * world_size()
            self.current_learning_iteration = it + 1
            changed = bool(self.resample_interval and (it + 1) % self.resample_interval == 0 and task.resample_motions())
            for cb in self.callbacks:
                changed |= bool(cb(self, it))
            if changed:  # other motions loaded (all envs restarted): stored references are stale
                alg.replay.stale_references()
                obs = task.get_observations()
                cur_ret.zero_()
                cur_len.zero_()
                agg.setdefault("motions/resampled", []).append(1.0)
            if (it + 1) % self.log_interval == 0:
                self._log(it + 1, agg, ep_logs, rewbuf, lenbuf, time.time() - t_log)
                agg, ep_logs, t_log = {}, {}, time.time()
            if self.log_dir and (it + 1) % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it + 1}.pt"))
        if self.log_dir:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _log(self, it, agg, ep_logs, rewbuf, lenbuf, dt):
        row = {k: statistics.fmean(_scalar(x) for x in v) for k, v in agg.items()}
        row.update({f"Episode/{k}": statistics.fmean(v) for k, v in ep_logs.items()})
        if rewbuf:
            row["Train/mean_reward"], row["Train/mean_episode_length"] = statistics.fmean(rewbuf), statistics.fmean(lenbuf)
        row.update({"Perf/total_steps": self.tot_steps, "Perf/gpu_hours": self.tot_time * world_size() / 3600,
                    "Perf/it_per_s": self.log_interval / max(dt, 1e-9)})  # fmt: skip
        if not is_main():
            return row
        if self.log_dir:
            self.jsonl.write(json.dumps({"it": it, **row}) + "\n")
            self.jsonl.flush()
        if self.writer:
            for k, v in row.items():
                self.writer.add_scalar(k, v, it)
        keys = ("wm/err_last", "policy/imag_reward", "policy/imag_survival", "policy/grad_norm", "critic/loss",
                "Train/mean_reward", "Train/mean_episode_length", "Perf/gpu_hours", "Perf/it_per_s")  # fmt: skip
        print(f"it {it} | " + " | ".join(f"{k.split('/')[-1]} {row[k]:.4g}" for k in keys if k in row), flush=True)
        return row

    # ------------------------------------------------------------------ checkpoints and inference
    def save(self, path: str, infos: dict | None = None):
        if not is_main():
            return
        d = self.alg.state_dict()
        d.update({"iter": self.current_learning_iteration, "tot_time": self.tot_time, "tot_steps": self.tot_steps, "infos": infos})
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(d, path + ".tmp")
        os.replace(path + ".tmp", path)

    def load(self, path: str, load_optimizer: bool = True, load_world_model: bool = True):
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.load_state_dict(d, load_optimizer, load_world_model)
        self.current_learning_iteration = d.get("iter", 0)
        self.tot_time, self.tot_steps = d.get("tot_time", 0.0), d.get("tot_steps", 0)
        return d.get("infos")

    def get_inference_policy(self, device=None):
        self.policy.eval()
        if device is not None:
            self.policy.to(device)
        return self.policy.act_inference
