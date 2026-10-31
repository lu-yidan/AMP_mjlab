"""The task protocol SuperMimic trains against: rsl_rl's VecEnv, plus what imagination needs.

rsl_rl locks ``num_envs / num_actions / get_observations() / step(a)``. SuperMimic keeps those and adds:

* ``get_state()``: the simulated state s the world model predicts (``StateLayout``, flat);
* ``ref(clip_ids, frames)``: the reference motion, indexable at any future frame of an imagined rollout;
* ``reward_fn / termination_fn / obs_fn``: the task's reward, termination and observations as functions
  of the state, so BPTT can push gradients through them. Written as plain tensor code they can run as
  one compiled graph per imagined step together with the policy and the world model (``CarryRollout``),
  which is what makes imagination fast.

``step()`` still drives the real simulator. It feeds the replay, and it is the ground truth the imagined
reward, termination and observations are checked against (``supermimic.utils.checks.verify_task``).

Tasks whose reward and observations live in a manager stack (Isaac Lab, mjlab) can skip the functional
route and override ``imagine()`` with their own ``ImaginedRollout``, e.g. one that writes the imagined state
into the env's buffers and re-runs the manager terms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from ..utils.rotations import mat_to_quat, quat_to_mat

Tensor = torch.Tensor


class StateLayout:
    """Floating-base state ``S = (p, R, v, w, q, qd)``: root position, root rotation matrix, world-frame
    root linear and angular velocity, joint positions and velocities, for ``num_joints`` simulated joints
    (include passive ones: the world model has to predict them too).

    Flat form: ``(p, quat_wxyz, v, w, q, qd)``, ``13 + 2 * num_joints`` wide.
    """

    def __init__(self, num_joints: int):
        self.num_joints = num_joints
        self.state_dim = 13 + 2 * num_joints
        self.feature_dim = 1 + 6 + 3 + 3 + 2 * num_joints  # world-model input features
        self.delta_dim = 12 + 2 * num_joints  # world-model output increments

    def flatten(self, S) -> Tensor:
        p, R, v, w, q, qd = S
        return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)

    def unflatten(self, x: Tensor):
        nj = self.num_joints
        return (x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13],
                x[..., 13:13 + nj], x[..., 13 + nj:13 + 2 * nj])  # fmt: skip


@dataclass
class ImaginedStep:
    reward: Tensor  # (N,) on the same scale as step() returns
    done: Tensor  # (N,) bool: the task's exact termination (not time-outs)
    done_soft: Tensor | None = None  # (N,) differentiable termination probability, the backward surrogate
    terms: dict[str, Tensor] = field(default_factory=dict)  # per-term rewards, for logging


class ImaginedRollout(ABC):
    """An imagined continuation of the live envs, made by ``TrackingTask.imagine()`` right after the real
    step whose observations start it. It owns everything beyond S that obs and reward read (reference
    cursor, previous actions, observation history, DR) and advances it per step. Both methods must be
    differentiable in S and the actions."""

    @abstractmethod
    def observe(self, S) -> dict[str, Tensor]:
        """Observations of imagined state S, same keys and layout as ``TrackingTask.get_observations()``."""

    @abstractmethod
    def step(self, S, action: Tensor, S1, aux: Tensor | None = None) -> ImaginedStep:
        """Reward and termination of the imagined transition S -action-> S1; then advance one step."""


class TrackingTask(ABC):
    num_envs: int
    num_actions: int
    layout: StateLayout
    step_dt: float
    device: torch.device

    # extra per-env inputs of the world model the policy does not see (DR: gains, masses, offsets ...)
    xi_dim: int = 0
    # extra world-model targets observed after each step that imagined rewards need (e.g. requested torque)
    aux_dim: int = 0
    # longest observation history (in steps) imagination must rebuild when starting from a stored state
    history_length: int = 0
    # functional route: frame of the reference a step's reward and termination use, relative to the frame
    # of the observation the step acted on. 1: the next frame (advance, then reward, as the toy's step()).
    # 0: the same frame (Isaac Lab / mjlab managers evaluate reward and termination before the command
    # advances, so the reference the next observation sees is one frame ahead of the one just rewarded).
    ref_offset: int = 1
    # the task holds only a subset of a larger motion library and must be resampled periodically (the runner
    # refuses to train such a task without ``motion_resample_interval``)
    motion_subset: bool = False
    # CarryRollout (functional route included): torch.compile mode of its functions when they run outside the
    # training loop's whole-step graph (gates, evaluation, SuperMimicCfg.compile=False). None: eager.
    compile_mdp: str | None = None

    # ------------------------------------------------------------------ simulator (rsl_rl VecEnv)
    @abstractmethod
    def get_observations(self) -> dict[str, Tensor]:
        """``{"policy": (N, ...), "critic": (N, ...)}``; "critic" is optional and falls back to "policy"."""

    @abstractmethod
    def step(self, actions: Tensor) -> tuple[dict[str, Tensor], Tensor, Tensor, dict]:
        """Step the real simulator. Returns ``obs, rewards (N,), dones (N,), extras``.

        extras may hold ``"time_outs"`` (N,) bool, ``"log"`` or ``"episode"`` ({name: scalar}),
        ``"terminal_state"`` (N, state_dim): the pre-reset state of the envs that are done, which lets the
        world model also learn the transitions into terminations, and ``"teleported"`` (N,) bool: envs
        whose state was moved inside step() without a done (e.g. a motion resampled at its end), whose
        transition is not dynamics, and ``"reward_not_imagined"`` (N,): the part of the reward from terms
        imagination cannot evaluate (contact counts ...), and ``"reward_terms"`` ({name: (N,)}), the real
        per-term rewards, both for the adapter gate.
        """

    def reset(self, env_ids: Tensor | None = None):
        raise NotImplementedError

    @property
    def state_dim(self) -> int:
        return self.layout.state_dim

    # ------------------------------------------------------------------ SuperMimic
    @abstractmethod
    def get_state(self) -> Tensor:
        """(N, layout.state_dim) flat state of the live envs."""

    def get_context(self) -> Tensor:
        return torch.zeros(self.num_envs, self.xi_dim, device=self.device)

    def get_wm_aux(self) -> Tensor | None:
        return None

    def resample_motions(self) -> bool:
        """Load a new subset of the motion library (``motion_subset`` tasks; the runner calls it every
        ``motion_resample_interval`` iterations). Returns True if the loaded motions changed: the reference
        cursors stored in the replay then point into other motions, so imagination stops starting from rows
        stored before it. Default: nothing to resample."""
        return False

    def with_context(self, obs: dict[str, Tensor], xi: Tensor) -> dict[str, Tensor]:
        """Observations with the DR context added where the task wants it (the manager wrappers append it to
        "critic"); applied to real and imagined observations alike. Default: unchanged."""
        return obs

    def get_side_state(self) -> dict[str, Tensor]:
        """Per-env tensors that, with the flat state, let imagination start from a stored state later: the
        reference cursor, previous actions ... (stored in the replay next to each transition). The default
        serves the functional route."""
        clip_ids, frames = self.get_ref_cursor()
        return {"clip": clip_ids, "frame": frames, "last_action": self.get_last_action()}

    def imagine(self, side: dict[str, Tensor] | None = None, past: dict | None = None) -> ImaginedRollout:
        """A rollout from the live envs (side=None) or from stored states whose side state is ``side``
        (as returned by ``get_side_state``, one row per env, sampled from the replay, plus ``"xi"``: the stored
        DR context of those rows). ``past`` holds the
        ``history_length`` stored rows ending at the start (see TransitionReplay.sample_starts) for tasks
        whose observations carry history."""
        return FunctionalRollout(self, side)

    # ------------------------------------------------------------------ functional route (default imagine())
    def get_ref_cursor(self) -> tuple[Tensor, Tensor]:
        """(clip_ids, frames) of the live envs: the reference frame the current state is compared with."""
        raise NotImplementedError

    def get_last_action(self) -> Tensor:
        raise NotImplementedError

    def ref(self, clip_ids: Tensor, frames: Tensor):
        """Reference at (clip, frame), any structure the task's functions read. Clamp past the clip end."""
        raise NotImplementedError

    def prepare(self, S):
        """What obs_fn / reward_fn / termination_fn read in place of the raw state: anything derived from S
        that several of them need (forward kinematics ...), computed once per imagined state. Default: S."""
        return S

    def obs_fn(self, P, ref, last_action: Tensor) -> dict[str, Tensor]:
        """Observations of the prepared state P (``prepare(S)``)."""
        raise NotImplementedError

    def reward_fn(self, P1, action: Tensor, last_action: Tensor, ref, P=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Reward of reaching P1 from P by ``action`` (after ``last_action``), compared with ``ref``. P is the
        prepared state the step started from, for terms that anchor the reference at the previous state
        (manager-based envs compute robot-relative references before the step); returns (total, {term: (N,)})."""
        raise NotImplementedError

    def termination_fn(self, P1, ref, P=None) -> tuple[Tensor, Tensor | None]:
        """(exact bool termination, differentiable termination probability or None) of reaching P1."""
        raise NotImplementedError


def _cloned(fn):
    """CUDA-graph replays overwrite the previous call's outputs; the rollout keeps them across steps."""
    def run(*a):
        return torch.utils._pytree.tree_map(lambda x: x.clone() if torch.is_tensor(x) else x, fn(*a))
    return run


class CarryRollout(ImaginedRollout):
    """An imagined rollout written as pure functions of a *carry*: a pytree of tensors holding everything beyond S
    that observations, rewards and terminations read (reference cursor, previous action, observation history,
    quantities derived from the current state such as its forward kinematics).

    This is the fast path. Because the functions are pure, ``SuperMimic`` compiles a whole imagined step
    (observation, policy, world model, wild-state filter, reward, termination, return bookkeeping) into one graph,
    forward and backward. Subclasses implement the static methods below; they are traced, so: tensor ops only,
    fixed shapes, no ``.item()`` or data-dependent Python branches, no writes to tensors outside them.

    ``observe`` / ``step`` (the ``ImaginedRollout`` interface, used by the gates and the generic loop) run the same
    functions, compiled per ``task.compile_mdp``. ``observe(S)`` observes the carry's current state: S must be the
    state the rollout is at (the start state, or the S1 of the last ``step``)."""

    def __init__(self, task: TrackingTask, carry: dict):
        self.task, self.carry, self._begun = task, carry, False

    @staticmethod
    def begin(task, carry: dict, S) -> dict:
        """The carry at the start state S (e.g. add what is derived from S). Default: unchanged."""
        return carry

    @staticmethod
    @abstractmethod
    def observe_fn(task, carry: dict, S) -> dict[str, Tensor]:
        """Observations at the carry's current state S."""

    @staticmethod
    @abstractmethod
    def step_fn(task, carry: dict, S, action: Tensor, S1, aux: Tensor | None):
        """The transition S -action-> S1: ``(carry at S1, (reward, done, done_soft or None, {term: (N,)}))``."""

    def _begin(self, S):
        if not self._begun:
            self.carry, self._begun = carry_graphs(self.task, type(self))[0](self.carry, S), True

    def observe(self, S):
        self._begin(S)
        return carry_graphs(self.task, type(self))[1](self.carry, S)

    def step(self, S, action, S1, aux=None):
        self._begin(S)
        self.carry, (r, done, done_soft, terms) = carry_graphs(self.task, type(self))[2](self.carry, S, action, S1, aux)
        return ImaginedStep(r, done, done_soft, terms)


def carry_graphs(task: TrackingTask, cls: type[CarryRollout]):
    """(begin, observe, step) of a CarryRollout class bound to ``task``, compiled per ``task.compile_mdp`` and cached
    on the task (eager use; the training loop compiles its own whole-step graph from the raw functions)."""
    mode = task.compile_mdp
    cache = task.__dict__.setdefault("_carry_graphs", {})
    if (cls, mode) not in cache:
        fns = (lambda carry, S: cls.begin(task, carry, S), lambda carry, S: cls.observe_fn(task, carry, S),
               lambda carry, S, a, S1, aux: cls.step_fn(task, carry, S, a, S1, aux))  # fmt: skip
        if mode:
            fns = tuple(torch.compile(f, mode=mode, dynamic=False, fullgraph=True) for f in fns)
            if mode == "reduce-overhead":
                fns = tuple(_cloned(f) for f in fns)
        cache[(cls, mode)] = fns
    return cache[(cls, mode)]


class FunctionalRollout(CarryRollout):
    """imagine() for tasks written on the functional route. Carry: reference cursor (clip, frame of the current
    observation), last action, and ``P = task.prepare(S)`` of the current state (computed once per imagined state).
    Per step: reward_fn / termination_fn of S -> S1 against the reference ``task.ref_offset`` frames past the
    observed one, then the cursor advances one frame."""

    def __init__(self, task: TrackingTask, side: dict | None = None):
        if side is None:
            side = task.get_side_state()
        frames = side["frame"].clone()
        clip = side["clip"].clone() if "clip" in side else torch.zeros_like(frames)
        carry = {"clip": clip, "frame": frames, "last_action": side["last_action"].clone()}
        if task.xi_dim > 0:  # the start's DR context, for task.with_context
            carry["xi"] = (side["xi"] if "xi" in side else task.get_context()).clone()
        super().__init__(task, carry)

    @staticmethod
    def begin(task, carry, S):
        return {**carry, "P": task.prepare(S)}

    @staticmethod
    def observe_fn(task, carry, S):
        obs = task.obs_fn(carry["P"], task.ref(carry["clip"], carry["frame"]), carry["last_action"])
        return task.with_context(obs, carry["xi"]) if "xi" in carry else obs

    @staticmethod
    def step_fn(task, carry, S, action, S1, aux):
        P1 = task.prepare(S1)
        ref = task.ref(carry["clip"], carry["frame"] + task.ref_offset)
        r, terms = task.reward_fn(P1, action, carry["last_action"], ref, carry["P"])
        done, done_soft = task.termination_fn(P1, ref, carry["P"])
        return {**carry, "frame": carry["frame"] + 1, "last_action": action, "P": P1}, (r, done, done_soft, terms)
