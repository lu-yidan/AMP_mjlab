"""RSL-RL configuration for the migrated G1 recovery task."""

from dataclasses import dataclass, field, replace

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from g1recovery_amp.tasks.recovery.rl.algorithm import AmpPpoCfg

AMP_PPO_CLASS_PATH = "g1recovery_amp.tasks.recovery.rl.algorithm:AmpPPO"
A6_AMP_STYLE_REWARD_SCALE = 50.0
A6_STABLE_STD_RANGE = (0.05, 1.5)
A6_STABLE_ENTROPY_COEF = 0.001
A6_STABLE_LEARNING_RATE = 2.25e-5
A6_CONSTRAINT_AWARE_AMP_SCALE = 0.05
A6_CONSTRAINT_AWARE_001_AMP_SCALE = 0.01


@dataclass
class RecoveryAmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Ordinary PPO settings plus the task-owned AMP settings."""

    class_name: str = AMP_PPO_CLASS_PATH
    amp_cfg: AmpPpoCfg = field(default_factory=AmpPpoCfg)


def g1_recovery_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Return source-shaped actor, critic, PPO, and AMP training settings."""

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RecoveryAmpPpoAlgorithmCfg(
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
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
        experiment_name="g1_amp_get_up_no_keybody_robust_separate",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=500_000,
        clip_actions=3.5,
        logger="tensorboard",
        upload_model=False,
    )


def g1_recovery_a6_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Keep checkpoint shapes and the native AMP reward scale for A6 recovery."""

    cfg = g1_recovery_ppo_runner_cfg()
    if not isinstance(cfg.algorithm, RecoveryAmpPpoAlgorithmCfg):
        raise TypeError("expected the recovery AMP algorithm configuration")
    amp_cfg = cfg.algorithm.amp_cfg
    cfg.algorithm.amp_cfg = replace(
        amp_cfg,
        discriminator=replace(
            amp_cfg.discriminator,
            style_reward_scale=A6_AMP_STYLE_REWARD_SCALE,
        ),
    )
    return cfg


def g1_recovery_a6_symmetric_005_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """A conservative resume setup for the symmetric 0.05 obstruction trial."""

    cfg = g1_recovery_a6_ppo_runner_cfg()
    if not isinstance(cfg.algorithm, RecoveryAmpPpoAlgorithmCfg):
        raise TypeError("expected the recovery AMP algorithm configuration")
    distribution_cfg = dict(cfg.actor.distribution_cfg or {})
    distribution_cfg["std_range"] = A6_STABLE_STD_RANGE
    cfg.actor = replace(cfg.actor, distribution_cfg=distribution_cfg)
    cfg.algorithm.entropy_coef = A6_STABLE_ENTROPY_COEF
    cfg.algorithm.learning_rate = A6_STABLE_LEARNING_RATE
    cfg.algorithm.schedule = "fixed"
    cfg.algorithm.amp_cfg = replace(
        cfg.algorithm.amp_cfg,
        style_reward_scale_obs_group="amp_reward_scale",
        obstructed_style_reward_scale=0.05,
        actor_std_range_on_load=A6_STABLE_STD_RANGE,
    )
    return cfg


def g1_recovery_a6_constraint_aware_005_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Use 5% AMP style under obstruction while retaining full safety costs."""

    cfg = g1_recovery_a6_symmetric_005_ppo_runner_cfg()
    if not isinstance(cfg.algorithm, RecoveryAmpPpoAlgorithmCfg):
        raise TypeError("expected the recovery AMP algorithm configuration")
    cfg.algorithm.amp_cfg = replace(
        cfg.algorithm.amp_cfg,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_AMP_SCALE,
        ppo_learning_rate_on_load=A6_STABLE_LEARNING_RATE,
    )
    return cfg


def g1_recovery_a6_constraint_aware_001_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Use 1% AMP style under obstruction while retaining full safety costs."""

    cfg = g1_recovery_a6_constraint_aware_005_ppo_runner_cfg()
    if not isinstance(cfg.algorithm, RecoveryAmpPpoAlgorithmCfg):
        raise TypeError("expected the recovery AMP algorithm configuration")
    cfg.algorithm.amp_cfg = replace(
        cfg.algorithm.amp_cfg,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_001_AMP_SCALE,
    )
    return cfg
