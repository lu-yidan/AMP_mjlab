"""Load only SuperMimic actor/critic weights; inference never needs expert data."""
import sys
from pathlib import Path
import torch


def load_policy(path, actor_obs, device='cuda:0', use_ema=True):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'third_party/mbrl_supermimic'))
    from supermimic.modules.actor_critic import ActorCritic
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    weights=checkpoint.get('policy_ema',checkpoint['policy']) if use_ema else checkpoint['policy']
    # A terrain-trained critic has extra inputs. It is restored strictly but
    # never evaluated when testing the proprioceptive actor on a flat task.
    width=weights['critic_norm.mean'].numel()
    assert weights['actor_norm.mean'].numel()==actor_obs.shape[-1]
    model=ActorCritic({'policy':actor_obs,'critic':actor_obs.new_zeros((1,width))},29).to(device)
    model.load_state_dict(weights,strict=True);model.eval()
    clip=checkpoint['cfg'].get('action_clip')
    def policy(obs):
        action=model.act_inference({'policy':obs})
        return action if clip is None else action.clamp(-clip,clip)
    return policy
