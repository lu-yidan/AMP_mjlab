"""K: shared discriminator, terrain-conditioned SuperMimic, proprioceptive actor."""
import torch
from supermimic.modules.world_model import WorldModel, invariant_features
from supermimic.env.tracking_task import CarryRollout
from .task import AMPTask
from .terrain import TerrainMap
from src.tasks.amp_loco.mdp.terrain_tracking import ground_height


class TerrainWorldModel(WorldModel):
    def __init__(self, task, **kwargs):
        n = task.terrain_map.offsets.shape[0]
        super().__init__(task.layout, task.num_actions, task.xi_dim-3+2*n, task.aux_dim, **kwargs)
        self.terrain_map = task.terrain_map

    def inputs(self, S, a, xi):
        scan, ground, valid = self.terrain_map.scan(S[0], S[1], xi[..., -3:])
        features = invariant_features(S)
        features = torch.cat((features[..., :1]-ground[:,None], features[..., 1:]), -1)
        return torch.cat((features, a, xi[..., :-3], scan, valid.float()), -1)


class TerrainAMPTask(AMPTask):
    def __init__(self, env, agent, cache, capture_verification=False):
        self.terrain_map = TerrainMap.from_env(env, cache)
        super().__init__(env, agent, capture_verification)
        self.xi_dim += 3
        assert self.disc.networks['rec'] is self.disc.networks['loco']
        assert self.disc.optimizers['rec'] is self.disc.optimizers['loco']

    def build_world_model(self, **kwargs):
        return TerrainWorldModel(self, **kwargs)

    def get_context(self):
        return torch.cat((super().get_context(), self.env.scene.env_origins), -1)

    def get_side_state(self):
        return {**super().get_side_state(), 'origin':self.env.scene.env_origins.clone()}

    def prepare(self, S, origin=None):
        P=super().prepare(S)
        if origin is None: origin=self.env.scene.env_origins
        scan, ground, valid=self.terrain_map.scan(S[0], S[1], origin)
        return self._terrain_features(P, ground, scan, valid)

    @staticmethod
    def _terrain_features(P, ground, scan, valid):
        P={**P, 'height':P['height']-ground, 'head':P['head']-ground,
           'scan':scan, 'scan_valid':valid}
        P['geometry']=torch.cat((P['head'][:,None],P['geometry'][:,1:]),-1)
        return P

    def prepare_real(self):
        P=super().prepare_real()
        sensor=self.env.scene['terrain_scan']
        scan=(sensor.data.pos_w[:,2:3]-sensor.data.hit_pos_w[...,2])
        valid=sensor.data.distances>=0
        scan=torch.where(valid,scan,30.).clamp(-20,20)
        return self._terrain_features(P,ground_height(self.env)-self.env.scene.env_origins[:,2],scan,valid)

    def frames(self, P, command, action, noise=False):
        a,c=super().frames(P,command,action,noise)
        return a,torch.cat((c,P['scan']),-1)

    def imagine(self, side=None, past=None):
        rollout=super().imagine(side,past)
        return TerrainRollout(self,rollout.carry)


class TerrainRollout(CarryRollout):
    @staticmethod
    def begin(task,carry,S):return {**carry,'P':task.prepare(S,carry['origin'])}
    @staticmethod
    def observe_fn(task,carry,S):
        return {'policy':carry['actor_hist'].flatten(1),'critic':carry['critic_hist'].flatten(1)}
    @staticmethod
    def step_fn(task,carry,S,action,S1,aux):
        return task.transition(carry,carry['P'],action,task.prepare(S1,carry['origin']),aux)
