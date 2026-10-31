import unittest
from types import SimpleNamespace as S
import torch
from src.tasks.amp_loco.mdp.terrain_tracking import ground_height,terrain_root_low

class TerrainHeightTests(unittest.TestCase):
    def test_critic_only_scan(self):
        import src.tasks
        from mjlab.tasks.registry import load_env_cfg
        for label in ['I-Shared-TerrainCritic','J-Dual-TerrainCritic']:
            cfg=load_env_cfg('Unitree-G1-AMP-'+label)
            self.assertNotIn('height_scan',cfg.observations['actor'].terms)
            self.assertIn('height_scan',cfg.observations['critic'].terms)
            self.assertEqual(cfg.observations['critic'].history_length,4)
            sensor=next(s for s in cfg.scene.sensors if s.name=='terrain_scan')
            self.assertEqual(sensor.include_geom_groups,(0,))
            self.assertEqual(sensor.ray_alignment,'yaw')

    def test_local_ground_and_miss(self):
        hit=torch.zeros(3,187,3);hit[:,93,2]=torch.tensor([0.,2.,9.])
        dist=torch.ones(3,187);dist[2,93]=-1
        env=S(scene={'terrain_scan':S(data=S(hit_pos_w=hit,distances=dist)),
                    'robot':S(data=S(root_link_pos_w=torch.tensor([[0.,0.,.7],[0.,0.,2.7],[0.,0.,3.4]])))})
        class Scene(dict):pass
        env.scene=Scene(env.scene);env.scene.env_origins=torch.tensor([[0.,0.,0.],[0.,0.,1.],[0.,0.,3.]])
        torch.testing.assert_close(ground_height(env),torch.tensor([0.,2.,3.]))
        torch.testing.assert_close(terrain_root_low(env,.5),torch.tensor([False,False,True]))

if __name__=='__main__':unittest.main()
