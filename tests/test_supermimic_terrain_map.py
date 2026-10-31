import torch
from src.amp_supermimic.terrain import TerrainMap


def test_pose_query_has_correct_world_origin_and_position_gradient():
    y,x=torch.meshgrid(torch.arange(9.),torch.arange(9.),indexing='ij')
    terrain=TerrainMap(.2*x+.1*y,[0.,0.],1.,torch.tensor([[-.5,0.],[0.,0.],[.5,0.]]))
    p=torch.tensor([[1.,1.,3.],[1.,1.,3.]],requires_grad=True)
    origin=torch.tensor([[1.,1.,0.],[3.,1.,0.]])
    R=torch.eye(3).repeat(2,1,1)
    scan,ground,valid=terrain.scan(p,R,origin)
    torch.testing.assert_close(ground,torch.tensor([.6,1.]))
    assert valid.all()
    scan[:,1].sum().backward()
    torch.testing.assert_close(p.grad,torch.tensor([[-.2,-.1,1.],[-.2,-.1,1.]]))
    moved=p.detach().clone();moved[:,0]+=1
    torch.testing.assert_close(terrain.scan(moved,R,origin)[0],scan.detach()-.2)


def test_yaw_rotates_pattern_and_outside_or_below_is_invalid():
    y,x=torch.meshgrid(torch.arange(9.),torch.arange(9.),indexing='ij')
    terrain=TerrainMap(.2*x,[0.,0.],1.,torch.tensor([[-1.,0.],[0.,0.],[1.,0.]]))
    p=torch.tensor([[3.,3.,2.]])
    R=torch.tensor([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]])
    scan,_,valid=terrain.scan(p,R,torch.zeros(1,3))
    torch.testing.assert_close(scan,torch.full((1,3),1.4));assert valid.all()
    p[:,2]=-1
    scan,_,valid=terrain.scan(p,R,torch.zeros(1,3))
    assert not valid.any();assert (scan==20).all()
    _,valid=terrain.sample(torch.tensor([[20.,20.]]));assert not valid.any()


def test_sensor_finalize_invalidates_early_reward_read(monkeypatch):
    from mjlab.sensor import RayCastSensor
    from src.tasks.amp_loco.mdp.terrain_sensor import FreshTerrainRayCastSensor
    sensor=object.__new__(FreshTerrainRayCastSensor)
    sensor._cached_data='previous step';sensor._cache_valid=True
    sensor._compute_data=lambda:'new scan'
    monkeypatch.setattr(RayCastSensor,'postprocess_rays',lambda self:None)
    assert sensor.data=='previous step'
    sensor.postprocess_rays()
    assert sensor.data=='new scan'
