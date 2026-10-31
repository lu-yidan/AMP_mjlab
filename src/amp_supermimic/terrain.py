"""Static collision-terrain raster queried at every predicted pose.

Bilinear interpolation is a differentiable approximation near stair edges.
The simulator's ray sensor remains the authority for actual transitions.
"""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TerrainMap(nn.Module):
    def __init__(self, heights, lower, spacing, offsets):
        super().__init__()
        self.register_buffer('heights', heights[None, None].float())
        self.register_buffer('lower', torch.as_tensor(lower, device=heights.device).float())
        self.register_buffer('offsets', offsets.float())
        self.spacing = spacing

    def sample(self, xy):
        shape = xy.shape[:-1]
        h, w = self.heights.shape[-2:]
        uv = (xy - self.lower) / self.spacing
        valid = (uv[..., 0] >= 0) & (uv[..., 0] <= w-1) & (uv[..., 1] >= 0) & (uv[..., 1] <= h-1)
        scale = uv.new_tensor([w-1, h-1])
        grid = (2 * uv / scale - 1).reshape(1, -1, 1, 2)
        z = F.grid_sample(self.heights, grid, align_corners=True, padding_mode='border').reshape(shape)
        return z, valid

    def scan(self, p, R, origin):
        # Match mjlab's yaw scanner fallback when the body's X axis is vertical.
        x = R[:, :2, 0]
        norm = x.norm(dim=-1, keepdim=True)
        y = R[:, :2, 1]
        y = y / y.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        forward = torch.where(norm < .1, torch.stack((y[:, 1], -y[:, 0]), -1), x / norm.clamp(min=1e-6))
        c, s = forward[:, :1], forward[:, 1:2]
        ox, oy = self.offsets[:, 0], self.offsets[:, 1]
        dx, dy = c*ox-s*oy, s*ox+c*oy
        xy = p[:, None, :2] + origin[:, None, :2] + torch.stack((dx, dy), -1)
        z, valid = self.sample(xy)
        clearance = p[:, 2:3] + origin[:, 2:3] - z
        # Downward rays cannot see an overhead surface. Match real scan clipping.
        valid = valid & (clearance >= 0) & (clearance <= 30)
        scan = torch.where(valid, clearance, 30.).clamp(-20, 20)
        return scan, z[:, z.shape[1]//2]-origin[:, 2], valid

    @classmethod
    def from_env(cls, env, cache, spacing=.05):
        import mujoco
        from pathlib import Path
        m = env.sim.mj_model
        tc = env.cfg.scene.terrain.terrain_generator
        # Generated terrain spans centered rows/columns, including the flat border.
        origins = env.scene.terrain.terrain_origins.detach().cpu().numpy().reshape(-1,3)
        lower = origins[:,:2].min(0)-np.asarray(tc.size)/2-tc.border_width
        upper = origins[:,:2].max(0)+np.asarray(tc.size)/2+tc.border_width
        lower += spacing / 2
        upper -= spacing / 2
        digest=hashlib.sha256()
        for name in ('geom_pos','geom_quat','geom_size','geom_type','geom_group','mesh_vert','mesh_face','hfield_data'):
            digest.update(np.asarray(getattr(m,name)).tobytes())
        digest.update(np.asarray([*lower,*upper,spacing]).tobytes())
        cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
        path=cache/(digest.hexdigest()+'.npz')
        if path.exists():
            z=np.load(path)['height']
        else:
            xs=np.arange(lower[0],upper[0]+spacing*.5,spacing)
            ys=np.arange(lower[1],upper[1]+spacing*.5,spacing)
            z=np.empty((len(ys),len(xs)),dtype=np.float32)
            d=mujoco.MjData(m);mujoco.mj_forward(m,d)
            groups=np.array([1,0,0,0,0,0],dtype=np.uint8)
            ray=np.array([0.,0.,-1.]);gid=np.empty(1,dtype=np.int32)
            start=np.array([0.,0.,20.])
            for j,y in enumerate(ys):
                start[1]=y
                for i,x in enumerate(xs):
                    start[0]=x
                    dist=mujoco.mj_ray(m,d,start,ray,groups,1,-1,gid)
                    if dist < 0:raise ValueError(f'Terrain raster miss at {x}, {y}')
                    z[j,i]=20.-dist
                if j%100==0:print(f'terrain raster row {j}/{len(ys)}',flush=True)
            np.savez_compressed(path,height=z)
        offsets=env.scene['terrain_scan']._local_offsets[:,:2].clone()
        obj=cls(torch.as_tensor(z,device=env.device),lower,spacing,offsets)
        obj.fingerprint=digest.hexdigest()
        return obj
