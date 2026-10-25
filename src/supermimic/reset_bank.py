"""Stratified initial states for unified recovery and locomotion (not demonstrations)."""
from pathlib import Path
import hashlib
import numpy as np
import torch

DEFAULT_BANK = str(Path(__file__).resolve().parents[1] / 'assets/motions/g1/reset_banks/l4_amp_train.npz')
DIRECTIONS = ('supine', 'prone', 'left_side_down', 'right_side_down')


class ResetBank:
    def __init__(self, path, joint_names, device):
        self.path = str(Path(path).resolve())
        self.sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            names = data['joint_names'].tolist()
            if len(set(names)) != len(names) or set(names) != set(joint_names):
                raise ValueError('Reset bank joint names do not match robot')
            q = data['qpos'].copy()
            self.sources = data['sources'].copy()
            self.stages = data['stages'].copy()
            self.labels = data['labels'].copy()
        if q.shape[1] != 7+len(joint_names) or not np.isfinite(q).all():
            raise ValueError('Invalid reset bank positions')
        if not np.allclose(np.linalg.norm(q[:,3:7],axis=-1),1,atol=1e-3):
            raise ValueError('Reset bank quaternions must be unit length')
        q[:,7:] = q[:,7:][:,[names.index(k) for k in joint_names]]
        self.qpos = torch.as_tensor(q, dtype=torch.float32, device=device)
        selectors = [(self.sources=='natural') & (self.stages==s) for s in ('late','middle')]
        selectors += [(self.stages=='low') & (self.labels==d) & (self.sources==source)
                      for d in DIRECTIONS for source in ('natural','procedural')]
        self.pools = [torch.as_tensor(np.flatnonzero(s),device=device) for s in selectors]
        if any(len(x)==0 for x in self.pools):
            raise ValueError('Reset bank must contain late/middle and four natural/procedural low directions')

    def sample(self, count):
        # 30% standing (group 0), 15% late, 15% middle, 40% low.
        # Each low direction gets 10%, with 75/25 natural/procedural sources.
        probs = torch.tensor([.30,.15,.15]+[.075,.025]*4,device=self.qpos.device)
        groups = torch.multinomial(probs,count,replacement=True)
        indices = torch.full((count,),-1,device=self.qpos.device,dtype=torch.long)
        for group,pool in enumerate(self.pools,1):
            mask=groups==group
            indices[mask]=pool[torch.randint(len(pool),(int(mask.sum()),),device=pool.device)]
        return groups,indices
