"""Load inference checkpoints across hosts without weakening semantic config checks."""
import hashlib
from pathlib import Path
import torch


def load_inference_checkpoint(runner, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    module = runner.alg.multi_amp
    if module is not None and 'multi_amp' in checkpoint:
        saved = checkpoint['multi_amp']['config']
        comparable = dict(saved)
        if 'expert_bank' in comparable:
            local_bank = module.cfg.get('expert_bank')
            if not local_bank or hashlib.sha256(Path(local_bank).read_bytes()).hexdigest() != saved.get('expert_bank_sha256'):
                raise ValueError('Relocated expert bank content differs from checkpoint')
            comparable['expert_bank'] = local_bank
        if comparable != module.cfg:
            raise ValueError('Inference discriminator config differs from checkpoint')
        module.cfg = dict(saved)
    del checkpoint
    return runner.load(path, load_optimizer=False)
