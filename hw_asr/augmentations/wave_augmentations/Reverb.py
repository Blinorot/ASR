from pathlib import Path

import torch_audiomentations
from hw_asr.augmentations.base import AugmentationBase
from torch import Tensor


class Reverb(AugmentationBase):
    def __init__(self, *args, **kwargs):
        data_dir = Path(__file__).absolute().resolve().parent.parent.parent.parent
        data_dir = data_dir / "data" / "reverb" / "mit"
        self._aug = torch_audiomentations.ApplyImpulseResponse(data_dir, *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
