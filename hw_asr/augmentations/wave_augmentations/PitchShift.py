import torch_audiomentations
from hw_asr.augmentations.base import AugmentationBase
from torch import Tensor


class PitchShift(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.PitchShift(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
