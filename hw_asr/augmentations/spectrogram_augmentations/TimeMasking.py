import random

from hw_asr.augmentations.base import AugmentationBase
from torch import Tensor
from torchaudio import transforms as T


class TimeMasking(AugmentationBase):
    def __init__(self, p, time_masking_p=1, *args, **kwargs):
        self.p = p
        self._aug = T.TimeMasking(*args, p=time_masking_p, **kwargs)

    def __call__(self, data: Tensor):
        if random.random() < self.p:
            x = data.unsqueeze(1)
            return self._aug(x).squeeze(1)
        else:
            return data
        
