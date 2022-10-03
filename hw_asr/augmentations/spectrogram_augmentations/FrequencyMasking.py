import random

from hw_asr.augmentations.base import AugmentationBase
from torch import Tensor
from torchaudio import transforms as T


class FrequencyMasking(AugmentationBase):
    def __init__(self, p, *args, **kwargs):
        self.p = p
        self._aug = T.FrequencyMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        if random.random() < self.p:
            x = data.unsqueeze(1)
            return self._aug(x).squeeze(1)
        else:
            return data
        
