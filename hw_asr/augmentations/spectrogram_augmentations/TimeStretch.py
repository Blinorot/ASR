import imp
import random

import torch
from hw_asr.augmentations.base import AugmentationBase
from torch import Tensor
from torchaudio import transforms as T


class TimeStretch(AugmentationBase):
    def __init__(self, p, min_stretch, max_stretch, *args, **kwargs):
        self.p = p
        self._aug = T.TimeStretch(*args, **kwargs)
        self.min_stretch = min_stretch
        self.max_stretch = max_stretch

    def __call__(self, data: Tensor):
        if random.random() < self.p:
            stretch = self.min_stretch + torch.rand(1) * (self.max_stretch - self.min_stretch)
            x = data.unsqueeze(1)
            return self._aug(x, stretch.item()).squeeze(1)
        else:
            return data
        
