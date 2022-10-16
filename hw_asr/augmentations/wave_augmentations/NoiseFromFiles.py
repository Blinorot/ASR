import random
from pathlib import Path

import numpy as np
import torch
import torch_audiomentations
import torchaudio
from hw_asr.augmentations.base import AugmentationBase
from torch import Tensor


class NoiseFromFiles(AugmentationBase):
    def __init__(self, *args, **kwargs):
        data_dir = Path(__file__).absolute().resolve().parent.parent.parent.parent
        self.data_dir = data_dir / "data" / "noise" / "fsd"
        self.noises = [fpath for fpath in (self.data_dir).iterdir()]
        self.p = kwargs["p"]
        self.sample_rate = kwargs["sample_rate"]
        self.max_snr_in_db = kwargs["max_snr_in_db"]
        assert self.max_snr_in_db > 3

    def __call__(self, data: Tensor):
        if random.random() > self.p:
            return data
        
        noise_path = np.random.choice(self.noises)
        noise, noise_sr = torchaudio.load(noise_path)
        if noise_sr != self.sample_rate:
            noise = torchaudio.functional.resample(noise, noise_sr, self.sample_rate)
        noise = noise[0:1, :]

        if noise.shape[1] < data.shape[1]:
            noise = noise.repeat(1, data.shape[1] // noise.shape[1] + 1)
        
        noise = noise[:, :data.shape[1]]
        
        snr_db = np.random.uniform(3, self.max_snr_in_db)

        speech_rms = data.norm(p=2)
        noise_rms = noise.norm(p=2)

        snr = 10 ** (snr_db / 20)
        scale = snr * noise_rms / speech_rms

        result = (scale * data + noise) / 2

        return result
