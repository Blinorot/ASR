import re
from typing import List, Union

import numpy as np
from torch import Tensor


class BaseTextEncoder:
    def encode(self, text) -> Tensor:
        raise NotImplementedError()

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item: int) -> str:
        raise NotImplementedError()

    @staticmethod
    def normalize_text(text: str, lng="en"):
        text = text.lower()
        if lng == "en":
            text = re.sub(r"[^a-z ]", "", text)
        elif lng == "ru":
            text = re.sub(r"[^а-я ]", "", text)
        else:
            raise NotImplementedError()
        return text
