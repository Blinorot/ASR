from collections import defaultdict
from typing import List

import numpy as np
import torch
from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_cer
from torch import Tensor


class ArgmaxCERMetric(BaseMetric):
    """
    CER Metric when predicted text is argmax of log_probs
    """
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    """
    CER Metric when predicted text is beam_search over log_probs
    """
    def __init__(self, text_encoder: BaseTextEncoder, beam_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        predictions = log_probs.detach().cpu().numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)

            sentence, proba = self._beam_search(log_prob_vec[:length])
            sentence = np.array(list(sentence[0]))

            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(sentence)
            else:
                pred_text = self.text_encoder.decode(sentence)
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)

    # based on seminar materials
    def _beam_search(self, probs):
        beam = {}

        for prob in probs:
            beam = self._extend_beam(beam, prob)
            beam = self._cut_beam(beam)
        
        sorted_beam = sorted(beam.items(), key=lambda x: -x[1])
        return sorted_beam[0]

    def _extend_beam(self, beam, prob):
        if len(beam) == 0:
            last_char = prob.argmax() # char in int format
            beam = {
                ((last_char,), last_char): prob[last_char].item()
            }
            return beam

        new_beam = defaultdict(float)
        
        for (sentence, last_char), v in beam.items():
            for i in range(len(prob)):
                if i == last_char:
                    new_beam[(sentence, last_char)] += v * prob[i]
                else:
                    new_beam[((*sentence, i), i)] += v * prob[i]

        return new_beam

    def _cut_beam(self, beam):
        return dict(sorted(beam.items(), key=lambda x: -x[1])[:self.beam_size])
        