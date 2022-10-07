from collections import defaultdict
from typing import List

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
            target_text = self.text_encoder.normalize_text(target_text, self.text_encoder.lng)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    """
    CER Metric when predicted text is beam_search over log_probs.
    Using Language Model is optional.
    """
    def __init__(self, text_encoder: BaseTextEncoder, beam_size: int, use_lm=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        self.use_lm = use_lm
        if self.use_lm:
            self.lm_a = kwargs['lm_a']
            self.lm_b = kwargs['lm_b']

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        predictions = log_probs.detach().cpu().numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text, self.text_encoder.lng)

            if not self.use_lm:
                if hasattr(self.text_encoder, "ctc_beam_search"):
                    hypos = self.text_encoder.ctc_beam_search(log_prob_vec[:length], self.beam_size)
                    pred_text = hypos[0].text
                else:
                    raise NotImplementedError()
            else:
                if hasattr(self.text_encoder, "ctc_lm_beam_search"):
                    hypos = self.text_encoder.ctc_lm_beam_search(log_prob_vec[:length],
                                                                 self.beam_size,
                                                                 self.lm_a, self.lm_b)
                    pred_text = hypos[0].text
                else:
                    raise NotImplementedError()

            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
        