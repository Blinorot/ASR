from typing import List

import torch
from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_wer, ctc_beam_search
from scipy.special import softmax
from torch import Tensor


class ArgmaxWERMetric(BaseMetric):
    """
    WER Metric when predicted text is argmax of log_probs
    """
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamSearchWERMetric(BaseMetric):
    """
    WER Metric when predicted text is beam_search over log_probs
    """
    def __init__(self, text_encoder: BaseTextEncoder, beam_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        predictions = log_probs.detach().cpu().numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)

            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = ctc_beam_search(softmax(log_prob_vec[:length], axis=1), self.beam_size,
                                            self.text_encoder)
            else:
                raise NotImplementedError()
                
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
