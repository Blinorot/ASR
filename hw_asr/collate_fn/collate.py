import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    batch_size = len(dataset_items)
    channels = dataset_items[0]['spectrogram'].shape[0]
    assert channels == 1

    spectrogram_freq = dataset_items[0]['spectrogram'].shape[1]
    lengths = [elem['spectrogram'].shape[2] for elem in dataset_items]
    max_spec_length = max(lengths)

    texts = [elem['text'] for elem in dataset_items]
    text_encoded_lengths = [elem['text_encoded'].shape[1] for elem in dataset_items]
    max_text_length = max(text_encoded_lengths)

    batch_spectrogram = torch.zeros((batch_size, spectrogram_freq, max_spec_length))
    batch_text_encoded = torch.zeros((batch_size, max_text_length))

    result_batch = {}
    for i in range(batch_size):
        batch_spectrogram[i, :, :lengths[i]] = dataset_items[i]['spectrogram'][0]
        batch_text_encoded[i, :text_encoded_lengths[i]] = dataset_items[i]['text_encoded'][0]

    result_batch['spectrogram'] = batch_spectrogram
    result_batch['spectrogram_length'] = torch.Tensor(lengths)
    result_batch['text_encoded'] = batch_text_encoded
    result_batch['text_encoded_length'] = torch.Tensor(text_encoded_lengths)
    result_batch['text'] = texts
    
    return result_batch
