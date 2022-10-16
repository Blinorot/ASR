import logging
from typing import List

import torch
from numpy import dtype

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items. Returns batch containing: spectrogram, its length,
    encoded text and its length, original text, audio and audio path
    
    :param dataset_items: list of dicts from dataset

    Dict contains the following keys:
        spectrogram - the spectrogram of the audio (may be in log scale) after augmentations
        text - the text from the audio
        text_encoder - encoded text from the audio
        audio_path - path to audio file
        audio - tensor loaded from audio_path after augmentations
        duration - the duration of an audio in seconds
        log_spec - whether the spectrogram is in log scale or not

    Note: for log scale padding is done with log(1e-5) not with zeros. This is because 
        data should be padded with silence and silence in log scale is log(0) ~ log(1e-5)
    """

    batch_size = len(dataset_items)
    channels = dataset_items[0]['spectrogram'].shape[0]

    spectrogram_freq = dataset_items[0]['spectrogram'].shape[1]
    lengths = [elem['spectrogram'].shape[2] for elem in dataset_items]
    max_spec_length = max(lengths)

    texts = [elem['text'] for elem in dataset_items]
    text_encoded_lengths = [elem['text_encoded'].shape[1] for elem in dataset_items]
    max_text_length = max(text_encoded_lengths)

    audio_paths = [elem['audio_path'] for elem in dataset_items]
    audio = [elem['audio'] for elem in dataset_items]

    batch_spectrogram = torch.zeros((batch_size, spectrogram_freq, max_spec_length))

    log_needed = dataset_items[0]['log_spec']

    if log_needed is not None and log_needed:
        batch_spectrogram = torch.log(batch_spectrogram + 1e-5) # to simulate silence in raw wave

    batch_text_encoded = torch.zeros((batch_size, max_text_length))

    result_batch = {}
    for i in range(batch_size):
        batch_spectrogram[i, :, :lengths[i]] = dataset_items[i]['spectrogram'][0]
        batch_text_encoded[i, :text_encoded_lengths[i]] = dataset_items[i]['text_encoded'][0]

    result_batch['spectrogram'] = batch_spectrogram
    result_batch['spectrogram_length'] = torch.tensor(lengths, dtype=torch.long)
    result_batch['text_encoded'] = batch_text_encoded
    result_batch['text_encoded_length'] = torch.tensor(text_encoded_lengths, dtype=torch.long)
    result_batch['text'] = texts
    result_batch['audio_path'] = audio_paths
    result_batch['audio'] = audio
    
    return result_batch
