import multiprocessing
from collections import defaultdict
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
import torch
from hw_asr.utils import ROOT_PATH
from pyctcdecode import build_ctcdecoder
from scipy.special import softmax
from tokenizers import Tokenizer
from torch import Tensor

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float

class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, lng: str = "en",
                 use_bpe: bool = False, use_lm: bool = False):
        super().__init__(alphabet, lng)
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.use_bpe = use_bpe
        if self.use_bpe:
            if lng == "en":
                tok_path = Path(__file__).absolute().resolve().parent.parent / 'bpe' / 'tokenizer.json'
            else:
                tok_path = Path(__file__).absolute().resolve().parent.parent / 'bpe' / 'ru_tokenizer.json'
            self.tokenizer = Tokenizer.from_file(str(tok_path))
            self.char2ind = self.tokenizer.get_vocab()
            self.ind2char = {v: k.lower() for k, v in self.char2ind.items()}
            self.vocab = [self.ind2char[ind] for ind in range(len(self.ind2char))]

        if lng == "en":
            self.KENLM = ROOT_PATH / 'data' / 'lm' / 'librispeech'/ '4-gram.arpa'
        else:
            self.KENLM = ROOT_PATH / 'data' / 'lm' / 'nvidia'/ '4gram-pruned-0_1_7_9-ru-lm-set-1.0.arpa'
        if use_lm:
            self.lm_decoder = self._create_lm_decoder()

    def _create_lm_decoder(self):
        vocab = self.vocab
        vocab[0] = ""

        if self.lng == "en":
            vocab = [elem.upper() for elem in vocab]
        else:
            vocab = [elem.lower() for elem in vocab]

        decoder = build_ctcdecoder(
            vocab,
            kenlm_model_path=str(self.KENLM)
        )

        return decoder

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text, self.lng)
        if not self.use_bpe:
            try:
                return Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
            except KeyError as e:
                unknown_chars = set([char for char in text if char not in self.char2ind])
                raise Exception(
                    f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'")
        else:
            if self.lng == "en":
                return Tensor(self.tokenizer.encode(text.upper()).ids).unsqueeze(0)
            else:
                return Tensor(self.tokenizer.encode(text.lower()).ids).unsqueeze(0)

    def ctc_decode(self, inds: List[int]) -> str:
        # based on seminar materials
        last_char = self.EMPTY_TOK
        text = []
        for ind in inds:
            if self.ind2char[ind] == last_char:
                continue
            char = self.ind2char[ind]
            if char != self.EMPTY_TOK:
                text.append(char)
            last_char = char
        return (''.join(text)).replace("'", "").replace("|", "").strip()

    # based on seminar material
    def ctc_beam_search(self, probs: np.array,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
    
        beam = defaultdict(float)

        probs = softmax(probs, axis=1) # to be sure

        for prob in probs:
            beam = self._extend_beam(beam, prob)
            beam = self._cut_beam(beam, beam_size)

        final_beam = defaultdict(float)
        for (sentence, last_char), v in beam.items():
            final_sentence = (sentence + last_char).strip().replace(self.EMPTY_TOK, "")\
                             .replace("'", "").replace("|", "")
            final_beam[final_sentence] += v
            
        sorted_beam = sorted(final_beam.items(), key=lambda x: -x[1])
        result = [Hypothesis(sentence, v) \
                  for sentence, v in sorted_beam]
        return result

    def ctc_lm_beam_search(self, probs: torch.tensor, lengths: torch.tensor,
            beam_size: int = 100) -> List[str]:
        """
        Performs beam search with language model and returns 
        a list of pairs (hypothesis, hypothesis probability).
        """

        probs = torch.nn.functional.log_softmax(probs, -1) # to be sure

        logits_list = [probs[i][:lengths[i]].numpy() for i in range(lengths.shape[0])]

        with multiprocessing.get_context("fork").Pool() as pool:
            text_list = self.lm_decoder.decode_batch(pool, logits_list, beam_width=beam_size)

        text_list = [elem.lower().replace("'", "").replace("|", "").replace("??", "")\
                    .strip() for elem in text_list]

        return text_list

    def _extend_beam(self, beam, prob):
        if len(beam) == 0:
            for i in range(len(prob)):
                last_char = self.ind2char[i]
                beam[('', last_char)] += prob[i]
            return beam

        new_beam = defaultdict(float)
        
        for (sentence, last_char), v in beam.items():
            for i in range(len(prob)):
                if self.ind2char[i] == last_char:
                    new_beam[(sentence, last_char)] += v * prob[i]
                else:
                    new_last_char = self.ind2char[i]
                    new_sentence = (sentence + last_char).replace(self.EMPTY_TOK, '')\
                                    .replace("'", "").replace("|", "")
                    new_beam[(new_sentence, new_last_char)] += v * prob[i]

        return new_beam

    def _cut_beam(self, beam, beam_size):
        return dict(sorted(beam.items(), key=lambda x: -x[1])[:beam_size])
