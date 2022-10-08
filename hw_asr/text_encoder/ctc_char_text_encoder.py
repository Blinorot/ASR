from collections import defaultdict
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
import torch
from scipy.special import softmax
from tokenizers import Tokenizer
from torch import Tensor
from torchaudio.models.decoder import (CTCHypothesis, ctc_decoder,
                                       download_pretrained_files)

from .char_text_encoder import CharTextEncoder

LM_FILES = download_pretrained_files("librispeech-4-gram")

class Hypothesis(NamedTuple):
    text: str
    prob: float

class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, lng: str = "en", use_bpe: bool = False):
        super().__init__(alphabet, lng)
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.LM_WEIGHT = 3.23
        self.WORD_SCORE = -0.26

        self.use_bpe = use_bpe
        if self.use_bpe:
            tok_path = Path(__file__).absolute().resolve().parent.parent / 'bpe' / 'tokenizer.json'
            self.tokenizer = Tokenizer.from_file(str(tok_path))
            self.char2ind = self.tokenizer.get_vocab()
            self.vocab = [key.lower() for key, _ in self.char2ind.items()]
            self.ind2char = {v: k.lower() for k, v in self.char2ind.items()}

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
            return Tensor(self.tokenizer.encode(text.upper()).ids).unsqueeze(0)

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

        probs = softmax(probs, axis=1) # in case of training mode

        for prob in probs:
            beam = self._extend_beam(beam, prob)
            beam = self._cut_beam(beam, beam_size)
            
        sorted_beam = sorted(beam.items(), key=lambda x: -x[1])
        result = [Hypothesis((sentence + last_char).strip().replace(self.EMPTY_TOK, '')\
                  .replace("'", "").replace("|", ""), v) \
                  for (sentence, last_char), v in sorted_beam]
        return result

    def ctc_lm_beam_search(self, probs: torch.tensor, lengths: torch.tensor,
            beam_size: int = 100) -> List[List[CTCHypothesis]]:
        """
        Performs beam search with language model and returns 
        a list of pairs (hypothesis, hypothesis probability).
        """

        beam_search_decoder = ctc_decoder(
            lexicon=LM_FILES.lexicon,
            tokens=self.vocab,
            lm=LM_FILES.lm,
            nbest=3,
            beam_size=beam_size,
            lm_weight=self.LM_WEIGHT,
            word_score=self.WORD_SCORE,
            blank_token="^",
            unk_word="<UNK>",
            sil_token="|",
        )

        probs = torch.nn.functional.log_softmax(probs, -1)
    
        beam_search_result = beam_search_decoder(probs, lengths)
        return beam_search_result

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
                    new_sentence = (sentence + last_char).replace(self.EMPTY_TOK, '')
                    new_beam[(new_sentence, new_last_char)] += v * prob[i]

        return new_beam

    def _cut_beam(self, beam, beam_size):
        return dict(sorted(beam.items(), key=lambda x: -x[1])[:beam_size])
