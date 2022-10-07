from collections import defaultdict
from typing import List, NamedTuple

import torch
from scipy.special import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, lng: str = "en"):
        super().__init__(alphabet, lng)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

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
        return (''.join(text)).strip()

    # based on seminar material
    def ctc_beam_search(self, probs: torch.tensor,
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
        result = [Hypothesis((sentence + last_char).strip().replace(self.EMPTY_TOK, ''), v) \
                  for (sentence, last_char), v in sorted_beam]
        return result

    def ctc_lm_beam_search(self, probs: torch.tensor,
                           beam_size: int = 100, lm_a=0.5, lm_b=1e-4) -> List[Hypothesis]:
        """
        Performs beam search with language model and returns 
        a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
    
        beam = defaultdict(float)

        lm_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        lm_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        lm_model.eval()

        probs = softmax(probs, axis=1) # in case of training mode

        for prob in probs:
            beam = self._extend_beam(beam, prob)
            beam = self._cut_beam(beam, beam_size)

        for k, v in beam.items():
            text = (k[0] + k[1]).strip().replace(self.EMPTY_TOK, '')
            text_prob = self._lm_analyze_text(lm_tokenizer, lm_model, text)
            beam[k] += lm_a * text_prob
            beam[k] += lm_b * len(text)
            
        sorted_beam = sorted(beam.items(), key=lambda x: -x[1])
        result = [Hypothesis((sentence + last_char).strip().replace(self.EMPTY_TOK, ''), v) \
                  for (sentence, last_char), v in sorted_beam]
        return result

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

    def _lm_analyze_text(self, lm_tokenizer, lm_model, text):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        tokenized_text = torch.tensor(lm_tokenizer.encode(text))[None, :].to(device)
        lm_model.to(device)

        output = lm_model(tokenized_text, labels=tokenized_text).loss
        if torch.isnan(output).sum() == 1:
            return 0
        else:
            return output.item()
