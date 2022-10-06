import torch
from hw_asr.base import BaseModel
from torch import nn


class LSTMModel(BaseModel):
    def __init__(self, n_feats, n_class, n_layers=10, fc_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.net = nn.LSTM(input_size=n_feats, hidden_size=fc_hidden,
                           num_layers=n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(fc_hidden, n_class)

    def forward(self, spectrogram, **batch):
        output, _ = self.net(spectrogram.transpose(1, 2))
        output = self.fc(output)
        return {"logits": output}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
