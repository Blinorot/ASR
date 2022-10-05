from math import floor
from unicodedata import bidirectional

import torch
from hw_asr.base import BaseModel
from torch import is_tensor, nn


class DeepSpeechV2Model(BaseModel):
    def __init__(self, n_feats, n_class, n_layers=3, fc_hidden=512,
                n_channels=[32, 32], kernel_size=[(11, 41), (11, 21)], 
                stride=[(2, 2), (1, 2)], **batch):
        super().__init__(n_feats, n_class, **batch)

        assert len(n_channels) == len(kernel_size)
        assert len(stride) == len(kernel_size)

        n_channels = [1] + n_channels

        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.stride = stride

        convs = []

        for i in range(len(kernel_size)):
            layer = nn.Sequential(
                nn.Conv2d(n_channels[i], n_channels[i + 1], kernel_size[i], stride[i]),
                nn.Hardtanh(0, 20, inplace=True),
                nn.BatchNorm2d(n_channels[i+1])
            )
            convs.append(layer)
        
        self.convs = nn.Sequential(*convs)

        input_size = self._compute_shapes_after_convs(n_feats, index=1) * n_channels[-1]

        self.rnn = nn.GRU(input_size=input_size, hidden_size=fc_hidden,
                          batch_first=True, bidirectional=True, num_layers=n_layers)

        self.fc = nn.Linear(fc_hidden * 2, n_class)

    def forward(self, spectrogram, **batch):
        spectrogram = torch.unsqueeze(spectrogram, 1)
        conv_out = self.convs(spectrogram.transpose(2, 3))
        conv_out = conv_out.view(conv_out.shape[0], conv_out.shape[2], -1)
        rnn_out, _ = self.rnn(conv_out)
        output = self.fc(rnn_out)

        if self.training:
            output = nn.functional.log_softmax(output, dim=-1)
        else:
            output = nn.functional.softmax(output, dim=-1)

        return {"logits": output}

    def transform_input_lengths(self, input_lengths):
        new_lengths = self._compute_shapes_after_convs(input_lengths, index=0)
        return new_lengths


    def _compute_shapes_after_convs(self, input_size, index):
        for i in range(len(self.kernel_size)):
            numerator = input_size - (self.kernel_size[i][index] - 1) - 1
            denominator = self.stride[i][index]
            if torch.is_tensor(input_size):
                input_size = torch.floor(numerator / denominator + 1).to(int)
            else:
                input_size = floor(numerator / denominator + 1)
        return input_size

