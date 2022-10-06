from math import floor
from turtle import forward
from unicodedata import bidirectional

import torch
from hw_asr.base import BaseModel
from torch import is_tensor, layer_norm, nn


class LayerNormBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.first_GRU = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                                batch_first=True, bidirectional=True, num_layers=1)
        other_GRU = []
        layer_norms = []
        for i in range(num_layers - 1):
            other_GRU.append(nn.GRU(input_size=hidden_size, hidden_size=hidden_size,
                                    batch_first=True, bidirectional=True, num_layers=1))
            layer_norms.append(nn.LayerNorm(hidden_size))
        self.other_GRU = nn.ModuleList(other_GRU)
        self.layer_norms = nn.ModuleList(layer_norms)

    def forward(self, input):
        output, h_n = self.first_GRU(input)
        output = self._convert_bi_output_to_uni(output)
        for i in range(len(self.other_GRU)):
            output = self.layer_norms[i](output)
            output, h_n = self.other_GRU[i](output, h_n)
            output = self._convert_bi_output_to_uni(output)
        return output
        
    def _convert_bi_output_to_uni(self, output):
        output = output.view(output.shape[0], output.shape[1], 2, -1)
        output = output.sum(dim=2)
        output = output.view(output.shape[0], output.shape[1], -1)
        return output
        

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

        self.rnn = LayerNormBiGRU(input_size=input_size, hidden_size=fc_hidden,
                                  num_layers=n_layers)

        self.fc = nn.Linear(fc_hidden, n_class)

    def forward(self, spectrogram, **batch):
        spectrogram = torch.unsqueeze(spectrogram, 1)
        conv_out = self.convs(spectrogram.transpose(2, 3))
        conv_out = conv_out.view(conv_out.shape[0], conv_out.shape[2], -1)
        rnn_out = self.rnn(conv_out)
        output = self.fc(rnn_out)

        if not self.training:
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

