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
            other_GRU.append(nn.GRU(input_size=2 * hidden_size, hidden_size=hidden_size,
                                    batch_first=True, bidirectional=True, num_layers=1))
            layer_norms.append(nn.LayerNorm(2 * hidden_size))
        self.other_GRU = nn.ModuleList(other_GRU)
        self.layer_norms = nn.ModuleList(layer_norms)

    def forward(self, input):
        output, h_n = self.first_GRU(input)
        for i in range(len(self.other_GRU)):
            output = self.layer_norms[i](output)
            output, h_n = self.other_GRU[i](output, h_n)
        return output    
        

class ResDeepSpeechV2Model(BaseModel):
    def __init__(self, n_feats, n_class, n_layers=5, fc_hidden=512,
                n_channels=[32, 32, 32], **batch):
        super().__init__(n_feats, n_class, **batch)

        self.n_channels = n_channels
        self.kernel_size = [(3, 3)] * (len(n_channels))
        self.stride = [(2, 2)] + [(1, 1)] * (len(n_channels) - 1)
        self.padding = [(1, 1)] * (len(n_channels))

        self.first_conv = nn.Sequential(
            nn.Conv2d(1, n_channels[0], 3, 2, 1),
            nn.GELU(),
            nn.BatchNorm2d(n_channels[0])
        )

        convs = []

        for i in range(len(n_channels) - 1):
            layer = nn.Sequential(
                nn.Conv2d(n_channels[i], n_channels[i + 1], 3, 1, 1),
                nn.GELU(),
                nn.BatchNorm2d(n_channels[i+1])
            )
            convs.append(layer)
        
        self.convs = nn.ModuleList(convs)

        self.out_nin = nn.Sequential(
            nn.Linear(n_feats // 2, n_feats // 8),
            nn.GELU(),
            nn.BatchNorm2d(n_channels[-1])
        )

        input_size = n_feats // 8 * n_channels[-1]

        self.rnn = LayerNormBiGRU(input_size=input_size, hidden_size=fc_hidden,
                                  num_layers=n_layers)

        self.fc = nn.Linear(2 * fc_hidden, n_class)

    def forward(self, spectrogram, **batch):
        spectrogram = torch.unsqueeze(spectrogram, 1)
        conv_out = self.first_conv(spectrogram.transpose(2, 3))
        for i in range(len(self.convs)):
            new_conv_out = self.convs[i](conv_out)
            conv_out = new_conv_out + conv_out

        conv_out = self.out_nin(conv_out)
        
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

            numerator = input_size + 2 * self.padding[i][index] - (self.kernel_size[i][index] - 1) - 1
            denominator = self.stride[i][index]
            if torch.is_tensor(input_size):
                input_size = torch.floor(numerator / denominator + 1).to(int)
            else:
                input_size = floor(numerator / denominator + 1)
        return input_size

