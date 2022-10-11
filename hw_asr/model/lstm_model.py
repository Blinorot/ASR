from hw_asr.base import BaseModel
from torch import nn


class LayerNormBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0):
        super().__init__()

        self.first_LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                batch_first=True, bidirectional=True, num_layers=1)
        other_LSTM = []
        layer_norms = []
        for i in range(num_layers - 1):
            other_LSTM.append(nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                                    batch_first=True, bidirectional=True, num_layers=1))
            layer_norms.append(nn.LayerNorm(hidden_size))
        self.other_LSTM = nn.ModuleList(other_LSTM)
        self.layer_norms = nn.ModuleList(layer_norms)
        self.dropout = dropout

    def forward(self, input):
        output, h_n = self.first_LSTM(input)
        output = self._convert_bi_output_to_uni(output)
        for i in range(len(self.other_LSTM)):
            output = self.layer_norms[i](output)
            output, h_n = self.other_LSTM[i](output, h_n)
            output = self._convert_bi_output_to_uni(output)
            output = nn.functional.dropout(output, p=self.dropout, training=self.training)
        return output
        
    def _convert_bi_output_to_uni(self, output):
        output = output.view(output.shape[0], output.shape[1], 2, -1)
        output = output.sum(dim=2)
        output = output.view(output.shape[0], output.shape[1], -1)
        return output


class LSTMModel(BaseModel):
    def __init__(self, n_feats, n_class, n_layers=10, fc_hidden=512, dropout=0, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.net = LayerNormBiLSTM(input_size=n_feats, hidden_size=fc_hidden,
                                   num_layers=n_layers, dropout=dropout)
        self.fc = nn.Linear(fc_hidden, n_class)

    def forward(self, spectrogram, **batch):
        output = self.net(spectrogram.transpose(1, 2))
        output = self.fc(output)
        return {"logits": output}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
