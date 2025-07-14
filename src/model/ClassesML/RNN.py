from typing import Dict
import torch
import torch.nn as nn

class WineRNN(nn.Module):
    '''Sequence‑aware binary classifier for French wine quality.

    forward expects:
        x_seq  : (B, T, F_seq)   météo sur T millésimes
        x_cat  : (B, 3)          indices [region, station, cépage]
        x_stat : (B, F_static)   prix ou autres scalaires

    Required hyper‑param keys:
        seq_len, num_seq_features, embedding_sizes,
        static_num_features, rnn_hidden_size (list), num_layers,
        dropout_rate, (optional) bidirectional, output_dim
    '''

    def __init__(self, hp: Dict):
        super().__init__()

        self.hidden_size   = hp['rnn_hidden_size'][0]
        self.num_layers    = hp['num_layers']
        self.bidirectional = hp.get('bidirectional', False)
        self.output_dim    = hp.get('output_dim', 2)
        self.embeddings    = nn.ModuleDict({
            name: nn.Embedding(n, d) for name, (n, d) in hp['embedding_sizes'].items()
        })
        emb_dim = sum(d for _, (_, d) in hp['embedding_sizes'].items())

        self.lstm = nn.LSTM(
            input_size = hp['input_dim'][0],
            hidden_size  = self.hidden_size,
            num_layers   = self.num_layers,
            batch_first  = True,
            dropout      = hp['dropout_rate'] if self.num_layers > 1 else 0.,
            bidirectional= self.bidirectional,
        )
        lstm_out = self.hidden_size * (2 if self.bidirectional else 1)

        fc_in = lstm_out + emb_dim + hp.get('static_num_features', 1)
        self.head = nn.Sequential(
            nn.Linear(fc_in, 64), nn.ReLU(), nn.Dropout(hp['dropout_rate']),
            nn.Linear(64, 32),    nn.ReLU(), nn.Dropout(hp['dropout_rate']),
            nn.Linear(32, self.output_dim)
        )

    def forward(self, x_seq, x_cat, x_stat):
        _, (h_n, _) = self.lstm(x_seq)         # (layers*dir, B, H)
        h_last = h_n[-1]                       # (B, H)

        emb = torch.cat([
            self.embeddings[k](x_cat[:, i]) for i, k in enumerate(self.embeddings)
        ], dim=1)

        if x_stat.ndim == 1:
            x_stat = x_stat.unsqueeze(1)

        feats = torch.cat([h_last, emb, x_stat], dim=1)
        return self.head(feats)
