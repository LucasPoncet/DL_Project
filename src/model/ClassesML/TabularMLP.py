import torch
import torch.nn as nn
from ClassesML.Blocks import DenseBlock
from Utils.Utilities import Utilities

class TabularMLP(nn.Module):

    def __init__(self, hyperparameters, embedding_sizes):
        super(TabularMLP, self).__init__()

        self.hidden_layers_size = hyperparameters["hidden_layers_size"]
        self.activation = Utilities.get_activation(hyperparameters["activation"])
        self.batch_normalization = hyperparameters["batch_normalization"]
        self.dropout_rate = hyperparameters["dropout_rate"]

        # Embeddings
        region_vocab_size, region_emb_dim = embedding_sizes['region']
        station_vocab_size, station_emb_dim = embedding_sizes['station']
        cepages_vocab_size, cepages_emb_dim = embedding_sizes['cepages']

        self.region_emb = nn.Embedding(region_vocab_size, region_emb_dim)
        self.station_emb = nn.Embedding(station_vocab_size, station_emb_dim)
        self.cepages_emb = nn.Embedding(cepages_vocab_size, cepages_emb_dim)

        # Calculate total input dimension: numerical + embeddings
        num_numeric_features = hyperparameters["num_numeric_features"]
        total_input_dim = num_numeric_features + region_emb_dim + station_emb_dim + cepages_emb_dim

        # Dense layers
        layers = []
        layers.append(DenseBlock(in_size=total_input_dim,
                                 out_size=self.hidden_layers_size[0],
                                 activation=self.activation,
                                 batch_normalization=self.batch_normalization,
                                 dropout_rate=self.dropout_rate))

        for i in range(1, len(self.hidden_layers_size)):
            layers.append(DenseBlock(in_size=self.hidden_layers_size[i-1],
                                     out_size=self.hidden_layers_size[i],
                                     activation=self.activation,
                                     batch_normalization=self.batch_normalization,
                                     dropout_rate=self.dropout_rate))

        layers.append(nn.Linear(self.hidden_layers_size[-1], hyperparameters["output_dim"]))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x_numeric, x_cat):
        x_emb = torch.cat([
            self.region_emb(x_cat[:, 0]),
            self.station_emb(x_cat[:, 1]),
            self.cepages_emb(x_cat[:, 2])
        ], dim=1)

        x = torch.cat([x_numeric, x_emb], dim=1)
        return self.classifier(x)
