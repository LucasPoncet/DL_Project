import torch
import torch.nn as nn 
import numpy as np
import math

class DenseBlock(nn.Module):

    def __init__(self, in_size,out_size,activation,
                 batch_normalization=False,dropout_rate=0.1):
        super(DenseBlock,self).__init__()
        
        self.linear_layer = nn.Linear(in_size, out_size)
        self.activation = activation

        if batch_normalization:
            self.batch_norm_layer = nn.BatchNorm1d(out_size)
        else :
            self.batch_norm_layer = None
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear_layer(x)
        if self.batch_norm_layer is not None :
            x = self.batch_norm_layer(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        return x
    

class FlattenDenseBlock(nn.Module):
    def __init__(self, in_size, out_size, activation,
                 batch_normalization=False,dropout_rate=0.1):
        super(FlattenDenseBlock,self).__init__()
        in_size_flatten = int(np.prod(in_size))
        self.flatten_layer = nn.Flatten()
        self.dense_layer = DenseBlock(in_size=in_size_flatten,
                                      out_size=out_size,
                                      activation=activation,
                                      batch_normalization=batch_normalization,
                                      dropout_rate=dropout_rate)
    def forward(self,x):
        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        return x
    
class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 activation, batch_normalization=False,
                 dropout_rate=0.1):
        super(Conv2DBlock,self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size, 
                                    padding="same")
        self.activation = activation
        if batch_normalization:
            self.batch_norm_layer = nn.BatchNorm2d(out_channels)
        else :
            self.batch_norm_layer = None
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self,x):
        x = self.conv_layer(x)
        if self.batch_norm_layer is not None :
            x = self.batch_norm_layer(x)
        if self.activation is not None :
            x = self.activation(x)
        x = self.dropout_layer(x)
        return x
    

class BasicResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, activation,
                 batch_normalization = False, dropout_rate = 0.1):
        super(BasicResNetBlock,self).__init__()

        self.conv_layer1= Conv2DBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        activation=activation,
                                        batch_normalization=batch_normalization,
                                        dropout_rate=dropout_rate)
        self.conv_layer2 = Conv2DBlock(in_channels=out_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        activation=activation,
                                        batch_normalization=batch_normalization,
                                        dropout_rate=dropout_rate)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = Conv2DBlock(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            activation=activation,
                                            batch_normalization=batch_normalization,
                                            dropout_rate=dropout_rate)
        
    def forward(self,x):
        residual = x
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        residual = self.shortcut(residual)
        x += residual
        return x
    
class UnflattenDenseBlock(nn.Module):

    def __init__(self, in_size, out_size, activation,
                 batch_normalization=False, dropout_rate=0.1):
        super(UnflattenDenseBlock,self).__init__()
        self.dense_layer = DenseBlock(in_size=in_size,
                                      out_size=np.prod(out_size),
                                      activation=activation,
                                      batch_normalization=batch_normalization,
                                      dropout_rate=dropout_rate)
        self.unflatten_layer = nn.Unflatten(dim=1, unflattened_size=out_size)

    def forward(self, x):
        x = self.dense_layer(x)
        x = self.unflatten_layer(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, num_embeddings, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(num_embeddings, d_model)
        position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register positional embedding as non-trainable parameters
        self.register_buffer('pe',pe)

    def forward(self, x):
        x = x + self.pe 
        x = self.dropout(x)
        return x
    
    def plot_positional_embedding(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.imshow(self.pe.T, aspect='auto', cmap='viridis') # Transpose for better visualization
        plt.colorbar()
        plt.title('Positional Embedding Visualization')
        plt.xlabel('Position Index')
        plt.ylabel('Embedding Dimension')
        plt.show()
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, expansion_factor: int = 2, 
                 activation = nn.ReLU(),dropout_rate = 0.0):
        super().__init__()

        self.mha_layer = nn.MultiheadAttention(embed_dim = input_dim, 
                                               num_heads=num_heads, 
                                               dropout=dropout_rate)
        self.attention_weights = None

        self.norm_layer1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        hidden_dim = input_dim * expansion_factor
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 activation,
                                 nn.Linear(hidden_dim, input_dim))
        self.norm_layer2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attention_out, self.attention_weights = self.mha_layer(x, x, x) # (B, S, D) -> (B, S, D)
        #Add and Norm
        attention_out = attention_out + x 
        out = self.norm_layer1(attention_out) # (B, S, D)
        out = self.dropout1(out) # (B, S, D)
        #MLP part
        ff_out = self.mlp(out) # (B, S, D)
        out = ff_out + out
        out = self.norm_layer2(out) # (B, S, D)
        out = self.dropout2(out) # (B, S, D)
        return out
    