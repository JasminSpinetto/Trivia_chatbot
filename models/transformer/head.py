import torch
from torch import nn
import math
from typing import Optional

class Classifier(nn.Module):
    """
    Simple multi-layer perceptron head for clasiffication. 
    Given the specified number of layers and classes it returns 
    the logits on which decide the final class.
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            num_classes
            ):
        super().__init__()

        layers=[]

        # Build network infrastructure
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, num_classes))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        # Applies the full linear block
        logits = self.block(x)
        return logits

class Aggregator(nn.Module):
    """
    Aggregator class to combine different embeddings into a 
    single one to feed the final classifier.
    """

    def __init__(
            self,
            use_concat,
            num_input: Optional[int] = None,
            input_size: Optional[int] = None,
            embed_dim: Optional[int] = None,
            num_heads: Optional[int] = None
            ):
        super().__init__()

        self.use_concat = use_concat

        if not use_concat:
            if any(p is None for p in [num_input, input_size, embed_dim, num_heads]):
                raise ValueError("num_input, input_size, embed_dim e num_heads must be specified if use_concat=False")

            # Adding learnable [CLS] token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, input_size))

            # Sinusoidal positional encoding
            self.register_buffer('positional_encoding', self._sinusoidal_encoding(num_input + 1, input_size))

            # Query, Key, Value
            self.query = nn.Linear(input_size, embed_dim)
            self.key = nn.Linear(input_size, embed_dim)
            self.value = nn.Linear(input_size, embed_dim)

            # Multihead attention block
            self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    @staticmethod
    def _sinusoidal_encoding(num_positions, embed_dim):
        pe = torch.zeros(num_positions, embed_dim)
        position = torch.arange(0, num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        # x: (batch_size, num_input, input_size)
        if self.use_concat:
            return x.flatten(start_dim=1)
        
        batch_size = x.size(0)

        # Add [CLS] token
        cls = self.cls_token.expand(batch_size, 1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add sinusoidal positional encoding
        x = x + self.positional_encoding

        # Project in embedding space
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Permute dimensions for Multihead attention
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        # Compute attention output
        attn_output, _ = self.attention(q, k, v)

        return attn_output[0]