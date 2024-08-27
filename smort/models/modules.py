from typing import Dict, OrderedDict

import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False) -> None:
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


# From https://github.com/Mathux/TMR/blob/master/src/model/actor.py
class ACTORStyleEncoder(nn.Module):
    # Similar to ACTOR but "action agnostic" and more general
    def __init__(
        self,
        nfeats: int,
        vae: bool,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.nfeats = nfeats
        self.projection = nn.Linear(nfeats, latent_dim)

        self.vae = vae
        self.nbtokens = 2 if vae else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout=dropout, batch_first=True
        )

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )

    def forward(self, x_dict: Dict) -> Tensor:
        x = x_dict["x"]
        mask = x_dict["mask"]

        x = self.projection(x)

        device = x.device
        bs = len(x)

        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        xseq = torch.cat((tokens, x), 1)

        token_mask = torch.ones((bs, self.nbtokens), dtype=bool, device=device)  # type: ignore
        # import pdb; pdb.set_trace()
        aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        return final[:, : self.nbtokens]


class ACTORStyleDecoder(nn.Module):
    # Similar to ACTOR Decoder

    def __init__(
        self,
        nfeats: int,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        output_feats = nfeats
        self.nfeats = nfeats

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout, batch_first=True
        )

        seq_trans_decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.seqTransDecoder = nn.TransformerDecoder(
            seq_trans_decoder_layer, num_layers=num_layers
        )

        self.final_layer = nn.Linear(latent_dim, output_feats)

    def forward(self, z_dict: Dict) -> Tensor:
        z = z_dict["z"]
        mask = z_dict["mask"]

        latent_dim = z.shape[1]
        bs, nframes = mask.shape

        z = z[:, None]  # sequence of 1 element for the memory

        # Construct time queries
        time_queries = torch.zeros(bs, nframes, latent_dim, device=z.device)
        time_queries = self.sequence_pos_encoding(time_queries)

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(
            tgt=time_queries, memory=z, tgt_key_padding_mask=~mask
        )

        output = self.final_layer(output)
        # import pdb; pdb.set_trace()
        # zero for padded area
        output[~mask.unsqueeze(-1).expand_as(output)] = 0
        return output


class ACTORStyleEncoderWithCA(ACTORStyleEncoder):
    def __init__(
        self,
        nfeats: int,
        n_context_feats: int,
        vae: bool,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__(
            nfeats,
            vae,
            latent_dim,
            ff_size,
            num_layers,
            num_heads,
            dropout,
            activation,
        )
        self.context_linear = nn.Linear(n_context_feats, n_context_feats)
        self.context_linear = nn.Linear(n_context_feats, n_context_feats)
        self.context_projection = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Linear(n_context_feats, n_context_feats)),
                    ("gelu1", nn.GELU()),
                    ("conv2", nn.Linear(n_context_feats, latent_dim)),
                ]
            )
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x_dict: Dict, context_dict: Dict) -> Tensor:
        x = x_dict["x"]
        mask = x_dict["mask"]

        # Project input features to the latent dimension
        x = self.projection(x)

        device = x.device
        bs = len(x)

        # Create tokens for VAE
        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        xseq = torch.cat((tokens, x), 1)

        token_mask = torch.ones((bs, self.nbtokens), dtype=bool, device=device)  # type: ignore
        aug_mask = torch.cat((token_mask, mask), 1)

        # Add positional encoding
        xseq = self.sequence_pos_encoding(xseq)

        # Cross-attention with context
        context = context_dict["x"]
        context_mask = context_dict["mask"]

        # Project context features to latent dimension
        context = self.context_projection(self.context_linear(context))
        context = self.sequence_pos_encoding(context)

        # Perform cross-attention
        attn_output, _ = self.cross_attention(
            query=xseq, key=context, value=context, key_padding_mask=~context_mask
        )

        # Pass through transformer encoder
        final = self.seqTransEncoder(attn_output, src_key_padding_mask=~aug_mask)

        return final[:, : self.nbtokens]
