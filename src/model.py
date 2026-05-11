import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class EventTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int = 0,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.max_len = max_len

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_idx,
        )

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids) * (self.d_model ** 0.5)
        x = self.pos_encoder(x)

        src_key_padding_mask = attention_mask == 0

        encoded = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )

        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(encoded.size(0), device=encoded.device)

        last_hidden = encoded[batch_idx, lengths]
        logits = self.classifier(last_hidden)

        return logits


def build_transformer_from_config(vocab_size: int, config: dict):
    return EventTransformer(
        vocab_size=vocab_size,
        pad_idx=config.get("pad_idx", 0),
        d_model=config.get("d_model", 128),
        nhead=config.get("nhead", 4),
        num_layers=config.get("num_layers", 2),
        dim_feedforward=config.get("dim_feedforward", 256),
        dropout=config.get("dropout", 0.1),
        max_len=config.get("max_len", 64),
    )
