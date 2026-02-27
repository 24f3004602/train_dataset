"""
Sign Language Translation Model
Transformer-based encoder-decoder architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        pe_slice = self.pe.narrow(1, 0, x.shape[1])
        x = x + pe_slice
        return self.dropout(x)


class SignLanguageTranslator(nn.Module):
    """Transformer model for Sign Language Translation"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        feature_dim: int = 1024,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Feature projection (I3D features -> model dimension)
        self.feature_proj = nn.Linear(feature_dim, d_model)
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encodings
        self.encoder_pos = PositionalEncoding(d_model, max_len, dropout)
        self.decoder_pos = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(
        self,
        features: torch.Tensor,
        features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode sign language features"""
        x = self.feature_proj(features)
        x = self.encoder_pos(x)
        
        # Create padding mask (True = ignore)
        if features_mask is not None:
            src_key_padding_mask = ~features_mask
        else:
            src_key_padding_mask = None
        
        memory = self.transformer.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )
        return memory
    
    def decode(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        tokens_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode to text tokens"""
        x = self.token_embedding(tokens) * math.sqrt(self.d_model)
        x = self.decoder_pos(x)
        
        # Create causal mask
        tgt_len = tokens.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(tokens.device)
        
        # Padding masks
        if tokens_mask is not None:
            tgt_key_padding_mask = ~tokens_mask
        else:
            tgt_key_padding_mask = None
        
        if memory_mask is not None:
            memory_key_padding_mask = ~memory_mask
        else:
            memory_key_padding_mask = None
        
        output = self.transformer.decoder(
            x,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        
        logits = self.output_proj(output)
        return logits
    
    def forward(
        self,
        features: torch.Tensor,
        tokens: torch.Tensor,
        features_mask: Optional[torch.Tensor] = None,
        tokens_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            features: (B, T_src, 1024) I3D features
            tokens: (B, T_tgt) target tokens
            features_mask: (B, T_src) True = valid
            tokens_mask: (B, T_tgt) True = valid
        
        Returns:
            logits: (B, T_tgt, vocab_size)
        """
        memory = self.encode(features, features_mask)
        logits = self.decode(tokens, memory, tokens_mask, features_mask)
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        features: torch.Tensor,
        features_mask: Optional[torch.Tensor] = None,
        max_len: int = 100,
        sos_token: int = 1,
        eos_token: int = 2,
    ) -> torch.Tensor:
        """Generate translation autoregressively"""
        self.eval()
        
        batch_size = features.size(0)
        device = features.device
        
        # Encode
        memory = self.encode(features, features_mask)
        
        # Start with SOS token
        tokens = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            logits = self.decode(tokens, memory, memory_mask=features_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Stop if all sequences have EOS
            if (next_token == eos_token).all():
                break
        
        return tokens


class CNNSeq2SeqTranslator(nn.Module):
    """CNN encoder + GRU decoder model for Sign Language Translation"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_encoder_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 512,
        feature_dim: int = 1024,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        self.feature_proj = nn.Linear(feature_dim, d_model)

        conv_layers = []
        for _ in range(n_encoder_layers):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.cnn_encoder = nn.ModuleList(conv_layers)

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_gru = nn.GRU(
            input_size=d_model * 2,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        self.init_hidden = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        features: torch.Tensor,
        features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.feature_proj(features)  # (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)

        for layer in self.cnn_encoder:
            x = x + layer(x)

        x = x.transpose(1, 2)  # (B, T, D)

        if features_mask is not None:
            x = x * features_mask.unsqueeze(-1).float()

        return x

    def _context_vector(
        self,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if memory_mask is None:
            return memory.mean(dim=1)

        mask = memory_mask.unsqueeze(-1).float()
        masked_sum = (memory * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return masked_sum / denom

    def decode(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        tokens_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del tokens_mask  # Unused in GRU decode path

        context = self._context_vector(memory, memory_mask)  # (B, D)
        emb = self.token_embedding(tokens) * math.sqrt(self.d_model)  # (B, T, D)

        context_seq = context.unsqueeze(1).expand(-1, emb.size(1), -1)  # (B, T, D)
        gru_in = torch.cat([emb, context_seq], dim=-1)  # (B, T, 2D)

        h0 = torch.tanh(self.init_hidden(context)).unsqueeze(0)  # (1, B, D)
        output, _ = self.decoder_gru(gru_in, h0)

        logits = self.output_proj(output)
        return logits

    def forward(
        self,
        features: torch.Tensor,
        tokens: torch.Tensor,
        features_mask: Optional[torch.Tensor] = None,
        tokens_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encode(features, features_mask)
        return self.decode(tokens, memory, tokens_mask, features_mask)

    @torch.no_grad()
    def generate(
        self,
        features: torch.Tensor,
        features_mask: Optional[torch.Tensor] = None,
        max_len: int = 100,
        sos_token: int = 1,
        eos_token: int = 2,
    ) -> torch.Tensor:
        self.eval()

        memory = self.encode(features, features_mask)
        context = self._context_vector(memory, features_mask)

        batch_size = features.size(0)
        device = features.device

        tokens = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)
        hidden = torch.tanh(self.init_hidden(context)).unsqueeze(0)

        for _ in range(max_len - 1):
            last_tok = tokens[:, -1:]
            emb = self.token_embedding(last_tok) * math.sqrt(self.d_model)  # (B, 1, D)
            step_in = torch.cat([emb, context.unsqueeze(1)], dim=-1)  # (B, 1, 2D)

            out, hidden = self.decoder_gru(step_in, hidden)
            logits = self.output_proj(out)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok == eos_token).all():
                break

        return tokens


def create_model(vocab_size: int, **kwargs) -> nn.Module:
    """Create model with default configuration"""
    default_config = {
        'model_type': 'transformer',
        'd_model': 512,
        'n_encoder_layers': 4,
        'n_decoder_layers': 4,
        'n_heads': 8,
        'd_ff': 1024,
        'dropout': 0.1,
        'feature_dim': 1024,
    }
    default_config.update(kwargs)

    model_type = default_config.pop('model_type')

    if model_type == 'cnn':
        model = CNNSeq2SeqTranslator(
            vocab_size=vocab_size,
            d_model=default_config.get('d_model', 512),
            n_encoder_layers=default_config.get('n_encoder_layers', 4),
            dropout=default_config.get('dropout', 0.1),
            feature_dim=default_config.get('feature_dim', 1024),
        )
    else:
        model = SignLanguageTranslator(vocab_size, **default_config)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model


if __name__ == '__main__':
    # Test the model
    vocab_size = 10000
    batch_size = 4
    src_len = 32
    tgt_len = 20
    
    model = create_model(vocab_size, model_type='transformer')
    
    # Dummy inputs
    features = torch.randn(batch_size, src_len, 1024)
    features_mask = torch.ones(batch_size, src_len, dtype=torch.bool)
    tokens = torch.randint(0, vocab_size, (batch_size, tgt_len))
    tokens_mask = torch.ones(batch_size, tgt_len, dtype=torch.bool)
    
    # Forward pass
    logits = model(features, tokens, features_mask, tokens_mask)
    print(f"Output shape: {logits.shape}")  # (batch_size, tgt_len, vocab_size)
    
    # Test generation
    print('Generation test skipped in static test block')
