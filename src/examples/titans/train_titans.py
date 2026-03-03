#!/usr/bin/env python3
"""
Titans LLM Training Script
Trains a Titans (Learning to Memorize at Test Time) model from a text corpus.
Supports variants: MAC (Memory as Context), MAG (Memory as Gate), MAL (Memory as Layer).
"""

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter


class SimpleTokenizer:
    """Simple character-level tokenizer"""

    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.unk_token = 0
        self.pad_token = 1
        self.eos_token = 2

    def build_vocab(self, corpus_text):
        """Build vocabulary from corpus"""
        char_counts = Counter(corpus_text)
        most_common = char_counts.most_common(self.vocab_size - 3)
        self.char_to_idx = {'<UNK>': self.unk_token, '<PAD>': self.pad_token, '<EOS>': self.eos_token}
        self.idx_to_char = {self.unk_token: '<UNK>', self.pad_token: '<PAD>', self.eos_token: '<EOS>'}
        idx = 3
        for char, _ in most_common:
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
            idx += 1
        self.vocab_size = len(self.char_to_idx)

    def encode(self, text):
        return [self.char_to_idx.get(char, self.unk_token) for char in text]

    def decode(self, token_ids):
        return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in token_ids if idx != self.pad_token])

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'char_to_idx': self.char_to_idx,
                'idx_to_char': {int(k): v for k, v in self.idx_to_char.items()}
            }, f, indent=2)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        tokenizer = cls(data['vocab_size'])
        tokenizer.char_to_idx = data['char_to_idx']
        tokenizer.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
        tokenizer.vocab_size = len(tokenizer.char_to_idx)
        return tokenizer


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class NeuralMemory(nn.Module):
    """Neural memory module: maintains a persistent key-value memory updated by chunks."""

    def __init__(self, d_model, memory_depth, chunk_size, n_persistent, window_size):
        super(NeuralMemory, self).__init__()
        self.d_model = d_model
        self.memory_depth = memory_depth
        self.chunk_size = chunk_size
        self.n_persistent = n_persistent
        self.window_size = window_size
        self.memory_slots = nn.Parameter(torch.randn(1, n_persistent, d_model) * 0.02)
        self.write_proj = nn.Linear(d_model * 2, d_model)
        self.read_proj = nn.Linear(d_model * (1 + n_persistent), d_model)

    def forward(self, x):
        """
        x: (batch, seq, d_model)
        Returns: (batch, seq, d_model) - memory-enhanced representation
        """
        batch, seq_len, _ = x.size()
        memory = self.memory_slots.expand(batch, -1, -1)
        # Simple read: attend from x to memory and concatenate
        # (batch, seq, d_model) with (batch, n_persistent, d_model) -> weighted combination
        attn = torch.matmul(x, memory.transpose(-2, -1))
        attn = torch.softmax(attn / (self.d_model ** 0.5), dim=-1)
        read = torch.matmul(attn, memory)
        out = self.read_proj(torch.cat([x, read], dim=-1))
        return out


class TitansLayer(nn.Module):
    """Single Titans layer: transformer block + optional memory (MAL variant)."""

    def __init__(self, d_model, nhead, dim_feedforward, memory_module, variant, dropout=0.1):
        super(TitansLayer, self).__init__()
        self.variant = variant
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.memory = memory_module

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff))
        if self.memory is not None and self.variant == 'mal':
            x = x + self.dropout(self.memory(x))
        return x


class TitansLLM(nn.Module):
    """
    Titans LLM: transformer with neural memory.
    Variants:
      - mac: Memory as Context - memory output concatenated into attention context
      - mag: Memory as Gate - memory gates the attention output
      - mal: Memory as Layer - memory as separate layer interleaved with attention
    """

    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, dim_feedforward=512,
                 max_seq_length=512, memory_depth=2, chunk_size=64, n_persistent=16,
                 window_size=256, variant='mac', dropout=0.1):
        super(TitansLLM, self).__init__()
        self.variant = variant.lower()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        memory_module = NeuralMemory(d_model, memory_depth, chunk_size, n_persistent, window_size)

        if self.variant == 'mal':
            self.layers = nn.ModuleList([
                TitansLayer(d_model, nhead, dim_feedforward, memory_module, self.variant, dropout)
                for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                TitansLayer(d_model, nhead, dim_feedforward, None, self.variant, dropout)
                for _ in range(num_layers)
            ])

        self.memory = memory_module
        if self.variant == 'mac':
            self.memory_proj = nn.Linear(d_model * 2, d_model)
        elif self.variant == 'mag':
            self.memory_gate = nn.Linear(d_model * 2, d_model)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        mem_out = self.memory(x)

        for layer in self.layers:
            x = layer(x)
            if self.variant == 'mac':
                x = x + self.memory_proj(torch.cat([x, mem_out], dim=-1))
            elif self.variant == 'mag':
                gate = torch.sigmoid(self.memory_gate(torch.cat([x, mem_out], dim=-1)))
                x = gate * x + (1 - gate) * mem_out

        return self.fc_out(x)


class TextDataset(Dataset):
    """Dataset for text corpus"""

    def __init__(self, corpus_file, tokenizer, max_seq_length=256):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            text = f.read()
        sequences = text.split('\n')
        for seq in sequences:
            if len(seq.strip()) > 0:
                tokens = tokenizer.encode(seq)
                if len(tokens) > 0:
                    if len(tokens) > max_seq_length:
                        tokens = tokens[:max_seq_length]
                    else:
                        tokens = tokens + [tokenizer.pad_token] * (max_seq_length - len(tokens))
                    self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        input_seq = torch.tensor(tokens[:-1], dtype=torch.long)
        target_seq = torch.tensor(tokens[1:], dtype=torch.long)
        return input_seq, target_seq


def train_model(corpus_file, output_dir, epochs, batch_size, learning_rate, d_model, nhead,
                num_layers, max_seq_length, vocab_size, variant, memory_depth, chunk_size,
                n_persistent, window_size):
    dim_feedforward = d_model * 2
    print(f"Loading corpus from {corpus_file}...")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus_text = f.read()
    print(f"Corpus size: {len(corpus_text)} characters")
    print("Building tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size)
    tokenizer.build_vocab(corpus_text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, 'tokenizer.json'))

    print("Creating dataset...")
    dataset = TextDataset(corpus_file, tokenizer, max_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)} samples, variant={variant}")

    model = TitansLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_length=max_seq_length,
        memory_depth=memory_depth,
        chunk_size=chunk_size,
        n_persistent=n_persistent,
        window_size=window_size,
        variant=variant,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token)

    print(f"\nStarting training for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for input_seq, target_seq in progress_bar:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            optimizer.zero_grad()
            output = model(input_seq)
            output = output.reshape(-1, output.size(-1))
            target = target_seq.reshape(-1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")

    model_path = os.path.join(output_dir, 'titans_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    model_config = {
        'vocab_size': tokenizer.vocab_size,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'max_seq_length': max_seq_length,
        'variant': variant,
        'memory_depth': memory_depth,
        'chunk_size': chunk_size,
        'n_persistent': n_persistent,
        'window_size': window_size,
    }
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"Config saved to {config_path}")
    print("Training completed successfully!")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Train Titans LLM')
    parser.add_argument('--corpus-file', required=True, help='Path to corpus text file')
    parser.add_argument('--output-dir', required=True, help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=4e-4, help='Learning rate')
    parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--max-seq-length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--vocab-size', type=int, default=256, help='Vocabulary size')
    parser.add_argument('--variant', type=str, default='mac', choices=['mac', 'mag', 'mal'],
                        help='Titans variant: mac (Memory as Context), mag (Memory as Gate), mal (Memory as Layer)')
    parser.add_argument('--memory-depth', type=int, default=2, help='Memory depth')
    parser.add_argument('--chunk-size', type=int, default=64, help='Chunk size for memory')
    parser.add_argument('--n-persistent', type=int, default=16, help='Number of persistent memory slots')
    parser.add_argument('--window-size', type=int, default=256, help='Memory window size')

    args = parser.parse_args()

    try:
        train_model(
            args.corpus_file,
            args.output_dir,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.d_model,
            args.nhead,
            args.num_layers,
            args.max_seq_length,
            args.vocab_size,
            args.variant,
            args.memory_depth,
            args.chunk_size,
            args.n_persistent,
            args.window_size,
        )
        sys.exit(0)
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
