"""
Training script for Sign Language Translation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import re
from contextlib import nullcontext

from slt_dataset import SLTDataset, collate_fn, create_dataloaders
from slt_model import create_model


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        vocab,
        device,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_epochs=100,
        save_dir='checkpoints',
        keep_last_n=3,
        save_optimizer_state=False,
        label_smoothing=0.1,
        warmup_steps=500,
        use_amp=True,
        log_interval=50,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.device = device
        self.max_epochs = max_epochs
        self.save_dir = save_dir
        self.keep_last_n = keep_last_n
        self.save_optimizer_state = save_optimizer_state
        self.base_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.use_amp = use_amp and device.type == 'cuda'
        self.log_interval = log_interval
        
        # Create inverse vocab for decoding
        self.inv_vocab = {v: k for k, v in vocab.items()}
        
        # Loss function (ignore padding)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocab['<pad>'],
            label_smoothing=label_smoothing,
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=1e-6,
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training state
        self.best_val_loss = float('inf')
        self.global_step = 0

    def _safe_save(self, obj, path):
        """Safely save checkpoint via temp file + atomic replace."""
        tmp_path = path + '.tmp'
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
            return True
        except Exception as exc:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            print(f"WARNING: Failed to save checkpoint at {path}: {exc}")
            return False

    def _cleanup_old_checkpoints(self):
        """Keep only the latest N epoch checkpoints to avoid disk exhaustion."""
        if self.keep_last_n is None or self.keep_last_n <= 0:
            return

        epoch_files = []
        for filename in os.listdir(self.save_dir):
            match = re.match(r'checkpoint_epoch_(\d+)\.pt$', filename)
            if match:
                epoch_files.append((int(match.group(1)), filename))

        epoch_files.sort(key=lambda x: x[0])
        while len(epoch_files) > self.keep_last_n:
            _, old_file = epoch_files.pop(0)
            old_path = os.path.join(self.save_dir, old_file)
            try:
                os.remove(old_path)
                print(f"Removed old checkpoint: {old_path}")
            except OSError as exc:
                print(f"WARNING: Failed to remove old checkpoint {old_path}: {exc}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            features = batch['features'].to(self.device)
            features_mask = batch['features_mask'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            tokens_mask = batch['tokens_mask'].to(self.device)
            
            # Shift tokens for teacher forcing
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            input_mask = tokens_mask[:, :-1]
            
            # Forward pass
            self.optimizer.zero_grad()
            amp_ctx = torch.cuda.amp.autocast() if self.use_amp else nullcontext()
            with amp_ctx:
                logits = self.model(features, input_tokens, features_mask, input_mask)

                # Compute loss
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens.reshape(-1),
                )
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.warmup_steps > 0 and self.global_step < self.warmup_steps:
                warmup_ratio = float(self.global_step + 1) / float(self.warmup_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.base_lr * warmup_ratio
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            features = batch['features'].to(self.device)
            features_mask = batch['features_mask'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            tokens_mask = batch['tokens_mask'].to(self.device)
            
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            input_mask = tokens_mask[:, :-1]
            
            amp_ctx = torch.cuda.amp.autocast() if self.use_amp else nullcontext()
            with amp_ctx:
                logits = self.model(features, input_tokens, features_mask, input_mask)

                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens.reshape(-1),
                )
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def decode_tokens(self, tokens):
        """Decode token ids to text"""
        words = []
        for tok in tokens:
            if tok == self.vocab['<eos>']:
                break
            if tok not in [self.vocab['<pad>'], self.vocab['<sos>']]:
                words.append(self.inv_vocab.get(tok, '<unk>'))
        return ' '.join(words)
    
    @torch.no_grad()
    def generate_samples(self, num_samples=3):
        """Generate sample translations"""
        self.model.eval()
        
        batch = next(iter(self.val_loader))
        features = batch['features'][:num_samples].to(self.device)
        features_mask = batch['features_mask'][:num_samples].to(self.device)
        translations = batch['translations'][:num_samples]
        
        generated = self.model.generate(
            features,
            features_mask,
            max_len=100,
            sos_token=self.vocab['<sos>'],
            eos_token=self.vocab['<eos>'],
        )
        
        print("\n--- Sample Generations ---")
        for i in range(num_samples):
            gen_text = self.decode_tokens(generated[i].cpu().tolist())
            print(f"Target: {translations[i][:100]}...")
            print(f"Generated: {gen_text[:100]}...")
            print()
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'vocab': self.vocab,
        }

        if self.save_optimizer_state:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        saved = self._safe_save(checkpoint, path)
        if saved:
            self._cleanup_old_checkpoints()
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            best_saved = self._safe_save(checkpoint, best_path)
            if best_saved:
                print(f"Saved best model with val_loss: {val_loss:.4f}")
    
    def train(self):
        """Full training loop"""
        print(f"Starting training for {self.max_epochs} epochs")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")

        train_total = getattr(self.train_loader.dataset, 'total_samples', len(self.train_loader.dataset))
        train_usable = getattr(self.train_loader.dataset, 'usable_samples', len(self.train_loader.dataset))
        train_missing = getattr(self.train_loader.dataset, 'missing_samples', max(train_total - train_usable, 0))

        val_total = getattr(self.val_loader.dataset, 'total_samples', len(self.val_loader.dataset))
        val_usable = getattr(self.val_loader.dataset, 'usable_samples', len(self.val_loader.dataset))
        val_missing = getattr(self.val_loader.dataset, 'missing_samples', max(val_total - val_usable, 0))

        print(
            f"Data quality (train): total={train_total}, usable={train_usable}, missing={train_missing}"
        )
        print(
            f"Data quality (val): total={val_total}, usable={val_usable}, missing={val_missing}"
        )
        
        for epoch in range(1, self.max_epochs + 1):
            print(
                f"Epoch {epoch} data status -> "
                f"train usable/missing: {train_usable}/{train_missing}, "
                f"val usable/missing: {val_usable}/{val_missing}"
            )
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            self.scheduler.step()
            
            print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Generate samples every 5 epochs
            if epoch % 5 == 0:
                self.generate_samples()
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
        
        print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train Sign Language Translation Model')
    
    # Data arguments
    parser.add_argument('--train_tsv', type=str, 
                        default='how2sign/tsv_files_how2sign/tsv_files_how2sign/cvpr23.fairseq.i3d.train.how2sign.tsv')
    parser.add_argument('--val_tsv', type=str,
                        default='how2sign/tsv_files_how2sign/tsv_files_how2sign/cvpr23.fairseq.i3d.val.how2sign.tsv')
    parser.add_argument('--feature_root', type=str,
                        default='how2sign/i3d_features_how2sign/i3d_features_how2sign')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='transformer', choices=['transformer', 'cnn'])
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_encoder_layers', type=int, default=4)
    parser.add_argument('--n_decoder_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--keep_last_n', type=int, default=3,
                        help='Keep only the most recent N epoch checkpoints (0 disables cleanup)')
    parser.add_argument('--save_optimizer_state', action='store_true',
                        help='Include optimizer/scheduler state in checkpoints (larger files)')
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision (AMP)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num_workers', type=int, default=0)
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading datasets...")
    
    # Custom feature paths for train/val
    train_feature_root = os.path.join(args.feature_root, 'train')
    val_feature_root = os.path.join(args.feature_root, 'val')
    
    train_dataset = SLTDataset(args.train_tsv, train_feature_root)
    val_dataset = SLTDataset(args.val_tsv, val_feature_root)
    val_dataset.vocab = train_dataset.vocab  # Share vocabulary
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    # Create model
    print("Creating model...")
    vocab_size = len(train_dataset.vocab)
    model = create_model(
        vocab_size=vocab_size,
        model_type=args.model_type,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=train_dataset.vocab,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        save_dir=args.save_dir,
        keep_last_n=args.keep_last_n,
        save_optimizer_state=args.save_optimizer_state,
        label_smoothing=args.label_smoothing,
        warmup_steps=args.warmup_steps,
        use_amp=not args.no_amp,
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
