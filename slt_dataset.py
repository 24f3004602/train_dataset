"""
Sign Language Translation Dataset
Loads I3D features and text translations for training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from typing import Optional, List, Dict

class SLTDataset(Dataset):
    """Dataset for Sign Language Translation using I3D features"""
    
    def __init__(
        self,
        tsv_path: str,
        feature_root: str,
        tokenizer=None,
        max_src_len: int = 512,
        max_tgt_len: int = 256,
    ):
        """
        Args:
            tsv_path: Path to TSV file with annotations
            feature_root: Root directory containing .npy feature files
            tokenizer: Tokenizer for text (if None, returns raw text)
            max_src_len: Maximum source sequence length (features)
            max_tgt_len: Maximum target sequence length (text tokens)
        """
        self.feature_root = feature_root
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        # Load TSV file
        self.data = pd.read_csv(tsv_path, sep='\t')
        print(f"Loaded {len(self.data)} samples from {tsv_path}")
        
        # Build vocabulary if no tokenizer provided
        if tokenizer is None:
            self.vocab = self._build_vocab()
        
    def _build_vocab(self) -> Dict[str, int]:
        """Build simple vocabulary from translations"""
        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        idx = 4
        
        for translation in self.data['translation']:
            if pd.isna(translation):
                continue
            for word in str(translation).lower().split():
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        
        print(f"Vocabulary size: {len(vocab)}")
        return vocab
    
    def _tokenize(self, text: str) -> List[int]:
        """Simple whitespace tokenization"""
        if pd.isna(text):
            return [self.vocab['<sos>'], self.vocab['<eos>']]
        
        tokens = [self.vocab['<sos>']]
        for word in str(text).lower().split():
            tokens.append(self.vocab.get(word, self.vocab['<unk>']))
        tokens.append(self.vocab['<eos>'])
        return tokens
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load I3D features
        sample_id = row['id']
        
        # Try to find the feature file
        feature_path = os.path.join(self.feature_root, f"{sample_id}.npy")
        
        if os.path.exists(feature_path):
            features = np.load(feature_path).astype(np.float32)
        else:
            # Return zero features if file not found
            features = np.zeros((1, 1024), dtype=np.float32)
        
        # Truncate or pad features to max_src_len
        if features.shape[0] > self.max_src_len:
            features = features[:self.max_src_len]
        
        # Get translation
        translation = row['translation']
        
        if self.tokenizer:
            tokens = self.tokenizer.encode(str(translation))
        else:
            tokens = self._tokenize(translation)
        
        # Truncate tokens
        if len(tokens) > self.max_tgt_len:
            tokens = tokens[:self.max_tgt_len-1] + [self.vocab['<eos>']]
        
        return {
            'id': sample_id,
            'features': torch.from_numpy(features),
            'features_len': features.shape[0],
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'tokens_len': len(tokens),
            'translation': str(translation) if not pd.isna(translation) else "",
        }


def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    
    # Get max lengths in this batch
    max_feat_len = max(item['features_len'] for item in batch)
    max_tok_len = max(item['tokens_len'] for item in batch)
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    features = torch.zeros(batch_size, max_feat_len, 1024)
    features_mask = torch.zeros(batch_size, max_feat_len, dtype=torch.bool)
    tokens = torch.zeros(batch_size, max_tok_len, dtype=torch.long)
    tokens_mask = torch.zeros(batch_size, max_tok_len, dtype=torch.bool)
    
    ids = []
    translations = []
    
    for i, item in enumerate(batch):
        feat_len = item['features_len']
        tok_len = item['tokens_len']
        
        features[i, :feat_len] = item['features']
        features_mask[i, :feat_len] = True
        tokens[i, :tok_len] = item['tokens']
        tokens_mask[i, :tok_len] = True
        
        ids.append(item['id'])
        translations.append(item['translation'])
    
    return {
        'ids': ids,
        'features': features,
        'features_mask': features_mask,
        'tokens': tokens,
        'tokens_mask': tokens_mask,
        'translations': translations,
    }


def create_dataloaders(
    train_tsv: str,
    val_tsv: str,
    feature_root: str,
    batch_size: int = 8,
    num_workers: int = 0,
):
    """Create train and validation dataloaders"""
    
    train_dataset = SLTDataset(train_tsv, feature_root)
    val_dataset = SLTDataset(val_tsv, feature_root)
    
    # Share vocabulary
    val_dataset.vocab = train_dataset.vocab
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader, train_dataset.vocab


if __name__ == '__main__':
    # Test the dataset
    tsv_path = 'how2sign/tsv_files_how2sign/tsv_files_how2sign/cvpr23.fairseq.i3d.train.how2sign.tsv'
    feature_root = 'how2sign/i3d_features_how2sign/i3d_features_how2sign/train'
    
    dataset = SLTDataset(tsv_path, feature_root)
    
    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample ID: {sample['id']}")
    print(f"Features shape: {sample['features'].shape}")
    print(f"Tokens: {sample['tokens'][:10]}...")
    print(f"Translation: {sample['translation'][:100]}...")
