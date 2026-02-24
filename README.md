# Sign Language Translation Training Pipeline

This repository contains the training pipeline for sign language to text translation using I3D features and Transformer-based architecture.

## Project Structure

```
├── feature_extractor.py      # Extract I3D features from videos
├── slt_dataset.py            # PyTorch Dataset for sign language data
├── slt_model.py              # Transformer model architecture
├── train_slt.py              # Training script
├── google_colab_setup.ipynb  # Google Colab setup for GPU training
├── requirements.txt          # Python dependencies
└── pytorch-i3d/              # I3D feature extraction model
```

## Features

- **I3D Feature Extraction**: Extracts spatial-temporal features from sign language videos
- **Transformer Architecture**: Encoder-decoder model for sequence-to-sequence translation
- **How2Sign Compatible**: Outputs features in (T, 1024) format matching How2Sign dataset
- **GPU Training**: Optimized for Google Colab with GPU acceleration

## Quick Start

### Local Feature Extraction

```bash
# Install dependencies
pip install -r requirements.txt

# Extract features from videos
python feature_extractor.py
```

### Google Colab Training

1. Upload this repository to Google Drive under `MyDrive/training_dataset/`
2. Upload your dataset files:
   - `how2sign/` folder with TSV files and features
   - or your custom dataset in the same format
3. Open `google_colab_setup.ipynb` in Google Colab
4. Select **GPU runtime** (Runtime → Change runtime type → GPU)
5. Run all cells sequentially

## Dataset Format

### TSV Format
```
id	translation	signs_file
video_001	Hello world	train/video_001.npy
video_002	How are you	train/video_002.npy
```

### Feature Format
- NumPy arrays with shape `(T, 1024)` where T = number of frames
- Float32 dtype
- Stored as `.npy` files

## Model Architecture

- **Input**: I3D features (T × 1024)
- **Encoder**: 4-layer Transformer (d_model=512, 8 heads)
- **Decoder**: 4-layer Transformer
- **Output**: Text sequence (vocabulary-based)

## Training Configuration

```python
model_config = {
    'd_model': 512,
    'num_encoder_layers': 4,
    'num_decoder_layers': 4,
    'num_heads': 8,
    'dim_feedforward': 1024,
    'dropout': 0.1
}

training_config = {
    'batch_size': 8,
    'learning_rate': 1e-4,
    'num_epochs': 20,
    'max_src_len': 512,
    'max_tgt_len': 128
}
```

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
opencv-python>=4.7.0
```

## Notes

⚠️ **Large Files Excluded**: Dataset files, model checkpoints, and feature files are not included in this repository due to size constraints. Upload them separately to Google Drive for Colab training.

## GPU Training Performance

- CPU: ~30 hours per epoch (not recommended)
- Google Colab GPU: ~2-3 hours per epoch (recommended)

## Citation

This project uses the I3D model from:
- Carreira & Zisserman. "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset." CVPR 2017.

## License

See LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact: 24f3004602@ds.study.iitm.ac.in
