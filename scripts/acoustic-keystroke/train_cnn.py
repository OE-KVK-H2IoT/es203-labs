#!/usr/bin/env python3
"""Train CNN keystroke classifier using PyTorch with GPU support.

Usage:
    python train_cnn.py                        # default keystroke_data/
    python train_cnn.py path/to/data_dir       # specify data directory
    python train_cnn.py --epochs 100           # more training epochs
    python train_cnn.py --export model.onnx    # export to ONNX after training

Automatically uses GPU (CUDA) if available, otherwise falls back to CPU.
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data

from features import RATE, detect_onsets, extract_keystroke_segment


# ── CNN Architecture ────────────────────────────────────────────────

class KeystrokeCNN(nn.Module):
    """Small CNN for mel spectrogram classification.

    Input: (batch, 1, n_mels, n_frames) — single-channel "image"
    Output: (batch, n_classes) — logits per class
    """
    def __init__(self, n_classes, n_mels=32, n_frames=50):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1 → 16 channels
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (16, n_mels//2, n_frames//2)

            # Block 2: 16 → 32 channels
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32, n_mels//4, n_frames//4)
        )

        flat_size = 32 * (n_mels // 4) * (n_frames // 4)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ── Dataset Builder ─────────────────────────────────────────────────

def build_mel_dataset(data_dir):
    """Load keystroke data and extract mel spectrograms.

    Returns:
        X: numpy array (n_samples, n_mels, n_frames)
        y: numpy array of integer labels
        label_names: list of key names
    """
    data_dir = Path(data_dir)
    X, y, label_names = [], [], []

    # Find all key_*.npy files (per-key collection format)
    key_files = sorted(data_dir.glob("key_*.npy"))
    if not key_files:
        print(f"No key_*.npy files found in {data_dir}")
        sys.exit(1)

    for key_file in key_files:
        key_name = key_file.stem.replace("key_", "")
        audio = np.load(key_file)

        # Detect keystrokes in this recording
        onsets = detect_onsets(audio, RATE)
        if len(onsets) == 0:
            print(f"  {key_name}: no onsets found, skipping")
            continue

        if key_name not in label_names:
            label_names.append(key_name)
        label_idx = label_names.index(key_name)

        for onset in onsets:
            segment = extract_keystroke_segment(audio, onset, RATE)
            mel = _compute_mel_spectrogram(segment, RATE)
            if mel is not None:
                X.append(mel)
                y.append(label_idx)

        print(f"  {key_name}: {sum(1 for yi in y if yi == label_idx)} samples")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"Total: {len(X)} samples, {len(label_names)} classes")
    return X, y, label_names


def _compute_mel_spectrogram(segment, rate, n_mels=32, n_fft=1024,
                              hop_length=256, n_frames=50):
    """Compute mel spectrogram suitable for CNN input."""
    try:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=segment.astype(np.float32), sr=rate,
            n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
    except ImportError:
        # Fallback without librosa: use scipy
        from scipy.signal import spectrogram as scipy_spectrogram
        _, _, Sxx = scipy_spectrogram(segment, fs=rate, nperseg=n_fft,
                                       noverlap=n_fft - hop_length)
        # Simple mel approximation: take first n_mels frequency bins
        mel_db = 10 * np.log10(Sxx[:n_mels] + 1e-10)

    # Pad or truncate to fixed n_frames
    if mel_db.shape[1] < n_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, n_frames - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :n_frames]

    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-10)
    return mel_db


# ── Training ────────────────────────────────────────────────────────

def train(X, y, n_classes, epochs=50, lr=0.001, batch_size=32):
    """Train CNN with automatic GPU detection."""

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("  (no GPU detected — using CPU)")
        print("  Tip: install PyTorch with CUDA support for faster training:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu124")

    # Convert to tensors
    X_tensor = torch.FloatTensor(X).unsqueeze(1)  # (N, 1, n_mels, n_frames)
    y_tensor = torch.LongTensor(y)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

    # 80/20 split
    n_val = max(1, len(dataset) // 5)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        pin_memory=(device.type == 'cuda'))  # faster GPU transfer
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size,
        pin_memory=(device.type == 'cuda'))

    # Model, optimizer, loss
    model = KeystrokeCNN(n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    print(f"  Training samples: {n_train}, validation: {n_val}")
    print()

    t_start = time.time()
    best_acc = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                _, predicted = torch.max(output, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_acc = correct / total if total > 0 else 0
        best_acc = max(best_acc, val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"loss={train_loss/len(train_loader):.4f}  "
                  f"val_acc={val_acc:.1%}  "
                  f"[{elapsed:.1f}s]")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s — best validation accuracy: {best_acc:.1%}")

    # Move model to CPU for saving/export (works everywhere)
    model = model.cpu()
    return model


# ── Export ──────────────────────────────────────────────────────────

def export_onnx(model, n_mels=32, n_frames=50, path="keystroke_cnn.onnx"):
    """Export trained model to ONNX format for deployment."""
    model.eval()
    dummy = torch.randn(1, 1, n_mels, n_frames)
    torch.onnx.export(model, dummy, path,
                      input_names=['spectrogram'],
                      output_names=['logits'],
                      dynamic_axes={'spectrogram': {0: 'batch'},
                                    'logits': {0: 'batch'}})
    print(f"Exported to {path}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train CNN keystroke classifier (GPU-accelerated)")
    parser.add_argument('data_dir', nargs='?', default='keystroke_data',
                        help='Directory with key_*.npy files')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--export', type=str, default=None,
                        help='Export to ONNX file after training')
    parser.add_argument('--save', type=str, default='keystroke_cnn.pt',
                        help='Save PyTorch model (default: keystroke_cnn.pt)')
    args = parser.parse_args()

    print("=== CNN Keystroke Classifier ===")
    print(f"Data: {args.data_dir}")

    # Load data
    X, y, label_names = build_mel_dataset(args.data_dir)
    if len(X) == 0:
        print("No data loaded. Check your data directory.")
        sys.exit(1)

    # Train
    model = train(X, y, len(label_names),
                  epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    # Save
    torch.save({
        'model_state': model.state_dict(),
        'label_names': label_names,
        'n_classes': len(label_names),
    }, args.save)
    print(f"Saved model to {args.save}")

    # Export ONNX if requested
    if args.export:
        export_onnx(model, path=args.export)


if __name__ == '__main__':
    main()
