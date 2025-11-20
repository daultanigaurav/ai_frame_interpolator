"""
Custom AI Frame Interpolation Model Training Script.
Trains a deep learning model for frame interpolation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from pathlib import Path
import logging
from typing import Tuple, List
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameInterpolationDataset(Dataset):
    """Dataset for frame interpolation training."""
    
    def __init__(self, data_dir: str, transform=None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing training data
            transform: Optional transforms
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Tuple[str, str]]:
        """Load sample pairs from data directory."""
        samples = []
        # Look for frame pairs in the directory
        frame_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Create pairs of consecutive frames
        for i in range(len(frame_files) - 1):
            frame1_path = os.path.join(self.data_dir, frame_files[i])
            frame2_path = os.path.join(self.data_dir, frame_files[i + 1])
            samples.append((frame1_path, frame2_path))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        frame1_path, frame2_path = self.samples[idx]
        
        # Load frames
        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)
        
        if frame1 is None or frame2 is None:
            raise ValueError(f"Failed to load frames: {frame1_path}, {frame2_path}")
        
        # Resize to standard size
        frame1 = cv2.resize(frame1, (256, 256))
        frame2 = cv2.resize(frame2, (256, 256))
        
        # Convert to RGB and normalize
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        frame1 = frame1.astype(np.float32) / 255.0
        frame2 = frame2.astype(np.float32) / 255.0
        
        # Create ground truth (middle frame - simple linear interpolation)
        alpha = 0.5
        ground_truth = (1 - alpha) * frame1 + alpha * frame2
        
        # Convert to tensors
        frame1_tensor = torch.from_numpy(frame1).permute(2, 0, 1).float()
        frame2_tensor = torch.from_numpy(frame2).permute(2, 0, 1).float()
        gt_tensor = torch.from_numpy(ground_truth).permute(2, 0, 1).float()
        
        # Concatenate input frames
        input_tensor = torch.cat([frame1_tensor, frame2_tensor], dim=0)
        
        return input_tensor, gt_tensor


class CustomInterpolationModel(nn.Module):
    """
    Custom-trained frame interpolation model.
    Deep CNN architecture for generating intermediate frames.
    """
    
    def __init__(self):
        super(CustomInterpolationModel, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_model(
    data_dir: str,
    model_save_path: str,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    device: str = None
):
    """
    Train the custom interpolation model.
    
    Args:
        data_dir: Directory containing training data
        model_save_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to use (cuda/cpu)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Training on device: {device}")
    
    # Create dataset and dataloader
    dataset = FrameInterpolationDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Initialize model
    model = CustomInterpolationModel().to(device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    training_history = []
    
    logger.info("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        training_history.append(avg_loss)
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model with loss: {best_loss:.6f}")
    
    # Save training history
    history_path = model_save_path.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'training_history': training_history,
            'best_loss': best_loss,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"Training completed! Best loss: {best_loss:.6f}")
    logger.info(f"Model saved to: {model_save_path}")
    
    return model, training_history


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train custom frame interpolation model')
    parser.add_argument('--data_dir', type=str, default='data/training', help='Training data directory')
    parser.add_argument('--model_path', type=str, default='models/trained_model.pth', help='Model save path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Train model
    train_model(
        data_dir=args.data_dir,
        model_save_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()

