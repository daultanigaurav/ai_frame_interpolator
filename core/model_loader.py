"""
Model loading and management for custom-trained frame interpolation model.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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


class CustomInterpolationModelWrapper:
    """
    Wrapper for the custom-trained frame interpolation model.
    Loads the trained model from models/trained_model.pth
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize custom interpolation model.
        
        Args:
            model_path: Path to trained model weights (default: models/trained_model.pth)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path or self._get_default_model_path()
        self._load_model()
    
    def _get_default_model_path(self) -> str:
        """Get default model path."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, "models", "trained_model.pth")
    
    def _load_model(self):
        """Load the custom trained model."""
        try:
            logger.info(f"Loading custom interpolation model from: {self.model_path}")
            logger.info(f"Using device: {self.device}")
            
            # Initialize model architecture
            self.model = CustomInterpolationModel().to(self.device)
            
            # Load trained weights if available
            if os.path.exists(self.model_path):
                try:
                    self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    logger.info(f"Successfully loaded trained model weights from {self.model_path}")
                except Exception as e:
                    logger.warning(f"Could not load trained weights: {e}")
                    logger.info("Using randomly initialized weights (model not trained yet)")
            else:
                logger.warning(f"Model weights not found at {self.model_path}")
                logger.info("Using randomly initialized weights. Train the model first using model_train.py")
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading custom model: {e}")
            raise
    
    def interpolate(self, frame1: np.ndarray, frame2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Interpolate between two frames using the custom trained model.
        
        Args:
            frame1: First frame (H, W, 3) in [0, 255]
            frame2: Second frame (H, W, 3) in [0, 255]
            alpha: Interpolation factor (0.0 = frame1, 1.0 = frame2)
            
        Returns:
            Interpolated frame (H, W, 3) in [0, 255]
        """
        self.model.eval()
        
        # Normalize to [0, 1]
        f1 = frame1.astype(np.float32) / 255.0
        f2 = frame2.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        f1_tensor = torch.from_numpy(f1).permute(2, 0, 1).unsqueeze(0).to(self.device)
        f2_tensor = torch.from_numpy(f2).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Concatenate frames
        input_tensor = torch.cat([f1_tensor, f2_tensor], dim=1)
        
        with torch.no_grad():
            # Get model prediction
            output = self.model(input_tensor)
            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Blend with alpha for smoother interpolation
            blended = (1 - alpha) * f1 + alpha * f2
            final = 0.7 * output_np + 0.3 * blended
        
        # Denormalize
        final = np.clip(final * 255.0, 0, 255).astype(np.uint8)
        
        return final


def load_model(model_path: Optional[str] = None):
    """
    Load the custom-trained frame interpolation model.
    
    Args:
        model_path: Optional path to model weights (default: models/trained_model.pth)
        
    Returns:
        Model instance
    """
    return CustomInterpolationModelWrapper(model_path)
