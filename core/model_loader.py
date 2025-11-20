import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CustomInterpolationModel(nn.Module):
    def __init__(self):
        super(CustomInterpolationModel, self).__init__()
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class CustomInterpolationModelWrapper:
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path or self._get_default_model_path()
        self._load_model()

    def _get_default_model_path(self) -> str:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, "models", "trained_model.pth")

    def _load_model(self):
        try:
            logger.info(f"Loading custom interpolation model from: {self.model_path}")
            logger.info(f"Using device: {self.device}")
            self.model = CustomInterpolationModel().to(self.device)
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
        self.model.eval()
        f1 = frame1.astype(np.float32) / 255.0
        f2 = frame2.astype(np.float32) / 255.0
        f1_tensor = torch.from_numpy(f1).permute(2, 0, 1).unsqueeze(0).to(self.device)
        f2_tensor = torch.from_numpy(f2).permute(2, 0, 1).unsqueeze(0).to(self.device)
        input_tensor = torch.cat([f1_tensor, f2_tensor], dim=1)
        with torch.no_grad():
            output = self.model(input_tensor)
            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            blended = (1 - alpha) * f1 + alpha * f2
            final = 0.7 * output_np + 0.3 * blended
        final = np.clip(final * 255.0, 0, 255).astype(np.uint8)
        return final


def load_model(model_path: Optional[str] = None):
    return CustomInterpolationModelWrapper(model_path)
