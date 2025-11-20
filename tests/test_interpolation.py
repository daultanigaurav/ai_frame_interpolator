"""
Unit tests for the frame interpolation system.
"""

import unittest
import os
import sys
import numpy as np
import cv2
import tempfile
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.interpolate import interpolate_frames
from core.model_loader import load_model, RIFEModel, FILMModel
from core.metrics import calculate_ssim, calculate_psnr, evaluate_interpolation_quality
from core.utils import load_image, save_image, resize_image, normalize_image, denormalize_image
from core.video_utils import extract_frames, create_video_from_frames


class TestInterpolation(unittest.TestCase):
    """Test cases for frame interpolation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary test images
        self.temp_dir = tempfile.mkdtemp()
        
        # Create two test frames
        self.frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        self.frame1_path = os.path.join(self.temp_dir, "frame1.png")
        self.frame2_path = os.path.join(self.temp_dir, "frame2.png")
        
        cv2.imwrite(self.frame1_path, self.frame1)
        cv2.imwrite(self.frame2_path, self.frame2)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_load_image(self):
        """Test image loading."""
        img = load_image(self.frame1_path)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, self.frame1.shape)
    
    def test_save_image(self):
        """Test image saving."""
        output_path = os.path.join(self.temp_dir, "test_output.png")
        save_image(self.frame1, output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # Verify saved image
        loaded = load_image(output_path)
        self.assertEqual(loaded.shape, self.frame1.shape)
    
    def test_resize_image(self):
        """Test image resizing."""
        target_size = (1280, 720)
        resized = resize_image(self.frame1, target_size)
        self.assertEqual(resized.shape[:2], (target_size[1], target_size[0]))
    
    def test_normalize_denormalize(self):
        """Test image normalization and denormalization."""
        normalized = normalize_image(self.frame1)
        self.assertTrue(np.all(normalized >= 0) and np.all(normalized <= 1))
        
        denormalized = denormalize_image(normalized)
        self.assertTrue(np.all(denormalized >= 0) and np.all(denormalized <= 255))
    
    def test_rife_model(self):
        """Test RIFE model initialization and interpolation."""
        model = RIFEModel()
        self.assertIsNotNone(model.model)
        
        # Test interpolation
        interpolated = model.interpolate(self.frame1, self.frame2, alpha=0.5)
        self.assertEqual(interpolated.shape, self.frame1.shape)
        self.assertEqual(interpolated.dtype, np.uint8)
    
    def test_film_model(self):
        """Test FILM model initialization and interpolation."""
        model = FILMModel()
        self.assertIsNotNone(model.model)
        
        # Test interpolation
        interpolated = model.interpolate(self.frame1, self.frame2, alpha=0.5)
        self.assertEqual(interpolated.shape, self.frame1.shape)
        self.assertEqual(interpolated.dtype, np.uint8)
    
    def test_load_model(self):
        """Test model loading function."""
        rife_model = load_model("rife")
        self.assertIsNotNone(rife_model)
        
        film_model = load_model("film")
        self.assertIsNotNone(film_model)
        
        with self.assertRaises(ValueError):
            load_model("invalid_model")
    
    def test_interpolate_frames(self):
        """Test frame interpolation function."""
        frames, video_path, metrics = interpolate_frames(
            frame1_path=self.frame1_path,
            frame2_path=self.frame2_path,
            num_interpolations=3,
            model_name="rife",
            resolution=(640, 480),
            fps=30,
            output_dir=self.temp_dir
        )
        
        # Check results
        self.assertGreater(len(frames), 2)  # Should have original + interpolated frames
        self.assertIn("total_frames", metrics)
        self.assertIn("interpolated_frames", metrics)
        self.assertIn("total_time", metrics)
    
    def test_calculate_ssim(self):
        """Test SSIM calculation."""
        # Same image should have SSIM = 1.0
        ssim_same = calculate_ssim(self.frame1, self.frame1)
        self.assertAlmostEqual(ssim_same, 1.0, places=2)
        
        # Different images should have SSIM < 1.0
        ssim_diff = calculate_ssim(self.frame1, self.frame2)
        self.assertLess(ssim_diff, 1.0)
        self.assertGreater(ssim_diff, 0.0)
    
    def test_calculate_psnr(self):
        """Test PSNR calculation."""
        # Same image should have high PSNR
        psnr_same = calculate_psnr(self.frame1, self.frame1)
        self.assertGreater(psnr_same, 40.0)  # Should be very high for identical images
        
        # Different images should have lower PSNR
        psnr_diff = calculate_psnr(self.frame1, self.frame2)
        self.assertGreater(psnr_diff, 0.0)
    
    def test_evaluate_interpolation_quality(self):
        """Test interpolation quality evaluation."""
        # Create interpolated frame
        model = RIFEModel()
        interpolated = model.interpolate(self.frame1, self.frame2, alpha=0.5)
        
        # Evaluate quality
        metrics = evaluate_interpolation_quality(
            self.frame1,
            self.frame2,
            interpolated,
            alpha=0.5
        )
        
        # Check metrics
        self.assertIn("ssim_avg", metrics)
        self.assertIn("psnr_avg", metrics)
        self.assertGreater(metrics["ssim_avg"], 0.0)
        self.assertLess(metrics["ssim_avg"], 1.0)
    
    def test_create_video_from_frames(self):
        """Test video creation from frames."""
        frames = [self.frame1, self.frame2]
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        
        create_video_from_frames(frames, video_path, fps=30)
        
        # Check video was created
        self.assertTrue(os.path.exists(video_path))
        self.assertGreater(os.path.getsize(video_path), 0)


class TestVideoUtils(unittest.TestCase):
    """Test cases for video utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_create_video_from_frames(self):
        """Test video creation."""
        # Create test frames
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        create_video_from_frames(frames, video_path, fps=30)
        
        self.assertTrue(os.path.exists(video_path))


if __name__ == "__main__":
    unittest.main()

