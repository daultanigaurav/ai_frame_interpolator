"""
Core interpolation logic for generating intermediate frames.
"""

import os
import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging
import time

from .model_loader import load_model
from .utils import (
    load_image, save_image, resize_image, 
    ensure_dir, get_project_root
)
from .video_utils import create_video_from_frames

logger = logging.getLogger(__name__)


def interpolate_frames(
    frame1_path: str,
    frame2_path: str,
    num_interpolations: int = 5,
    resolution: Tuple[int, int] = (1280, 720),
    fps: int = 30,
    output_dir: Optional[str] = None
) -> Tuple[List[np.ndarray], str, str, dict]:
    """
    Generate intermediate frames between two images using AI interpolation.
    
    Args:
        frame1_path: Path to first frame
        frame2_path: Path to second frame
        num_interpolations: Number of intermediate frames to generate
        Uses custom trained interpolation model
        resolution: Output resolution (width, height)
        fps: Frames per second for output video
        output_dir: Directory to save outputs (optional)
        
    Returns:
        Tuple of (list of frames, output video path, metrics dict)
    """
    start_time = time.time()
    
    # Setup output directories
    project_root = get_project_root()
    if output_dir is None:
        output_dir = os.path.join(project_root, "output", "interpolated_frames")
    ensure_dir(output_dir)
    
    # Setup preview directory
    preview_dir = os.path.join(project_root, "output", "previews")
    ensure_dir(preview_dir)
    
    # Load frames
    logger.info(f"Loading frames: {frame1_path} and {frame2_path}")
    frame1 = load_image(frame1_path)
    frame2 = load_image(frame2_path)
    
    # Resize frames to target resolution
    frame1 = resize_image(frame1, resolution)
    frame2 = resize_image(frame2, resolution)
    
    # Load custom trained model
    logger.info("Loading custom trained interpolation model...")
    model = load_model()
    
    # Generate interpolated frames
    logger.info(f"Generating {num_interpolations} intermediate frames...")
    all_frames = [frame1.copy()]
    
    interpolation_times = []
    
    for i in range(1, num_interpolations + 1):
        alpha = i / (num_interpolations + 1)
        
        frame_start = time.time()
        interpolated = model.interpolate(frame1, frame2, alpha)
        frame_time = time.time() - frame_start
        interpolation_times.append(frame_time)
        
        all_frames.append(interpolated)
        
        # Save individual frame to both directories
        frame_path = os.path.join(output_dir, f"interpolated_frame_{i:04d}.png")
        preview_path = os.path.join(preview_dir, f"interpolated_frame_{i:04d}.png")
        save_image(interpolated, frame_path)
        save_image(interpolated, preview_path)
        
        logger.info(f"Generated frame {i}/{num_interpolations} (alpha={alpha:.3f}, time={frame_time:.3f}s)")
    
    all_frames.append(frame2.copy())
    
    # Create video from frames
    logger.info("Creating output video...")
    video_output_dir = os.path.join(os.path.dirname(output_dir), "..")
    video_path = os.path.join(video_output_dir, "output_video.mp4")
    preview_video_path = os.path.join(preview_dir, "output_video.mp4")
    ensure_dir(os.path.dirname(video_path))
    
    create_video_from_frames(all_frames, video_path, fps)
    create_video_from_frames(all_frames, preview_video_path, fps)
    
    # Calculate metrics
    total_time = time.time() - start_time
    avg_frame_time = np.mean(interpolation_times) if interpolation_times else 0
    
    metrics = {
        "total_frames": len(all_frames),
        "interpolated_frames": num_interpolations,
        "total_time": total_time,
        "avg_frame_time": avg_frame_time,
        "fps": fps,
        "resolution": resolution,
        "model": "custom_trained"
    }
    
    logger.info(f"Interpolation complete! Total time: {total_time:.2f}s")
    logger.info(f"Average time per frame: {avg_frame_time:.3f}s")
    logger.info(f"Preview files saved to: {preview_dir}")
    
    return all_frames, video_path, preview_video_path, metrics


def interpolate_video(
    video_path: str,
    num_interpolations: int = 5,
    resolution: Tuple[int, int] = (1280, 720),
    fps: int = 30,
    output_dir: Optional[str] = None
) -> Tuple[str, str, dict]:
    """
    Interpolate frames for an entire video.
    
    Args:
        video_path: Path to input video
        num_interpolations: Number of intermediate frames per pair
        resolution: Output resolution
        fps: Output FPS
        output_dir: Output directory
        
    Returns:
        Tuple of (output video path, preview video path, metrics dict)
    """
    from .video_utils import extract_frames
    
    # Extract frames from video
    frames = extract_frames(video_path)
    
    if len(frames) < 2:
        raise ValueError("Video must contain at least 2 frames")
    
    # Setup output
    if output_dir is None:
        project_root = get_project_root()
        output_dir = os.path.join(project_root, "output", "interpolated_frames")
    ensure_dir(output_dir)
    
    # Load custom trained model
    model = load_model()
    
    all_interpolated_frames = []
    interpolation_times = []
    start_time = time.time()
    
    # Interpolate between consecutive frame pairs
    for i in range(len(frames) - 1):
        frame1 = resize_image(frames[i], resolution)
        frame2 = resize_image(frames[i + 1], resolution)
        
        all_interpolated_frames.append(frame1.copy())
        
        # Generate intermediate frames
        for j in range(1, num_interpolations + 1):
            alpha = j / (num_interpolations + 1)
            
            frame_start = time.time()
            interpolated = model.interpolate(frame1, frame2, alpha)
            frame_time = time.time() - frame_start
            interpolation_times.append(frame_time)
            
            all_interpolated_frames.append(interpolated)
        
        logger.info(f"Processed frame pair {i+1}/{len(frames)-1}")
    
    all_interpolated_frames.append(resize_image(frames[-1], resolution))
    
    # Create output video
    video_output_dir = os.path.join(os.path.dirname(output_dir), "..")
    video_path = os.path.join(video_output_dir, "output_video.mp4")
    preview_video_path = os.path.join(preview_dir, "output_video.mp4")
    ensure_dir(os.path.dirname(video_path))
    
    create_video_from_frames(all_interpolated_frames, video_path, fps)
    create_video_from_frames(all_interpolated_frames, preview_video_path, fps)
    
    total_time = time.time() - start_time
    avg_frame_time = np.mean(interpolation_times) if interpolation_times else 0
    
    metrics = {
        "total_frames": len(all_interpolated_frames),
        "interpolated_frames": len(interpolation_times),
        "total_time": total_time,
        "avg_frame_time": avg_frame_time,
        "fps": fps,
        "resolution": resolution,
        "model": "custom_trained"
    }
    
    return video_path, preview_video_path, metrics

