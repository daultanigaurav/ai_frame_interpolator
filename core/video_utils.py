"""
Video utilities for frame extraction and video generation.
"""

import cv2
import numpy as np
from typing import List, Optional
import os
import logging
# MoviePy import removed - using OpenCV for video creation

logger = logging.getLogger(__name__)


def extract_frames(video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (None for all)
        
    Returns:
        List of frames as numpy arrays
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        frame_count += 1
        
        if max_frames and frame_count >= max_frames:
            break
    
    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {video_path}")
    
    return frames


def create_video_from_frames(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30,
    codec: str = 'mp4v'
) -> str:
    """
    Create a video file from a list of frames.
    
    Args:
        frames: List of frames (numpy arrays)
        output_path: Path to save the video
        fps: Frames per second
        codec: Video codec to use
        
    Returns:
        Path to created video
    """
    if not frames:
        raise ValueError("No frames provided")
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Ensure frame is correct size
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        out.write(frame)
    
    out.release()
    logger.info(f"Created video: {output_path} ({len(frames)} frames, {fps} fps)")
    
    return output_path


def create_comparison_gif(
    frames1: List[np.ndarray],
    frames2: List[np.ndarray],
    output_path: str,
    fps: int = 10
) -> str:
    """
    Create a side-by-side comparison GIF from two frame sequences.
    
    Args:
        frames1: First sequence of frames
        frames2: Second sequence of frames
        output_path: Path to save the GIF
        fps: Frames per second for GIF
        
    Returns:
        Path to created GIF
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("PIL not available, skipping GIF creation")
        return output_path
    
    # Ensure same number of frames
    min_frames = min(len(frames1), len(frames2))
    frames1 = frames1[:min_frames]
    frames2 = frames2[:min_frames]
    
    # Create side-by-side frames
    comparison_frames = []
    for f1, f2 in zip(frames1, frames2):
        # Resize if needed
        h1, w1 = f1.shape[:2]
        h2, w2 = f2.shape[:2]
        h = max(h1, h2)
        w = w1 + w2
        
        # Create combined frame
        combined = np.zeros((h, w, 3), dtype=np.uint8)
        combined[:h1, :w1] = f1
        combined[:h2, w1:] = f2
        
        # Convert to PIL Image
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(combined_rgb)
        comparison_frames.append(pil_image)
    
    # Save as GIF
    comparison_frames[0].save(
        output_path,
        save_all=True,
        append_images=comparison_frames[1:],
        duration=1000 // fps,
        loop=0
    )
    
    logger.info(f"Created comparison GIF: {output_path}")
    
    return output_path


def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)
    
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": 0
    }
    
    if info["fps"] > 0:
        info["duration"] = info["frame_count"] / info["fps"]
    
    cap.release()
    
    return info

