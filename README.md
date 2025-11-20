# 🎬 AI-Based Video Frame Interpolation System

A professional-grade, research-quality Python application for generating intermediate video frames using a **custom-trained deep learning model**. This system uses an in-house AI model specifically designed and trained for frame interpolation to create smooth, high-quality interpolated frames between consecutive video frames.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Custom Model](#custom-model)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Model Training](#model-training)
- [Usage](#usage)
- [Preview Persistence](#preview-persistence)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Research Applications](#research-applications)
- [Future Work](#future-work)
- [References](#references)

## 🎯 Overview

This project implements an end-to-end AI-powered video frame interpolation system that:

- Takes two consecutive frames (or a video file) as input
- Generates multiple intermediate frames using a **custom-trained deep learning model**
- Provides configurable interpolation parameters (number of frames, resolution, FPS)
- Evaluates interpolation quality using SSIM and PSNR metrics
- Outputs smooth, high-quality interpolated videos
- **Permanently stores previews** in `/output/previews/` for persistent viewing

The system is designed for research, academic projects, and practical applications in video processing, slow-motion generation, and frame rate upscaling.

## ✨ Features

### Core Functionality
- **Custom AI Model**: In-house trained deep learning model for frame interpolation
- **Flexible Input**: Accept two frames or entire video files
- **Configurable Parameters**: 
  - Number of interpolated frames (1-10)
  - Output resolution (customizable)
  - Frame rate (FPS) control
- **Quality Metrics**: SSIM and PSNR evaluation
- **Visualization**: Optical flow visualization and comparison tools
- **Preview Persistence**: All previews stored permanently and remain visible after download

### User Interface
- **Modern Web Interface**: Clean, intuitive Streamlit-based UI
- **Real-time Progress**: Progress bars and status updates
- **Persistent Previews**: Video and frame previews remain visible and functional after download
- **Metrics Dashboard**: Visual charts and statistics
- **Documentation**: Built-in help and documentation

### Technical Features
- **Modular Architecture**: Clean, reusable code structure
- **Error Handling**: Robust error handling and logging
- **Performance Optimization**: Efficient frame processing
- **Offline Operation**: Fully offline, no external API calls required
- **Preview Storage**: All previews automatically saved to `/output/previews/`

## 🤖 Custom Model

### Model Architecture

This system uses a **custom-trained deep learning model** built specifically for frame interpolation. The model architecture consists of:

- **Encoder**: Multi-layer CNN for feature extraction from input frame pairs
  - Convolutional layers with ReLU activation
  - Max pooling for dimensionality reduction
  - Feature extraction at multiple scales

- **Decoder**: Transposed convolutions for frame reconstruction
  - Upsampling layers to restore original resolution
  - Convolutional layers for refinement
  - Sigmoid activation for output normalization

### Model Training

The model is trained using `model_train.py`:

```bash
python model_train.py --data_dir data/training --epochs 50 --batch_size 8
```

**Training Process:**
1. Prepare training data: Place consecutive frame pairs in `data/training/`
2. Run training script: `python model_train.py`
3. Model weights saved to: `models/trained_model.pth`
4. Training history saved to: `models/trained_model_history.json`

**Training Parameters:**
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Learning Rate**: 0.001 (with step decay)
- **Batch Size**: 8 (configurable)
- **Epochs**: 50 (configurable)

### Model Loading

The system automatically loads the trained model from `models/trained_model.pth`:
- If trained model exists: Loads trained weights
- If no trained model: Uses randomly initialized weights (for testing)

**Note**: For best results, train the model on your specific data before using the system.

## 🏗️ System Architecture

```
┌─────────────────┐
│  Input Frames   │
│  (Frame 1 & 2)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Custom Model   │
│  Loader         │
│  (trained_model)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Interpolation  │
│    Engine       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Video Generator│
│  (OpenCV)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Video   │
│  + Previews     │
│  + Metrics      │
└─────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, for faster processing)
- 4GB+ RAM recommended

### Step 1: Clone or Download the Project

```bash
cd ai_frame_interpolator
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; import cv2; import streamlit; print('Installation successful!')"
```

## 🎓 Model Training

### Preparing Training Data

1. Create a training data directory:
   ```bash
   mkdir -p data/training
   ```

2. Place consecutive frame pairs in the directory:
   - Frame pairs should be named sequentially (e.g., `frame_001.png`, `frame_002.png`)
   - The system will automatically pair consecutive frames

### Training the Model

```bash
python model_train.py --data_dir data/training --epochs 50 --batch_size 8 --lr 0.001
```

**Training Options:**
- `--data_dir`: Path to training data directory (default: `data/training`)
- `--model_path`: Path to save trained model (default: `models/trained_model.pth`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 8)
- `--lr`: Learning rate (default: 0.001)

### Training Output

After training, you'll have:
- `models/trained_model.pth`: Trained model weights
- `models/trained_model_history.json`: Training history and metrics

## 🚀 Usage

### Running the Web Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**:
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

3. **Use the application**:
   - Upload two frames or a video file
   - Configure settings (number of frames, resolution, FPS)
   - Click "Generate Interpolated Video"
   - Preview and download the result

### Command-Line Usage (Python API)

```python
from core.interpolate import interpolate_frames

# Interpolate between two frames
frames, video_path, preview_video_path, metrics = interpolate_frames(
    frame1_path="path/to/frame1.png",
    frame2_path="path/to/frame2.png",
    num_interpolations=5,
    resolution=(1920, 1080),
    fps=60
)

print(f"Generated {len(frames)} frames")
print(f"Video saved to: {video_path}")
print(f"Preview saved to: {preview_video_path}")
print(f"Metrics: {metrics}")
```

## 📸 Preview Persistence

### How It Works

All generated previews are **permanently stored** in `/output/previews/`:

- **Video Preview**: `output/previews/output_video.mp4`
- **Frame Previews**: `output/previews/interpolated_frame_XXXX.png`

### Preview Behavior

- **Previews remain visible**: After generation, previews stay visible in the UI
- **Previews persist after download**: Downloading does not remove or reload previews
- **Permanent storage**: All previews are saved to disk and can be reloaded
- **Dynamic reloading**: UI automatically loads previews from storage

### Preview Locations

```
output/
├── previews/              # Permanent preview storage
│   ├── output_video.mp4   # Video preview
│   ├── interpolated_frame_0001.png
│   ├── interpolated_frame_0002.png
│   └── ...
├── interpolated_frames/   # Full resolution frames
└── output_video.mp4       # Final output video
```

## 📊 Evaluation Metrics

### SSIM (Structural Similarity Index)

- **Range**: 0.0 to 1.0 (higher is better)
- **Measures**: Structural similarity between frames
- **Interpretation**: 
  - > 0.9: Excellent
  - 0.8-0.9: Good
  - < 0.8: Needs improvement

### PSNR (Peak Signal-to-Noise Ratio)

- **Range**: 0 to ∞ dB (higher is better)
- **Measures**: Image quality in decibels
- **Interpretation**:
  - > 40 dB: Excellent
  - 30-40 dB: Good
  - < 30 dB: Acceptable

### Processing Time

- **Average Frame Time**: Time per interpolated frame
- **Total Time**: End-to-end processing time
- **Performance**: Varies with resolution, model, and hardware

## 📁 Project Structure

```
ai_frame_interpolator/
│
├── app.py                      # Streamlit web application
├── model_train.py              # Model training script
│
├── core/                       # Core modules
│   ├── __init__.py
│   ├── interpolate.py         # Main interpolation logic
│   ├── model_loader.py        # Custom model loading
│   ├── metrics.py             # SSIM, PSNR, evaluation
│   ├── utils.py               # Helper functions
│   └── video_utils.py          # Video processing utilities
│
├── models/                     # Model weights
│   ├── trained_model.pth      # Trained model (auto-loaded)
│   └── trained_model_history.json
│
├── output/                     # Output directory
│   ├── previews/              # Permanent preview storage
│   │   ├── output_video.mp4
│   │   └── interpolated_frame_*.png
│   ├── interpolated_frames/   # Generated frames
│   └── output_video.mp4        # Final output video
│
├── data/                       # Training data (optional)
│   └── training/              # Training frame pairs
│
├── tests/                      # Unit tests
│   └── test_interpolation.py
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Technical Details

### Technologies Used

- **PyTorch**: Deep learning framework for model training and inference
- **OpenCV**: Image and video processing
- **Streamlit**: Web interface
- **NumPy**: Numerical computations
- **scikit-image**: Image metrics (SSIM, PSNR)
- **Plotly**: Interactive visualizations

### Key Algorithms

1. **Frame Interpolation**: CNN-based intermediate frame generation
2. **Optical Flow**: Farneback method for motion estimation
3. **Temporal Blending**: Alpha-based frame blending
4. **Quality Metrics**: SSIM and PSNR calculation

### Performance Considerations

- **GPU Acceleration**: Automatic GPU detection and usage
- **Memory Management**: Efficient frame processing
- **Batch Processing**: Support for multiple frame pairs
- **Resolution Scaling**: Automatic resolution handling

## 🔧 Troubleshooting

### Preview or Video Not Displaying

If previews or videos fail to display in the Streamlit interface:

1. **Check File Paths**:
   - Verify that `output/previews/` directory exists
   - Check that files are being saved correctly
   - Ensure file permissions allow reading

2. **Check File Format**:
   - Ensure video codec is supported (MP4 with H.264)
   - Verify image formats are PNG or JPG
   - Check file sizes are not too large

3. **Browser Issues**:
   - Clear browser cache
   - Try a different browser (Chrome, Firefox, Edge)
   - Check browser console for errors

4. **Streamlit Issues**:
   - Restart Streamlit server: `streamlit run app.py`
   - Check Streamlit logs for errors
   - Verify all dependencies are installed

5. **File Loading**:
   - Check if preview files exist in `output/previews/`
   - Verify file paths in session state
   - Ensure files are not corrupted

6. **Video Playback**:
   - Try downloading the video and playing locally
   - Check video codec compatibility
   - Verify video file is not corrupted

### Model Not Loading

If the custom model fails to load:

1. **Check Model File**:
   - Verify `models/trained_model.pth` exists
   - Check file permissions
   - Ensure model file is not corrupted

2. **Model Architecture Mismatch**:
   - If you modified the model architecture, retrain the model
   - Ensure model architecture matches training configuration

3. **Fallback Behavior**:
   - System will use randomly initialized weights if model not found
   - This allows testing without a trained model

### Performance Issues

1. **Slow Processing**:
   - Use GPU if available (CUDA)
   - Reduce resolution or number of frames
   - Close other applications

2. **Memory Issues**:
   - Reduce batch size during training
   - Process smaller videos
   - Use lower resolution

## 🔬 Research Applications

### Academic Use Cases

1. **Video Frame Rate Upscaling**: Convert 30fps to 60fps or higher
2. **Slow-Motion Generation**: Create smooth slow-motion effects
3. **Video Restoration**: Enhance low-frame-rate videos
4. **Motion Analysis**: Study motion patterns and interpolation quality
5. **Computer Vision Research**: Benchmark interpolation algorithms

### Practical Applications

- **Video Production**: Enhance video quality and smoothness
- **Gaming**: Frame interpolation for game recordings
- **Security**: Enhance surveillance video quality
- **Medical Imaging**: Interpolate medical video sequences
- **Animation**: Generate intermediate animation frames

## 🚧 Future Work

### Planned Enhancements

- [ ] Advanced model architectures (attention mechanisms)
- [ ] Multi-frame interpolation (more than 2 input frames)
- [ ] Real-time video streaming support
- [ ] Advanced quality metrics (LPIPS, FVD)
- [ ] GPU optimization and multi-GPU support
- [ ] Model fine-tuning capabilities
- [ ] Export to various video formats
- [ ] API endpoint for programmatic access

### Research Directions

- [ ] Temporal consistency improvements
- [ ] Handling of occlusions and disocclusions
- [ ] Adaptive interpolation based on motion magnitude
- [ ] Quality-aware interpolation
- [ ] Perceptual loss functions

## 📚 References

### Papers

1. **Frame Interpolation**: Various deep learning approaches to frame interpolation
2. **CNN Architectures**: Encoder-decoder architectures for image generation

### Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-image Documentation](https://scikit-image.org/)

## 📄 License

This project is provided for educational and research purposes.

## 👥 Authors

**AI Research Team**

## 🙏 Acknowledgments

- Open-source community for excellent tools and libraries
- PyTorch team for the deep learning framework

## 📧 Contact

For questions, issues, or contributions, please open an issue in the project repository.

---

**Version**: 2.0.0  
**Last Updated**: 2025

**Note**: This system uses a custom-trained model. For best results, train the model on your specific data before use. Previews are permanently stored and remain visible after download.
