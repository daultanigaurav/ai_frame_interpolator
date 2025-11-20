import streamlit as st
import os
import sys
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import tempfile

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.interpolate import interpolate_frames, interpolate_video
from core.metrics import calculate_ssim, calculate_psnr, calculate_optical_flow, evaluate_frame_sequence
from core.video_utils import extract_frames, get_video_info, create_comparison_gif
from core.utils import ensure_dir, get_project_root

st.set_page_config(page_title="AI Frame Interpolation System", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

css_path = os.path.join(project_root, "static", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>.main-header{font-size:3rem;font-weight:bold;text-align:center;color:#1f77b4;margin-bottom:2rem}.sub-header{font-size:1.5rem;color:#666;text-align:center;margin-bottom:2rem}.metric-card{background-color:#f0f2f6;padding:1rem;border-radius:0.5rem;margin:0.5rem 0}</style>", unsafe_allow_html=True)

if 'frames' not in st.session_state:
    st.session_state.frames = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'preview_video_path' not in st.session_state:
    st.session_state.preview_video_path = None
if 'preview_frames_dir' not in st.session_state:
    st.session_state.preview_frames_dir = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None


def main():
    st.markdown('<h1 class="main-header">🎬 AI-Based Video Frame Interpolation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate smooth intermediate frames using custom-trained deep learning model</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Settings")

        num_interpolations = st.slider(
            "Number of Interpolated Frames",
            min_value=1,
            max_value=10,
            value=5,
            help="How many intermediate frames to generate between two frames"
        )

        st.subheader("Output Resolution")
        col1, col2 = st.columns(2)
        with col1:
            width = st.number_input("Width", min_value=320, max_value=7680, value=1280, step=160)
        with col2:
            height = st.number_input("Height", min_value=240, max_value=4320, value=720, step=90)

        resolution = (width, height)

        fps = st.number_input(
            "Output FPS",
            min_value=1,
            max_value=120,
            value=30,
            help="Frames per second for output video"
        )

        st.divider()

        with st.expander("ℹ️ About This Project"):
            st.markdown("""
            ### AI Frame Interpolation System
            
            This system uses deep learning models to generate intermediate frames 
            between two consecutive video frames, creating smooth motion.
            
            **Models:**
            - **RIFE**: Real-Time Intermediate Flow Estimation - Fast and efficient
            - **FILM**: Frame Interpolation for Large Motion - Better for large movements
            
            **Features:**
            - High-quality frame interpolation
            - SSIM and PSNR quality metrics
            - Optical flow visualization
            - Video export capabilities
            
            **Use Cases:**
            - Video frame rate upscaling
            - Slow-motion generation
            - Video restoration
            - Research and development
            """)

    tab1, tab2, tab3 = st.tabs(["📤 Upload & Generate", "📊 Metrics & Analysis", "📖 Documentation"])

    with tab1:
        st.header("Upload Frames or Video")

        input_method = st.radio(
            "Input Method",
            ["Two Frames", "Video File"],
            horizontal=True
        )

        if input_method == "Two Frames":
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Frame 1")
                frame1_file = st.file_uploader(
                    "Upload First Frame",
                    type=['png', 'jpg', 'jpeg', 'bmp', 'jpg'],
                    key="frame1"
                )
                if frame1_file:
                    frame1_bytes = frame1_file.read()
                    frame1_np = np.frombuffer(frame1_bytes, np.uint8)
                    frame1_img = cv2.imdecode(frame1_np, cv2.IMREAD_COLOR)
                    st.image(cv2.cvtColor(frame1_img, cv2.COLOR_BGR2RGB), caption="Frame 1", use_container_width=True)

            with col2:
                st.subheader("Frame 2")
                frame2_file = st.file_uploader(
                    "Upload Second Frame",
                    type=['png', 'jpg', 'jpeg', 'bmp', 'jpg'],
                    key="frame2"
                )
                if frame2_file:
                    frame2_bytes = frame2_file.read()
                    frame2_np = np.frombuffer(frame2_bytes, np.uint8)
                    frame2_img = cv2.imdecode(frame2_np, cv2.IMREAD_COLOR)
                    st.image(cv2.cvtColor(frame2_img, cv2.COLOR_BGR2RGB), caption="Frame 2", use_container_width=True)

            if st.button("🚀 Generate Interpolated Video", type="primary", use_container_width=True):
                if frame1_file and frame2_file:
                    with st.spinner("Generating interpolated frames..."):
                        project_root = get_project_root()
                        temp_dir = os.path.join(project_root, "output", "temp")
                        ensure_dir(temp_dir)

                        frame1_path = os.path.join(temp_dir, "frame1.png")
                        frame2_path = os.path.join(temp_dir, "frame2.png")

                        cv2.imwrite(frame1_path, frame1_img)
                        cv2.imwrite(frame2_path, frame2_img)

                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        try:
                            status_text.text("Loading custom trained model...")
                            progress_bar.progress(10)

                            status_text.text("Generating intermediate frames...")
                            frames, video_path, preview_video_path, metrics = interpolate_frames(
                                frame1_path=frame1_path,
                                frame2_path=frame2_path,
                                num_interpolations=num_interpolations,
                                resolution=resolution,
                                fps=fps
                            )

                            progress_bar.progress(90)
                            status_text.text("Finalizing video...")

                            st.session_state.frames = frames
                            st.session_state.video_path = video_path
                            st.session_state.preview_video_path = preview_video_path
                            st.session_state.preview_frames_dir = os.path.join(get_project_root(), "output", "previews")
                            st.session_state.metrics = metrics

                            progress_bar.progress(100)
                            status_text.text("Complete!")

                            st.success("✅ Interpolation complete!")

                            if os.path.exists(video_path):
                                if 'video_download_bytes' not in st.session_state:
                                    with open(video_path, "rb") as video_file:
                                        st.session_state.video_download_bytes = video_file.read()

                                st.download_button(
                                    label="📥 Download Video",
                                    data=st.session_state.video_download_bytes,
                                    file_name="output_video.mp4",
                                    mime="video/mp4",
                                    use_container_width=True,
                                    key="download_video_main"
                                )
                            else:
                                st.warning(f"Video file not found at: {video_path}")

                            st.subheader("🖼️ All Generated Frames")

                            display_frames = st.session_state.frames if st.session_state.frames is not None else frames

                            if len(display_frames) > 0:
                                num_frames = len(display_frames)
                                num_cols = 3
                                num_rows = (num_frames + num_cols - 1) // num_cols

                                for row in range(num_rows):
                                    cols = st.columns(num_cols)
                                    for col_idx in range(num_cols):
                                        frame_idx = row * num_cols + col_idx
                                        if frame_idx < num_frames:
                                            with cols[col_idx]:
                                                frame_rgb = cv2.cvtColor(display_frames[frame_idx], cv2.COLOR_BGR2RGB)
                                                if frame_idx == 0:
                                                    caption = "📷 Original Frame 1"
                                                elif frame_idx == num_frames - 1:
                                                    caption = f"📷 Original Frame 2"
                                                else:
                                                    interp_num = frame_idx
                                                    caption = f"✨ Interpolated Frame {interp_num}"

                                                st.image(frame_rgb, caption=caption, use_container_width=True)

                                                frame_key = f"frame_bytes_{frame_idx}"
                                                if frame_key not in st.session_state:
                                                    st.session_state[frame_key] = cv2.imencode('.png', display_frames[frame_idx])[1].tobytes()

                                                st.download_button(
                                                    label=f"📥 Download",
                                                    data=st.session_state[frame_key],
                                                    file_name=f"frame_{frame_idx:04d}.png",
                                                    mime="image/png",
                                                    key=f"download_frame_{frame_idx}",
                                                    use_container_width=True
                                                )

                            if len(display_frames) > 2:
                                st.subheader("✨ Interpolated Frames Only")
                                interpolated_frames = display_frames[1:-1]

                                num_interp = len(interpolated_frames)
                                num_cols = 3
                                num_rows = (num_interp + num_cols - 1) // num_cols

                                for row in range(num_rows):
                                    cols = st.columns(num_cols)
                                    for col_idx in range(num_cols):
                                        interp_idx = row * num_cols + col_idx
                                        if interp_idx < num_interp:
                                            with cols[col_idx]:
                                                actual_frame_idx = interp_idx + 1
                                                frame_rgb = cv2.cvtColor(display_frames[actual_frame_idx], cv2.COLOR_BGR2RGB)

                                                caption = f"✨ Interpolated Frame {interp_idx + 1} of {num_interp}"
                                                st.image(frame_rgb, caption=caption, use_container_width=True)

                                                interp_frame_key = f"interp_frame_bytes_{interp_idx}"
                                                if interp_frame_key not in st.session_state:
                                                    st.session_state[interp_frame_key] = cv2.imencode('.png', display_frames[actual_frame_idx])[1].tobytes()

                                                st.download_button(
                                                    label=f"📥 Download Frame {interp_idx + 1}",
                                                    data=st.session_state[interp_frame_key],
                                                    file_name=f"interpolated_frame_{interp_idx + 1:04d}.png",
                                                    mime="image/png",
                                                    key=f"download_interp_{interp_idx}",
                                                    use_container_width=True
                                                )

                        except Exception as e:
                            st.error(f"Error during interpolation: {str(e)}")
                            st.exception(e)
                else:
                    st.warning("⚠️ Please upload both frames to proceed.")
        else:
            st.subheader("Upload Video")
            video_file = st.file_uploader(
                "Upload Video File",
                type=['mp4', 'avi', 'mov', 'mkv'],
                key="video"
            )

            if video_file:
                project_root = get_project_root()
                temp_dir = os.path.join(project_root, "output", "temp")
                ensure_dir(temp_dir)

                video_path = os.path.join(temp_dir, "input_video.mp4")
                with open(video_path, "wb") as f:
                    f.write(video_file.read())

                video_info = get_video_info(video_path)
                st.info(f"📹 Video Info: {video_info['width']}x{video_info['height']}, {video_info['fps']:.2f} FPS, {video_info['frame_count']} frames")

                if st.button("🚀 Generate Interpolated Video", type="primary", use_container_width=True):
                    with st.spinner("Processing video..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        try:
                            status_text.text("Extracting frames from video...")
                            progress_bar.progress(20)

                            status_text.text("Interpolating frames...")
                            video_output_path, preview_video_path, metrics = interpolate_video(
                                video_path=video_path,
                                num_interpolations=num_interpolations,
                                resolution=resolution,
                                fps=fps
                            )

                            st.session_state.preview_video_path = preview_video_path

                            progress_bar.progress(90)
                            status_text.text("Finalizing...")

                            st.session_state.video_path = video_output_path
                            st.session_state.metrics = metrics

                            progress_bar.progress(100)
                            status_text.text("Complete!")

                            st.success("✅ Video processing complete!")

                            if os.path.exists(video_output_path):
                                if 'video_upload_download_bytes' not in st.session_state:
                                    with open(video_output_path, "rb") as output_video:
                                        st.session_state.video_upload_download_bytes = output_video.read()

                                st.download_button(
                                    label="📥 Download Video",
                                    data=st.session_state.video_upload_download_bytes,
                                    file_name="output_video.mp4",
                                    mime="video/mp4",
                                    use_container_width=True,
                                    key="download_video_upload"
                                )
                            else:
                                st.warning(f"Video file not found at: {video_output_path}")

                        except Exception as e:
                            st.error(f"Error during video processing: {str(e)}")
                            st.exception(e)

    with tab2:
        st.header("📊 Metrics & Analysis")

        if st.session_state.metrics:
            metrics = st.session_state.metrics

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Frames", metrics.get("total_frames", 0))
            with col2:
                st.metric("Interpolated Frames", metrics.get("interpolated_frames", 0))
            with col3:
                st.metric("Total Time", f"{metrics.get('total_time', 0):.2f}s")
            with col4:
                st.metric("Avg Time/Frame", f"{metrics.get('avg_frame_time', 0):.3f}s")

            if st.session_state.frames and len(st.session_state.frames) >= 2:
                st.subheader("Quality Metrics")

                frames = st.session_state.frames
                ssim_scores = []
                psnr_scores = []

                for i in range(len(frames) - 1):
                    ssim_val = calculate_ssim(frames[i], frames[i + 1])
                    psnr_val = calculate_psnr(frames[i], frames[i + 1])
                    ssim_scores.append(ssim_val)
                    psnr_scores.append(psnr_val)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average SSIM", f"{np.mean(ssim_scores):.4f}")
                with col2:
                    st.metric("Average PSNR", f"{np.mean(psnr_scores):.2f} dB")

                st.subheader("Metrics Visualization")

                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("SSIM Scores", "PSNR Scores"),
                    vertical_spacing=0.1
                )

                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(ssim_scores))),
                        y=ssim_scores,
                        mode='lines+markers',
                        name='SSIM',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(psnr_scores))),
                        y=psnr_scores,
                        mode='lines+markers',
                        name='PSNR',
                        line=dict(color='green', width=2)
                    ),
                    row=2, col=1
                )

                fig.update_xaxes(title_text="Frame Pair Index", row=2, col=1)
                fig.update_yaxes(title_text="SSIM", row=1, col=1)
                fig.update_yaxes(title_text="PSNR (dB)", row=2, col=1)
                fig.update_layout(height=600, showlegend=True)

                st.plotly_chart(fig, use_container_width=True)

                if len(frames) >= 2:
                    st.subheader("Optical Flow Visualization")
                    flow_vis = calculate_optical_flow(frames[0], frames[-1])
                    st.image(cv2.cvtColor(flow_vis, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.info("👆 Generate a video first to see metrics and analysis.")

    with tab3:
        st.header("📖 Documentation")

        st.markdown("""
        ## AI-Based Video Frame Interpolation System
        
        ### Overview
        
        This system uses deep learning models to generate intermediate frames between two consecutive 
        video frames, creating smooth motion and enabling frame rate upscaling.
        
        ### Models
        
        #### RIFE (Real-Time Intermediate Flow Estimation)
        - **Speed**: Fast, real-time capable
        - **Quality**: High quality for most scenarios
        - **Best for**: General purpose interpolation, real-time applications
        
        #### FILM (Frame Interpolation for Large Motion)
        - **Speed**: Moderate
        - **Quality**: Excellent for large movements
        - **Best for**: Complex motion, large frame differences
        
        ### System Architecture
        
        ```
        Input Frames → Model Loading → Frame Interpolation → Video Generation → Output
                              ↓
                        Quality Metrics (SSIM, PSNR)
        ```
        
        ### Installation
        
        1. Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
        
        2. Run the application:
        ```bash
        streamlit run app.py
        ```
        
        ### Usage
        
        1. **Upload Frames**: Upload two consecutive frames or a video file
        2. **Configure Settings**: Choose model, number of interpolations, resolution, and FPS
        3. **Generate**: Click "Generate Interpolated Video"
        4. **Download**: Download the output video
        
        ### Evaluation Metrics
        
        - **SSIM (Structural Similarity Index)**: Measures structural similarity (0-1, higher is better)
        - **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality in dB (higher is better)
        - **Processing Time**: Time taken per frame and total processing time
        
        ### Technical Details
        
        - **Framework**: PyTorch
        - **Video Processing**: OpenCV, MoviePy
        - **Web Interface**: Streamlit
        - **Metrics**: scikit-image
        
        ### Research Applications
        
        - Video frame rate upscaling
        - Slow-motion generation
        - Video restoration
        - Motion analysis
        - Computer vision research
        
        ### Future Work
        
        - Support for more interpolation models
        - Batch processing capabilities
        - Real-time video streaming
        - Advanced quality metrics
        - GPU acceleration optimization
        
        ### References
        
        - RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation
        - FILM: Frame Interpolation for Large Motion (Google Research)
        """)

        st.divider()
        st.markdown("**Project Version**: 1.0.0")
        st.markdown("**Author**: AI Research Team")


if __name__ == "__main__":
    main()


"""
Streamlit Web Application for AI-Based Video Frame Interpolation System.
"""

import streamlit as st
import os
import sys
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import tempfile

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.interpolate import interpolate_frames, interpolate_video
from core.metrics import calculate_ssim, calculate_psnr, calculate_optical_flow, evaluate_frame_sequence
from core.video_utils import extract_frames, get_video_info, create_comparison_gif
from core.utils import ensure_dir, get_project_root

# Page configuration
st.set_page_config(page_title="AI Frame Interpolation System", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

# Load external CSS
css_path = os.path.join(project_root, "static", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>.main-header{font-size:3rem;font-weight:bold;text-align:center;color:#1f77b4;margin-bottom:2rem}.sub-header{font-size:1.5rem;color:#666;text-align:center;margin-bottom:2rem}.metric-card{background-color:#f0f2f6;padding:1rem;border-radius:0.5rem;margin:0.5rem 0}</style>", unsafe_allow_html=True)

# Initialize session state
if 'frames' not in st.session_state:
    st.session_state.frames = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'preview_video_path' not in st.session_state:
    st.session_state.preview_video_path = None
if 'preview_frames_dir' not in st.session_state:
    st.session_state.preview_frames_dir = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🎬 AI-Based Video Frame Interpolation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate smooth intermediate frames using custom-trained deep learning model</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Number of interpolations
        num_interpolations = st.slider(
            "Number of Interpolated Frames",
            min_value=1,
            max_value=10,
            value=5,
            help="How many intermediate frames to generate between two frames"
        )
        
        # Resolution
        st.subheader("Output Resolution")
        col1, col2 = st.columns(2)
        with col1:
            width = st.number_input("Width", min_value=320, max_value=7680, value=1280, step=160)
        with col2:
            height = st.number_input("Height", min_value=240, max_value=4320, value=720, step=90)
        
        resolution = (width, height)
        
        # FPS
        fps = st.number_input(
            "Output FPS",
            min_value=1,
            max_value=120,
            value=30,
            help="Frames per second for output video"
        )
        
        st.divider()
        
        # About section
        with st.expander("ℹ️ About This Project"):
            st.markdown("""
            ### AI Frame Interpolation System
            
            This system uses deep learning models to generate intermediate frames 
            between two consecutive video frames, creating smooth motion.
            
            **Models:**
            - **RIFE**: Real-Time Intermediate Flow Estimation - Fast and efficient
            - **FILM**: Frame Interpolation for Large Motion - Better for large movements
            
            **Features:**
            - High-quality frame interpolation
            - SSIM and PSNR quality metrics
            - Optical flow visualization
            - Video export capabilities
            
            **Use Cases:**
            - Video frame rate upscaling
            - Slow-motion generation
            - Video restoration
            - Research and development
            """)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Generate", "📊 Metrics & Analysis", "📖 Documentation"])
    
    with tab1:
        st.header("Upload Frames or Video")
        
        # Input method selection
        input_method = st.radio(
            "Input Method",
            ["Two Frames", "Video File"],
            horizontal=True
        )
        
        if input_method == "Two Frames":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Frame 1")
                frame1_file = st.file_uploader(
                    "Upload First Frame",
                    type=['png', 'jpg', 'jpeg', 'bmp', 'jpg'],
                    key="frame1"
                )
                if frame1_file:
                    frame1_bytes = frame1_file.read()
                    frame1_np = np.frombuffer(frame1_bytes, np.uint8)
                    frame1_img = cv2.imdecode(frame1_np, cv2.IMREAD_COLOR)
                    st.image(cv2.cvtColor(frame1_img, cv2.COLOR_BGR2RGB), caption="Frame 1", use_container_width=True)
            
            with col2:
                st.subheader("Frame 2")
                frame2_file = st.file_uploader(
                    "Upload Second Frame",
                    type=['png', 'jpg', 'jpeg', 'bmp', 'jpg'],
                    key="frame2"
                )
                if frame2_file:
                    frame2_bytes = frame2_file.read()
                    frame2_np = np.frombuffer(frame2_bytes, np.uint8)
                    frame2_img = cv2.imdecode(frame2_np, cv2.IMREAD_COLOR)
                    st.image(cv2.cvtColor(frame2_img, cv2.COLOR_BGR2RGB), caption="Frame 2", use_container_width=True)
            
            # Generate button
            if st.button("🚀 Generate Interpolated Video", type="primary", use_container_width=True):
                if frame1_file and frame2_file:
                    with st.spinner("Generating interpolated frames..."):
                        # Save uploaded files temporarily
                        project_root = get_project_root()
                        temp_dir = os.path.join(project_root, "output", "temp")
                        ensure_dir(temp_dir)
                        
                        frame1_path = os.path.join(temp_dir, "frame1.png")
                        frame2_path = os.path.join(temp_dir, "frame2.png")
                        
                        cv2.imwrite(frame1_path, frame1_img)
                        cv2.imwrite(frame2_path, frame2_img)
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            # Interpolate frames
                            status_text.text("Loading custom trained model...")
                            progress_bar.progress(10)
                            
                            status_text.text("Generating intermediate frames...")
                            frames, video_path, preview_video_path, metrics = interpolate_frames(
                                frame1_path=frame1_path,
                                frame2_path=frame2_path,
                                num_interpolations=num_interpolations,
                                resolution=resolution,
                                fps=fps
                            )
                            
                            progress_bar.progress(90)
                            status_text.text("Finalizing video...")
                            
                            # Store all data in session state
                            st.session_state.frames = frames
                            st.session_state.video_path = video_path
                            st.session_state.preview_video_path = preview_video_path
                            st.session_state.preview_frames_dir = os.path.join(get_project_root(), "output", "previews")
                            st.session_state.metrics = metrics
                            
                            progress_bar.progress(100)
                            status_text.text("Complete!")
                            
                            st.success("✅ Interpolation complete!")
                            
                            # Display download button immediately without showing video preview
                            if os.path.exists(video_path):
                                # Store video bytes in session state to prevent reload
                                if 'video_download_bytes' not in st.session_state:
                                    with open(video_path, "rb") as video_file:
                                        st.session_state.video_download_bytes = video_file.read()
                                
                                st.download_button(
                                    label="📥 Download Video",
                                    data=st.session_state.video_download_bytes,
                                    file_name="output_video.mp4",
                                    mime="video/mp4",
                                    use_container_width=True,
                                    key="download_video_main"
                                )
                            else:
                                st.warning(f"Video file not found at: {video_path}")
                            
                            # Display all interpolated frames separately - always show from session state
                            st.subheader("🖼️ All Generated Frames")
                            
                            # Use frames from session state to ensure they persist
                            display_frames = st.session_state.frames if st.session_state.frames is not None else frames
                            
                            if len(display_frames) > 0:
                                # Display all frames in a grid
                                num_frames = len(display_frames)
                                
                                # Determine grid layout (3 columns)
                                num_cols = 3
                                num_rows = (num_frames + num_cols - 1) // num_cols
                                
                                # Display frames in grid
                                for row in range(num_rows):
                                    cols = st.columns(num_cols)
                                    for col_idx in range(num_cols):
                                        frame_idx = row * num_cols + col_idx
                                        if frame_idx < num_frames:
                                            with cols[col_idx]:
                                                frame_rgb = cv2.cvtColor(display_frames[frame_idx], cv2.COLOR_BGR2RGB)
                                                
                                                # Label frames appropriately
                                                if frame_idx == 0:
                                                    caption = "📷 Original Frame 1"
                                                elif frame_idx == num_frames - 1:
                                                    caption = f"📷 Original Frame 2"
                                                else:
                                                    # Show interpolated frame number (1-indexed for user)
                                                    interp_num = frame_idx
                                                    caption = f"✨ Interpolated Frame {interp_num}"
                                                
                                                st.image(frame_rgb, caption=caption, use_container_width=True)
                                                
                                                # Add download button for each frame - preview remains visible
                                                # Store frame bytes in session state to prevent reload
                                                frame_key = f"frame_bytes_{frame_idx}"
                                                if frame_key not in st.session_state:
                                                    st.session_state[frame_key] = cv2.imencode('.png', display_frames[frame_idx])[1].tobytes()
                                                
                                                st.download_button(
                                                    label=f"📥 Download",
                                                    data=st.session_state[frame_key],
                                                    file_name=f"frame_{frame_idx:04d}.png",
                                                    mime="image/png",
                                                    key=f"download_frame_{frame_idx}",
                                                    use_container_width=True
                                                )
                            
                            # Display only interpolated frames (excluding originals) - use display_frames
                            if len(display_frames) > 2:
                                st.subheader("✨ Interpolated Frames Only")
                                interpolated_frames = display_frames[1:-1]  # Exclude first and last (original frames)
                                
                                num_interp = len(interpolated_frames)
                                num_cols = 3
                                num_rows = (num_interp + num_cols - 1) // num_cols
                                
                                for row in range(num_rows):
                                    cols = st.columns(num_cols)
                                    for col_idx in range(num_cols):
                                        interp_idx = row * num_cols + col_idx
                                        if interp_idx < num_interp:
                                            with cols[col_idx]:
                                                # Get the actual frame (interpolated_frames is 1-indexed relative to display_frames)
                                                actual_frame_idx = interp_idx + 1
                                                frame_rgb = cv2.cvtColor(display_frames[actual_frame_idx], cv2.COLOR_BGR2RGB)
                                                
                                                caption = f"✨ Interpolated Frame {interp_idx + 1} of {num_interp}"
                                                st.image(frame_rgb, caption=caption, use_container_width=True)
                                                
                                                # Add download button - store in session state to prevent reload
                                                interp_frame_key = f"interp_frame_bytes_{interp_idx}"
                                                if interp_frame_key not in st.session_state:
                                                    st.session_state[interp_frame_key] = cv2.imencode('.png', display_frames[actual_frame_idx])[1].tobytes()
                                                
                                                st.download_button(
                                                    label=f"📥 Download Frame {interp_idx + 1}",
                                                    data=st.session_state[interp_frame_key],
                                                    file_name=f"interpolated_frame_{interp_idx + 1:04d}.png",
                                                    mime="image/png",
                                                    key=f"download_interp_{interp_idx}",
                                                    use_container_width=True
                                                )
                        
                        except Exception as e:
                            st.error(f"Error during interpolation: {str(e)}")
                            st.exception(e)
                else:
                    st.warning("⚠️ Please upload both frames to proceed.")
        
        else:  # Video file input
            st.subheader("Upload Video")
            video_file = st.file_uploader(
                "Upload Video File",
                type=['mp4', 'avi', 'mov', 'mkv'],
                key="video"
            )
            
            if video_file:
                # Save uploaded video temporarily
                project_root = get_project_root()
                temp_dir = os.path.join(project_root, "output", "temp")
                ensure_dir(temp_dir)
                
                video_path = os.path.join(temp_dir, "input_video.mp4")
                with open(video_path, "wb") as f:
                    f.write(video_file.read())
                
                # Display video info
                video_info = get_video_info(video_path)
                st.info(f"📹 Video Info: {video_info['width']}x{video_info['height']}, {video_info['fps']:.2f} FPS, {video_info['frame_count']} frames")
                
                # Generate button
                if st.button("🚀 Generate Interpolated Video", type="primary", use_container_width=True):
                    with st.spinner("Processing video..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            status_text.text("Extracting frames from video...")
                            progress_bar.progress(20)
                            
                            status_text.text("Interpolating frames...")
                            video_output_path, preview_video_path, metrics = interpolate_video(
                                video_path=video_path,
                                num_interpolations=num_interpolations,
                                resolution=resolution,
                                fps=fps
                            )
                            
                            # Store preview paths
                            st.session_state.preview_video_path = preview_video_path
                            
                            progress_bar.progress(90)
                            status_text.text("Finalizing...")
                            
                            st.session_state.video_path = video_output_path
                            st.session_state.metrics = metrics
                            
                            progress_bar.progress(100)
                            status_text.text("Complete!")
                            
                            st.success("✅ Video processing complete!")
                            
                            # Display download button immediately without showing video preview
                            if os.path.exists(video_output_path):
                                # Store video bytes in session state to prevent reload
                                if 'video_upload_download_bytes' not in st.session_state:
                                    with open(video_output_path, "rb") as output_video:
                                        st.session_state.video_upload_download_bytes = output_video.read()
                                
                                st.download_button(
                                    label="📥 Download Video",
                                    data=st.session_state.video_upload_download_bytes,
                                    file_name="output_video.mp4",
                                    mime="video/mp4",
                                    use_container_width=True,
                                    key="download_video_upload"
                                )
                            else:
                                st.warning(f"Video file not found at: {video_output_path}")
                        
                        except Exception as e:
                            st.error(f"Error during video processing: {str(e)}")
                            st.exception(e)
    
    with tab2:
        st.header("📊 Metrics & Analysis")
        
        if st.session_state.metrics:
            metrics = st.session_state.metrics
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Frames", metrics.get("total_frames", 0))
            with col2:
                st.metric("Interpolated Frames", metrics.get("interpolated_frames", 0))
            with col3:
                st.metric("Total Time", f"{metrics.get('total_time', 0):.2f}s")
            with col4:
                st.metric("Avg Time/Frame", f"{metrics.get('avg_frame_time', 0):.3f}s")
            
            # Quality metrics (if frames available)
            if st.session_state.frames and len(st.session_state.frames) >= 2:
                st.subheader("Quality Metrics")
                
                # Calculate SSIM and PSNR
                frames = st.session_state.frames
                ssim_scores = []
                psnr_scores = []
                
                for i in range(len(frames) - 1):
                    ssim_val = calculate_ssim(frames[i], frames[i + 1])
                    psnr_val = calculate_psnr(frames[i], frames[i + 1])
                    ssim_scores.append(ssim_val)
                    psnr_scores.append(psnr_val)
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average SSIM", f"{np.mean(ssim_scores):.4f}")
                with col2:
                    st.metric("Average PSNR", f"{np.mean(psnr_scores):.2f} dB")
                
                # Visualization
                st.subheader("Metrics Visualization")
                
                # Create plots
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("SSIM Scores", "PSNR Scores"),
                    vertical_spacing=0.1
                )
                
                # SSIM plot
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(ssim_scores))),
                        y=ssim_scores,
                        mode='lines+markers',
                        name='SSIM',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
                
                # PSNR plot
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(psnr_scores))),
                        y=psnr_scores,
                        mode='lines+markers',
                        name='PSNR',
                        line=dict(color='green', width=2)
                    ),
                    row=2, col=1
                )
                
                fig.update_xaxes(title_text="Frame Pair Index", row=2, col=1)
                fig.update_yaxes(title_text="SSIM", row=1, col=1)
                fig.update_yaxes(title_text="PSNR (dB)", row=2, col=1)
                fig.update_layout(height=600, showlegend=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Optical flow visualization
                if len(frames) >= 2:
                    st.subheader("Optical Flow Visualization")
                    flow_vis = calculate_optical_flow(frames[0], frames[-1])
                    st.image(cv2.cvtColor(flow_vis, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.info("👆 Generate a video first to see metrics and analysis.")
    
    with tab3:
        st.header("📖 Documentation")
        
        st.markdown("""
        ## AI-Based Video Frame Interpolation System
        
        ### Overview
        
        This system uses deep learning models to generate intermediate frames between two consecutive 
        video frames, creating smooth motion and enabling frame rate upscaling.
        
        ### Models
        
        #### RIFE (Real-Time Intermediate Flow Estimation)
        - **Speed**: Fast, real-time capable
        - **Quality**: High quality for most scenarios
        - **Best for**: General purpose interpolation, real-time applications
        
        #### FILM (Frame Interpolation for Large Motion)
        - **Speed**: Moderate
        - **Quality**: Excellent for large movements
        - **Best for**: Complex motion, large frame differences
        
        ### System Architecture
        
        ```
        Input Frames → Model Loading → Frame Interpolation → Video Generation → Output
                              ↓
                        Quality Metrics (SSIM, PSNR)
        ```
        
        ### Installation
        
        1. Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
        
        2. Run the application:
        ```bash
        streamlit run app.py
        ```
        
        ### Usage
        
        1. **Upload Frames**: Upload two consecutive frames or a video file
        2. **Configure Settings**: Choose model, number of interpolations, resolution, and FPS
        3. **Generate**: Click "Generate Interpolated Video"
        4. **Download**: Download the output video
        
        ### Evaluation Metrics
        
        - **SSIM (Structural Similarity Index)**: Measures structural similarity (0-1, higher is better)
        - **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality in dB (higher is better)
        - **Processing Time**: Time taken per frame and total processing time
        
        ### Technical Details
        
        - **Framework**: PyTorch
        - **Video Processing**: OpenCV, MoviePy
        - **Web Interface**: Streamlit
        - **Metrics**: scikit-image
        
        ### Research Applications
        
        - Video frame rate upscaling
        - Slow-motion generation
        - Video restoration
        - Motion analysis
        - Computer vision research
        
        ### Future Work
        
        - Support for more interpolation models
        - Batch processing capabilities
        - Real-time video streaming
        - Advanced quality metrics
        - GPU acceleration optimization
        
        ### References
        
        - RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation
        - FILM: Frame Interpolation for Large Motion (Google Research)
        """)
        
        st.divider()
        st.markdown("**Project Version**: 1.0.0")
        st.markdown("**Author**: AI Research Team")


if __name__ == "__main__":
    main()