import streamlit as st
import cv2
import tempfile
import time
import os
import pandas as pd
from traffic_controller import TrafficController
from visualizer import Visualizer
from config import DEFAULT_ROIS

st.set_page_config(
    page_title="AI Smart Traffic Signal System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for vibrant, premium dashboard
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .metric-card {
        background-color: #1e222b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3440;
        text-align: center;
    }
    .emergency-alert {
        background-color: #ff0000;
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        animation: blinker 1.5s linear infinite;
    }
    @keyframes blinker {
        50% { opacity: 0.3; }
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚦 AI Traffic Signal Optimization Dashboard")
st.subheader("Computer Vision & Dynamic Signal Timing Analytics")

# Sidebar Configuration
st.sidebar.header("🔧 System Configurations")
yolo_model = st.sidebar.text_input("YOLO weights path/model", "yolov8n.pt")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

# Video Ingestion Options
st.sidebar.header("📁 Input Feed Source")
video_source_type = st.sidebar.radio("Source Type", ["Upload Video", "Default Sample Video"])

video_file_path = None
uploaded_file = None

if video_source_type == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a traffic video (MP4/AVI)", type=["mp4", "avi"])
    if uploaded_file is not None:
        # Save uploaded file to temp file to retrieve its path
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_file_path = tfile.name
else:
    # Look for sample videos in workspace
    default_videos = [f for f in os.listdir(".") if f.endswith((".mp4", ".avi"))]
    if default_videos:
        selected_video = st.sidebar.selectbox("Select Sample Video", default_videos)
        video_file_path = os.path.abspath(selected_video)
    else:
        st.sidebar.info("No sample videos found in directory. Please upload one.")

# Run control buttons
col1, col2 = st.sidebar.columns(2)
start_btn = col1.button("▶️ Start Simulation")
stop_btn = col2.button("⏸️ Stop / Reset")

# Main Dashboard Layout
m1, m2, m3, m4 = st.columns(4)
metric_phase = m1.empty()
metric_timer = m2.empty()
metric_congestion = m3.empty()
metric_emergency = m4.empty()

# Real-time Video frame container
video_container = st.empty()

# Table and stats display
table_container = st.empty()

# Define processing loop
if start_btn and video_file_path:
    # Initialize controller and visualizer
    try:
        controller = TrafficController(model_path=yolo_model, confidence=conf_threshold)
        visualizer = Visualizer()
        
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            st.error("Error: Could not open video source.")
            st.stop()
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
            
        frame_idx = 0
        
        # Stream frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame to optimize processing speed
            frame = cv2.resize(frame, (850, 480))
            
            # Process frame using TrafficController
            detections_by_lane, raw_detections, signal_colors = controller.process_frame(
                frame, 
                rois=DEFAULT_ROIS, 
                frame_idx=frame_idx, 
                fps=fps
            )
            
            # Render visual overlays
            annotated_frame = visualizer.draw_all(
                frame, 
                detections_by_lane, 
                raw_detections, 
                signal_colors, 
                controller
            )
            
            # Convert BGR (OpenCV) to RGB (Streamlit display)
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Update Video feed UI
            video_container.image(rgb_frame, channels="RGB", use_container_width=True)
            
            # Update Metric Tiles
            active_phase = controller.optimizer.phases[controller.optimizer.active_phase_idx]["name"]
            timer_left = int(controller.optimizer.timer)
            current_state = controller.optimizer.current_state
            
            metric_phase.markdown(f"""
                <div class="metric-card">
                    <h4>Active Phase</h4>
                    <h2 style='color:#00ff55;'>{active_phase}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            timer_color = "#ffff00" if current_state == "YELLOW" else ("#ff0000" if current_state == "RED_ALL" else "#00ff55")
            metric_timer.markdown(f"""
                <div class="metric-card">
                    <h4>Signal Timer</h4>
                    <h2 style='color:{timer_color};'>{timer_left}s <span style='font-size:16px;'>[{current_state}]</span></h2>
                </div>
            """, unsafe_allow_html=True)
            
            total_density = sum(controller.current_densities.values())
            congestion_level = "LOW" if total_density < 3 else ("MEDIUM" if total_density < 6 else "HIGH")
            congestion_color = "#00ff55" if congestion_level == "LOW" else ("#ffff00" if congestion_level == "MEDIUM" else "#ff0000")
            metric_congestion.markdown(f"""
                <div class="metric-card">
                    <h4>Congestion Level</h4>
                    <h2 style='color:{congestion_color};'>{congestion_level} <span style='font-size:14px;'>({total_density:.1f} PCU)</span></h2>
                </div>
            """, unsafe_allow_html=True)
            
            if controller.optimizer.emergency_override_active:
                metric_emergency.markdown(f"""
                    <div class="emergency-alert">
                        ⚠️ EMERGENCY ACTIVE
                        <p style='font-size:12px; margin:0;'>Lane: {controller.optimizer.emergency_lane}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                metric_emergency.markdown("""
                    <div class="metric-card">
                        <h4>Emergency Alert</h4>
                        <h2 style='color:#aaaaaa;'>None</h2>
                    </div>
                """, unsafe_allow_html=True)
                
            # Update live stats table
            stats_df = pd.DataFrame({
                "Lane Direction": list(controller.current_counts.keys()),
                "Vehicle Count": list(controller.current_counts.values()),
                "Traffic Density (PCU)": list(controller.current_densities.values()),
                "Signal Color": [signal_colors.get(lane, 'RED') for lane in controller.current_counts.keys()]
            })
            table_container.dataframe(stats_df, hide_index=True, use_container_width=True)
            
            frame_idx += 1
            # Add short delay to simulate real-time playback
            time.sleep(1.0 / fps)
            
        cap.release()
        st.success("Simulation finished! Output metrics stored in log.")
        
        # Export CSV report
        csv_filename = "traffic_analytics.csv"
        if controller.export_csv(csv_filename):
            st.download_button(
                label="📥 Download Traffic Optimization Report",
                data=open(csv_filename, 'rb').read(),
                file_name=csv_filename,
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"Failed to execute simulation pipeline: {e}")
        logger.error(f"Inference error in dashboard loop: {e}")
elif start_btn:
    st.warning("Please upload a video or select a sample video first.")
