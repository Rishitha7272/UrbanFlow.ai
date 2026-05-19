# AI-Powered Traffic Signal Optimization System

An intelligent, real-time transportation system designed to automatically analyze vehicle congestion at intersections and optimize green light signals to reduce travel delays, fuel consumption, and waiting times.

---

## 🌟 Key Features

1. **Computer Vision Vehicle Ingestion**: Detects and counts vehicle classes (Cars, Trucks, Buses, Motorcycles) from intersection video feeds using **Ultralytics YOLOv8**.
2. **Traffic Flow Modeling (PCU)**: Employs Passenger Car Unit (PCU) weighting to model traffic volume dynamically (e.g., Heavy Trucks/Buses are weighted higher than motorcycles).
3. **Adaptive Signal Control Logic**:
   - **Dynamic Green Phase**: Adjusts green time duration dynamically based on the current lane density ratio.
   - **Yellow & Red-All Transitions**: Enforces standard safety transition phases (Yellow, Red-all buffer) to prevent accidents.
4. **Starvation Prevention**: Tracks waiting times for all lanes; if a lane waits longer than a maximum threshold, it receives overriding green priority.
5. **Emergency Vehicle Priority Override**: Detects emergency lights (e.g., ambulances) using an HSV color-strobe heuristic, triggering a safe transition to clear the active emergency lane immediately.
6. **Live Interactive Dashboard**: Streamlit interface containing live frame feeds, real-time metric tiles (active phase, timer, congestion index), emergency alerts, and raw data export.
7. **Developer Mock Simulation Fallback**: Auto-falls back to a lightweight simulated traffic engine if YOLO weights or camera hardware are unavailable, making the repository fully runnable on any machine out-of-the-box.

---

## 📂 Project Structure

```
traffic/
│
├── app.py                     # Main CLI command entrypoint
├── dashboard.py               # Streamlit-based interactive web UI
├── detector.py                # YOLOv8 object detection & polygon ROI filtering
├── signal_logic.py            # Phase state machine & dynamic green scheduling
├── traffic_controller.py      # Core coordinator linking detection & signals
├── logger_config.py           # Structured logging setup (writes to traffic_system.log)
├── config.py                  # Core variables (ROI coordinates, PCU weights, timing bounds)
├── utils.py                   # Custom synthetic traffic video generator
├── requirements.txt           # Python library requirements
├── .env.example               # Config environment template
├── .env                       # Local environment configurations
├── traffic_system.log         # Auto-generated application logs
└── traffic_analytics.csv      # Auto-generated simulation report
```

---

## 🚀 Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Installation
Clone this repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Adjust variables in `.env` to configure default thresholds or turn off GUI rendering if running on headless server environments.

---

## 💻 How to Run

### Command Line Interface (CLI)

#### A. Run with YOLOv8 on an existing video:
Provide your custom video file:
```bash
python3 app.py --video your_traffic_video.mp4 --model yolov8n.pt
```

#### B. Run the Developer Mock Simulation:
Runs the entire system using a programmatically generated simulation video and mock traffic logic. Perfect for quick validation without downloading heavy ML weights:
```bash
python3 app.py --model mock
```

---

### Streamlit Dashboard

Launch the browser-based visualization dashboard:
```bash
streamlit run dashboard.py
```
This launches a local dashboard page (usually at `http://localhost:8501`) showing:
- Real-time video player displaying ROI overlays, bounding boxes, and traffic status.
- Current active phase name and countdown timer.
- Live vehicle counting logs and congestion level metrics.
- PDF/CSV exporter of timing history data.

---

## 📊 Analytics and Logs
- **System logs**: Track operational steps and AI explainability reasons in `traffic_system.log`.
- **CSV Data Reports**: Analyze vehicle throughput and calculated signal intervals inside `traffic_analytics.csv`.
