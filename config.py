import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# YOLO settings
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.3))

# Detection classes of interest (COCO indices)
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Passenger Car Unit (PCU) weights for traffic density calculations
PCU_WEIGHTS = {
    "car": 1.0,
    "motorcycle": 0.5,
    "bus": 2.5,
    "truck": 2.5,
    "ambulance": 3.0  # Custom priority class
}

# Default timing parameters (seconds)
MIN_GREEN_TIME = int(os.getenv("MIN_GREEN_TIME", 10))
MAX_GREEN_TIME = int(os.getenv("MAX_GREEN_TIME", 60))
YELLOW_TIME = int(os.getenv("YELLOW_TIME", 3))
RED_ALL_TIME = int(os.getenv("RED_ALL_TIME", 2))
MAX_WAITING_THRESHOLD = int(os.getenv("MAX_WAITING_THRESHOLD", 120))

# Path configurations
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output_videos")
LOG_FILE = os.getenv("LOG_FILE", "traffic_system.log")

# GUI options
SHOW_GUI = os.getenv("SHOW_GUI", "False").lower() in ("true", "1", "yes")
SAVE_OUTPUT = os.getenv("SAVE_OUTPUT", "True").lower() in ("true", "1", "yes")

# Default ROI Polygons for multi-lane or single-camera intersection processing.
# Coordinates are normalized (x, y) where (0.0, 0.0) is top-left and (1.0, 1.0) is bottom-right.
# This makes it resolution-independent!
DEFAULT_ROIS = {
    "North": [
        (0.45, 0.10),
        (0.55, 0.10),
        (0.58, 0.40),
        (0.42, 0.40)
    ],
    "South": [
        (0.40, 0.60),
        (0.60, 0.60),
        (0.70, 0.95),
        (0.30, 0.95)
    ],
    "East": [
        (0.60, 0.42),
        (0.90, 0.35),
        (0.90, 0.65),
        (0.60, 0.58)
    ],
    "West": [
        (0.10, 0.35),
        (0.40, 0.42),
        (0.40, 0.58),
        (0.10, 0.65)
    ]
}

# Signal Phase mapping
# Usually we pair opposite directions (e.g., North-South together, East-West together)
# to maximize efficiency and prevent conflicting crossings.
PHASES = [
    {
        "id": 0,
        "name": "North-South Green",
        "lanes": ["North", "South"]
    },
    {
        "id": 1,
        "name": "East-West Green",
        "lanes": ["East", "West"]
    }
]

# Color map for signals
COLORS = {
    "red": (0, 0, 255),       # BGR
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "roi": (255, 255, 0),     # Cyan-blue/Yellow
    "bbox": (0, 165, 255),    # Orange
    "text": (255, 255, 255)
}
