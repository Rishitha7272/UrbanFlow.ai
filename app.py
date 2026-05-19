import argparse
import os
import cv2
from traffic_controller import TrafficController
from visualizer import Visualizer
from utils import generate_mock_video
from config import DEFAULT_ROIS
from logger_config import logger

def parse_args():
    parser = argparse.ArgumentParser(description="AI Traffic Signal Optimization System")
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to input traffic video file. If not provided, a simulated one will be generated."
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="YOLO model path or name (e.g. yolov8n.pt) or 'mock' to run in mock mode."
    )
    parser.add_argument(
        "--conf", type=float, default=0.3,
        help="Confidence threshold for vehicle detection."
    )
    parser.add_argument(
        "--output", type=str, default="output_traffic.mp4",
        help="Path to save the annotated output video."
    )
    parser.add_argument(
        "--export", type=str, default="traffic_analytics.csv",
        help="Path to export the CSV analytics report."
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Enable real-time GUI window using OpenCV imshow (requires a screen/X11)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Initializing AI Traffic Signal Optimization System...")
    
    # Resolve input video source
    video_path = args.video
    if not video_path:
        video_path = "sample_traffic.mp4"
        if not os.path.exists(video_path):
            logger.info("No video file provided. Generating simulated traffic video...")
            generate_mock_video(video_path, duration_sec=30, fps=30)
            
    if not os.path.exists(video_path):
        logger.error(f"Error: Video file '{video_path}' not found.")
        return

    # Initialize components
    try:
        controller = TrafficController(model_path=args.model, confidence=args.conf)
        visualizer = Visualizer()
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video '{video_path}'.")
        return

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # We will resize internally to 850x480 for consistency and visual layout
    target_width, target_height = 850, 480
    logger.info(f"Input Video: {video_path} ({width}x{height} @ {fps} FPS, {total_frames} frames)")
    logger.info(f"Internal Target Processing Resolution: {target_width}x{target_height}")

    # Setup video writer
    out_writer = None
    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output, fourcc, fps, (target_width, target_height))
        logger.info(f"Writing annotated output to: {args.output}")

    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame to standardized processing scale
            processed_frame = cv2.resize(frame, (target_width, target_height))
            
            # Process frame using Controller (detection + optimization state machine)
            detections_by_lane, raw_detections, signal_colors = controller.process_frame(
                processed_frame, 
                rois=DEFAULT_ROIS, 
                frame_idx=frame_idx, 
                fps=fps
            )
            
            # Draw visual feedback overlay
            annotated_frame = visualizer.draw_all(
                processed_frame, 
                detections_by_lane, 
                raw_detections, 
                signal_colors, 
                controller
            )
            
            # Save frame to output video
            if out_writer:
                out_writer.write(annotated_frame)
                
            # Show live OpenCV GUI if flag is enabled
            if args.gui:
                cv2.imshow("AI Smart Traffic Optimization", annotated_frame)
                # Press 'q' to quit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Execution stopped by user.")
                    break
                    
            frame_idx += 1
            if frame_idx % 100 == 0:
                percent = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                logger.info(f"Processing progress: {frame_idx} frames ({percent:.1f}%)")

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by KeyboardInterrupt.")
    except Exception as e:
        logger.error(f"Inference run error: {e}")
    finally:
        # Cleanup
        cap.release()
        if out_writer:
            out_writer.release()
        if args.gui:
            cv2.destroyAllWindows()
            
        # Export final CSV stats
        if args.export:
            controller.export_csv(args.export)
            
        logger.info(f"Processing finished. Total frames processed: {frame_idx}")

if __name__ == "__main__":
    main()
