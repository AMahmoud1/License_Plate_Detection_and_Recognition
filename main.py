import argparse
from src.license_plates_detector import LicensePlatesDetector

if __name__ == "__main__":
    # Define Argument Parser
    parser = argparse.ArgumentParser(description="License Plate Detection and Recognition")
    parser.add_argument(
        "--vehicle-detection-model-name", 
        type=str, 
        default="yolov8n.pt", 
        help="Path to the vehicle detection model (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--license-plate-detector-model-name", 
        type=str, 
        default="license_plate_detector.pt", 
        help="Path to the license plate detector model (default: license_plate_detector.pt)"
    )
    parser.add_argument(
        "--input-video-path", 
        type=str, 
        default="input_video.mp4", 
        help="Path to the input video file (default: input_video.mp4)"
    )
    parser.add_argument(
        "--output-video-path", 
        type=str, 
        default="output_video.avi", 
        help="Path to the output video file (default: output_video.avi)"
    )

    # Parse Arguments
    args = parser.parse_args()

    # Create License Plates Detector Instance
    license_plates_detector = LicensePlatesDetector(
        vehicle_detection_model_name=args.vehicle_detection_model_name,
        license_plate_detector_model_name=args.license_plate_detector_model_name,
        input_video_path=args.input_video_path,
        output_video_path=args.output_video_path,
    )

    # Run License Plates Detector
    license_plates_detector.run()
