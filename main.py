from src.license_plates_detector import LicensePlatesDetector

if __name__ == "__main__":
    # Define Argument
    vehicle_detection_model_name='yolov8n.pt'
    license_plate_detector_model_name='license_plate_detector.pt'
    input_video_path = "input_video.mp4"

    # Create License Plates Detector Instance
    license_plates_detector = LicensePlatesDetector(
        vehicle_detection_model_name=vehicle_detection_model_name,
        license_plate_detector_model_name=license_plate_detector_model_name,
        input_video_path=input_video_path,        
    )

    # Run License Plates Detector
    license_plates_detector.run()