from src.license_plates_detector import LicensePlatesDetector

if __name__ == "__main__":
    # Define Argument
    vehicle_detection_model_name='yolov8n.pt'
    license_plate_detector_model_name='license_plate_detector.pt'

    # Create License Plates Detector Instance
    license_plates_detector = LicensePlatesDetector(
        vehicle_detection_model_name=vehicle_detection_model_name,
        license_plate_detector_model_name=license_plate_detector_model_name,
        
    )