from src.tracker import Sort
from ultralytics import YOLO
import os


class LicensePlatesDetector:
    def __init__(self, vehicle_detection_model_name, license_plate_detector_model_name):
        # Initialize SORT Tracker
        self.tracker = Sort()

        # Load Vehicle Detection Model
        self.vehicle_detection_model = YOLO(vehicle_detection_model_name)

        # Load License Plate Detection Model
        self.license_plate_detector_model = YOLO(os.path.join("models", license_plate_detector_model_name))