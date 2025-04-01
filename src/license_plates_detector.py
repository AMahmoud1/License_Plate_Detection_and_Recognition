import os

import cv2
import numpy as np
from ultralytics import YOLO

from src.ocr_reader import OCRReader
from src.tracker import Sort


class LicensePlatesDetector:
    def __init__(
        self,
        vehicle_detection_model_name,
        license_plate_detector_model_name,
        input_video_path,
        output_video_path,
    ):
        # Set Class Attributes
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

        # Initialize SORT Tracker
        self.tracker = Sort()

        # Load Vehicle Detection Model
        self.vehicle_detection_model = YOLO(vehicle_detection_model_name)

        # Load License Plate Detection Model
        self.license_plate_detector_model = YOLO(
            os.path.join("models", license_plate_detector_model_name)
        )

        # Initialize OCR Reader
        self.ocr_reader = OCRReader(gpu=True)

        # Define Vehicles Class id in COCO Dataset
        self.vehicles_classes_id = [2, 3, 5, 7]

    def run(self):
        # Initialize Results Dictionary to store the results
        results_dictionary = {}

        # Load Video
        cap = cv2.VideoCapture(self.input_video_path)

        # Get Video properties (e.g., frame size, FPS) for output video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Initialize VideoWriter to save the output video
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Or use 'mp4v' for .mp4 output
        out = cv2.VideoWriter(
            self.output_video_path, fourcc, fps, (frame_width, frame_height)
        )

        # Read Frames
        frame_counter = 0
        ret = True

        # Main Loop
        while ret:
            frame_counter += 1
            ret, frame = cap.read()
            if ret:
                # Initialize Results Dictionary instance for this frame
                results_dictionary[frame_counter] = {}

                # Detect Vehicles
                detections = self.vehicle_detection_model(frame)[0]

                # Extract Vehicle Detections
                detections_list = self.extract_vehicle_detections(detections)

                # Track Vehicles
                track_ids = self.tracker.update(np.asarray(detections_list))

                # Detect License Plates
                license_plates = self.license_plate_detector_model(frame)[0]
                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, _ = license_plate

                    # assign license plate to car
                    xcar1, ycar1, xcar2, ycar2, car_id = self.filter_cars(
                        license_plate, track_ids
                    )

                    if car_id != -1:
                        license_plate_text, license_plate_text_score = (
                            self.ocr_reader.read_license_plate(
                                frame[int(y1) : int(y2), int(x1) : int(x2), :]
                            )
                        )

                        if license_plate_text is not None:
                            results_dictionary[frame_counter][car_id] = {
                                "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                                "license_plate": {
                                    "bbox": [x1, y1, x2, y2],
                                    "text": license_plate_text,
                                    "bbox_score": score,
                                    "text_score": license_plate_text_score,
                                },
                            }

                            # For the car ID
                            cv2.rectangle(
                                frame,
                                (int(xcar1), int(ycar1)),
                                (int(xcar2), int(ycar2)),
                                (0, 255, 0),  # Green color for the bounding box
                                6,  # Further increased thickness for bolder bounding box
                            )

                            # Adding a background rectangle to the label
                            label_text = f"Car ID: {car_id}"
                            (label_width, label_height), _ = cv2.getTextSize(
                                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3  # Increased font scale and thickness
                            )
                            cv2.rectangle(
                                frame,
                                (int(xcar1), int(ycar1) - 40),  # Adjusted positioning for larger font
                                (int(xcar1) + label_width, int(ycar1) - 10),
                                (0, 255, 0),  # Background color same as bbox (green)
                                -1,  # Filled rectangle for background
                            )
                            cv2.putText(
                                frame,
                                label_text,
                                (int(xcar1), int(ycar1) - 10),  # Label position
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,  # Increased font scale
                                (255, 255, 255),  # White text on green background
                                3,  # Increased text thickness
                            )

                            # For the license plate
                            cv2.rectangle(
                                frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (255, 0, 0),  # Red color for the bounding box
                                6,  # Further increased thickness for bolder bounding box
                            )

                            # Adding a background rectangle to the license plate label
                            license_plate_label = f"{license_plate_text}"
                            (label_width, label_height), _ = cv2.getTextSize(
                                license_plate_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3  # Increased font scale and thickness
                            )
                            cv2.rectangle(
                                frame,
                                (int(x1), int(y1) - 40),  # Adjusted positioning for larger font
                                (int(x1) + label_width, int(y1) - 10),
                                (255, 0, 0),  # Background color same as bbox (red)
                                -1,  # Filled rectangle for background
                            )
                            cv2.putText(
                                frame,
                                license_plate_label,
                                (int(x1), int(y1) - 10),  # Label position
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,  # Increased font scale
                                (255, 255, 255),  # White text on red background
                                3,  # Increased text thickness
                            )

            # Write the annotated frame to the output video
            out.write(frame)

        # Release Video Capture
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def filter_cars(self, license_plate, vehicle_track_ids):
        """
        Retrieve the vehicle coordinates and ID based on the license plate coordinates.

        Args:
            license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
            vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

        Returns:
            tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
        """
        # Extract License Plate Coordinates
        x1, y1, x2, y2, _, _ = license_plate

        # Filter Cars
        vehicle_found_flag = False
        for j in range(len(vehicle_track_ids)):
            # Extract Car Coordinates
            xcar1, ycar1, xcar2, ycar2, _ = vehicle_track_ids[j]

            # Check if the license plate is inside the car bounding box
            if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
                car_indx = j
                vehicle_found_flag = True
                break

        # Return Vehicle Coordinates and ID
        if vehicle_found_flag:
            return vehicle_track_ids[car_indx]

        # Return -1 if no vehicle is found
        return -1, -1, -1, -1, -1

    def extract_vehicle_detections(self, detections):
        """
        Extract vehicle detections from the given detections."

        Args:
            detections (list): List of vehicle detections.

        Returns:
            list: List of filtered vehicle detections.
        """
        detections_list = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in self.vehicles_classes_id:
                detections_list.append([x1, y1, x2, y2, score])
        return detections_list
