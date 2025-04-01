#!/bin/bash

# Default values
vehicle_detection_model_name="yolov8n.pt"
license_plate_detector_model_name="license_plate_detector.pt"
input_video_path="input_video.mp4"
output_video_path="output_video.avi"

# Name of the Conda virtual environment
VENV_NAME="license_plate_venv"

# Check if the Conda environment exists
env_exists=$(conda env list | grep "^$VENV_NAME")

if [ -z "$env_exists" ]; then
    echo "Conda environment '$VENV_NAME' does not exist. Creating..."
    conda create -y -n $VENV_NAME python=3.9
else
    echo "Conda environment '$VENV_NAME' already exists."
fi

# Activate the Conda environment
source activate $VENV_NAME || conda activate $VENV_NAME

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt
echo "All packages installed."

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --vehicle-detection-model-name) vehicle_detection_model_name="$2"; shift ;;
        --license-plate-detector-model-name) license_plate_detector_model_name="$2"; shift ;;
        --input-video-path) input_video_path="$2"; shift ;;
        --output-video-path) output_video_path="$2"; shift ;;
        *) echo "Unknown option $1"; exit 1 ;;
    esac
    shift
done

# Run The Application
echo "Running License Plate Detection and Recognition Application..."
python3 main.py --vehicle-detection-model-name "$vehicle_detection_model_name" \
                --license-plate-detector-model-name "$license_plate_detector_model_name" \
                --input-video-path "$input_video_path" \
                --output-video-path "$output_video_path"
echo "License Plate Detection and Recognition Application Finished Executing Successfully!"
