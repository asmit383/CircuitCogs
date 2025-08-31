# Fire Detection and Analysis System
## Overview
This project is a comprehensive fire detection and analysis system that uses computer vision and machine learning to detect fires, analyze flame characteristics, and identify burning materials. The system includes a GUI interface built with CustomTkinter and a Flask API server for remote access.

## Features
- Fire Detection : Uses a custom YOLO model to detect fire regions in images
- Fire Segmentation : Segments fire regions using a custom YOLO segmentation model
- Flame Analysis : Applies multiple clustering techniques (K-Means++, GMM, DBSCAN, Mean-Shift) to analyze flame characteristics
- Material Identification : Identifies burning materials based on flame color analysis
- Extinguisher Recommendation : Suggests appropriate fire extinguishers based on detected materials
- Flask API : Provides RESTful API endpoints for remote fire analysis
- GUI Interface : User-friendly interface for uploading and analyzing images
## Requirements
- Python 3.x
- OpenCV
- NumPy
- TensorFlow/PyTorch (for YOLO models)
- scikit-learn
- Flask
- CustomTkinter
- Ultralytics YOLO
- Other dependencies listed in the imports


## API Endpoints
- /api/status : Check API status
- /api/results : Get the latest analysis results
## Project Structure
- main.py : Main application with GUI and analysis logic
- robot.py : Additional functionality
- screen.py : Screen management
- APIreq.py : API request handling
- flame_dataset.json : Database of flame characteristics for different materials
- OBJ_best.pt : Custom YOLO detection model
- SEG_best.pt : Custom YOLO segmentation model

This README provides an overview of your fire detection and analysis system, including its features, installation instructions, and usage guidelines. The .gitignore file is configured to ignore Python cache files, the fire directory, specific image files (5.png, 6.jpeg, 7.jpeg), and other common files while preserving the git files needed for pushing to a repository.
