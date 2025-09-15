## Accident Detector (YOLOv8 + OpenCV)

An end-to-end project that detects vehicle accidents in videos using a custom-trained YOLOv8 model. The repo includes:
- A runnable detection script (main.py) that performs inference on a video and visualizes detections.
- A frame extraction utility (img.py) to generate images from a video for dataset creation.
- A training/evaluation notebook (yolov8_object_detection_on_custom_dataset.ipynb).
- The trained model weights (best.pt) and example assets.

### Key Capabilities
- Real-time accident detection on video streams (file input by default; webcam supported with a small tweak).
- Bounding boxes and class labels overlay with color-coding (red for accident, green otherwise).
- Custom class list loaded from coco1.txt.
- Dataset scaffolding included under diff/ for reference (images and YOLO-format labels for training/validation).

---

## Table of Contents
- Overview
- Repository Structure
- Requirements
- Setup
- How It Works
- Usage
  - Run on a video file
  - Run on a webcam
  - Extract frames for dataset creation
- Dataset and Training
- Configuration Notes
- Performance Tips
- Troubleshooting
- FAQ
- Credits

---

## Overview
This project demonstrates an accident detection pipeline built on Ultralytics YOLOv8 and OpenCV. A pre-trained model (best.pt) is used to detect the presence of accidents in each video frame, draw bounding boxes, and display results interactively.

The repository also contains a notebook showcasing how to train or fine-tune YOLOv8 on a custom dataset (e.g., accident vs non-accident). Example dataset structure is included to illustrate YOLO-format labels.

---

## Repository Structure
```

Accident-Detector-Python-main/
├─ best.pt                      # Trained YOLOv8 weights (custom)
├─ coco1.txt                    # Class names (one label per line; includes 'accident')
├─ cr.mp4                       # Example input video
├─ data.txt                     # Ancillary data file (not used by main.py)
├─ diff/                        # Example dataset structure (images + labels)
│  ├─ images/
│  │  ├─ training/
│  │  └─ validation/
│  └─ labels/
│     ├─ training/
│     └─ validation/
├─ images/                      # Additional images (mirrors training/validation split)
├─ img.py                       # Frame extraction utility from video
├─ main.py                      # Accident detection (inference + visualization)
├─ yolov8_object_detection_on_custom_dataset.ipynb  # Training/eval notebook
└─ README.md
```

Notes:
- best.pt is required by main.py. Without it, inference will fail.
- coco1.txt must include the class name accident for color-coding to work as intended.

---

## Requirements
- Python 3.9 or newer (3.8+ typically works, but 3.9+ is recommended)
- OS: Windows, macOS, or Linux (repo paths/examples geared to Windows)

Python packages:

pip install ultralytics opencv-python cvzone pandas numpy jupyter

If you plan to use GPU acceleration with CUDA, install a PyTorch build compatible with your CUDA version before installing ultralytics.

---

## Setup
1. Ensure Python is installed and available in your PATH.
2. Create and activate a virtual environment (recommended).
3. Install the dependencies listed above.
4. Place your model weights as best.pt in the repo root (already present in this repo).
5. Ensure coco1.txt contains your class names, one per line. Include accident exactly if you expect red boxes for that class.

---

## How It Works
At a high level, main.py:
1. Loads the YOLOv8 model from best.pt using Ultralytics.
2. Opens a video stream (cr.mp4 by default) using OpenCV.
3. Reads frames in a loop, optionally downsampling by processing every 3rd frame to improve speed.
4. Runs model.predict(frame) to obtain bounding boxes, class indices, and scores.
5. Maps class indices to class names from coco1.txt.
6. Draws bounding boxes and labels with cv2.rectangle and cvzone.putTextRect.
   - If the detected class string contains accident, the box is red; otherwise green.
7. Displays the result in a window named Accident Detector.
8. Press Esc to exit. If the video ends, it loops back to the start.

Important implementation details from main.py:
- Frames are resized to 1020x500 for display.
- A simple mouse move callback prints the current pointer coordinates in the window (useful for quick inspection).
- The script does not save output video by default. It only displays the visualization.

img.py is a helper that:
- Opens cr.mp4, resizes frames to 1080x500, and writes every 3rd frame to disk.
- By default, it saves to a hard-coded path. Update the path to your environment before running.

---

## Usage

### Run on a video file (default)
Ensure cr.mp4 is in the repository root or change the path in main.py.

python main.py

Controls:
- Press Esc to quit the display window.
- The video loops automatically when it reaches the end.

### Run on a webcam
Edit the video capture line in main.py:
python
cap = cv2.VideoCapture(0)  # 0 or 1 depending on your camera index

Then run:

python main.py


### Extract frames for dataset creation
Update the output directory in img.py to a directory on your machine, then run:

python img.py

This will write frames like car_0.jpg, car_1.jpg, ... to the specified path.

---

## Dataset and Training
- The diff/ directory shows a typical YOLO dataset layout with images and corresponding YOLO-format labels.
- Labels are text files with lines of the form: class_id x_center y_center width height normalized to [0, 1].
- Use the notebook yolov8_object_detection_on_custom_dataset.ipynb to train/evaluate a model on your dataset.

Typical YOLOv8 training commands (from a terminal) look like:

# Example (adjust paths and settings to your dataset)
yolo task=detect mode=train model=yolov8n.pt data=your_data.yaml imgsz=640 epochs=50 batch=16

# Evaluate
 yolo task=detect mode=val model=path/to/weights.pt data=your_data.yaml imgsz=640

# Inference on images or videos
yolo task=detect mode=predict model=path/to/weights.pt source=path/to/images_or_video

If using the notebook, ensure your dataset YAML (data=...) correctly points to your train, val image directories and includes the class names list.

Tips for reproducing best.pt:
- Start from a YOLOv8 family checkpoint (e.g., yolov8n.pt, yolov8s.pt).
- Curate positive accident samples and hard negatives.
- Balance classes and consider data augmentation (flips, brightness, blur).
- Validate frequently and monitor precision/recall. Export the best checkpoint.

---

## Configuration Notes
- Model path: change in main.py if your weights have a different name/location:
  python
  model = YOLO('path/to/your_best.pt')
  
- Input source: change cv2.VideoCapture('cr.mp4') to your file path or camera index.
- Class list: update coco1.txt to match your trained model classes. The detection overlay logic checks if 'accident' in c to color boxes red.
- Frame stride: main.py processes every 3rd frame. Adjust the modulo to trade accuracy for speed:
  python
  if count % 3 != 0:
      continue
  
- Display size: frames are resized to 1020x500 for inference/display. Adjust as needed.

---

## Performance Tips
- Use a smaller model variant (e.g., yolov8n.pt) for CPU-only devices.
- Enable GPU: install CUDA-capable PyTorch and run with a CUDA device; Ultralytics will auto-detect.
- Reduce input resolution or increase the frame skip to boost FPS.
- Avoid unnecessary UI logging (mouse callback) for faster console output.

---

## Troubleshooting
- Ultralytics import error: ensure pip install ultralytics and compatible PyTorch are installed.
- OpenCV cannot open video: verify the path to cr.mp4 or camera index; try using absolute paths on Windows.
- Blank or closed window immediately: check if frames are being read (ret is True). Some codecs require installing additional decoders (e.g., K-Lite on Windows) or re-encoding the video.
- No red boxes for accidents: confirm coco1.txt contains the exact class string your model predicts (e.g., accident).
- Poor accuracy: retrain on more data, verify labels, and use a model variant appropriate to your hardware.

---

## FAQ
- Does this save annotated videos? By default, no; it displays results. You can add a cv2.VideoWriter in main.py if you need to save.
- Can I run on images? Yes: modify main.py to read images from a folder and call model.predict per image, or use yolo mode=predict with the CLI.
- Is the dataset included? Only sample structure is provided under diff/ and images/. Use your own dataset or construct one using img.py and manual labeling.

---

## Credits
- Ultralytics YOLOv8 for detection (ultralytics package).
- OpenCV for video I/O and visualization.
- cvzone for convenient text box overlays.
- Notebook adapted for training on a custom dataset.

If you use this project, consider citing Ultralytics and linking back to this repository. Contributions and improvements are welcome.
