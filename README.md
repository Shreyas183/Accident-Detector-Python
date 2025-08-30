# 🚗 Accident Detector using Python + YOLOv8  

An AI-powered system that detects road accidents in real-time using **YOLOv8** and **OpenCV**.  
The project processes live video streams or recorded footage, identifies accident scenarios, and highlights them with bounding boxes for easy monitoring and safety analysis.  

---

## 📌 Features  
- 🎥 **Real-time accident detection** from live camera feeds or videos  
- 🖼️ **Image detection support** – works on both video streams and images  
- ✅ **Trained YOLOv8 model** with ~92% accuracy  
- 📦 **OpenCV integration** for video frame processing  
- 💾 Saves processed output videos/images with detections  
- ⚡ Lightweight and fast – runs on CPU/GPU  

---

## 🛠️ Tech Stack  
- **Python 3.9+**  
- **YOLOv8 (Ultralytics)**  
- **OpenCV**  
- **NumPy**  
- **Jupyter Notebook / .py scripts**  

---

## ⚙️ How It Works  
1. **Input**: Accepts a video file, live webcam feed, or images.  
2. **Frame Processing**: Each frame is passed through the trained **YOLOv8 model**.  
3. **Accident Detection**: If an accident is detected, bounding boxes and labels are drawn.  
4. **Output**: The processed video/image is displayed and can be saved locally.  

---

## 📂 Project Structure  

```bash
Accident-Detector-Python/
│── best.pt             # Trained YOLOv8 model weights
│── main.py             # Main script for running detection
│── detect.ipynb        # Jupyter notebook version for testing
│── requirements.txt    # Python dependencies
│── sample_videos/      # Example input videos/images
│── outputs/            # Saved results after detection

