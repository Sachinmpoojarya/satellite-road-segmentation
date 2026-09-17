# 🛰️ Satellite Road Segmentation & GIS Vectorization  
### 🚀 Extracting Roads from Satellite Imagery using Deep Learning  

---

## 📌 Project Overview  

This project focuses on extracting **road networks from satellite imagery** using Deep Learning and converting them into **GIS-ready vector formats**.

The system uses a **ResNet-UNet model** to detect roads and generates:

- Binary road masks  
- GeoJSON / KML vector files  
- Road statistics (length, area)  
- Automated PDF reports  

---

## 🎯 Objective  

To develop an intelligent system that:

- Identifies roads from satellite images  
- Converts raster outputs into vector polygons  
- Provides GIS-compatible outputs for real-world applications  

---

## 🧠 Model Details  

- **Model**: ResNet-34 based U-Net  
- **Loss Function**: Focal Tversky Loss  
- **Framework**: PyTorch  
- **Image Processing**: OpenCV  

---

## 🛠️ Tech Stack  

- **API**: Flask  
- **Machine Learning**: PyTorch  
- **Image Processing**: OpenCV  
- **Geospatial Processing**: Shapely  
- **Report Generation**: ReportLab  

---

## 📊 Results  

| Metric            | Value   |
|------------------|--------|
| Pixel Accuracy   | 96.77% |
| Precision        | 74.78% |
| F1 Score (Dice)  | 59.69% |
| Mean IoU         | 42.54% |

---

## 📂 Project Structure  

Satellite-Road-Segmentation/
│── model/ # Trained model files
│── dataset/ # (Not included)
│── app.py # Flask API
│── predict.py # Inference script
│── utils/ # Helper functions
│── outputs/ # Masks & vectors
│── reports/ # PDF reports
│── requirements.txt
│── README.md



---

## ⚙️ Installation & Setup  

### 1. Clone Repository  

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Create Environment
conda create -n satellite python=3.10
conda activate satellite

3. Install Dependencies
pip install -r requirements.txt

▶️ Usage
Run Prediction
python predict.py --image input.jpg
Run Flask App
python app.py
📤 Output

The system generates:

Segmented road mask images
GeoJSON / KML files
Road length & area calculations
PDF reports
🌍 Applications
Smart city planning
Road infrastructure monitoring
Disaster management
Navigation systems
GIS data generation
🔄 Workflow
Satellite Image → Model → Road Mask → Vectorization → GIS Output
⚠️ Notes
Dataset is not included due to size limitations
Pretrained model should be downloaded separately
👨‍💻 Author

Sachin M Poojar
Computer Science Student | Aspiring AI Engineer
