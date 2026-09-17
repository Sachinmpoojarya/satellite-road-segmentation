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
