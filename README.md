🚀 Satellite Road Segmentation & GIS Vectorization
📌 Project Overview

This project focuses on extracting road networks from satellite imagery using Deep Learning and converting them into GIS-ready vector formats.

The system uses a ResNet-UNet model to detect roads from satellite images and generates outputs such as:

Binary road masks
GeoJSON / KML vector files
Road statistics (length, area)
Automated PDF reports
🎯 Objective

To develop an intelligent system that:

Identifies roads from satellite images
Converts raster outputs into vector polygons
Provides GIS-compatible outputs for real-world applications
🧠 Model Details
Model: ResNet-34 based U-Net
Loss Function: Focal Tversky Loss
Framework: PyTorch
Image Processing: OpenCV
🛠️ Tech Stack
Frontend / API: Flask
Backend / ML: PyTorch
Image Processing: OpenCV
Geospatial Processing: Shapely
Report Generation: ReportLab
📊 Results
Metric	Value
Pixel Accuracy	96.77%
Precision	74.78%
F1 Score (Dice)	59.69%
Mean IoU	42.54%
📂 Project Structure
Satellite-Road-Segmentation/
│── model/                # Trained model files
│── dataset/              # (Optional) Dataset (not included in repo)
│── app.py                # Flask API
│── predict.py            # Inference script
│── utils/                # Helper functions
│── outputs/              # Generated masks & vectors
│── reports/              # PDF reports
│── requirements.txt
│── README.md
⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2️⃣ Create Virtual Environment
conda create -n satellite python=3.10
conda activate satellite
3️⃣ Install Dependencies
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
PDF report with analysis
🌍 Applications
Smart city planning
Road infrastructure monitoring
Disaster management
Navigation systems
GIS data generation
📸 Sample Workflow
Input satellite image
Model predicts road segmentation
Mask converted to vector format
GIS-ready output generated
🔐 Notes
Dataset is not included due to size limitations
Pretrained model can be downloaded separately
👨‍💻 Author

Sachin M Poojar
Computer Science Student | Aspiring AI Engineer

📜 License

This project is for academic and research purposes.
