🛰️ Satellite Road Segmentation & GIS Vectorization
🚀 Extracting Roads from Space with Deep Learning

🌟 Overview

This project builds an AI-powered system that automatically detects and extracts road networks from satellite imagery and converts them into GIS-ready vector formats.

🔍 From raw satellite images → 📍 to structured road maps → 🌍 ready for real-world use.

🎯 Key Features

✨ Deep Learning Road Detection

Uses ResNet-UNet for high-quality segmentation

🗺️ GIS Vector Output

Converts masks into GeoJSON / KML

📊 Analytics & Reporting

Calculates road length & area
Generates automated PDF reports

⚡ API Ready

Integrated with Flask for easy deployment
🧠 Model Architecture
🧩 Model: ResNet-34 based U-Net
🎯 Loss Function: Focal Tversky Loss
⚙️ Framework: PyTorch
🖼️ Image Processing: OpenCV
📊 Performance Metrics
Metric	Score
🟢 Pixel Accuracy	96.77%
🔵 Precision	74.78%
🟡 F1 Score	59.69%
🔴 Mean IoU	42.54%
🛠️ Tech Stack
Category	Tools
🤖 ML / AI	PyTorch
🌐 API	Flask
🖼️ Image Processing	OpenCV
🌍 GIS Processing	Shapely
📄 Reports	ReportLab
📂 Project Structure
Satellite-Road-Segmentation/
│── model/            # Trained model
│── dataset/          # (Excluded from repo)
│── app.py            # Flask API
│── predict.py        # Prediction script
│── utils/            # Helper functions
│── outputs/          # Results (masks + vectors)
│── reports/          # Generated PDFs
│── requirements.txt
│── README.md
⚙️ Setup Guide
1️⃣ Clone Repo
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2️⃣ Create Environment
conda create -n satellite python=3.10
conda activate satellite
3️⃣ Install Requirements
pip install -r requirements.txt
▶️ Run the Project
🔹 Predict Roads
python predict.py --image input.jpg
🔹 Start API Server
python app.py
🔄 Workflow
Satellite Image → Segmentation Model → Road Mask → Vectorization → GIS Output
🌍 Real-World Applications

🏙️ Smart City Planning
🛣️ Road Infrastructure Analysis
🚨 Disaster Management
🧭 Navigation Systems
🌐 GIS Data Generation

📌 Important Notes

⚠️ Dataset not included (large size)
📥 Pretrained model should be downloaded separately

👨‍💻 Author

Sachin M Poojar
🎓 Computer Science Student
🤖 Aspiring AI Engineer
🚀 Passionate about AI & Emerging Technologies
