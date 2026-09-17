import os
import json
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
from predict import predict_road

app = Flask(__name__)

import shutil

# Config directories
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(app.root_path, 'static', 'outputs')

# Config Retraining Dataset Storage Directory (images/ folder)
RETRAIN_DATASET_DIR = os.path.join(app.root_path, 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(RETRAIN_DATASET_DIR, exist_ok=True)

def save_sample_for_retraining(input_path, mask_path):
    """
    Saves input satellite image and predicted binary road mask strictly in JPG format (.jpg)
    directly inside the images/ directory for future model training. No .npz files are generated.
    """
    try:
        sample_id = f"sample_{int(time.time() * 1000)}"
        
        dst_sat = os.path.join(RETRAIN_DATASET_DIR, f"{sample_id}_sat.jpg")
        dst_mask = os.path.join(RETRAIN_DATASET_DIR, f"{sample_id}_mask.jpg")

        # Save Satellite Image as high-quality JPG
        sat_img = Image.open(input_path).convert("RGB")
        sat_img.save(dst_sat, "JPEG", quality=95)

        # Save Binary Mask as JPG
        mask_img = Image.open(mask_path).convert("RGB")
        mask_img.save(dst_mask, "JPEG", quality=95)

        print(f"[Dataset Engine]: Saved JPG sample pair -> {sample_id}_sat.jpg & {sample_id}_mask.jpg in {RETRAIN_DATASET_DIR}")
        return sample_id
    except Exception as e:
        print(f"[Dataset Engine Error]: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def handle_prediction():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    threshold = float(request.form.get('threshold', 0.25))

    # Save uploaded input image
    input_path = os.path.join(UPLOAD_FOLDER, 'input_image.png')
    file.save(input_path)

    # Output file paths
    mask_path = os.path.join(OUTPUT_FOLDER, 'predicted_road_mask.png')
    overlay_path = os.path.join(OUTPUT_FOLDER, 'predicted_road_overlay.png')
    geojson_path = os.path.join(OUTPUT_FOLDER, 'predicted_road_features.geojson')
    kml_path = os.path.join(OUTPUT_FOLDER, 'predicted_road_features.kml')

    try:
        # Execute road segmentation model prediction
        _, _, _, _, feature_count, total_length_km, total_area_sqm = predict_road(
            image_path=input_path,
            output_mask_path=mask_path,
            output_overlay_path=overlay_path,
            output_geojson_path=geojson_path,
            output_kml_path=kml_path,
            threshold=threshold
        )

        # Save sample for future model retraining
        save_sample_for_retraining(input_path, mask_path)

        return jsonify({
            'success': True,
            'original_url': '/static/uploads/input_image.png',
            'mask_url': '/static/outputs/predicted_road_mask.png',
            'overlay_url': '/static/outputs/predicted_road_overlay.png',
            'geojson_url': '/static/outputs/predicted_road_features.geojson',
            'kml_url': '/static/outputs/predicted_road_features.kml',
            'feature_count': feature_count,
            'total_length_km': total_length_km,
            'total_area_sqm': total_area_sqm
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

import cv2
import numpy as np

# Import new GIS engine functions
from gis_utils import (
    search_location_nominatim,
    reverse_geocode_nominatim,
    fetch_and_stitch_satellite_tiles,
    calculate_advanced_gis_statistics,
    generate_ai_gis_report_data,
    export_pdf_report,
    export_csv_statistics,
    export_shapefile_zip
)

# Global in-memory cache for latest prediction report
latest_report_data = {}

@app.route('/api/search-location', methods=['GET'])
def handle_location_search():
    """Geocodes location query via Nominatim API."""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'success': False, 'error': 'Query parameter q is required'}), 400

    results = search_location_nominatim(query)
    if results:
        return jsonify({'success': True, 'results': results})
    return jsonify({'success': False, 'error': 'Location not found'}), 404

@app.route('/api/reverse-geocode', methods=['GET'])
def handle_reverse_geocode():
    """Reverse geocodes latitude & longitude into a location name."""
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Invalid lat or lon parameter'}), 400

    result = reverse_geocode_nominatim(lat, lon)
    return jsonify({'success': True, 'result': result})


@app.route('/api/fetch-satellite-bbox', methods=['POST'])
def handle_satellite_bbox_inference():
    """
    Accepts latitude/longitude bounding box or AOI, downloads & stitches satellite tiles,
    executes existing U-Net road detection model without modification, calculates GIS stats,
    and returns predictions.
    """
    data = request.get_json(force=True, silent=True) or {}
    bounds = data.get('bounds')
    location_name = data.get('location_name', 'Searched Location')
    threshold = float(data.get('threshold', 0.25))

    if not bounds:
        return jsonify({'success': False, 'error': 'Bounding box bounds required'}), 400

    south = float(bounds.get('south', 12.96))
    west = float(bounds.get('west', 77.58))
    north = float(bounds.get('north', 12.98))
    east = float(bounds.get('east', 77.60))

    start_t = time.time()
    input_path = os.path.join(UPLOAD_FOLDER, 'input_image.png')

    try:
        # 1. Fetch & stitch satellite tiles
        fetch_and_stitch_satellite_tiles(south, west, north, east, zoom=16, output_image_path=input_path)

        # Output file paths
        mask_path = os.path.join(OUTPUT_FOLDER, 'predicted_road_mask.png')
        overlay_path = os.path.join(OUTPUT_FOLDER, 'predicted_road_overlay.png')
        geojson_path = os.path.join(OUTPUT_FOLDER, 'predicted_road_features.geojson')
        kml_path = os.path.join(OUTPUT_FOLDER, 'predicted_road_features.kml')

        # 2. Run existing road detection model
        predict_road(
            image_path=input_path,
            output_mask_path=mask_path,
            output_overlay_path=overlay_path,
            output_geojson_path=geojson_path,
            output_kml_path=kml_path,
            threshold=threshold,
            bounds={'south': south, 'west': west, 'north': north, 'east': east}
        )

        # Save sample for future model retraining
        save_sample_for_retraining(input_path, mask_path)

        # 3. Calculate advanced GIS statistics
        binary_mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if binary_mask_np is None:
            binary_mask_np = np.zeros((512, 512), dtype=np.uint8)

        stats = calculate_advanced_gis_statistics(binary_mask_np, bounds={'south': south, 'west': west, 'north': north, 'east': east})
        elapsed = round(time.time() - start_t, 2)

        # 4. Generate AI GIS report
        global latest_report_data
        latest_report_data = generate_ai_gis_report_data(location_name, {'south': south, 'west': west, 'north': north, 'east': east}, stats, detection_time_sec=elapsed)

        return jsonify({
            'success': True,
            'original_url': '/static/uploads/input_image.png',
            'mask_url': '/static/outputs/predicted_road_mask.png',
            'overlay_url': '/static/outputs/predicted_road_overlay.png',
            'geojson_url': '/static/outputs/predicted_road_features.geojson',
            'kml_url': '/static/outputs/predicted_road_features.kml',
            'bounds': {'south': south, 'west': west, 'north': north, 'east': east},
            'statistics': stats,
            'report': latest_report_data
        })
    except Exception as e:
        print(f"[GIS Engine Error]: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate-report', methods=['POST', 'GET'])
def handle_generate_report():
    """Returns AI GIS report JSON for the latest prediction."""
    global latest_report_data
    if not latest_report_data:
        # Fallback dummy report
        latest_report_data = generate_ai_gis_report_data("Current Region", {'south': 12.96, 'west': 77.58, 'north': 12.98, 'east': 77.60}, calculate_advanced_gis_statistics(np.zeros((512, 512))))
    return jsonify({'success': True, 'report': latest_report_data})

@app.route('/api/export/pdf', methods=['GET'])
def export_pdf():
    """Generates and serves downloadable PDF summary report."""
    global latest_report_data
    pdf_path = os.path.join(OUTPUT_FOLDER, 'road_detection_report.pdf')
    export_pdf_report(latest_report_data or {}, pdf_path)
    return send_from_directory(OUTPUT_FOLDER, 'road_detection_report.pdf', as_attachment=True)

@app.route('/api/export/csv', methods=['GET'])
def export_csv():
    """Generates and serves downloadable CSV statistics file."""
    global latest_report_data
    csv_path = os.path.join(OUTPUT_FOLDER, 'road_statistics.csv')
    export_csv_statistics(latest_report_data or {}, csv_path)
    return send_from_directory(OUTPUT_FOLDER, 'road_statistics.csv', as_attachment=True)

@app.route('/api/export/geojson', methods=['GET'])
def export_geojson():
    """Serves GeoJSON feature collection file."""
    return send_from_directory(OUTPUT_FOLDER, 'predicted_road_features.geojson', as_attachment=True)

@app.route('/api/export/shapefile', methods=['GET'])
def export_shapefile():
    """Generates and serves ESRI Shapefile ZIP package."""
    geojson_path = os.path.join(OUTPUT_FOLDER, 'predicted_road_features.geojson')
    zip_path = os.path.join(OUTPUT_FOLDER, 'predicted_road_shapefile.zip')
    export_shapefile_zip(geojson_path, zip_path)
    return send_from_directory(OUTPUT_FOLDER, 'predicted_road_shapefile.zip', as_attachment=True)

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" GeoSeg AI — Satellite Road Segmentation & GIS Engine Running!")
    print(" Access UI in browser at: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=False)

