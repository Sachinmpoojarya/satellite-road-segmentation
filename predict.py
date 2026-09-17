import os
import sys
import json
import torch
import cv2
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms

# Import UNet from local module
from unet_model import UNet

def load_road_model(checkpoint_path=None):
    if checkpoint_path is None:
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "road_model"))
        best_path = os.path.join(model_dir, "best_road_seg_unet.pth")
        latest_path = os.path.join(model_dir, "latest_road_seg_unet.pth")

        if os.path.exists(best_path):
            checkpoint_path = best_path
        elif os.path.exists(latest_path):
            checkpoint_path = latest_path
        else:
            pth_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pth")]
            if pth_files:
                checkpoint_path = pth_files[0]
            else:
                checkpoint_path = best_path
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=3, out_channels=1, pretrained=True)
    
    if os.path.exists(checkpoint_path):
        print(f"[Model Engine] Loading weights from: {checkpoint_path}")
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"[Model Engine Warning] Existing checkpoint weights mismatch ({e}). Using initialized backbone.")
    else:
        print(f"[Model Engine Warning] Checkpoint not found at '{checkpoint_path}'. Initialized model with ResNet34 backbone.")
        
    model.to(device)
    model.eval()
    return model, device


def enhance_satellite_image(pil_img):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space
    to bring out low-contrast road structures in satellite imagery.
    """
    img_np = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    enhanced_img = Image.fromarray(enhanced_np)
    enhanced_img = transforms.functional.adjust_contrast(enhanced_img, 1.25)
    return enhanced_img

def save_geojson_and_kml(binary_mask_np, output_geojson_path="predicted_road_features.geojson", output_kml_path="predicted_road_features.kml", bounds=None):
    """
    Converts binary road mask numpy array into GeoJSON & KML vector files and calculates road length and area metrics.
    If bounds dict is provided ({south, west, north, east}), pixel coordinates are projected into real Lat/Lon geographic coordinates.
    """
    h, w = binary_mask_np.shape[:2]
    contours, _ = cv2.findContours(binary_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = []
    total_length_px = 0.0
    total_area_px = 0.0

    has_bounds = bounds and 'south' in bounds and 'west' in bounds and 'north' in bounds and 'east' in bounds
    if has_bounds:
        south, west, north, east = bounds['south'], bounds['west'], bounds['north'], bounds['east']
    
    for idx, cnt in enumerate(contours):
        approx = cv2.approxPolyDP(cnt, epsilon=1.0, closed=True)
        if len(approx) < 3:
            continue
            
        area = float(cv2.contourArea(cnt))
        length = float(cv2.arcLength(cnt, False)) / 2.0  # Approx centerline length
        
        total_length_px += length
        total_area_px += area
        
        if has_bounds:
            coords = []
            for pt in approx:
                px_x = float(pt[0][0])
                px_y = float(pt[0][1])
                geo_lon = round(west + (px_x / float(w)) * (east - west), 6)
                geo_lat = round(north - (px_y / float(h)) * (north - south), 6)
                coords.append([geo_lon, geo_lat])
        else:
            coords = [[float(pt[0][0]), float(pt[0][1])] for pt in approx]

        if coords[0] != coords[-1]:
            coords.append(coords[0])
            
        feature = {
            "type": "Feature",
            "id": idx + 1,
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            },
            "properties": {
                "class": "road",
                "area_px": area,
                "length_px": length
            }
        }
        features.append(feature)

    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)

    # Generate KML vector file for Google Earth
    kml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '  <Document>',
        '    <name>AI Extracted Road Network</name>',
        '    <Style id="roadStyle">',
        '      <LineStyle><color>ff0000ff</color><width>3</width></LineStyle>',
        '      <PolyStyle><color>7f0000ff</color></PolyStyle>',
        '    </Style>'
    ]
    for idx, feat in enumerate(features):
        poly_coords = " ".join([f"{pt[0]},{pt[1]},0" for pt in feat["geometry"]["coordinates"][0]])
        kml_lines.extend([
            '    <Placemark>',
            f'      <name>Road Segment #{idx+1}</name>',
            '      <styleUrl>#roadStyle</styleUrl>',
            '      <Polygon><outerBoundaryIs><LinearRing>',
            f'        <coordinates>{poly_coords}</coordinates>',
            '      </LinearRing></outerBoundaryIs></Polygon>',
            '    </Placemark>'
        ])
    kml_lines.extend(['  </Document>', '</kml>'])
    
    with open(output_kml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(kml_lines))
        
    # Scale estimated pixel dimensions to real-world kilometers (0.5m/px satellite scale)
    total_length_km = round((total_length_px * 0.5) / 1000.0, 2)
    total_area_sqm = round(total_area_px * 0.25, 1)

    return output_geojson_path, output_kml_path, len(features), total_length_km, total_area_sqm

def predict_road(image_path, output_mask_path="predicted_road_mask.png", output_overlay_path="predicted_road_overlay.png", output_geojson_path="predicted_road_features.geojson", output_kml_path="predicted_road_features.kml", threshold=0.25, image_size=512, bounds=None):
    model, device = load_road_model()
    
    # Load raw satellite image and enhance road contrast with CLAHE
    raw_image = Image.open(image_path).convert("RGB")
    image = enhance_satellite_image(raw_image)
    original_size = raw_image.size # (width, height)
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Pass 1: Original Image view
        p1 = model(image_tensor)
        
        # Pass 2: Horizontal Flip view
        img_hflip = torch.flip(image_tensor, dims=[3])
        p2 = torch.flip(model(img_hflip), dims=[3])
        
        # Pass 3: Vertical Flip view
        img_vflip = torch.flip(image_tensor, dims=[2])
        p3 = torch.flip(model(img_vflip), dims=[2])
        
        # Average probability maps across all 3 view angles for high-precision output
        output = (p1 + p2 + p3) / 3.0
        
    mask = output[0, 0].cpu().numpy()
    max_val = float(mask.max())
    
    # Advanced Hybrid Adaptive Thresholding: Combines Otsu & relative scaling for ultra-sharp mask edges
    if max_val > 0.08:
        # Scale mask relative to max probability map
        mask_scaled = (mask / max_val * 255.0).astype(np.uint8)
        otsu_val, _ = cv2.threshold(mask_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        calibrated_thresh = max_val * min(threshold, max(0.10, (otsu_val / 255.0) * 0.80))
        effective_thresh = min(threshold, max(0.04, calibrated_thresh))
    elif max_val < 0.02:
        effective_thresh = max(0.003, max_val * 0.45)
    else:
        effective_thresh = min(threshold, max(0.025, max_val * 0.35))

    binary_mask = (mask > effective_thresh).astype(np.uint8) * 255

    # Multi-Stage Morphological Refinement:
    # Stage 1: Close small gaps along roads (trees/building shadows)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Stage 2: Remove small noise specks
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
    
    # Stage 3: Smooth contours for sharp road masks
    binary_mask = cv2.GaussianBlur(binary_mask, (3, 3), 0)
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Convert and resize binary mask to original image resolution
    mask_img = Image.fromarray(binary_mask).convert("L").resize(original_size, Image.NEAREST)
    mask_img.save(output_mask_path)
    
    # High-contrast Red Overlay Visualization
    overlay = raw_image.copy()
    red_mask = Image.new("RGB", original_size, (255, 0, 0))
    overlay = Image.composite(red_mask, overlay, mask_img)
    blended = Image.blend(raw_image, overlay, alpha=0.50)
    blended.save(output_overlay_path)
    
    # Generate GeoJSON and KML vector files with geographic projection support
    mask_np_resized = np.array(mask_img)
    geojson_file, kml_file, feat_count, length_km, area_sqm = save_geojson_and_kml(mask_np_resized, output_geojson_path, output_kml_path, bounds=bounds)
    
    print(f"Road segmentation completed successfully!")
    print(f" - Binary mask saved to: {output_mask_path}")
    print(f" - Overlay saved to:     {output_overlay_path}")
    print(f" - Vector GeoJSON:       {geojson_file}")
    print(f" - Vector KML:           {kml_file}")
    print(f" - Extracted Metrics:    {feat_count} segments | {length_km} km total length | {area_sqm} m2 area")
    
    return output_mask_path, output_overlay_path, geojson_file, kml_file, feat_count, length_km, area_sqm

if __name__ == "__main__":
    threshold = 0.25 # default threshold for road segmentation
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if len(sys.argv) > 2:
            threshold = float(sys.argv[2])
    else:
        # Create a sample test satellite image if none provided
        img_path = "sample_test_image.jpg"
        if not os.path.exists(img_path):
            sample_img = Image.new("RGB", (512, 512), color=(100, 140, 80)) # green background
            for i in range(512):
                for j in range(240, 270):
                    sample_img.putpixel((i, j), (120, 120, 120))
            sample_img.save(img_path)
            print(f"Created sample test image: {img_path}")
            
    predict_road(img_path, threshold=threshold)
