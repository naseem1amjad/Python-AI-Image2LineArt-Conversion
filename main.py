# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import cv2
import uuid
import shutil
import numpy as np
from PIL import Image
import io

# --- FastAPI App Setup ---
app = FastAPI()

# --- Folder Configuration ---
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Static Files and Frontend ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# --- Image Processing Functions ---
def generate_black_and_white_lineart(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    edges = cv2.adaptiveThreshold(gray_smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7)
    kernel = np.ones((3,3), np.uint8)
    thick_lines = cv2.dilate(edges, kernel, iterations=1)
    inverted_lines = cv2.bitwise_not(thick_lines)
    lineart_bgr = cv2.cvtColor(inverted_lines, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_not(lineart_bgr)

def generate_colored_cartoon(img: np.ndarray) -> np.ndarray:
    line_art = generate_black_and_white_lineart(img)
    pixel_vals = img.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    k = 10
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    color_quantized_img = segmented_data.reshape((img.shape))
    cartoon = cv2.bitwise_and(color_quantized_img, line_art)
    return cartoon

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > 1024 * 1024:
        return {"error": "File too large. Max allowed size is 1MB."}

    try:
        img_pil = Image.open(io.BytesIO(contents))
        width, height = img_pil.size
        if width > 2560 or height > 1440:
            return {"error": f"Image too large. Max resolution is 2560x1440 (yours: {width}x{height})"}
    except Exception:
        return {"error": "Invalid image format."}

    filename_base = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_FOLDER, f"{filename_base}.jpg")
    with open(input_path, "wb") as buffer:
        buffer.write(contents)

    img = cv2.imread(input_path)
    if img is None:
        return {"error": "Could not read image with OpenCV."}

    bw_output_path = os.path.join(OUTPUT_FOLDER, f"{filename_base}_bw.png")
    color_output_path = os.path.join(OUTPUT_FOLDER, f"{filename_base}_color.png")

    bw_image = generate_black_and_white_lineart(img)
    color_image = generate_colored_cartoon(img)

    cv2.imwrite(bw_output_path, bw_image)
    cv2.imwrite(color_output_path, color_image)

    return {
        "bw_url": f"/download/{os.path.basename(bw_output_path)}",
        "color_url": f"/download/{os.path.basename(color_output_path)}"
    }

@app.get("/download/{filename}")
async def download(filename: str):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.isfile(path):
        return {"error": "File not found."}
    return FileResponse(path, media_type="image/png", filename=filename)