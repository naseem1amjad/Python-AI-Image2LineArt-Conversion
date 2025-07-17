from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import cv2
import uuid
import shutil
import numpy as np


app = FastAPI()

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Mount static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at root
@app.get("/")
async def root():
    return FileResponse("static/index.html")


def generate_black_and_white_lineartOld(img):
    import numpy as np
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply median blur to smooth
    gray_blur = cv2.medianBlur(gray, 5)

    # Edge detection (black lines on white)
    edges = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,  # Invert to make lines black
        9,
        9
    )

    # Denoise small dots
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    # Convert to 3-channel
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

def generate_black_and_white_lineart(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    # Detect edges: lines become black, background white
    edges = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=9,
        C=9
    )

    # Ensure white background and black lines
    _, clean = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)

    return cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)



def generate_colored_cartoon(img):
    color = cv2.bilateralFilter(img, d=9, sigmaColor=200, sigmaSpace=200)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)

    edges = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 9, 9)

    edges_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges_mask)

    # Optional brighten
    white = np.full_like(cartoon, 255)
    cartoon = cv2.addWeighted(cartoon, 1.2, white, 0.3, 0)

    return cartoon




@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    filename_base = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_FOLDER, f"{filename_base}.jpg")
    bw_output_path = os.path.join(OUTPUT_FOLDER, f"{filename_base}_bw.jpg")
    color_output_path = os.path.join(OUTPUT_FOLDER, f"{filename_base}_color.jpg")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.imread(input_path)

    bw = generate_black_and_white_lineart(img)
    color = generate_colored_cartoon(img)

    cv2.imwrite(bw_output_path, bw)
    cv2.imwrite(color_output_path, color)

    return {
        "bw_url": f"/download/{filename_base}_bw.jpg",
        "color_url": f"/download/{filename_base}_color.jpg"
    }


@app.get("/download/{filename}")
async def download(filename: str):
    path = os.path.join(OUTPUT_FOLDER, filename)
    return FileResponse(path, media_type="image/jpeg", filename=filename)
