"""image_loader.py
Handles direct image uploads (PNG, JPG, TIFF, WebP).
Normalizes dimensions to the sweet spot (max 1800px on long edge)
to prevent CPU thread thrashing and memory exhaustion in PaddleOCR.
"""
import os
from PIL import Image

MAX_DIMENSION = 1800  # Sweet spot for A4-proportioned scans on 4-core CPU
MIN_DIMENSION = 600   # Avoid upscaling low-res thumbnails unnecessarily


def preprocess_uploaded_image(image_path: str, output_path: str = None) -> str:
    """
    Checks dimensions of an uploaded image.
    If the width or height exceeds MAX_DIMENSION, downscales it 
    proportionally using high-quality Lanczos resampling.
    
    Returns the path to the ready-to-OCR image.
    """
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_preprocessed{ext}"

    with Image.open(image_path) as img:
        # Convert RGBA/palette images to clean RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        max_edge = max(w, h)
        if max_edge > MAX_DIMENSION:
            scale = MAX_DIMENSION / float(max_edge)
            new_w = int(w * scale)
            new_h = int(h * scale)
            print(f"[*] Rescaling uploaded image from {w}x{h} to {new_w}x{new_h} (CPU sweet spot)...")
            resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized_img.save(output_path, quality=95)
            return output_path
        else:
            # Image is already within optimal bounds
            return image_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        out = preprocess_uploaded_image(sys.argv[1])
        print(f"[+] Output ready at: {out}")
