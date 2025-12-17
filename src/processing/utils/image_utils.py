# src/processing/utils/image_utils.py

import io
from PIL import Image
import fitz

def pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    """Convert PyMuPDF pixmap to a PIL Image consistently."""
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))