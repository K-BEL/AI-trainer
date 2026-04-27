import cv2
import numpy as np
from pathlib import Path
import logging

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

logger = logging.getLogger("trainer")

class LicensePlateOCR:
    def __init__(self, use_gpu=False):
        if not PADDLE_AVAILABLE:
            raise ImportError("PaddleOCR not found. Please install it with 'pip install paddleocr'.")
        
        # Initialize PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu, show_log=False)

    def extract(self, image_path_or_array):
        """
        Extracts text from a license plate image.
        """
        if isinstance(image_path_or_array, (str, Path)):
            img = cv2.imread(str(image_path_or_array))
        else:
            img = image_path_or_array

        if img is None:
            return None

        # Perform OCR
        result = self.ocr.ocr(img, cls=True)
        
        # Parse results
        extracted_texts = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                confidence = line[1][1]
                extracted_texts.append({"text": text, "confidence": confidence})
        
        return extracted_texts

def get_ocr_engine():
    if not PADDLE_AVAILABLE:
        return None
    return LicensePlateOCR()
