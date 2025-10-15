import os
import base64
from io import BytesIO
from PIL import Image
import warnings

TARGET_IMAGE_SIZE = (768, 768)
IMAGE_FORMAT = "JPEG"

def configure_environment():
    """Thiết lập biến môi trường và cấu hình cơ bản."""
    warnings.filterwarnings("ignore")
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDPIYi9iuhzzg6wfx6tY2dPuknQY_FGtm4" 
    print("Môi trường đã được cấu hình.")

def encode_image_to_base64(image_path: str) -> str:
    """Mở, thay đổi kích thước, và mã hóa ảnh sang chuỗi Base64."""
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize ảnh để tối ưu hóa token/tốc độ
        img = img.resize(TARGET_IMAGE_SIZE, Image.LANCZOS)
        
        buf = BytesIO()
        img.save(buf, format=IMAGE_FORMAT)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
