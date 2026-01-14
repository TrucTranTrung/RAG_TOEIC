import base64
from io import BytesIO
import re
from PIL import Image

TARGET_IMAGE_SIZE = (768, 768)
IMAGE_FORMAT = "JPEG"


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
    

def pre_validate_part1_context(transcript: str) -> str:
    """
    Xác nhận transcript có đủ 4 tùy chọn (A), (B), (C), và (D).
    Trả về '1' nếu hợp lệ, nếu không trả về thông báo lỗi tiếng Việt.
    """
    if not transcript:
        return "Bạn cung cấp không đủ đáp án (A)(B)(C), vui lòng nhập lại."
    all_options = set(re.findall(r'\([A-Z]\)', transcript.upper()))
    target_options = {'(A)', '(B)', '(C)','(D)'}
    if len(all_options) == 4 and all_options == target_options:
        return "1"  
    else:
        return "Bạn cung cấp không đủ đáp án (A)(B)(C)(D), vui lòng nhập lại."
    
def extract_text(result) -> str:
        try:
            return result["output"][0]["text"].strip()
        except Exception:
           return ""

def format_answer_only(text: str) -> str:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("Đáp án:"):
                return "\n".join(lines[i:]).strip()
        return text.strip()
