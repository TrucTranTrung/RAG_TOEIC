import base64
from io import BytesIO
import re
from PIL import Image
    
def pre_validate_part4_context(transcript: str) -> str:
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
