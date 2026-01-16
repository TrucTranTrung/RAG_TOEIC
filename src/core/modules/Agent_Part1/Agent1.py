import os
import asyncio
import logging
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from ..Agent_Base import BaseAgent 
from .utils import encode_image_to_base64, pre_validate_part1_context, extract_text, format_answer_only
from .prompts import TOEIC_REACT_SYSTEM_PROMPT
from .models import model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def get_image_data_tool(image_path: str) -> str:
    """Lấy dữ liệu Base64 của ảnh. Bắt buộc gọi khi Question có Image."""
    try:
        clean_path = image_path.strip().strip("'").strip('"')
        if not os.path.exists(clean_path):
            return f"Lỗi: Không tìm thấy file tại {clean_path}"
        
        base64_data = encode_image_to_base64(clean_path)
        image_url = f"data:image/jpeg;base64,{base64_data}"
        
        print(f"\n>>> [KÍCH HOẠT TOOL ẢNH]: Đã nạp ảnh vào bộ nhớ của Gemini...")
        
        return (
            f"Dữ liệu hình ảnh (Base64): {image_url}\n\n"
            "HƯỚNG DẪN XỬ LÝ NGỮ NGHĨA:\n"
            "1. Bạn hãy xem ảnh này để viết 'Image Transcript' chi tiết.\n"
            "2. So sánh ảnh với Transcript được cung cấp.\n"
            "3. TRƯỜNG HỢP ĐẶC BIỆT: Nếu sau khi xem ảnh, bạn thấy cả 4 đáp án (A,B,C,D) đều mô tả sai hành động/vật thể, "
            "hãy ghi 'Đáp án: Không có câu nào đúng cả' và dùng dữ liệu ảnh để chứng minh tại sao chúng sai."
        )
    except Exception as e:
        return f"Lỗi xử lý ảnh: {str(e)}"

# --- AGENT CLASS ---
class TOEICPart1Agent(BaseAgent):
    def _get_tools(self) -> List:
        return [get_image_data_tool]

    def _get_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_template(TOEIC_REACT_SYSTEM_PROMPT)

async def main():
    image1 = r"src/core/modules/data_test/mo_ta_tranh1.png"
    toeic_agent = TOEICPart1Agent(model=model)

    sample_transcript_ok = "(A) The man is using a screwdriver. (B) The man is hammering something. (C) The man is making the frame by hand. (D) The man is wearing protective glasses."
    sample_transcript_all_wrong = "(A) The man is sleeping on the sofa. (B) The woman is painting a picture. (C) They are swimming in the ocean. (D) The man is driving a car."
    sample_transcript_invalid = "(A) No. (B) No. (C) No."

    scenarios = [
        ("KỊCH BẢN 1 (HỢP LỆ)", sample_transcript_ok),
        ("KỊCH BẢN 2 (TẤT CẢ SAI)", sample_transcript_all_wrong),
        ("KỊCH BẢN 3 (KHÔNG HỢP LỆ)", sample_transcript_invalid)
    ]

    for name, transcript in scenarios:
        logger.info("\n" + "=" * 80)
        logger.info(f"--- ĐANG CHẠY: {name} ---")

        validate_result = pre_validate_part1_context(transcript)
        if validate_result != "1":
            print(f"\n--- KẾT QUẢ {name} ---")
            print(validate_result)
            continue

        input_data = f"Sử dụng tool để lấy ảnh từ đường dẫn: {image1}. Sau đó đối chiếu với các lựa chọn: {transcript}"
        
        try:
            res = await toeic_agent.ainvoke({"input": input_data})
            output = format_answer_only(extract_text(res))
            
            print(f"\n--- KẾT QUẢ {name} ---")
            print(output if output else "Agent không trả về nội dung. Kiểm tra API hoặc Prompt.")
            
        except Exception as e:
            logger.error(f"Lỗi thực thi {name}: {e}")
            
    print("\n" + "=" * 80 + "\n--- CHƯƠNG TRÌNH HOÀN TẤT ---")

if __name__ == "__main__":
    asyncio.run(main())