import re
import os
import asyncio
import logging
import sys
from typing import List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.tools import tool, Tool

# --- XỬ LÝ PATH & IMPORT ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.dirname(SCRIPT_DIR) 
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

from Agent_Base.Agent import BaseAgent 
from utils import encode_image_to_base64, pre_validate_part1_context, extract_text, format_answer_only
from prompts import ANALYSIS_PROMPT_TEXT, TOEIC_REACT_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(dotenv_path="config/.env")

@tool
def get_image_data_tool(image_path: str) -> str:
    """Lấy dữ liệu Base64 của ảnh. Bắt buộc gọi khi Question có Image."""
    try:
        clean_path = image_path.strip().strip("'").strip('"')
        if not os.path.exists(clean_path):
            return f"Lỗi: Không tìm thấy file tại {clean_path}"
        return encode_image_to_base64(clean_path)
    except Exception as e:
        return f"Lỗi xử lý ảnh: {str(e)}"

# --- AGENT CLASS ---
class TOEICPart1Agent(BaseAgent):
    def _get_tools(self) -> List:
        return [get_image_data_tool]

    def _get_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_template(TOEIC_REACT_SYSTEM_PROMPT)

async def main():
    logger.info("--- Khởi tạo Gemini 2.5 Flash Model ---")
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    
    # Để temperature cực thấp để tránh AI viết thêm lời dẫn thừa
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        api_key=google_api_key,
        temperature=0.0 
    )

    image1 = r"D:\Github\RAG_TOEIC1\src\core\modules\data_test\mo_ta_tranh1.png"
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

        # 1. Validation nội bộ
        validate_result = pre_validate_part1_context(transcript)
        if validate_result != "1":
            print(f"\n--- KẾT QUẢ {name} ---")
            print(validate_result)
            continue

        # 2. Chạy Agent
        input_data = f"Image: {image1}, Transcript: {transcript}"
        try:
            res = await toeic_agent.ainvoke({"input": input_data})
            output = format_answer_only(extract_text(res))
            print(f"\n--- KẾT QUẢ {name} ---")
            print(output)
            
        except Exception as e:
            logger.error(f"Lỗi thực thi {name}: {e}")
    print("\n" + "=" * 80 + "\n--- CHƯƠNG TRÌNH HOÀN TẤT ---")

if __name__ == "__main__":
    asyncio.run(main())