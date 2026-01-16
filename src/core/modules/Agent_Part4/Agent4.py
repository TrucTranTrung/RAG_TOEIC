import os
import re
import sys
import asyncio
import logging
from dotenv import load_dotenv
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

# --- XỬ LÝ PATH ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.dirname(SCRIPT_DIR) 
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

from Agent_Base.Agent import BaseAgent 
from utils import extract_text, format_answer_only, pre_validate_part4_context
from prompts import TOEIC_REACT_SYSTEM_PROMPT_P4, ANALYSIS_PROMPT_TEXT_P4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(dotenv_path="config/.env")

@tool
def summarize_transcript_tool(transcript: str) -> str:
    """Sử dụng khi câu hỏi yêu cầu suy luận hoặc tìm ý chính."""
    logger.info("Agent is calling summarize_transcript_tool")
    return (
        f"Dữ liệu bài nghe: {transcript}\n\n"
        "Nhiệm vụ: Để trả lời câu hỏi suy luận này, bạn hãy thực hiện:\n"
        "1. Tóm tắt nội dung theo cấu trúc: Đối tượng - Hành động - Thời gian/Địa điểm.\n"
        "2. Tìm các 'từ khóa điều kiện' (ví dụ: 'if', 'unless', 'only', 'monitoring', 'closely').\n"
        "3. Nếu đáp án không nằm trong tóm tắt hoặc ý chính, hãy quét lại dữ liệu gốc phía trên để tìm manh mối suy luận."
    )

class TOEICPart4Agent(BaseAgent):
    def _get_tools(self) -> List:
        return [summarize_transcript_tool]

    def _get_prompt(self) -> PromptTemplate:
        full_template = TOEIC_REACT_SYSTEM_PROMPT_P4 + "\n\n" + ANALYSIS_PROMPT_TEXT_P4
        return PromptTemplate.from_template(full_template)

async def main():
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        api_key=google_api_key,
        temperature=0.0,
        max_retries=0
    )

    agent = TOEICPart4Agent(model=model)

    transcript_test = """
    Our company picnic will be held this Saturday at Oak Ridge Park.
    Lunch will be served at 12:30 PM, followed by a team-building activity.
    Please bring your own blankets and sunscreen.
    The organizers are monitoring the weather forecast closely.
    """

    scenarios = [
        (
            "KỊCH BẢN 1: GIẢI ĐỀ TRỰC TIẾP (KHÔNG TÓM TẮT)", 
            f"{transcript_test}\nQuestion: What time will lunch start?\n(A) 10:00 AM (B) 12:30 PM (C) 1:00 PM (D) 2:30 PM"
        ),
        (
            "KỊCH BẢN 2: TÓM TẮT VÀ GIẢI ĐỀ (GỌI TOOL)", 
            f"{transcript_test}\nQuestion:What can be inferred about the event?\n(A) It will take place indoors (B) It may be postponed depending on the weather (C) Employees must bring their own lunch (D) Attendance is optional"
        ),
        (
            "KỊCH BẢN 3: TẤT CẢ ĐỀU SAI (TRUTH ONLY)", 
            f"{transcript_test}\nQuestion: Where is the picnic?\n(A) At the beach (B) In the office (C) At the mall (D) At the airport"
        ),
        (
            "KỊCH BẢN 4: LỖI VALIDATE (THIẾU ĐÁP ÁN)", 
            f"{transcript_test}\nQuestion: What should people bring?\n(A) Blankets (B) Food (C) Money" 
        )
    ]

    for name, query in scenarios:
        logger.info("\n" + "=" * 80)
        logger.info(f"--- THỰC THI {name} ---")

        validate_result = pre_validate_part4_context(query)
        if validate_result != "1":
            print(f"\n--- KẾT QUẢ {name} ---")
            print(validate_result)
            continue

        # --- BƯỚC 2: GỌI AGENT XỬ LÝ ---
        try:
            res = await agent.ainvoke({"input": query})
            raw_text = extract_text(res)
            output = format_answer_only(raw_text)

            print(f"\n--- KẾT QUẢ {name} ---")
            print(output)
            
        except Exception as e:
            logger.error(f"Lỗi kịch bản {name}: {e}")
    print("\n" + "=" * 80 + "\n--- CHƯƠNG TRÌNH HOÀN TẤT ---")

if __name__ == "__main__":
    asyncio.run(main())
