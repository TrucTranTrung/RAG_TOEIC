import requests
import logging
import asyncio

from random import sample
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from ..Agent_Base import BaseAgent
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.tools import BaseTool, tool
from langchain_core.prompts import PromptTemplate, BasePromptTemplate

from utils import clean_and_extract_passage_simple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@tool
def vocab_search(word: str, full: bool = False) -> dict:
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()[0]

        results = []
        meanings = data.get("meanings", [])
        if not full:
            meanings = meanings[:2]

        for m in meanings:
            defs = m.get("definitions", [])
            if not defs:
                continue

            d = defs[0]  
            results.append({
                "partOfSpeech": m.get("partOfSpeech"),
                "definition": d.get("definition"),
                "example": d.get("example")
            })

        return {
            "word": word,
            "meanings": results
        }

    except Exception:
        return {
            "word": word,
            "meanings": [],
            "error": "Definition not found"
        }

# --- ĐỊNH NGHĨA CÁC CLASS AGENT (SỬ DỤNG GEMINI API) ---
class LanguageAgentPart6(BaseAgent):
    """Agent này dùng để sử lí part6 để chọn từ thích hợp điền vào chỗ trống trong đoạn văn"""
    def _get_tools(self) -> List[BaseTool]:
        """
        Cung cấp danh sách các tools CHUYÊN BIỆT cho Reading Part 6.
        """
        logger.info("ReadingAgent: Cung cấp tools [check_grammar_tool, find_synonym_tool]")
        return [vocab_search]


    def _get_prompt(self) -> BasePromptTemplate:
        """
        Cung cấp một prompt template CỤ THỂ cho ReadingAgent Part 6.
        """
        prompt_string = """
        Bạn là một trợ lý AI chuyên gia về Đọc hiểu và Ngữ pháp Tiếng Anh (TOEIC Reading).

        NHIỆM VỤ:
        1. Với mỗi câu hỏi tiếng Anh được cung cấp, hãy:
        - Chọn đáp án đúng (A, B, C hoặc D).
        - Viết **giải thích đầy đủ nhưng ngắn gọn bằng tiếng Việt** cho lý do tại sao đáp án đó đúng (2–4 câu, tối đa 80 từ).
        - Viết **lý do ngắn gọn** cho từng đáp án sai (mỗi đáp án 1 dòng, tối đa 20 từ, tập trung vào lỗi ngữ pháp hoặc ngữ nghĩa).
        2. Nếu có nhiều câu hỏi, hãy trả lời lần lượt theo thứ tự.
        3. Tất cả phần trả lời và giải thích PHẢI bằng tiếng Việt, dễ hiểu, ngắn gọn.
        4. Nếu cần, bạn có thể sử dụng các công cụ sau:
        {tools}
        (Tên công cụ: {tool_names})

        BẮT BUỘC TUÂN THEO ĐỊNH DẠNG DƯỚI ĐÂY:

        Question: <Câu hỏi và lựa chọn>
        Thought: <suy nghĩ ngắn về hướng giải>
        Action: <tên công cụ hoặc "none">
        Action Input: <input cho công cụ hoặc "N/A">
        Observation: <kết quả công cụ (do hệ thống cung cấp)>

        Final Answer:
        **Đáp án đúng:** (X)
        **Giải thích:** [Giải thích bằng tiếng Việt, giải thích đầy đủ chi tiết vì sao chọn đáp án đó đúng, dựa vào đâu trong câu hỏi để xác định]
        **Lý do các đáp án sai:**
        (A) ...
        (B) ...
        (C) ...
        (D) ...

        --- (Lặp lại cho từng câu hỏi) ---

        Cuối cùng, tổng hợp lại ngắn gọn:
        Final Summary:
        <Liệt kê tất cả số câu và đáp án đúng, ví dụ:>
        131: (A) — động từ chia đúng thì với chủ ngữ.
        132: (C) — giới từ phù hợp ngữ cảnh.

        BẮT ĐẦU!

        Question: {input}

        {agent_scratchpad}
        """
        logger.info("ReadingAgent: Cung cấp prompt chuyên về Reading")

        return PromptTemplate.from_template(prompt_string)


# --- CHẠY DEMO ĐA TOOL/AGENT (TÍCH HỢP) ---
async def main():
    # print(vocab_search("address"))
    # print(vocab_search("anxiety"))
    """Hàm chạy chính (bất đồng bộ) để test agent."""

    # --- Initialize Gemini 2.5 Flash Model ---
    logger.info("--- Khởi tạo Gemini 2.5 Flash Model ---")

    google_api_key = "AI"
    if not google_api_key:
        print("="*50)
        print("LỖI: Vui lòng đặt biến môi trường GOOGLE_API_KEY để chạy ví dụ này.")
        print("Lệnh (Terminal/PowerShell): set GOOGLE_API_KEY=YOUR_API_KEY_HERE")
        print("="*50)
        return

    try:
        # Sử dụng model "gemini-2.5-flash"
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=google_api_key,
            convert_system_message_to_human=True
        )
    except Exception as e:
        logger.error(f"Không thể khởi tạo Gemini model: {e}")
        return

    # --- Initialize ReadingAgent with Model ---
    print("\n--- Khởi tạo ReadingAgent với Model ---")

    # Pass model to BaseAgent 
    reading_agent = LanguageAgentPart6(model=model)

    print("\n--- Bắt đầu chạy ReadingAgent (Async) ---")

    # --- DỮ LIỆU VÍ DỤ PART 6 ---
    PART_6_PASSAGE = """
    The annual departmental retreat will be held next month. Please check the attachment for the detailed schedule.
    We hope everyone _______ this important team-building event.
    (A) attends
    (B) attending
    (C) to attend
    (D) attendance
    Could you choose the correct answer and explain why?
    """
    s = clean_and_extract_passage_simple(PART_6_PASSAGE)
    # print(f"Test input: {s}")
    try:
        # Sử dụng .ainvoke() (bất đồng bộ) vì đây là API call thật
        result = await reading_agent.ainvoke({
            "input": s
        })

        print("\n--- Kết quả từ ReadingAgent (Gemini) ---")
        print(f"Câu trả lời cuối cùng: {result['output'][0]['text']}")

    except Exception as e:
        logger.error(f"Lỗi khi chạy agent: {e}")

# Chạy hàm main bất đồng bộ
if __name__ == "__main__":
    asyncio.run(main())