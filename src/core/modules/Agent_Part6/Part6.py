import requests
import logging
import asyncio
import os

from random import sample
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langchain_core.prompts import PromptTemplate, BasePromptTemplate

from ..Agent_Base import BaseAgent
from .utils import clean_and_extract_passage_simple
from .prompts import prompt_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="config/.env")


@tool
def vocab_search(word: str, full: bool = False) -> dict:

    """Searches for vocabulary definitions from an online dictionary API."""

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
        logger.info("ReadingAgent: Cung cấp tools [vocab_search]")
        return [vocab_search]


    def _get_prompt(self) -> BasePromptTemplate:
        """
        Cung cấp một prompt template CỤ THỂ cho ReadingAgent Part 6.
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

    openai_api_key = os.envziron.get("openai_api_key")
    if not openai_api_key:
        print("="*50)
        print("LỖI: Vui lòng đặt biến môi trường openai_api_key để chạy ví dụ này.")
        print("Lệnh (Terminal/PowerShell): set openai_api_key=YOUR_API_KEY_HERE")
        print("="*50)
        return

    try:
        model = ChatOpenAI(
            model="gpt-4o-mini",   # hoặc gpt-4.1 / gpt-4o
            api_key=openai_api_key,
            temperature=0.2
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
        print(f"Câu trả lời cuối cùng: {result['output']}")

    except Exception as e:
        logger.error(f"Lỗi khi chạy agent: {e}")

# Chạy hàm main bất đồng bộ
if __name__ == "__main__":
    asyncio.run(main())