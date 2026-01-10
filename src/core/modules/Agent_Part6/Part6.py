import requests
import logging
import asyncio

from typing import List
from langchain_core.tools import tool, BaseTool
from langchain_core.prompts import PromptTemplate, BasePromptTemplate

from ..Agent_Base import BaseAgent
from .utils import clean_and_extract_passage_simple
from .prompts import prompt_string
from .models import llm_mistral

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def vocab_search(word: str, full: bool = False) -> dict:
    """
    Searches for vocabulary definitions from an online dictionary API.
    Returns definitions and examples.
    """

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
        # print(word, results)
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
    
# @tool
# # def Summarize(word: str, full: bool = False) -> dict:
# def summarize(text: str, max_len=60) -> str:
#     """ 
#        this is a text summarization tool given a text input, it returns a summarized version of the text
#     """
#     inp = "summarize: " + text
#     ids = tok.encode(inp, return_tensors="pt", truncation=True)
#     out = model.generate(
#         ids,
#         max_length=max_len,
#         min_length=20,
#         num_beams=4,
#         length_penalty=1.5,
#         early_stopping=True
#     )
#     return tok.decode(out[0], skip_special_tokens=True)

# --- ĐỊNH NGHĨA CÁC CLASS AGENT ---
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
    """Hàm chạy chính (bất đồng bộ) để test agent."""

    # --- Initialize Qwen3 Flash Model ---
    PART_6_PASSAGE = """For your protection, we suggest you ship via UPS. A replacement will be made and if the shoe style you returned is not available, a comparable style will be substituted. We guarantee to match the quality of the shoes you used to _______.
    A. wear
    B. wearing
    C. worn
    D. be worn"""
    s = clean_and_extract_passage_simple(PART_6_PASSAGE)
    # print(s)

    # --- Initialize ReadingAgent with Model ---
    print("\n--- Khởi tạo ReadingAgent với Model ---")
    reading_agent = LanguageAgentPart6(model=llm_mistral)  # Pass the pre-loaded model

    print("\n--- Bắt đầu chạy ReadingAgent (Async) ---")

    try:
        # Sử dụng .ainvoke() (bất đồng bộ) vì đây là API call thật
        result = await reading_agent.ainvoke({
            "input": s
        })

        print(f"Câu trả lời cuối cùng: {result['output']}")

    except Exception as e:
        logger.error(f"Lỗi khi chạy agent: {e}")

# Chạy hàm main bất đồng bộ
if __name__ == "__main__":
    asyncio.run(main())