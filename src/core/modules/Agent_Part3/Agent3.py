import os
import asyncio
import logging
import assemblyai as aai
# import ollama
from dotenv import load_dotenv
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
logging.getLogger("assemblyai").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

from ..Agent_Base import BaseAgent
from .models import llm_mistral
from .utils import label_transcript_gender, extract_text, format_answer_only, pre_validate_part3_context
from .prompts import UNIFIED_LISTENING_REACT_PROMPT 

# --- 2. CONFIG ---
load_dotenv(dotenv_path="config/.env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
if ASSEMBLYAI_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY

_TRANSCRIPT_CACHE = {}

# --- 3. TOOLS ---
@tool
def call_assemblyai_transcribe(audio_path: str) -> str:
    """Dùng để chuyển audio (.mp3) thành văn bản có gán nhãn giới tính [M]/[F]."""
    if not ASSEMBLYAI_API_KEY: return "LỖI: Thiếu API Key."
    
    audio_key = os.path.abspath(audio_path)
    if audio_key in _TRANSCRIPT_CACHE:
        return _TRANSCRIPT_CACHE[audio_key]

    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(speaker_labels=True)
    
    try:
        raw_obj = transcriber.transcribe(audio_path, config=config)
        labeled = label_transcript_gender(raw_obj, audio_path)
        _TRANSCRIPT_CACHE[audio_key] = labeled
        return labeled
    except Exception as e:
        return f"Lỗi: {str(e)}"


# @tool
# def summarize_transcript_tool(transcript: str) -> str:
#     """BẮT BUỘC dùng cho câu hỏi suy luận, mục đích, ý chính."""
#     logger.info("--- [Action]: đang tóm tắt bài nghe ---")
    
#     prompt = f"""
#     Analyze this TOEIC transcript for inference questions.
#     Provide a brief summary in Vietnamese covering:
#     - Main Topic/Purpose
#     - Key Entities (Speaker roles, locations)
#     - Specific Clues (Dates, numbers, reasons)
#     Transcript: {transcript}
#     """
    
#     try:
#         response = ollama.chat(
#             model='llama3.2', 
#             messages=[{'role': 'user', 'content': prompt}],
#             options={'temperature': 0} 
#         )
#         return response['message']['content']
#     except Exception as e:
#         return f"Lỗi tóm tắt: {str(e)}"
    
# --- 4. CLASS AGENT ---
class TOEICListeningAgent(BaseAgent):
    def _get_tools(self) -> List:
        return [call_assemblyai_transcribe]

    def _get_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_template(UNIFIED_LISTENING_REACT_PROMPT)

# --- 5. MAIN ---
async def main():
    agent = TOEICListeningAgent(model=llm_mistral)

    AUDIO_P3 = r"src/core/modules/data_test/Ld1lt.mp3"
    AUDIO_P4 = r"src/core/modules/data_test/4luSC.mp3"

    scenarios = [
        ("P3_CASE_1", AUDIO_P3, "What does the woman say about her phone service?\n A. She is unhappy with Z Mobile's service. \nB. She pays a monthly fee of 70 dollars. \n C. She gets 700 unlimited minutes with everyone. \n D. She recently moved to Canada for free calls."),
        ("P3_CASE_2", AUDIO_P3, "Who recorded the message?\n(A) Tenant (B) Security (C) Dispatcher (D) Post worker"),
        ("P3_CASE_3", AUDIO_P3, "How old is this building?\n(A) 10 years (B) Office"), # Case lỗi thiếu đáp án
        ("P4_CASE_1", AUDIO_P4, "When is the new addition to the library scheduled to open?\n(A) At 9 a.m. this Friday (B) At 10 a.m. next Monday (C) At 9 a.m. next Monday (D) At 10 a.m. this Friday")
    ]

    for name, audio, problem in scenarios:
        logger.info("\n" + "=" * 80)
        logger.info(f"--- ĐANG CHẠY: {name} ---")

        # --- VALIDATE LOGIC ---
        validate_result = pre_validate_part3_context(problem)
        if validate_result != "1":
            print(f"\n--- KẾT QUẢ {name} ---")
            print(validate_result)
            continue

        input_data = f"Audio file: {audio}\nProblem: {problem}"
        try:
            res = await agent.ainvoke({"input": input_data})
            output = format_answer_only(extract_text(res))
            print(f"\n[KẾT QUẢ {name}]:\n{output}")
        except Exception as e:
            logger.error(f"Lỗi: {e}")

if __name__ == "__main__":
    asyncio.run(main())