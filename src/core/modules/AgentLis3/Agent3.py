import os
import sys
import asyncio
import logging
import assemblyai as aai
from dotenv import load_dotenv
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

# --- 1. XỬ LÝ PATH ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.dirname(SCRIPT_DIR)
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

from Agent_Base.Agent import BaseAgent
from AgentLis3.utils import label_transcript_gender, pre_validate_part3_context, extract_text,format_answer_only
from AgentLis3.prompts import ANALYSIS_PROMPT_TEXT_P3, REACT_TEMPLATE_P3

# --- 2. CẤU HÌNH LOGGING & ENV ---
load_dotenv(dotenv_path="../../../config/.env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
print(ASSEMBLYAI_API_KEY)
if ASSEMBLYAI_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY

# --- 3. ĐỊNH NGHĨA TOOLS ---
@tool
def call_assemblyai_transcribe(audio_path: str):
    """
    Dùng công cụ này khi đầu vào là một đường dẫn file âm thanh (ví dụ: .mp3).
    Nó sẽ chuyển đổi âm thanh thành văn bản và gán nhãn giới tính [M]/[F] cho người nói.
    """
    if not ASSEMBLYAI_API_KEY:
        return "LỖI: Chưa cấu hình API Key cho AssemblyAI."
    
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(speaker_labels=True)
    try:
        # Agent sẽ tự động gọi bước này khi nó thấy input là file path
        logger.info(f"--- [Agent Action]: Đang sử dụng Tool để transcribe file: {audio_path} ---")
        raw_obj = transcriber.transcribe(audio_path, config=config)
        
        # Gán nhãn giới tính ngay trong tool để cung cấp dữ liệu sạch cho Agent suy luận
        labeled_text = label_transcript_gender(raw_obj, audio_path)
        return labeled_text
    except Exception as e:
        return f"LỖI KẾT NỐI: {str(e)}"

# --- 4. ĐỊNH NGHĨA AGENT CLASS ---
class TOEICPart3Agent(BaseAgent):
    def _get_tools(self) -> List:
        return [call_assemblyai_transcribe]

    def _get_prompt(self) -> PromptTemplate:
        full_content = ANALYSIS_PROMPT_TEXT_P3 + "\n" + REACT_TEMPLATE_P3
        return PromptTemplate.from_template(full_content)

# --- 5. HÀM CHẠY CHÍNH ---
async def main():
    """Hàm chạy chính (bất đồng bộ) để test agent."""

    # --- Initialize Gemini 2.5 Flash Model ---
    logger.info("--- Khởi tạo Gemini 2.5 Flash Model ---")

    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        print("="*50)
        print("LỖI: Vui lòng đặt biến môi trường GOOGLE_API_KEY.")
        print("="*50)
        return

    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=google_api_key,
            convert_system_message_to_human=True,
            temperature=0.2,
            max_retries=0
        )
    except Exception as e:
        logger.error(f"Lỗi khởi tạo model: {e}")
        return

    # --- CONTEXT ---
    AUDIO_FILE_PATH_TEST = "/home/daniel/Documents/RAG_TOEIC/src/core/modules/data_test/Ld1lt.mp3"
    #AUDIO_FILE_PATH_TEST = "D:\\Github\\RAG_TOEIC1\\src\\core\\modules\\data_test\\6KXxh.mp3"

    # CONTEXT_OK = """
    # What is the purpose of the man's call?
    # A. To ask for the company's services
    # B. To cancel a business meeting
    # C. To request a job interview
    # D. To promote a new product
    # """

    CONTEXT_OK = """
    What does the woman say about her phone service?
    A. She is unhappy with Z Mobile's service.
    B. She pays a monthly fee of 70 dollars.
    C. She gets 700 unlimited minutes with everyone.
    D. She recently moved to Canada for free calls."""

    CONTEXT_NONE_CORRECT = """
    Who has recorded the message?
    (A) A tenant who is locked out
    (B) A parking garage security guard
    (C) A tow truck company dispatcher
    (D) A local post office worker"""

    CONTEXT_INVALID = """
    How old is this building?
    (A) To transport some materials.
    (B) About ten years old.
    (C) I think it's the company office."""

    agent_instance = TOEICPart3Agent(model=model)

    # --- CHẠY CÁC KỊCH BẢN ---
    scenarios = [
        ("KỊCH BẢN 1 (HỢP LỆ)", CONTEXT_OK),
        ("KỊCH BẢN 2 (KHÔNG CÂU ĐÚNG)", CONTEXT_NONE_CORRECT),
        ("KỊCH BẢN 3 (KHÔNG HỢP LỆ)", CONTEXT_INVALID),
    ]

    for name, context in scenarios:
        logger.info("\n" + "=" * 80)
        logger.info(f"--- ĐANG CHẠY: {name} ---")

        validate_result = pre_validate_part3_context(context)
        if validate_result != "1":
            print(f"\n--- KẾT QUẢ {name} ---")
            print(validate_result)
            continue


        input_data = f"Audio file: {AUDIO_FILE_PATH_TEST}\nProblem: {context}"
        res = await agent_instance.ainvoke({"input": input_data})

        output = format_answer_only(extract_text(res))

        print(f"\n--- KẾT QUẢ {name} ---")
        print(output)

    print("\n" + "=" * 80 + "\n--- CHƯƠNG TRÌNH HOÀN TẤT ---")

if __name__ == "__main__":
    asyncio.run(main())