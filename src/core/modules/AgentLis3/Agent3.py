import os
import warnings
import sys
import asyncio
import logging
from dotenv import load_dotenv
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.tools import BaseTool, Tool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.dirname(SCRIPT_DIR)
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

from Agent_Base.Agent import BaseAgent
from AgentLis3.utils import process_and_label_gender, pre_validate_part3_context
from AgentLis3.prompts import ANALYSIS_PROMPT_TEXT_P3, REACT_TEMPLATE_P3

# Cấu hình
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(dotenv_path="config/.env")

class TOEICPart3Agent(BaseAgent):
    """
    Agent chuyên biệt cho TOEIC Part 3.
    Thừa kế hoàn toàn từ BaseAgent.
    """
    def __init__(self, model: BaseChatModel):
        logger.info(f"[{self.__class__.__name__}]: Khởi tạo...")
        warnings.filterwarnings("ignore")
        self.llm = model
        super().__init__(llm=self.llm)

    def __analyze_logic_handler(self, tool_input: str) -> str:
        """Logic cốt lõi của Tool: Validate -> Gọi LLM -> Trả kết quả."""
        logger.info(f"\n[Tool Called]: analyze_part3_problem_tool")

        # 1. Kiểm tra định dạng (A, B, C, D)
        validation = pre_validate_part3_context(tool_input)
        if validation != "1":
            return validation

        # 2. Xử lý logic qua LLM (Sử dụng llm từ lớp cha BaseAgent)
        try:
            final_prompt = ANALYSIS_PROMPT_TEXT_P3 + f"\n{tool_input}"
            response = self.llm.predict(final_prompt)
            return response.strip() 
        except Exception as e:
            logger.error(f"Lỗi thực thi Tool: {e}")
            return f"Đã có lỗi xảy ra: {e}"

    def _get_tools(self) -> List[BaseTool]:
        """Cung cấp danh sách công cụ."""
        return [
            Tool(
                name="analyze_part3_problem_tool", 
                func=self.__analyze_logic_handler,
                description="Analyzes Part 3 context (Transcript + Q&A) to provide answers."
            )
        ]

    def _get_prompt(self) -> BasePromptTemplate:
        """Cung cấp Template ReAct."""
        return PromptTemplate.from_template(REACT_TEMPLATE_P3)


async def main():
    # 1. Khởi tạo Model
    google_api_key = os.environ.get("openai_api_key")
    if not google_api_key:
        print("="*50)
        print("LỖI: Vui lòng đặt biến môi trường GOOGLE_API_KEY để chạy ví dụ này.")
        print("="*50)
        return

    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=google_api_key,
            convert_system_message_to_human=True
        )
    except Exception as e:
        logger.error(f"Không thể khởi tạo Gemini model: {e}")
        return

    # 2. Dữ liệu Input & Đường dẫn Audio
    AUDIO_FILE_PATH_TEST = "D:\\Github\\RAG_TOEIC1\\src\\core\\modules\\data_test\\Ld1lt.mp3" 
    #AUDIO_FILE_PATH_TEST = "D:\\Github\\RAG_TOEIC1\\src\\core\\modules\\data_test\\6KXxh.mp3"

    # CONTEXT_OK = """
    # What is the purpose of the man's call?
    # A. To ask for the company's services
    # B. To cancel a business meeting
    # C. To request a job interview
    # D. To promote a new product
    # """

    CONTEXT_OK = """
    Question: What does the woman say about her phone service?
    A. She is unhappy with Z Mobile's service.
    B. She pays a monthly fee of 70 dollars.
    C. She gets 700 unlimited minutes with everyone.
    D. She recently moved to Canada for free calls.
    """
    CONTEXT_NONE_CORRECT = """
    Q: Who has recorded the message?
    (A) A tenant who is locked out
    (B) A parking garage security guard
    (C) A tow truck company dispatcher
    (D) A local post office worker
    """
    CONTEXT_INVALID = """
    Q: How old is this building?
    (A) To transport some materials.
    (B) About ten years old.
    (C) I think it's the company office.
    """

    # 3. Tiền xử lý Audio (Sử dụng hàm từ utils)
    logger.info("\n" + "="*80)
    logger.info("--- BƯỚC 1: TIỀN XỬ LÝ AUDIO ĐỂ LẤY TRANSCRIPT ---")
    
    if not os.path.exists(AUDIO_FILE_PATH_TEST):
        labeled_transcript = f"LỖI: Không tìm thấy file âm thanh tại đường dẫn: {AUDIO_FILE_PATH_TEST}"
    else:
        labeled_transcript = process_and_label_gender(AUDIO_FILE_PATH_TEST)
    
    if labeled_transcript.startswith("LỖI"):
        print(f"LỖI NGHIÊM TRỌNG TRONG TIỀN XỬ LÝ AUDIO: {labeled_transcript}")
        return

    # 4. Chuẩn bị Contexts và Khởi tạo Agent
    full_context_ok = f"{labeled_transcript}\n{CONTEXT_OK}"
    full_context_none_correct = f"{labeled_transcript}\n{CONTEXT_NONE_CORRECT}"
    full_context_invalid = f"{labeled_transcript}\n{CONTEXT_INVALID}"
    
    agent_instance = TOEICPart3Agent(model=model)
    
    
    logger.info("\n" + "="*80)
    logger.info("--- CHẠY KỊCH BẢN 1 (HỢP LỆ) ---")
    result1 = await agent_instance.ainvoke({"input": full_context_ok})
    print("\n--- KẾT QUẢ CUỐI CÙNG (KỊCH BẢN 1) ---")
    print(result1["output"].strip())

    logger.info("\n" + "="*80)
    logger.info("--- CHẠY KỊCH BẢN 2 (KHÔNG CÓ CÂU ĐÚNG) ---")
    result2 = await agent_instance.ainvoke({"input": full_context_none_correct})
    print("\n--- KẾT QUẢ CUỐI CÙNG (KỊCH BẢN 2) ---")
    print(result2["output"].strip())
    
    logger.info("\n" + "="*80)
    logger.info("--- CHẠY KỊCH BẢN 3 (KHÔNG HỢP LỆ) ---")
    result3 = await agent_instance.ainvoke({"input": full_context_invalid})
    print("\n--- KẾT QUẢ CUỐI CÙNG (KỊCH BẢN 3) ---")
    print(result3["output"].strip())

if __name__ == "__main__":
    asyncio.run(main())