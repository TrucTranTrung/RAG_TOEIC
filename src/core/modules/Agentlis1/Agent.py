import re
import os
import asyncio
import logging
import sys
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.tools import BaseTool, Tool
from utils import encode_image_to_base64 
from langchain.schema.messages import HumanMessage
from dotenv import load_dotenv 

# Thiết lập đường dẫn
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.dirname(SCRIPT_DIR) 
sys.path.append(MODULES_DIR)
from Agent_Base.Agent import BaseAgent 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(dotenv_path="config/.env")

def pre_validate_part1_context(transcript: str) -> str:
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


ANALYSIS_PROMPT_TEXT = """
You are a precise and accurate TOEIC Part 1 analyst. Your goal is to analyze the image and transcript to find the single best answer among the given options.
Follow these instructions STRICTLY:
1. Carefully observe the main action, people, objects, and setting in the image.
2. Read all four provided options (A, B, C, D).
3. Your task is to select the option that **BEST DESCRIBES** the image.
4. You must structure your output in Vietnamese using one of the two exact formats below. Do not add any text before or after this structure.

---
You MUST follow this exact output format.

**EXAMPLE OF YOUR OUTPUT FORMAT:**
Đáp án: B. About ten years old.
Giải thích:
- (B) là câu trả lời phù hợp nhất...
- (A) không đúng vì...
- (C) không đúng vì...
- (D) không đúng vì...


**RULE FOR NO CORRECT ANSWER:**
- If no option is correct, the `Đáp án:` line MUST be exactly: "Không có câu nào đúng cả."
- The `Giải thích:` block must STILL contain 4 bullets explaining why (A), (B), AND (C) are ALL incorrect.

Provide NO other text before or after this format.

Now, analyze the following data based on the strict rules above.
"""

class TOEICPart1Agent(BaseAgent):
    """
    Agent này chuyên xử lý các tác vụ TOEIC Listening Part 1, sử dụng 1 Tool duy nhất với Internal Validation.
    """
    
    llm: BaseChatModel 

    def __init__(self, model: BaseChatModel):
        logger.info(f"[{self.__class__.__name__}]: Khởi tạo với model...")
        self.llm = model
        super().__init__(llm=self.llm) 

    def __analyze_logic_handler(self, full_input_string: str) -> str:
        """
        Hàm logic nội bộ cho Tool: Phân tích và validate nội bộ.
        """
        logger.debug(f"[Tool Called]: _analyze_logic_handler (Analyze & Validate)") 
        try:
            image_match = re.search(r"Image: (.*?)(?:, Transcript:|$)", full_input_string, re.DOTALL)
            transcript_match = re.search(r"Transcript: (.*)", full_input_string, re.DOTALL)
            
            if not image_match or not transcript_match:
                raise ValueError("Input phải chứa 'Image: [path]' và 'Transcript: [text]'")
                
            image_path = image_match.group(1).strip()
            transcript = transcript_match.group(1).strip()
            
            if not image_path or not transcript:
                raise ValueError("Missing 'image_path' or 'transcript' sau khi parse.")
            
            logger.debug(f"[Parsed]: Image Path: {image_path}")
            logger.debug(f"[Parsed]: Transcript: {transcript}")

            # --- BƯỚC 1: VALIDATION NỘI BỘ ---
            validation = pre_validate_part1_context(transcript)
            if validation != "1":
                logger.error(f"[Validation Failed]: {validation}") 
                return validation 

            # --- BƯỚC 2: PHÂN TÍCH VỚI HÌNH ẢNH (Chỉ chạy khi validation OK) ---
            image_b64 = encode_image_to_base64(image_path)
            final_prompt = ANALYSIS_PROMPT_TEXT + f"\nTranscript:\n{transcript}"
            message = HumanMessage(content=[
                {"type": "text", "text": final_prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
            ])

            response = self.llm.invoke([message])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng trong _analyze_logic_handler: {e}")
            return f"Đã có lỗi xảy ra trong quá trình phân tích: {e}"

    def _get_tools(self) -> List[BaseTool]:
        """
        [TRIỂN KHAI TỪ BASEAGENT]
        Cung cấp duy nhất 1 tool phân tích.
        """
        logger.debug(f"{self.__class__.__name__}: Cung cấp tools [analyze_image_and_transcript_tool]")

        tool_analysis = Tool(
            name="analyze_image_and_transcript_tool",
            func=self.__analyze_logic_handler,
            description=(
                "Analyzes an image and its transcript to determine the best TOEIC Part 1 answer. "
                "This tool performs internal validation for options (A), (B), (C), (D) first. "
                "The input for this tool MUST BE THE ENTIRE ORIGINAL 'Question' STRING, "
                "which contains both 'Image:' and 'Transcript:' parts."
            )
        )

        return [tool_analysis]


    def _get_prompt(self) -> BasePromptTemplate:
        """
        [TRIỂN KHAI TỪ BASEAGENT]
        Sửa lại prompt ReAct, loại bỏ Rule 2 (chuyển từ 2 tools sang 1 tool).
        """
        
        REACT_TEMPLATE = """
        You are a specialized TOEIC Part 1 Agent. Your goal is to solve the question using the available tools.
        Available tools: {tools}
                
        Use the following format:
        Question: the input question you must answer
        Thought: analyze the current state and determine the next action
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input for the action
        Observation: the result of the action
        ... (repeat Thought/Action/Action Input/Observation)

        **CRITICAL RULES (QUY TẮC TỐI QUAN TRỌNG):**
        1. Always use the 'analyze_image_and_transcript_tool' first.
        2. If the 'Observation' from the tool is an error message (e.g., "Bạn cung cấp không đủ...", "Lỗi: Không tìm thấy..."), your next Thought MUST be "I have the final answer" and the 'Final Answer' MUST be the **EXACT, UNTRANSLATED** text from the 'Observation'. (Nếu Observation là lỗi, Final Answer PHẢI LÀ Y HỆT LỖI ĐÓ, KHÔNG ĐƯỢC DỊCH).
        3. The input for the tool MUST be the ENTIRE Question string.
        
        Thought: I have the final answer. DO NOT CALL ANY TOOL AGAIN after this point.
        Final Answer: the final answer or error message (Phải tuân thủ CRITICAL RULES)

        Begin!
        Question: {input}
        Thought: {agent_scratchpad}
        """
        logger.debug(f"[{self.__class__.__name__}]: Cung cấp prompt (Đã cập nhật cho 1 Tool)...")
        return PromptTemplate.from_template(REACT_TEMPLATE)

async def main():
    """Hàm chạy chính (bất đồng bộ) để test agent."""
    logger.debug("--- Khởi tạo Gemini 2.5 Flash Model ---") 

    google_api_key = os.environ.get("GOOGLE_API_KEY")
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

    print("\n--- Khởi tạo TOEICPart1Agent với Model (1 Tool) ---")
    toeic_agent = TOEICPart1Agent(model=model)
    print("\n--- Bắt đầu chạy TOEICPart1Agent (Async) ---")

    try:
        image1 = os.path.join(SCRIPT_DIR, "mo_ta_tranh1.png") 
        if not os.path.exists(image1):
            logger.error(f"Lỗi: Không tìm thấy file ảnh tại: {image1}")
            print(f"Lỗi: Không tìm thấy file ảnh tại: {image1}")
            return
    except NameError:
        logger.warning("Không thể tìm __file__, giả sử ảnh 'mo_ta_tranh1.png' nằm cùng thư mục.")
        image1 = "mo_ta_tranh1.png"
        if not os.path.exists(image1):
            logger.error(f"Lỗi: Không tìm thấy file ảnh tại: {image1}")
            print(f"Lỗi: Không tìm thấy file ảnh tại: {image1}")
            return
    
    # Định nghĩa các transcript mẫu
    sample_transcript_ok = "(A) The man is using a screwdriver. (B) The man is hammering something. (C) The man is making the frame by hand. (D) The man is wearing protective glasses."
    sample_transcript_all_wrong = "(A) The man is sleeping on the sofa. (B) The woman is painting a picture. (C) They are swimming in the ocean. (D) The man is driving a car."
    sample_transcript_invalid = "(A) This is option one. (B) This is option two. (C) They are swimming in the ocean. (D) The man is driving a car. (E). They are cooking"

    try:
        # --- KỊCH BẢN 1: ĐẦY ĐỦ---
        logger.info("\n" + "="*80)
        logger.info("--- CHẠY KỊCH BẢN 1: ĐẦY ĐỦ DỮ LIỆU HỢP LỆ ---")
        input_1 = f"Image: {image1}, Transcript: {sample_transcript_ok}"
        logger.debug(f"Input 1: {input_1}") 
        result1 = await toeic_agent.ainvoke({
            "input": input_1
        })
        print("\n--- KẾT QUẢ CUỐI CÙNG (KỊCH BẢN 1) ---")
        print(result1["output"].strip())


        # --- KỊCH BẢN 2: TẤT CẢ 4 CÂU ĐỀU SAI---
        logger.info("\n" + "="*80)
        logger.info("--- CHẠY KỊCH BẢN 2: TẤT CẢ 4 CÂU ĐỀU SAI ---")
        input_2 = f"Image: {image1}, Transcript: {sample_transcript_all_wrong}"
        logger.debug(f"Input 2: {input_2}")
        result2 = await toeic_agent.ainvoke({
            "input": input_2
        })
        print("\n--- KẾT QUẢ CUỐI CÙNG (KỊCH BẢN 2) ---")
        print(result2["output"].strip())


        # --- KỊCH BẢN 3: TRANSCRIPT KHÔNG HỢP LỆ (Bị chặn bởi Internal Validation) ---
        logger.info("\n" + "="*80)
        logger.info("--- CHẠY KỊCH BẢN 3: TRANSCRIPT KHÔNG HỢP LỆ (Async) ---")
        input_3 = f"Image: {image1}, Transcript: {sample_transcript_invalid}"
        logger.debug(f"Input 3: {input_3}")
        result3 = await toeic_agent.ainvoke({
            "input": input_3
        })
        print("\n--- KẾT QUẢ CUỐI CÙNG (KỊCH BẢN 3) ---")
        print(result3["output"].strip())
        
    except Exception as e:
        logger.error(f"Lỗi khi chạy agent: {e}")

if __name__ == "__main__":
    asyncio.run(main())