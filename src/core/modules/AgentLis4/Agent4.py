import os
import warnings
import re
import sys
import asyncio
import logging
from dotenv import load_dotenv
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.tools import BaseTool, Tool
from langchain.schema.messages import HumanMessage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(MODULES_DIR)
from Agent_Base.Agent import BaseAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(dotenv_path="config/.env")

ANALYSIS_PROMPT_TEXT_P4 = """
You are a precise TOEIC Part 4 analyst. Your task is to analyze the given data (which is guaranteed valid if validation passes) and provide a detailed explanation in VIETNAMESE.

You MUST follow this exact output format.

**EXAMPLE OF YOUR OUTPUT FORMAT (This is a Part 4 example):**
Đáp án: D. A building management office
Giải thích:
- (D) là câu trả lời phù hợp nhất...
- (A) không đúng vì...
- (B) không đúng vì...
- (C) không đúng vì...

**RULE FOR NO CORRECT ANSWER:**
- If no option is correct, the `Đáp án:` line MUST be exactly: "Không có câu nào đúng cả."
- The `Giải thích:` block must STILL contain 4 bullets explaining why (A), (B), (C), AND (D) are ALL incorrect.

Provide NO other text before or after this format.
Analyze the following data:
"""

def pre_validate_part4_context(problem_context: str) -> str:
    """Validate chỉ có đúng 4 đáp án (A)(B)(C)(D). Trả về '1' nếu hợp lệ, else thông báo lỗi."""
    if not problem_context:
        return "Bạn cung cấp không đủ đáp án (A)(B)(C), vui lòng nhập lại."
    all_options = set(re.findall(r'\([A-Z]\)', problem_context.upper()))
    target_options = {'(A)', '(B)', '(C)','(D)'}
    if len(all_options) == 4 and all_options == target_options:
        return "1"  
    else:
        return "Bạn cung cấp không đủ đáp án (A)(B)(C)(D), vui lòng nhập lại."

class TOEICPart4Agent(BaseAgent):
    llm: BaseChatModel

    def __init__(self, model: BaseChatModel):
        logger.info(f"[{self.__class__.__name__}]: Khởi tạo với model...")
        warnings.filterwarnings("ignore")
        self.llm = model
        super().__init__(llm=self.llm)

    def __analyze_logic_handler(self, tool_input: str) -> str:
        logger.info(f"\n[Tool Called]: analyze_part4_problem_tool")

        validation = pre_validate_part4_context(tool_input)
        if validation != "1":
            return validation

        try:
            final_prompt = ANALYSIS_PROMPT_TEXT_P4 + f"\n{tool_input}"
            message = HumanMessage(content=[{"type": "text", "text": final_prompt}])
            response = self.llm.invoke([message])
            return response.content.strip() 
            
        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng trong _analyze_logic_handler: {e}")
            return f"Đã có lỗi xảy ra trong quá trình phân tích: {e}"

    def _get_tools(self) -> List[BaseTool]:
        logger.info(f"[{self.__class__.__name__}]: Cung cấp tools...")
        tool_2 = Tool(
            name="analyze_part4_problem_tool",
            func=self.__analyze_logic_handler,
            description="Analyzes the 'problem_context' to provide the final answer, including internal validation."
        )
        return [tool_2]

    def _get_prompt(self) -> BasePromptTemplate:
        REACT_TEMPLATE_P4 = """
        You are a specialized, step-by-step TOEIC Part 4 Agent.
        Available tools: {tools}
        Question: {input}
        Thought: [Your thought process based on the rules]
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input for the action
        Observation: the result of the action
        ...
        **RULES:**
        1. Always use 'analyze_part4_problem_tool' first.
        2. If Observation is an error, Final Answer = error text.
        3. Else return final answer from tool.
        Thought: I have the final answer. The Final Answer MUST strictly follow the 'Đáp án: ... Giải thích: ...' format provided in the analysis prompt, ensuring all parts are present.
        Final Answer: the final answer or error message
        Begin!
        Question: {input}
        Thought: {agent_scratchpad}"""

        return PromptTemplate.from_template(REACT_TEMPLATE_P4)

async def main():
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

    part4_agent = TOEICPart4Agent(model=model)

    sample_context_ok = """
    M-Au You have reached the information line for the Cranbury Apartments management office...
    Who has recorded the message?
    (A) A city mayor’s office
    (B) A maintenance department
    (C) An automobile dealership
    (D) A building management office
    """
    sample_context_invalid = """
    How old is this building?
    (A) To transport some materials.
    (B) About ten years old.
    (C) I think it's the company office.
    """
    sample_context_none_correct = """
    M-Au You have reached the information line for the Cranbury Apartments management office...
    Who has recorded the message?
    (A) A tenant who is locked out
    (B) A parking garage security guard
    (C) A tow truck company dispatcher
    (D) A local post office worker
    """

    # Kịch bản 1
    logger.info("\n" + "="*80)
    logger.info("--- CHẠY KỊCH BẢN 1 (HỢP LỆ) ---")
    result1 = await part4_agent.ainvoke({"input": sample_context_ok})
    print("\n--- KẾT QUẢ CUỐI CÙNG (KỊCH BẢN 1) ---")
    print(result1["output"].strip())

    # Kịch bản 2
    logger.info("\n" + "="*80)
    logger.info("--- CHẠY KỊCH BẢN 2 (KHÔNG CÓ CÂU ĐÚNG) ---")
    result2 = await part4_agent.ainvoke({"input": sample_context_none_correct})
    print("\n--- KẾT QUẢ CUỐI CÙNG (KỊCH BẢN 2) ---")
    print(result2["output"].strip())

    # Kịch bản 3
    logger.info("\n" + "="*80)
    logger.info("--- CHẠY KỊCH BẢN 3 (KHÔNG HỢP LỆ) ---")
    result3 = await part4_agent.ainvoke({"input": sample_context_invalid})
    print("\n--- KẾT QUẢ CUỐI CÙNG (KỊCH BẢN 3) ---")
    print(result3["output"].strip())

if __name__ == "__main__":
    asyncio.run(main())