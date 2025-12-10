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

ANALYSIS_PROMPT_TEXT_P2 = """
You are a precise TOEIC Part 2 analyst. Your task is to analyze the given data (which is guaranteed valid) and provide a detailed explanation in VIETNAMESE.

You MUST follow this exact output format.

**STRICT FORMAT REQUIREMENT:** The output must start EXACTLY with 'Đáp án:' on the first line, followed by 'Giải thích:' on the second line. Use the hyphen (-) bullet point style for the explanations.

**EXAMPLE OF YOUR OUTPUT FORMAT:**
Đáp án: B. About ten years old.
Giải thích:
- (B) là câu trả lời phù hợp nhất vì ....
- (A) không đúng vì ....
- (C) không đúng vì ...

**RULE FOR NO CORRECT ANSWER:**
- If no option is correct, the `Đáp án:` line MUST be exactly: "Không có câu nào đúng cả."
- The `Giải thích:` block must STILL contain 3 bullets explaining why (A), (B), AND (C) are ALL incorrect.

Provide NO other text before or after this format.
Analyze the following data:
"""


def pre_validate_part2_context(problem_context: str) -> str:
    """Kiểm tra CHỈ có A, B, C và không có đáp án nào khác."""
    if not problem_context:
        return "Bạn cung cấp không đủ đáp án (A)(B)(C), vui lòng nhập lại."
    all_options = set(re.findall(r'\([A-Z]\)', problem_context.upper()))
    target_options = {'(A)', '(B)', '(C)'}
    if len(all_options) == 3 and all_options == target_options:
        return "1"  
    else:
        return "Bạn cung cấp không đủ đáp án (A)(B)(C), vui lòng nhập lại."
    
class TOEICPart2Agent(BaseAgent):
    llm: BaseChatModel 

    def __init__(self, model: BaseChatModel):
        logger.info(f"[{self.__class__.__name__}]: Khởi tạo với model...")
        warnings.filterwarnings("ignore")
        self.llm = model
        super().__init__(llm=self.llm)

    def __analyze_logic_handler(self, tool_input: str) -> str:
        logger.info(f"\n[Tool Called]: validate_context")
        validation = pre_validate_part2_context(tool_input)
        if validation != "1":
            return validation

        try:
            final_prompt = ANALYSIS_PROMPT_TEXT_P2 + f"\n{tool_input}"
            message = HumanMessage(content=[{"type": "text", "text": final_prompt}])
            response = self.llm.invoke([message])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng trong _analyze_logic_handler: {e}")
            return f"Đã có lỗi xảy ra trong quá trình phân tích: {e}"

    def _get_tools(self) -> List[BaseTool]:
        logger.info(f"[{self.__class__.__name__}]: Cung cấp tools...")
        tool_2 = Tool(
            name="analyze_part2_problem_tool",
            func=self.__analyze_logic_handler, 
            description="Analyzes the 'problem_context' to provide the final answer, including internal validation."
        )
        return [tool_2]

    def _get_prompt(self) -> BasePromptTemplate:
        REACT_TEMPLATE_P2 = """
        You are a specialized, step-by-step TOEIC Part 2 Agent.
        Available tools: {tools}
        Question: {input}
        Thought: [Your thought process based on the rules]
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input for the action
        Observation: the result of the action
        ...
        **RULES:**
        1. Always use 'analyze_part2_problem_tool' first.
        2. If Observation contains "Lỗi" or "Bạn cung cấp không đủ", Final Answer = Observation text.
        3. Else return final answer from tool.
        Thought: I have the final answer.
        Final Answer: the final answer or error message
        Begin!
        Question: {input}
        Thought: {agent_scratchpad}"""
        return PromptTemplate.from_template(REACT_TEMPLATE_P2)

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
            convert_system_message_to_human=True,
            temperature=0.0 
        )
    except Exception as e:
        logger.error(f"Không thể khởi tạo Gemini model: {e}")
        return

    part2_agent = TOEICPart2Agent(model=model)

    sample_context_ok = """
    How old is this building?
    (A) To transport some materials.
    (B) About ten years old.
    (C) I think it's the company office.
    """
    sample_context_invalid = """
    How old is this building?
    (A) To transport some materials.
    (B) About ten years old.
    (C) I think it's the company office.
    (D) Yes, it is.
    """
    sample_context_none_correct = """
    Who is the new manager?
    (A) Yes, he is.
    (B) It's on the second floor.
    (C) The meeting was canceled.
    """

    # Kịch bản 1
    result1 = await part2_agent.ainvoke({"input": sample_context_ok})
    print("\n--- KẾT QUẢ CUỐI CÙNG (KỊCH BẢN 1) ---")
    print(result1["output"].strip())

    # Kịch bản 2
    result2 = await part2_agent.ainvoke({"input": sample_context_none_correct})
    print("\n--- KẾT QUẢ CUỐI CÙNG (KỊCH BẢN 2) ---")
    print(result2["output"].strip())

    # Kịch bản 3
    result3 = await part2_agent.ainvoke({"input": sample_context_invalid})
    print("\n--- KẾT QUẢ CUỐI CÙNG (KỊCH BẢN 3) ---")
    print(result3["output"].strip())

if __name__ == "__main__":
    asyncio.run(main())