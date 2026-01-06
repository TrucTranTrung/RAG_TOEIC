from typing import Annotated, TypedDict, List
from langchain_core.tools import tool, BaseTool
from langchain_core.prompts import PromptTemplate, BasePromptTemplate
from langchain_core.tools import tool
# from langfuse.callback import CallbackHandler

from .prompts import prompt_string
from ...Agent_Base.Agent import BaseAgent
from .utils import is_chitchat_reply
from ..OCR_modules.OCR_api import perform_ocr


# ================================ catch chit chat wrapper  =================================
def chitchat_func(text: str):
    """
    Tool for the host bot: detects casual or social conversations (chitchat).
    - If the user message is a greeting / thank you / small talk -> return a friendly Vietnamese reply.
    - If it's not chitchat -> return "NOT_CHITCHAT" so the host can route the message to another agent.
    """
    text = (text or "").strip()
    if not text:
        return "Dường như bạn chưa nhập gì cả. Bạn có thể hỏi mình về TOEIC nhé!" # empty input -> friendly reply

    reply = is_chitchat_reply(text)
    if reply:
        return reply # it's chit-chat, return a friendly reply
    return 1 # call agent
# =============================== End catch chit chat wrapper ==============================

tests = [
        "Xin chào bạn!",
        "Hôm nay bạn khỏe không?",
        "Complete the sentence: She ____ to school. A. goes B. gone C. went D. gone",
        "Are you sure?",
        "What's the weather like?",
        "Điền từ thích hợp A. B. C. D.",
        "Hi, could you help me? Complete the sentence: He ___ the book.",
        "Bạn có thể giúp tôi trả lời câu hỏi TOEIC không?",
        "Cảm ơn bạn nhé!",
        "Mình mệt quá, hôm nay không muốn học"
    ]
for t in tests:
    print(f"{t!r} -> {chitchat_func(t)!r}")

@tool
def classify_toeic_part(question: str) -> str:
    """
    Classifies the TOEIC question into the appropriate part based on the content.
    Returns the part number (e.g., 'Listening Part 1', 'Reading Part 5') or 'Unknown' if unclear.
    """
    question_lower = question.lower()
    if any(keyword in question_lower for keyword in ["photograph", "picture", "what do you see", "describe the picture"]):
        return "Listening Part 1"
    elif any(keyword in question_lower for keyword in ["question-response", "short conversation", "what does the man/woman say"]):
        return "Listening Part 2"
    elif any(keyword in question_lower for keyword in ["conversation", "longer conversation", "what is the main purpose"]):
        return "Listening Part 3"
    elif any(keyword in question_lower for keyword in ["talk", "lecture", "announcement", "what is the topic"]):
        return "Listening Part 4"
    elif any(keyword in question_lower for keyword in ["incomplete sentence", "choose the best word", "a. b. c. d."]):
        return "Reading Part 5"
    elif any(keyword in question_lower for keyword in ["text completion", "paragraph", "fill in the blank"]):
        return "Reading Part 6"
    elif any(keyword in question_lower for keyword in ["reading comprehension", "passage", "what is the best title"]):
        return "Reading Part 7"
    else:
        return "Unknown"

# =============================== Bot Hosting ================================
class LanguageAgentPart6(BaseAgent):
    """Agent này dùng để sử lí part6 để chọn từ thích hợp điền vào chỗ trống trong đoạn văn"""
    def _get_tools(self) -> List[BaseTool]:
        """
        Cung cấp danh sách các tools CHUYÊN BIỆT cho Reading Part 6.
        """
        # logger.info("ReadingAgent: Cung cấp tools [vocab_search, summarize]")
        return [classify_toeic_part]


    def _get_prompt(self) -> BasePromptTemplate:
        """
        Cung cấp một prompt template CỤ THỂ cho ReadingAgent Part 6.
        """
        # logger.info("ReadingAgent: Cung cấp prompt chuyên về Reading")

        return PromptTemplate.from_template(prompt_string)
    


# grammar_agent = AgentClass(
#     name="grammar_agent",
#     prompt=grammar_agent_prompt
# )
# intent = detect_intent("Hôm nay bạn thế nào?")
# response = host.handle("Hôm nay bạn thế nào?", intent)
# =============================== Bot Hosting ================================