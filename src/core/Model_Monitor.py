import gradio as gr
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.tools import tool
# from langfuse.callback import CallbackHandler
from utils import is_chitchat_reply


# ================================ Tool wrapper  =================================
@tool
def chitchat_tool(text: str) -> str:
    """
    Tool for the host bot: detects casual or social conversations (chitchat).
    - If the user message is a greeting / thank you / small talk -> return a friendly Vietnamese reply.
    - If it's not chitchat -> return "NOT_CHITCHAT" so the host can route the message to another agent.
    """
    text = (text or "").strip()
    if not text:
        return "NOT_CHITCHAT"

    reply = is_chitchat_reply(text)
    if reply:
        return reply
    return "NOT_CHITCHAT"
# =============================== End Tools wrapper ==============================

# tests = [
#         "Xin chào bạn!",
#         "Hôm nay bạn khỏe không?",
#         "Complete the sentence: She ____ to school. A. goes B. gone C. went D. gone",
#         "Are you sure?",
#         "What's the weather like?",
#         "Điền từ thích hợp A. B. C. D.",
#         "Hi, could you help me? Complete the sentence: He ___ the book.",
#         "Cảm ơn bạn nhé!",
#         "Mình mệt quá, hôm nay không muốn học"
#     ]
# for t in tests:
#     print(f"{t!r} -> {chitchat_tool(t)!r}")

# =============================== Bot Hosting ================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    remaining_steps: int
    
class AgentClass():
    def __init__(self, name,prompt):
        self.name = name
        self.llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.prompt=prompt
        self.agent = create_react_agent(
            model=self.llm,
            tools=[self.grammar_analysis_tool,self.vision_analysis_tool],
            prompt=self.prompt,
            name=self.name,
            state_schema=AgentState
        )

    def handle(self, message, intent):
        # intent được detect trước đó (VD: "chitchat", "fashion", ...)
        if intent in self.agents:
            return self.agents[intent].run(message)
        elif intent in self.tools:
            return self.tools[intent].run(message)
        else:
            return "Xin lỗi, tôi chưa hiểu câu hỏi của bạn."

    @tool
    def Embbeding_Router(question: str) -> str:
      """
      Analyzes a TOEIC grammar or vocabulary question and provides a detailed explanation for the correct answer.
      Input should be the full question with all options (A, B, C, D).
      """
      # Ta có thể thêm logic phức tạp hơn ở đây, nhưng chỉ cần tool tồn tại là đủ
      # để agent tuân theo luồng ReAct. LLM sẽ tự suy luận để đưa ra câu trả lời.
      return "The user's question has been passed for analysis. The agent should now provide the final answer and explanation."
    
    @tool
    def Clarify_tool(self, user_utterance: str) -> str:
        """
        Purpose: Provide a concise explanation when user asks "Are you sure?" or "Why?"
        Input: user_utterance (string)
        Output: Explanation about why the original question is considered out-of-domain,
                and instructions for how to rephrase into a TOEIC-related question.
        """
        # Giả sử self.state lưu:
        # self.last_user_question, self.last_tool_decision (e.g., "out_of_domain"), self.last_decision_reason
        if hasattr(self, "last_tool_decision") and self.last_tool_decision == "out_of_domain":
            reason = getattr(self, "last_decision_reason", "This question seems outside TOEIC domain.")
            return f"{reason} Nếu bạn muốn, hãy gửi câu hỏi liên quan đến TOEIC (ví dụ: incomplete sentence with options A/B/C/D)."
        else:
            # Nếu không có ngữ cảnh, đưa hướng dẫn chung
            return "Mình chưa có thông tin trước đó — bạn có thể dán lại câu hỏi TOEIC (hoặc hỏi 'Why do you think it's out of domain?')."

    # LLM Router

grammar_agent_prompt = """You are a master of TOEIC grammar and vocabulary. Your task is to analyze an incomplete sentence and choose the best option among A, B, C, D.
INSTRUCTIONS: First, identify the grammatical rule or vocabulary concept being tested. Second, analyze each option and explain why it is correct or incorrect based on the rule.
Finally, state the correct answer and provide a clear, concise explanation. Respond ONLY with your analysis and the final answer. DO NOT add any conversational text."""

# grammar_agent = AgentClass(
#     name="grammar_agent",
#     prompt=grammar_agent_prompt
# )
# intent = detect_intent("Hôm nay bạn thế nào?")
# response = host.handle("Hôm nay bạn thế nào?", intent)
# =============================== Bot Hosting ================================