import os
import base64
import json
import re
from io import BytesIO
from PIL import Image, ImageDraw
import warnings

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent

warnings.filterwarnings("ignore")

os.environ["GOOGLE_API_KEY"] = "AIzaSyCdpAhCGYziOUfaIsjuQ9YhWvqzc2AjbXM"

# --- ✅ GLOBAL LLM INSTANCES (khởi tạo một lần duy nhất) ---
_llm_for_agent = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
_llm_for_analysis = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# ✅ Tool 1: validate_transcript
@tool
def validate_transcript(transcript: str) -> str:
    """ Validates that the transcript has four options: (A), (B), (C), and (D). 
    This is a critical first step and must be used before 'analyze_image_and_transcript'. 
    Returns '1' if valid, otherwise returns a Vietnamese error message. """

    print(f"\n[Tool Called]: validate_transcript")
    if transcript is None:
        transcript = ""
    options = set(re.findall(r'\([A-D]\)', transcript.upper()))
    if len(options) == 4:
        return "1"
    else:
        return "Bạn cung cấp không đủ đáp án, vui lòng nhập lại"


# ✅ Tool 2: analyze_image_and_transcript
@tool
def analyze_image_and_transcript(tool_input: str) -> str:
    """ Analyzes the image and transcript to provide the final answer for a TOEIC Part 1 question. Use this tool LAST, only when:
      1. A valid 'image_path' exists.
      2. The 'transcript' has been successfully validated by 'validate_transcript'. 
      Input must be a JSON string or dict with 'image_path' and 'transcript' keys. 
      Returns the answer and explanation (in Vietnamese). """
    print(f"\n[Tool Called]: analyze_image_and_transcript")

    prompt_text = """
    **Task: Analyze the image and options. Select ONLY the option that is 100% true based on the image.**
    **CORE RULE: DO NOT INFER OR ASSUME.** If any detail in an option is not explicitly visible, the option is WRONG.
    
    **REQUIRED OUTPUT FORMAT (in Vietnamese - Short version):**
    Đáp án: [Full correct option]
    Giải thích: [Explain why the chosen answer is 100% correct and why others are wrong, based on visual evidence. **Keep this explanation very brief and concise.**]
    ---
    **SPECIAL CASES:**
    1. Add **NO** other lines besides "Đáp án:" and "Giải thích:".
    2. **If NO option is 100% true based on the CORE RULE, return the EXACT string: "Không có câu nào chính xác cả"** (Do not start with "Đáp án:").
    
    Now, analyze the following data:
    """

    try:
        # --- Parse input an toàn ---
        if isinstance(tool_input, str):
            try:
                data = json.loads(tool_input)
            except json.JSONDecodeError:
                raise ValueError("Tool input is not a valid JSON string.")
        elif isinstance(tool_input, dict):
            data = tool_input
        else:
            raise ValueError("Tool input must be JSON string or dict.")

        image_path = data.get("image_path")
        transcript = data.get("transcript")

        if not image_path or not transcript:
            raise ValueError("Missing 'image_path' or 'transcript'.")

        # --- Encode ảnh sang base64 ---
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((768, 768), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # --- Chuẩn bị message multimodal ---
        final_prompt = prompt_text + f"\nTranscript:\n{transcript}"
        message = HumanMessage(content=[
            {"type": "text", "text": final_prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
        ])

        # ✅ Dùng global LLM riêng cho tool (tối ưu đúng chuẩn)
        global _llm_for_analysis
        response = _llm_for_analysis.invoke([message])
        return response.content.strip()

    except Exception as e:
        return f"Đã có lỗi xảy ra trong quá trình phân tích: {e}"


# ✅ Agent class — dùng LLM riêng cho reasoning text
class TOEICPart1Agent:
    def __init__(self):
        global _llm_for_agent
        tools = [validate_transcript, analyze_image_and_transcript]

        react_template = """You are a specialized TOEIC Part 1 Agent. Your goal is to solve the question using the available tools.
        
        Available tools: {tools}
        
        Use the following format:
        Question: the input question you must answer
        Thought: analyze the current state and determine the next action
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input for the action
        Observation: the result of the action
        ... (repeat Thought/Action/Action Input/Observation)
        Thought: I have the final answer
        Final Answer: the final answer or error message
        
        Begin!
        Question: {input}
        Thought:{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(react_template)
        agent = create_react_agent(_llm_for_agent, tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
        print(" TOEIC Part 1 Agent initialized and ready.")

    def run(self, image_path=None, transcript=None):
        transcript_for_agent = transcript if transcript is not None else ""
        agent_input = (
            "Solve this TOEIC Part 1 question. "
            "1. You **MUST** use 'validate_transcript'. "
            "2. If validation returns an error, your Final Answer MUST BE that message. "
            "3. If validation is '1', use 'analyze_image_and_transcript'. "
            f"- Image Path: '{image_path if image_path else 'not provided'}'\n"
            f"- Transcript: '{transcript_for_agent}'"
        )

        print("\n Bắt đầu xử lý...")
        try:
            result = self.agent_executor.invoke({"input": agent_input})
            print("\n--- KẾT QUẢ CUỐI CÙNG ---")
            print(result["output"].strip())
        except Exception as e:
            print(f"Đã xảy ra lỗi nghiêm trọng trong quá trình xử lý của agent: {e}")

# ---- 4️⃣ Usage and full scenario testing ----
if __name__ == "__main__":
    toeic_agent = TOEICPart1Agent()

    image1 = "mo_ta_tranh1.png"
    if not os.path.exists(image1):
        print(f"Tạo file ảnh mẫu '{image1}'...")
        try:
            img = Image.new("RGB", (600, 400), color="white")
            draw = ImageDraw.Draw(img)
            draw.rectangle((50, 50, 550, 350), outline="blue", width=10)
            img.save(image1)
            print(f"Đã tạo ảnh mẫu '{image1}' thành công.")
        except Exception as e:
            print(f"Không thể tạo ảnh mẫu: {e}.")

    sample_transcript_ok = "(A) The man is using a screwdriver. (B) The man is hammering something. (C) The man is making the frame by hand. (D) The man is wearing protective glasses."
    sample_transcript_invalid = "(A) This is option one. (B) This is option two."
    sample_transcript_wrong = "(A) The man is swimming. (B) The man is baking cookies. (C) The man is driving a car. (D) The man is holding a fish."
    
    # --- Scenario 1: Complete and valid data ---
    print("\n" + "="*80)
    print("--- CHẠY KỊCH BẢN 1: ĐẦY ĐỦ DỮ LIỆU HỢP LỆ (ĐÃ RÚT GỌN ĐẦU RA) ---")
    toeic_agent.run(image_path=image1, transcript=sample_transcript_ok)

    # --- Scenario 2: Missing transcript (Sẽ trả về lỗi validate) ---
    print("\n" + "="*80)
    print("--- CHẠY KỊCH BẢN 2: THIẾU TRANSCRIPT ---")
    toeic_agent.run(image_path=image1) 

    # --- Scenario 4: Transcript with insufficient options ---
    print("\n" + "="*80)
    print("--- CHẠY KỊCH BẢN 4: TRANSCRIPT KHÔNG HỢP LỆ (LỖI VALIDATE) ---")
    toeic_agent.run(image_path=image1, transcript=sample_transcript_invalid)