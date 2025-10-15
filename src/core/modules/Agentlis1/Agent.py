import re
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from prompts import REACT_TEMPLATE, ANALYSIS_PROMPT_TEXT
from utils import configure_environment, encode_image_to_base64

configure_environment()

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

class TOEICPart1Agent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
        @tool
        def analyze_image_and_transcript_tool(tool_input: str) -> str:
            """ Analyzes the image and transcript to provide the final answer for a TOEIC Part 1 question. Use this tool LAST, only when:
        1. A valid 'image_path' exists.
        2. The 'transcript' has been successfully validated by 'validate_transcript'.
        Input must be a JSON string or dict with 'image_path' and 'transcript' keys.
        Returns the answer and explanation (in Vietnamese). """
            return self._analyze_logic_handler(tool_input)
        tools = [validate_transcript, analyze_image_and_transcript_tool]
        
        prompt = PromptTemplate.from_template(REACT_TEMPLATE)
        agent = create_react_agent(self.llm, tools, prompt)
        
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=False,
            handle_parsing_errors=True
        )
        print("TOEIC Part 1 Agent initialized and ready (Final Version).")

   
    def _analyze_logic_handler(self, tool_input: str) -> str:
        print(f"\n[Tool Called]: analyze_image_and_transcript")
        try:
            if isinstance(tool_input, str):
                data = json.loads(tool_input)
            elif isinstance(tool_input, dict):
                data = tool_input
            else:
                raise ValueError("Tool input must be JSON string or dict.")

            image_path = data.get("image_path")
            transcript = data.get("transcript")
            
            if not image_path or not transcript:
                 raise ValueError("Missing 'image_path' or 'transcript'.")
            
            image_b64 = encode_image_to_base64(image_path)
            final_prompt = ANALYSIS_PROMPT_TEXT + f"\nTranscript:\n{transcript}"
            message = HumanMessage(content=[
                {"type": "text", "text": final_prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
            ])

            response = self.llm.invoke([message])
            return response.content.strip()
        except Exception as e:
            return f"Đã có lỗi xảy ra trong quá trình phân tích: {e}"

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
  
if __name__ == "__main__":
    toeic_agent = TOEICPart1Agent()

    image1 = "mo_ta_tranh1.png"
    sample_transcript_ok = "(A) The man is using a screwdriver. (B) The man is hammering something. (C) The man is making the frame by hand. (D) The man is wearing protective glasses."
    sample_transcript_wrong = "(A) The man is using a screwdriver. (B) The man is finding something. (C) The man is making the cookies by hand. (D) The man is wearing protective glasses."
    sample_transcript_invalid = "(A) This is option one. (B) This is option two."
    
    # --- Scenario 1: Complete and valid data ---
    print("\n" + "="*80)
    print("--- CHẠY KỊCH BẢN 1: ĐẦY ĐỦ DỮ LIỆU HỢP LỆ ---")
    toeic_agent.run(image_path=image1, transcript=sample_transcript_ok)

    # --- Scenario 2: Missing transcript (Sẽ trả về lỗi validate) ---
    print("\n" + "="*80)
    print("--- CHẠY KỊCH BẢN 2: THIẾU TRANSCRIPT ---")
    toeic_agent.run(image_path=image1) 

    # --- Scenario 3: Transcript with insufficient options ---
    print("\n" + "="*80)
    print("--- CHẠY KỊCH BẢN 3: TRANSCRIPT KHÔNG HỢP LỆ (LỖI VALIDATE) ---")
    toeic_agent.run(image_path=image1, transcript=sample_transcript_invalid)