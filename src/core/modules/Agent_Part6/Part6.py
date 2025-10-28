from ast import List
import os
import spacy
from dotenv import load_dotenv
from ..Agent_Base import BaseAgent
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.llms.base import BaseLLM
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.tools import BaseTool, tool
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CẤU HÌNH VÀ KHỞI TẠO HỆ THỐNG ---

# Tải biến môi trường (GEMINI_API_KEY)
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

try:
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    
    # Khởi tạo Gemini Client dùng chung
    gemini_client = genai.Client(api_key=API_KEY)
    
    # Tải mô hình Spacy cho phân tích cú pháp cục bộ
    nlp = spacy.load("en_core_web_sm")
    
except (ValueError, IOError) as e:
    print(f"Lỗi khởi tạo hệ thống: {e}")
    print("Vui lòng kiểm tra API Key và đảm bảo đã chạy 'python -m spacy download en_core_web_sm'")
    exit()

# --- DỮ LIỆU VÍ DỤ PART 6 ---
PART_6_PASSAGE = """
The annual departmental retreat will be held next month. Please check the attachment for the detailed schedule.
We hope everyone _______ this important team-building event.
(A) attends
(B) attending
(C) to attend
(D) attendance
"""
TARGET_SENTENCE = PART_6_PASSAGE.split('\n')[3].strip() 

# --- DANH SACH TOOL ---

@tool
def analyze_grammar_spacy(sentence):
    """Hàm này sử dụng câu chứa khoản trắng của đoạn văn để phân tích."""
    doc = nlp(sentence.replace("_____", "[BLANK]"))
    # print("\n[Tool Cục bộ] --- Phân tích Ngữ pháp (Spacy) ---")
    # for token in doc:
    #     print(f"Token: {token.text}\t POS: {token.pos_}\t DEPR: {token.dep_}")
    # print("-" * 60)
    return doc

@tool
def vocab_search(query: str) -> str:
    """
    Sử dụng Google Search để tìm kiếm định nghĩa, ví dụ, và từ đồng nghĩa 
    của một từ khóa trong ngữ cảnh công sở.
    
    Args:
        query: Từ hoặc cụm từ cần tra cứu (ví dụ: 'retreat business definition').
        
    Returns:
        Kết quả tìm kiếm web.
    """
    # Lệnh gọi tool Google Search thực tế được mô hình thực hiện.
    # Trong code này, ta chỉ cần định nghĩa function signature và mô tả 
    # để mô hình Gemini biết nó có thể gọi Google Search.
    
    # KHI BẠN CHẠY TRÊN MÁY: Gemini SDK sẽ tự động xử lý và
    # thực hiện tìm kiếm này nếu mô hình quyết định gọi nó.
    
    # Để đơn giản hóa, ta sẽ truyền trực tiếp tool 'google_search' vào Agent.
    # Tuy nhiên, nếu bạn muốn một hàm riêng (như yêu cầu), ta cần một bước trung gian:
    
    # Hàm này CHỈ DÙNG để định nghĩa Signature cho Tool Calling.
    # Trong luồng Part 6, ta sẽ dùng cách 1 (tức là gộp và yêu cầu tìm kiếm trực tiếp).
    
    return f"Đã chuẩn bị tìm kiếm định nghĩa cho: {query}"


# --- ĐỊNH NGHĨA CÁC CLASS AGENT (SỬ DỤNG GEMINI API) ---

class LanguageAgent(BaseAgent):
    """Agent này dùng để sử lí part6 để chọn từ thích hợp điền vào chỗ trống trong đoạn văn"""
    def _get_tools(self) -> List[BaseTool]:
        """
        [TRIỂN KHAI TỪ BASEAGENT]
        Cung cấp danh sách các tools CHUYÊN BIỆT cho Reading.
        """
        logger.info(
            "ReadingAgent: Cung cấp tools [check_grammar_tool, find_synonym_tool]")
        return [analyze_grammar_spacy,vocab_search]


    def _get_prompt(self) -> BasePromptTemplate:
        """
        [TRIỂN KHAI TỪ BASEAGENT]
        Cung cấp một prompt template (hướng dẫn) CỤ THỂ cho ReadingAgent.
        """

        prompt_string = """
        Bạn là một trợ lý AI chuyên gia về đọc hiểu và ngữ pháp Tiếng Anh (TOEIC Reading).
        Nhiệm vụ của bạn là trả lời các câu hỏi liên quan đến ngữ pháp, từ vựng, và đọc hiểu.
        
        Bạn CÓ THỂ sử dụng các tools sau:
        {tools}
        (Tên tools: {tool_names})

        HÃY SỬ DỤNG FORMAT SAU ĐỂ TRẢ LỜI:

        Question: Câu hỏi đầu vào bạn cần trả lời.
        Thought: Suy nghĩ của bạn về việc cần làm.
        Action: Hành động bạn sẽ thực hiện. PHẢI LÀ MỘT TRONG CÁC TOOLS SAU: [{tool_names}].
        Action Input: Đầu vào cho hành động đó.
        Observation: Kết quả của hành động (phần này do hệ thống cung cấp).

        ... (Chuỗi Thought/Action/Action Input/Observation có thể lặp lại) ...

        Thought: Bây giờ tôi đã có đủ thông tin để trả lời câu hỏi của người dùng.
        Câu trả lời cuối cùng của bạn cho người dùng (bằng Tiếng Việt).

        BẮT ĐẦU!

        Question: {input}

        {agent_scratchpad}
        
        """
        logger.info("ReadingAgent: Cung cấp prompt chuyên về Reading",
                    PromptTemplate.from_template(prompt_string))

        return PromptTemplate.from_template(prompt_string)


# --- 1. Agent Giải quyết Part 6 TỔNG HỢP ---
class Part6SolverAgent(LanguageAgent):
    """
    Agent tổng hợp giải quyết Part 6, có khả năng phân tích Ngữ pháp, Từ vựng, Collocation.
    """
    def __init__(self):
        system_instruction = (
            "Bạn là một chuyên gia luyện thi TOEIC (Part 6) toàn diện. Nhiệm vụ của bạn là: "
            "1. Phân tích ngữ cảnh và chọn đáp án chính xác. "
            "2. Giải thích chi tiết về đáp án chính (ngữ pháp và ngữ nghĩa). "
            "3. Phân tích tại sao các lựa chọn sai lại sai. "
            "4. Cung cấp phân tích từ vựng và collocation cho câu đã hoàn chỉnh. "
            "Đầu ra phải tuân thủ nghiêm ngặt định dạng 3 phần: **Đáp án**, **Giải thích Ngữ Pháp/Từ Loại**, **Phân Tích Chuyên Sâu**."
        )
        super().__init__(system_instruction)


# --- CHẠY DEMO ĐA TOOL/AGENT (TÍCH HỢP) ---
if __name__ == "__main__":
    """Hàm chạy chính (bất đồng bộ) để test agent."""

    # --- Khởi tạo LLM thật (Gemini 2.5 Flash) ---
    logger.info("--- Khởi tạo Gemini 2.5 Flash LLM ---")

    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        print("="*50)
        print("LỖI: Vui lòng đặt biến môi trường GOOGLE_API_KEY để chạy ví dụ này.")
        print("Lệnh (Terminal/PowerShell): set GOOGLE_API_KEY=YOUR_API_KEY_HERE")
        print("="*50)

    try:
        # Sử dụng model "gemini-2.5-flash"
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=google_api_key,
            convert_system_message_to_human=True
        )
    except Exception as e:
        logger.error(f"Không thể khởi tạo Gemini LLM: {e}")

   
    # Chỉ cần truyền LLM thật vào, BaseAgent sẽ tự động lo phần còn lại

    Part6_agent = LanguageAgent(llm=llm)

    print("\n--- Bắt đầu chạy ReadingAgent (Async) ---")

    test_input = "Câu 'He go to school' có đúng ngữ pháp không?"
    logger.info(f"Test input: {test_input}")

    try:
        # Sử dụng .ainvoke() (bất đồng bộ) vì đây là API call thật
        result = Part6_agent.ainvoke({
            "input": test_input
        })

        print("\n--- Kết quả từ ReadingAgent (Gemini) ---")
        print(f"Câu trả lời cuối cùng: {result.get('output')}")

    except Exception as e:
        logger.error(f"Lỗi khi chạy agent: {e}")


