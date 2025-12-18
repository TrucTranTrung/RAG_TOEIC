import os
from google import genai
from google.genai import types
import spacy
from dotenv import load_dotenv

# --- Cấu hình và Khởi tạo Hệ thống ---

# Tải biến môi trường (GEMINI_API_KEY)
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

try:
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    
    # Khởi tạo Gemini Client dùng chung cho tất cả các Agent
    gemini_client = genai.Client(api_key=API_KEY)
    
    # Tải mô hình Spacy cho phân tích cú pháp cục bộ
    # Đây là "Tool Grammar" cơ sở, hoạt động cục bộ mà không gọi API
    nlp = spacy.load("en_core_web_sm")
    
except (ValueError, IOError) as e:
    print(f"Lỗi khởi tạo hệ thống: {e}")
    print("Vui lòng kiểm tra API Key và đảm bảo đã chạy 'python -m spacy download en_core_web_sm'")
    exit()


# --- Dữ liệu Ví dụ Part 6 ---
PART_6_PASSAGE = """
The annual departmental retreat will be held next month. Please check the attachment for the detailed schedule.
We hope everyone _______ this important team-building event.
(A) attends
(B) attending
(C) to attend
(D) attendance
"""

# --- Hàm Phân tích Cú pháp Cục bộ (Sử dụng Spacy) ---
def analyze_grammar_spacy(sentence):
    """
    TOOL GRAMMAR CỤC BỘ: Sử dụng Spacy để phân tích cú pháp câu chứa chỗ trống.
    Kết quả này giúp xác định từ loại tiềm năng.
    """
    doc = nlp(sentence.replace("_______", "[BLANK]"))
    print("\n[Tool Cục bộ] --- Phân tích Ngữ pháp (Spacy) ---")
    for token in doc:
        # In Token, Loại từ (POS), và Mối quan hệ cú pháp (DEP)
        print(f"Token: {token.text}\t POS: {token.pos_}\t DEPR: {token.dep_}")
    
    # Logic sơ bộ: Tìm động từ chính của mệnh đề
    main_verbs = [token.text for token in doc if token.dep_ == "ROOT"]
    print(f"Động từ chính (ROOT): {main_verbs}")
    print("Gợi ý Spacy: Phân tích cho thấy chỗ trống cần được điền bởi một từ trong mệnh đề phụ.")
    print("-" * 60)
    return doc


# --- Định nghĩa các Class Agent (Sử dụng Gemini API) ---

class LanguageAgent:
    """Class cơ sở (Base Class) cho tất cả các Agent ngôn ngữ."""
    def __init__(self, system_instruction, model="gemini-2.5-flash"):
        self.system_instruction = system_instruction
        self.model = model

    def run(self, prompt):
        """Phương thức chung để gọi Gemini API với vai trò Agent cụ thể."""
        agent_name = self.__class__.__name__.replace("Agent", "")
        print(f"\n[{agent_name} Agent đang thực hiện nhiệm vụ...]")
        
        # Cấu hình cho vai trò cụ thể của Agent
        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction
        )
        
        try:
            response = gemini_client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
            return response.text
        except Exception as e:
            return f"Lỗi trong quá trình gọi API cho {agent_name} Agent: {e}"

# --- 1. Agent Giải quyết Part 6 (Kết hợp RAG/Grammar) ---
class Part6SolverAgent(LanguageAgent):
    """
    Agent chuyên giải quyết câu hỏi Part 6, hoạt động như một RAG/Grammar Tool
    để đưa ra đáp án chính xác và lời giải thích.
    """
    def __init__(self):
        system_instruction = (
            "Bạn là một chuyên gia luyện thi TOEIC (Part 6). Nhiệm vụ của bạn là phân tích ngữ cảnh, "
            "áp dụng các quy tắc ngữ pháp và chọn từ/câu thích hợp nhất để điền vào chỗ trống. "
            "Đầu ra phải tuân thủ nghiêm ngặt định dạng: **Đáp án:** [A/B/C/D] và **Giải thích:** [Lời giải chi tiết]."
        )
        super().__init__(system_instruction)

# --- 2. Grammar Check Tool Agent (Phân tích các lựa chọn) ---
class GrammarCheckAgent(LanguageAgent):
    """Agent chuyên kiểm tra và giải thích ngữ pháp của các lựa chọn."""
    def __init__(self):
        system_instruction = (
            "Bạn là chuyên gia ngữ pháp. Nhiệm vụ của bạn là phân tích các lựa chọn được cung cấp "
            "trong một câu hỏi trắc nghiệm, giải thích tại sao mỗi lựa chọn (A, B, C, D) lại đúng hoặc sai về mặt ngữ pháp."
        )
        super().__init__(system_instruction)

# --- 3. Vocab Tool Agent (Phân tích từ vựng) ---
class VocabAgent(LanguageAgent):
    """Agent chuyên phân tích từ vựng quan trọng trong đoạn văn."""
    def __init__(self):
        system_instruction = (
            "Bạn là một chuyên gia từ vựng. Nhiệm vụ của bạn là nhận một câu/đoạn văn và phân tích các từ khóa "
            "quan trọng, cung cấp định nghĩa, ví dụ và từ đồng nghĩa liên quan đến môi trường công sở."
        )
        super().__init__(system_instruction)

# --- 4. Collocation Tool Agent (Phân tích kết hợp từ) ---
class CollocationAgent(LanguageAgent):
    """Agent chuyên kiểm tra và đề xuất các kết hợp từ tự nhiên (collocations)."""
    def __init__(self):
        system_instruction = (
            "Bạn là chuyên gia về Collocation. Nhiệm vụ của bạn là phân tích cách kết hợp từ trong một câu "
            "để xác định tính tự nhiên. Đề xuất các collocations chuẩn cho các từ khóa chính."
        )
        super().__init__(system_instruction)


# --- CHẠY DEMO ĐA AGENT ---
if __name__ == "__main__":
    
    # Lấy câu cần phân tích cho Spacy
    target_sentence = PART_6_PASSAGE.split('\n')[3].strip() 
    
    print("==================================================")
    print("       HỆ THỐNG ĐA AGENT GIẢI QUYẾT PART 6        ")
    print("==================================================")

    # 1. Khởi tạo các Agent
    solver_agent = Part6SolverAgent()
    grammar_agent = GrammarCheckAgent()
    vocab_agent = VocabAgent()
    collocation_agent = CollocationAgent()

    # 2. Bước 1: Phân tích Cục bộ (Spacy)
    analyze_grammar_spacy(target_sentence)

    # 3. Bước 2: Giải quyết Câu hỏi Chính (Part6SolverAgent - RAG/Grammar Logic)
    # Đây là nơi hệ thống đưa ra đáp án cuối cùng
    solution_prompt = f"Giải quyết câu hỏi Part 6 sau: \n\n{PART_6_PASSAGE}"
    main_solution = solver_agent.run(solution_prompt)
    
    print("\n=================== ĐÁP ÁN CHÍNH (SOLVER AGENT) ===================")
    print(main_solution)
    print("==================================================================")

    # Lấy câu hoàn chỉnh để các agent khác phân tích sâu hơn
    completed_sentence = "We hope everyone attends this important team-building event."

    # 4. Bước 3: Phân tích Chuyên sâu (Các Agent Hỗ trợ)
    
    # 4a. Grammar Check Agent (Phân tích các lựa chọn và ngữ pháp câu)
    grammar_analysis_prompt = (
        f"Phân tích lý do ngữ pháp của các lựa chọn sau cho câu: '{target_sentence}'\n"
        "Lựa chọn: (A) attends, (B) attending, (C) to attend, (D) attendance."
    )
    grammar_result = grammar_agent.run(grammar_analysis_prompt)
    print("\n=================== PHÂN TÍCH NGỮ PHÁP (GRAMMAR AGENT) ===================")
    print(grammar_result)
    print("=========================================================================")

    # 4b. Vocab Agent (Phân tích từ vựng)
    vocab_analysis_prompt = f"Phân tích từ vựng quan trọng (như 'retreat', 'team-building') trong đoạn văn sau: \n\n{PART_6_PASSAGE}"
    vocab_result = vocab_agent.run(vocab_analysis_prompt)
    print("\n=================== PHÂN TÍCH TỪ VỰNG (VOCAB AGENT) ===================")
    print(vocab_result)
    print("=====================================================================")

    # 4c. Collocation Agent (Phân tích kết hợp từ)
    collocation_analysis_prompt = f"Phân tích các collocation trong câu đã hoàn chỉnh: '{completed_sentence}'"
    collocation_result = collocation_agent.run(collocation_analysis_prompt)
    print("\n=================== PHÂN TÍCH COLLOCATION (COLLOCATION AGENT) ===================")
    print(collocation_result)
    print("===============================================================================")
