import os
from google import genai
from google.genai import types
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
    
except ValueError as e:
    print(f"Lỗi khởi tạo hệ thống: {e}")
    print("Vui lòng kiểm tra tệp .env và API Key.")
    exit()


# --- Dữ liệu Ví dụ Part 7 (Đoạn văn Kép) ---
PART_7_PASSAGE = """
Document 1: Announcement Email
To: All Sales Staff (sales@corp.com)
From: Alex Chen, Business Manager (alex.chen@corp.com)
Date: October 15
Subject: Q4 Commission Policy Update

Dear Sales Team,
Please be informed that, effective November 1, we will be adjusting the commission structure for all contracts exceeding $50,000. The base commission rate will increase from 5% to 7%. The purpose of this change is to encourage the team to focus on larger, higher-value projects for the company. Please refer to the attached file for the detailed breakdown of the new structure.

Document 2: Excerpt from Employee Handbook
Section 4: Salary and Compensation Regulations
All changes to compensation policies must be announced at least two weeks prior to the effective date. Any inquiries regarding compensation should be directed to Human Resources (HR@corp.com) instead of department managers.
"""

# --- Class Base (Cơ sở) ---
class LanguageAgent:
    """Class cơ sở cho tất cả các Agent ngôn ngữ sử dụng Gemini API."""
    def __init__(self, system_instruction, model="gemini-2.5-flash"):
        self.system_instruction = system_instruction
        self.model = model

    def run(self, prompt):
        """Phương thức chung để gọi Gemini API với vai trò Agent cụ thể."""
        agent_name = self.__class__.__name__.replace("Agent", "")
        print(f"\n[{agent_name} Agent đang thực hiện nhiệm vụ...]")
        
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


# --- 1. Summarize / Extract Information Tool Agent ---
class InformationExtractorAgent(LanguageAgent):
    """
    Agent chuyên tóm tắt và trích xuất thông tin mục tiêu từ đoạn văn Part 7.
    Chức năng này giải quyết nhu cầu 'Summarize *tools' và 'Extract information'.
    """
    def __init__(self):
        system_instruction = (
            "Bạn là một chuyên gia phân tích và trích xuất thông tin cho các bài đọc hiểu (Part 7). "
            "Nhiệm vụ của bạn là nhận một hoặc nhiều tài liệu, sau đó tóm tắt các điểm chính và "
            "trích xuất các dữ liệu cụ thể (ví dụ: ngày, tên người, tỷ lệ, mục đích) theo định dạng bullet point rõ ràng."
        )
        super().__init__(system_instruction)

# --- 2. Question Tool Agent ---
class QuestionGeneratorAgent(LanguageAgent):
    """
    Agent chuyên tạo và trả lời các câu hỏi đọc hiểu phức tạp.
    Chức năng này giải quyết nhu cầu 'Extract information about question tool' (Trả lời) và tạo câu hỏi.
    """
    def __init__(self):
        system_instruction = (
            "Bạn là một chuyên gia tạo câu hỏi đọc hiểu. Nhiệm vụ của bạn là nhận các tài liệu (Part 7) "
            "và tạo ra 3 loại câu hỏi TOEIC: (1) Câu hỏi chi tiết (Fact/Detail), (2) Câu hỏi Suy luận (Inference), "
            "và (3) Câu hỏi Văn bản Kép (kết nối thông tin giữa các tài liệu). "
            "Sau khi tạo câu hỏi, hãy cung cấp đáp án chính xác cho từng câu."
        )
        super().__init__(system_instruction)


# --- DEMO SỬ DỤNG HỆ THỐNG ĐA AGENT ---
if __name__ == "__main__":
    
    print("==================================================")
    print("  HỆ THỐNG AGENT PHÂN TÍCH ĐỌC HIỂU (PART 7)      ")
    print("==================================================")

    # Khởi tạo các Agent
    extractor_agent = InformationExtractorAgent()
    question_agent = QuestionGeneratorAgent()

    # --- 1. Chạy Information Extractor Agent (Tóm tắt & Trích xuất) ---
    extract_prompt = (
        "Phân tích các tài liệu sau. Hãy tóm tắt ngắn gọn mục đích của Email và trích xuất các thông tin sau: "
        "1. Ngày có hiệu lực của chính sách mới. 2. Tỷ lệ hoa hồng mới. 3. Ai là người nhận thắc mắc về lương thưởng.\n\n"
        f"Tài liệu:\n{PART_7_PASSAGE}"
    )
    extraction_result = extractor_agent.run(extract_prompt)
    
    print("\n================== KẾT QUẢ TRÍCH XUẤT THÔNG TIN ==================")
    print(extraction_result)
    print("==================================================================")

    # --- 2. Chạy Question Generator Agent (Tạo Câu hỏi) ---
    question_prompt = (
        "Dựa trên các tài liệu đã cho (Email và Sổ tay), hãy tạo 3 câu hỏi TOEIC Part 7: "
        "1. Một câu hỏi chi tiết về Tài liệu 1. 2. Một câu hỏi suy luận về Tài liệu 1. 3. Một câu hỏi kết nối Tài liệu 1 và Tài liệu 2. "
        "Cung cấp đáp án (A/B/C/D) cho mỗi câu hỏi.\n\n"
        f"Tài liệu:\n{PART_7_PASSAGE}"
    )
    question_result = question_agent.run(question_prompt)

    print("\n=================== KẾT QUẢ TẠO CÂU HỎI VÀ ĐÁP ÁN ===================")
    print(question_result)
    print("=====================================================================")
