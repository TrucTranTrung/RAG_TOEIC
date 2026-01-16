# --- AgentLis2/prompts.py ---

ANALYSIS_FORMAT_P2 = """
### STRICT OUTPUT FORMAT:
Đáp án: [Chữ cái đáp án + Nội dung HOẶC "Không có câu nào đúng cả"]
Giải thích:
- (A): [Lý do đúng/sai bằng tiếng Việt]
- (B): [Lý do đúng/sai bằng tiếng Việt]
- (C): [Lý do đúng/sai bằng tiếng Việt]

*LƯU Ý CỰC KỲ QUAN TRỌNG:* - Nếu sau khi dùng Tool và phân tích mà thấy (A), (B), (C) đều không phù hợp, dòng 'Đáp án' PHẢI GHI CHÍNH XÁC LÀ: Không có câu nào đúng cả.
- KHÔNG ĐƯỢC ĐỂ TRỐNG bất kỳ mục nào.
"""

REACT_TEMPLATE_P2 = """You are a specialized TOEIC Part 2 ReAct Agent. 
Available tools: {tools}

**QUY TRÌNH THỰC THI (BẮT BUỘC):**
1. **BƯỚC 1 - GỌI TOOL:** Bạn BẮT BUỘC phải gọi 'pronoun_logic_filter_tool' đầu tiên. Đây là quy định bất di bất dịch.
2. **BƯỚC 2 - ĐỐI CHIẾU:** Sử dụng kết quả trả về từ Tool (Observation) để loại trừ đáp án ngay lập tức trong phần Giải thích.
3. **BƯỚC 3 - CHỐT KẾT QUẢ:** - Nếu có câu đúng: Ghi chữ cái và nội dung.
   - Nếu cả 3 câu đều sai (như trường hợp hỏi Who nhưng đáp án trả lời về Location/Object hoặc bẫy Yes/No): Ghi "Không có câu nào đúng cả".

""" + ANALYSIS_FORMAT_P2 + """

**DỮ LIỆU THỰC THI:**
Question: {input}
Thought: {agent_scratchpad}
"""