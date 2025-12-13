from random import sample
from typing import List, Dict, Optional, Tuple
import os
import spacy
from dotenv import load_dotenv
from ..Agent_Base import BaseAgent
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.tools import BaseTool, tool
from langchain_core.prompts import PromptTemplate, BasePromptTemplate
import logging
import asyncio
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import re
from typing import List, Dict

_BLANK_RE = re.compile(r'_{2,}|\[BLANK\]|\<BLANK\>', flags=re.I)
_HEADER_RE = re.compile(r'^\s*(To:|From:|Subject:|Date:)\b', flags=re.I)
_BLOCK_NUM_LINE = re.compile(r'^\s*(\d{1,3})\.\s*$', flags=re.M)
_OPTION_LINE = re.compile(r'^\s*\(?[A-Da-d]\)?[\.\)]\s+.*')  
_GARBAGE_TOKEN_RE = re.compile(r'\(\(+\d+\)+\)|Choices:?', flags=re.I)

def _split_lines_keep(text: str) -> List[str]:
    return [ln for ln in text.replace('\r\n','\n').split('\n')]

def _is_header_line(line: str) -> bool:
    return bool(_HEADER_RE.match(line))

def _is_choice_block_start(line: str) -> bool:
    return bool(re.match(r'^\s*\d{1,3}\.\s*$', line))

def _clean_choice_lines(lines: List[str]) -> List[str]:
    cleaned = []
    seen = set()
    for ln in lines:
        if not ln or ln.strip()=="":
            continue
        ln2 = _GARBAGE_TOKEN_RE.sub('', ln).strip()
        ln2 = re.sub(r'^\s*\d{1,3}\.\s*', '', ln2).strip()
        if not ln2:
            continue
        if ln2 not in seen:
            cleaned.append(ln2)
            seen.add(ln2)
    return cleaned

def _collect_choice_blocks(lines: List[str], start_idx: int) -> Dict[str, List[str]]:
    blocks = {}
    i = start_idx
    n = len(lines)
    current_num = None
    current_buf = []
    while i < n:
        ln = lines[i].rstrip()
        m = _BLOCK_NUM_LINE.match(ln)
        if m:
            if current_num is not None:
                blocks[current_num] = _clean_choice_lines(current_buf)
            current_num = m.group(1)
            current_buf = []
        else:
            current_buf.append(ln)
        i += 1
    if current_num is not None:
        blocks[current_num] = _clean_choice_lines(current_buf)
    return blocks

def clean_and_extract_passage_simple(passage: str) -> str:
    """
    Simpler & robust: if a line has a blank, return that line and any immediate following
    option lines (A/B/C/D). If no inline options present, fallback to parsing numeric choice blocks.
    """
    if not passage:
        return ""

    lines = _split_lines_keep(passage)
    # find first numeric choice block if exists
    first_choice_idx = None
    for idx, ln in enumerate(lines):
        if _is_choice_block_start(ln):
            first_choice_idx = idx
            break

    # header_lines: keep leading To/From/Subject/Date lines
    header_lines = []
    i = 0
    while i < len(lines) and _is_header_line(lines[i]):
        header_lines.append(lines[i].rstrip())
        i += 1

    # split body and choice lines
    if first_choice_idx is not None:
        body_lines = lines[:first_choice_idx]
        choice_lines = lines[first_choice_idx:]
    else:
        body_lines = lines
        choice_lines = []

    out_parts = []
    if header_lines:
        out_parts.extend([ln.rstrip() for ln in header_lines])
        out_parts.append("")

    # scan body_lines: for each line that contains a blank, capture it and any following option lines
    used_body_indices = set()
    for idx, ln in enumerate(body_lines):
        if _BLANK_RE.search(ln):
            # capture this line
            out_parts.append(ln.strip())
            used_body_indices.add(idx)
            # capture immediate following option lines (A/B/C/D), stop on first non-option or blank line
            j = idx + 1
            found_opt = False
            while j < len(body_lines):
                next_ln = body_lines[j].strip()
                if _OPTION_LINE.match(next_ln):
                    out_parts.append(next_ln)
                    used_body_indices.add(j)
                    found_opt = True
                    j += 1
                    continue
                # also accept lines like "A) attends" without parentheses, handled by regex above
                # stop if next line is empty or another header/metadata line
                break
            out_parts.append("")  # blank line after each question block

    # If no inline options were found for any question, fall back to parsing numeric choice blocks
    if not any(_OPTION_LINE.match(l.strip()) for l in body_lines):
        if choice_lines:
            choices_map = _collect_choice_blocks(choice_lines, 0)
            # append cleaned choice blocks
            if choices_map:
                for num in sorted(choices_map, key=lambda x: int(x)):
                    out_parts.append(f"{num}.")
                    for o in choices_map[num]:
                        out_parts.append(o)
        else:
            # nothing to append
            pass
    else:
        # if some inline options were captured already, we keep them and do NOT duplicate numeric blocks
        pass
    return "\n".join(out_parts).strip()


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
    print("hehe")
    return f"Đã chuẩn bị tìm kiếm định nghĩa cho: {query}"


# --- ĐỊNH NGHĨA CÁC CLASS AGENT (SỬ DỤNG GEMINI API) ---
class LanguageAgentPart6(BaseAgent):
    """Agent này dùng để sử lí part6 để chọn từ thích hợp điền vào chỗ trống trong đoạn văn"""
    def _get_tools(self) -> List[BaseTool]:
        """
        Cung cấp danh sách các tools CHUYÊN BIỆT cho Reading Part 6.
        """
        logger.info("ReadingAgent: Cung cấp tools [check_grammar_tool, find_synonym_tool]")
        return [vocab_search]


    def _get_prompt(self) -> BasePromptTemplate:
        """
        Cung cấp một prompt template (hướng dẫn) CỤ THỂ cho ReadingAgent Part 6.
        """
        prompt_string = """
        Bạn là một trợ lý AI chuyên gia về Đọc hiểu và Ngữ pháp Tiếng Anh (TOEIC Reading).

        NHIỆM VỤ:
        1. Với mỗi câu hỏi tiếng Anh được cung cấp, hãy:
        - Chọn đáp án đúng (A, B, C hoặc D).
        - Viết **giải thích đầy đủ nhưng ngắn gọn bằng tiếng Việt** cho lý do tại sao đáp án đó đúng (2–4 câu, tối đa 80 từ).
        - Viết **lý do ngắn gọn** cho từng đáp án sai (mỗi đáp án 1 dòng, tối đa 20 từ, tập trung vào lỗi ngữ pháp hoặc ngữ nghĩa).
        2. Nếu có nhiều câu hỏi, hãy trả lời lần lượt theo thứ tự.
        3. Tất cả phần trả lời và giải thích PHẢI bằng tiếng Việt, dễ hiểu, ngắn gọn.
        4. Nếu cần, bạn có thể sử dụng các công cụ sau:
        {tools}
        (Tên công cụ: {tool_names})

        BẮT BUỘC TUÂN THEO ĐỊNH DẠNG DƯỚI ĐÂY:

        Question: <Câu hỏi và lựa chọn>
        Thought: <suy nghĩ ngắn về hướng giải>
        Action: <tên công cụ hoặc "none">
        Action Input: <input cho công cụ hoặc "N/A">
        Observation: <kết quả công cụ (do hệ thống cung cấp)>

        Final Answer:
        **Đáp án đúng:** (X)
        **Giải thích:** [Giải thích bằng tiếng Việt, giải thích đầy đủ chi tiết vì sao chọn đáp án đó đúng, dựa vào đâu trong câu hỏi để xác định]
        **Lý do các đáp án sai:**
        (A) ...
        (B) ...
        (C) ...
        (D) ...

        --- (Lặp lại cho từng câu hỏi) ---

        Cuối cùng, tổng hợp lại ngắn gọn:
        Final Summary:
        <Liệt kê tất cả số câu và đáp án đúng, ví dụ:>
        131: (A) — động từ chia đúng thì với chủ ngữ.
        132: (C) — giới từ phù hợp ngữ cảnh.

        BẮT ĐẦU!

        Question: {input}

        {agent_scratchpad}
        """
        logger.info("ReadingAgent: Cung cấp prompt chuyên về Reading")

        return PromptTemplate.from_template(prompt_string)


# --- CHẠY DEMO ĐA TOOL/AGENT (TÍCH HỢP) ---
async def main():
    """Hàm chạy chính (bất đồng bộ) để test agent."""

    # --- Initialize Gemini 2.5 Flash Model ---
    logger.info("--- Khởi tạo Gemini 2.5 Flash Model ---")

    google_api_key = "AIzaSyCWjJtVscHUG5KY"
    if not google_api_key:
        print("="*50)
        print("LỖI: Vui lòng đặt biến môi trường GOOGLE_API_KEY để chạy ví dụ này.")
        print("Lệnh (Terminal/PowerShell): set GOOGLE_API_KEY=YOUR_API_KEY_HERE")
        print("="*50)
        return

    try:
        # Sử dụng model "gemini-2.5-flash"
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=google_api_key,
            convert_system_message_to_human=True
        )
    except Exception as e:
        logger.error(f"Không thể khởi tạo Gemini model: {e}")
        return

    # --- Initialize ReadingAgent with Model ---
    print("\n--- Khởi tạo ReadingAgent với Model ---")

    # Pass model to BaseAgent 
    reading_agent = LanguageAgentPart6(model=model)

    print("\n--- Bắt đầu chạy ReadingAgent (Async) ---")

    # --- DỮ LIỆU VÍ DỤ PART 6 ---
    passage = """To: Customer Service Department
    From: Kenneth Venkman
    Subject: Incorrect billing statement
    Date: July 20
    I am contacting you once ____ (131)  after receiving no reply to my prior e-mail regarding the problem with my gas bill for the month of June. After receiving my bill for last month, I queried the charge that was included for unpaid gas in May.
    When I called your department about the issue, I ____ (132 ) that the incorrect charges would be removed and that a new bill would be sent to me by e-mail.
    However, the new bill still includes these erroneous charges, which I have no intention of paying. Therefore, I have attached an image of a receipt ____  (133) that I paid my bill for the month of May in full on June 12. ____ (134).
    Sincerely,
    Kenneth Venkman

    131.
    much
    before
    previously
    again
    132.
    will be informed
    would have informed
    was informed
    had been informing
    133.
    collaborating
    confirming
    convincing
    converting
    134
    I will send the requested payment at my earliest convenience.
    I will send the requested payment at my earliest convenience.
    I will send the requested payment at my earliest convenience.
    I would like you to rectify this situation as soon as possible"""
    PART_6_PASSAGE = """
    The annual departmental retreat will be held next month. Please check the attachment for the detailed schedule.
    We hope everyone _______ this important team-building event.
    (A) attends
    (B) attending
    (C) to attend
    (D) attendance
    Could you choose the correct answer and explain why?
    """
    s = clean_and_extract_passage_simple(PART_6_PASSAGE)
    # print(f"Test input: {s}")
    try:
        # Sử dụng .ainvoke() (bất đồng bộ) vì đây là API call thật
        result = await reading_agent.ainvoke({
            "input": s
        })

        print("\n--- Kết quả từ ReadingAgent (Gemini) ---")
        print(f"Câu trả lời cuối cùng: {result['output'][0]['text']}")

    except Exception as e:
        logger.error(f"Lỗi khi chạy agent: {e}")

# Chạy hàm main bất đồng bộ
if __name__ == "__main__":
    asyncio.run(main())