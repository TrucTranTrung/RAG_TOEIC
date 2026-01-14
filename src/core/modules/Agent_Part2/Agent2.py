import sys
import asyncio
import logging
from typing import List
import re

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from ..Agent_Base.Agent import BaseAgent 
from .utils import pre_validate_part2_context, extract_text, format_answer_only
from .prompts import REACT_TEMPLATE_P2
from .models import llm_mistral

# --- CONFIG ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- TOOLS ---
@tool
def pronoun_logic_filter_tool(tool_input: str) -> str:
    """Lọc nhanh đáp án sai logic đối tượng (Người/Vật)."""
    # Tách câu hỏi và các đáp án
    parts = re.split(r'\n|\s*(?=\([A-C]\))', tool_input)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 2: return "Data sạch."

    q = parts[0].lower()
    options = parts[1:]
    
    # Xác định loại đối tượng
    is_p = any(w in q for w in ["who", "mr", "ms", "man", "woman", "staff"])
    forbidden = {"it", "its"} if is_p else {"he", "she", "him", "her", "his"}
    
    # Lọc đáp án lỗi
    bad_list = []
    for opt in options:
        o_words = set(re.findall(r'\b\w+\b', opt.lower()))
        if o_words & forbidden:
            bad_list.append(opt[:3])

    # --- DÒNG LOG XÁC NHẬN ---
    # if bad_list:
    #     print(f"\n>>> [FILTER] ĐÃ LOẠI: {', '.join(bad_list)}")
    # else:
    #     print("\n>>> [FILTER] KHÔNG LOẠI CÂU NÀO")
    # ------------------------

    if not bad_list: return "Data sạch. suy luận cả 3 câu."
    return f"LOẠI: {', '.join(bad_list)}. chỉ suy luận các câu còn lại."

# --- 4. AGENT CLASS ---
class TOEICPart2Agent(BaseAgent):
    def _get_tools(self) -> List:
        return [pronoun_logic_filter_tool]

    def _get_prompt(self) -> PromptTemplate:
        # Không cộng thêm ANALYSIS_PROMPT_TEXT_P2 nữa vì đã gộp trong REACT_TEMPLATE_P2 rồi
        return PromptTemplate.from_template(REACT_TEMPLATE_P2)

# --- 5. MAIN (FORMAT CHUẨN ĐẸP) ---
async def main():
    agent = TOEICPart2Agent(model=llm_mistral)

    # GIỮ NGUYÊN NỘI DUNG KỊCH BẢN TEST CỦA PART 2
    scenarios = [
        ("P2_CASE_1", """How old is this building? (A) To transport some materials. (B) About ten years old. (C) I think it's the company office."""),
        ("P2_CASE_2", """Who is the new manager? (A) Yes, he is. (B) It's on the second floor. (C) The meeting was canceled."""),
        ("P2_CASE_3", """How old is this building? (A) To transport some materials. (B) About ten years old. (C) I think it's the company office. (D) Yes, it is.""") 
    ]

    for name, problem in scenarios:
        logger.info("\n" + "=" * 80)
        logger.info(f"--- ĐANG CHẠY: {name} ---")

        # --- VALIDATE LOGIC (Part 2) ---
        validate_result = pre_validate_part2_context(problem)
        if validate_result != "1":
            print(f"\n--- KẾT QUẢ {name} ---")
            print(validate_result)
            continue
        # Input data giữ đúng format text của Part 2
        try:
            res = await agent.ainvoke({"input": problem})
            raw_text = extract_text(res)
            output = format_answer_only(raw_text)

            print(f"\n--- KẾT QUẢ {name} ---")
            print(output)
            
        except Exception as e:
            logger.error(f"Lỗi kịch bản {name}: {e}")
    print("\n" + "=" * 80 + "\n--- CHƯƠNG TRÌNH HOÀN TẤT ---")


if __name__ == "__main__":
    asyncio.run(main())