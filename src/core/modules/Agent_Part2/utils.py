import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


def pre_validate_part2_context(problem_context: str) -> str:
    """Kiểm tra CHỈ có A, B, C và không có đáp án nào khác."""
    if not problem_context:
        return "Bạn cung cấp không đủ đáp án (A)(B)(C), vui lòng nhập lại."
    all_options = set(re.findall(r'\([A-Z]\)', problem_context.upper()))
    target_options = {'(A)', '(B)', '(C)'}
    if len(all_options) == 3 and all_options == target_options:
        return "1"
    else:
        return "Bạn cung cấp không đủ đáp án (A)(B)(C), vui lòng nhập lại."


def extract_text(result: Any) -> str:
    try:
        return result["output"].strip() if isinstance(result, dict) and "output" in result else str(result).strip()
    except Exception:
        return ""


def format_answer_only(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("Đáp án:"):
            return "\n".join(lines[i:]).strip()
    return text.strip()
