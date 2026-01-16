import re
def pre_validate_part7_context(problem_context: str) -> str:
    """Validate 4 đáp án (A)(B)(C)(D)."""
    if not problem_context: return "Bạn cung cấp không đủ dữ liệu."
    all_options = set(re.findall(r'[\(\s]([A-D])[\)\.]', problem_context.upper()))
    target_options = {'A', 'B', 'C','D'} 
    
    if len(all_options) == 4 and all_options == target_options:
        return "1"  
    else:
        return "Bạn cung cấp không đủ đáp án (A)(B)(C)(D) hoặc sai định dạng."

def extract_text(result) -> str:
    output = result.get("output", "")
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, list):
        return output[0].get("text", "").strip()
    return ""

def format_answer_only(text: str) -> str:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("Đáp án:"):
                return "\n".join(lines[i:]).strip()
        return text.strip()
