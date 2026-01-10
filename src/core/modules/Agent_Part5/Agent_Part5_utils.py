import re
from typing import List, Tuple, Optional

# Regex patterns cho Part 5
_BLANK_RE = re.compile(r'_{2,}|\[BLANK\]|\<BLANK\>|---', flags=re.I)
_QUESTION_NUM_RE = re.compile(r'^\s*(\d{1,3})[\.\)]\s*', flags=re.M)
_OPTION_LINE = re.compile(r'^\s*\(?([A-Da-d])\)?[\.\)]\s+(.+)')
_GARBAGE_TOKEN_RE = re.compile(r'\(\(+\d+\)+\)|Choices:?|Options:?', flags=re.I)


def _split_lines_keep(text: str) -> List[str]:
    """Split text thành lines, giữ nguyên empty lines"""
    return [ln for ln in text.replace('\r\n', '\n').split('\n')]


def _extract_question_number(line: str) -> Optional[str]:
    """Trích xuất số thứ tự câu hỏi từ dòng"""
    match = _QUESTION_NUM_RE.match(line)
    if match:
        return match.group(1)
    return None


def _parse_option_line(line: str) -> Optional[Tuple[str, str]]:
    """
    Parse dòng option thành (letter, content)
    Ví dụ: "(A) attending" -> ("A", "attending")
    """
    match = _OPTION_LINE.match(line.strip())
    if match:
        letter = match.group(1).upper()
        content = match.group(2).strip()
        return (letter, content)
    return None


def clean_part5_question(text: str) -> str:
    """
    Làm sạch và chuẩn hóa câu hỏi Part 5.
    
    Part 5 thường có format:
    101. The manager _______ the report yesterday.
    (A) submit
    (B) submits
    (C) submitted
    (D) submitting
    
    Hoặc không có số:
    The manager _______ the report yesterday.
    (A) submit
    (B) submits
    (C) submitted
    (D) submitting
    """
    if not text:
        return ""
    
    # Loại bỏ garbage tokens
    text = _GARBAGE_TOKEN_RE.sub('', text)
    
    lines = _split_lines_keep(text)
    
    result_parts = []
    question_number = None
    question_sentence = []
    options = {}
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
        
        # Check nếu là số câu hỏi
        num = _extract_question_number(line)
        if num:
            question_number = num
            # Loại bỏ số ra khỏi line
            line = _QUESTION_NUM_RE.sub('', line).strip()
        
        # Check nếu là option line
        option_data = _parse_option_line(line)
        if option_data:
            letter, content = option_data
            options[letter] = content
            i += 1
            continue
        
        # Nếu có blank trong line, đó là câu hỏi
        if _BLANK_RE.search(line) and line:
            question_sentence.append(line)
        elif line and not option_data:
            # Có thể là phần tiếp theo của câu hỏi
            question_sentence.append(line)
        
        i += 1
    
    # Build output
    if question_number:
        result_parts.append(f"{question_number}.")
    
    if question_sentence:
        # Join câu hỏi, ensure có blank
        full_sentence = " ".join(question_sentence)
        result_parts.append(full_sentence)
    
    # Add options theo thứ tự A, B, C, D
    if options:
        for letter in ['A', 'B', 'C', 'D']:
            if letter in options:
                result_parts.append(f"({letter}) {options[letter]}")
    
    return "\n".join(result_parts).strip()


def extract_multiple_part5_questions(text: str) -> List[str]:
    """
    Tách nhiều câu hỏi Part 5 từ một đoạn text.
    Trả về list các câu hỏi đã được clean.
    """
    if not text:
        return []
    
    lines = _split_lines_keep(text)
    questions = []
    current_question = []
    
    for line in lines:
        # Check nếu bắt đầu câu hỏi mới (có số thứ tự)
        if _QUESTION_NUM_RE.match(line.strip()):
            # Save câu hỏi cũ nếu có
            if current_question:
                q_text = "\n".join(current_question)
                cleaned = clean_part5_question(q_text)
                if cleaned:
                    questions.append(cleaned)
                current_question = []
        
        current_question.append(line)
    
    # Save câu hỏi cuối cùng
    if current_question:
        q_text = "\n".join(current_question)
        cleaned = clean_part5_question(q_text)
        if cleaned:
            questions.append(cleaned)
    
    return questions


def validate_part5_format(text: str) -> bool:
    """
    Kiểm tra xem text có format hợp lệ của Part 5 không.
    Cần có: blank và ít nhất 2 options.
    """
    has_blank = bool(_BLANK_RE.search(text))
    
    # Check for options - both formats: (A) or A)
    option_pattern = re.compile(r'\([A-Da-d]\)|[A-Da-d]\)', flags=re.I)
    option_count = len(option_pattern.findall(text))
    
    return has_blank and option_count >= 2


def normalize_blank(text: str, blank_symbol: str = "______") -> str:
    """
    Chuẩn hóa tất cả các dạng blank về một dạng thống nhất.
    """
    return _BLANK_RE.sub(blank_symbol, text)


def extract_output_text(output) -> str:
    """Extract clean text from agent output (handles various formats)."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        texts = []
        for item in output:
            if isinstance(item, dict) and 'text' in item:
                texts.append(item['text'])
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts)
    if isinstance(output, dict) and 'text' in output:
        return output['text']
    return str(output)