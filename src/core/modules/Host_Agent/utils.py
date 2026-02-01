# utils.py
# -*- coding: utf-8 -*-
"""
Utility Functions for Host Agent

Common helper functions used across the Host Agent module.
"""

import re
import unicodedata
from typing import Optional, List, Tuple, Any, Dict
import logging

logger = logging.getLogger(__name__)


# ==================== TEXT NORMALIZATION ====================

def normalize_text(text: str) -> str:
    """
    Normalize unicode text
    
    - NFC normalization
    - Unify apostrophes
    - Lowercase
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    text = unicodedata.normalize("NFC", text)
    text = text.replace("'", "'").replace("'", "'")
    return text.lower()


def clean_text(text: str) -> str:
    """
    Clean text for processing
    
    - Normalize unicode
    - Remove extra whitespace
    - Strip leading/trailing whitespace
    
    Args:
        text: Input text
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_file_paths(text: str) -> str:
    """
    Remove file path patterns from text
    
    Args:
        text: Input text potentially containing file paths
    
    Returns:
        Text with file paths removed
    """
    # Common file path patterns
    patterns = [
        r'(?:image|ảnh|hình)[:\s]+[^\s]+\.(?:jpg|png|jpeg|gif|webp)',
        r'(?:audio|âm thanh)[:\s]+[^\s]+\.(?:mp3|wav|m4a|ogg)',
        r'[A-Za-z]:\\[^\s]+',  # Windows paths
        r'/[^\s]+\.[a-zA-Z]{2,4}',  # Unix paths
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return clean_text(text)


# ==================== TOEIC SPECIFIC ====================

def count_options(text: str) -> int:
    """
    Count TOEIC answer options (A), (B), (C), (D)
    
    Args:
        text: Input text
    
    Returns:
        Number of unique options found (0-4)
    """
    pattern = r'\([A-D]\)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return len(set(m.upper() for m in matches))


def extract_options(text: str) -> Dict[str, str]:
    """
    Extract answer options with their content
    
    Args:
        text: Input text containing options
    
    Returns:
        Dict mapping option letter to content
        e.g., {"A": "submit", "B": "submits", ...}
    """
    # Pattern: (A) content or A. content or A) content
    pattern = r'\(?([A-D])\)?[\.\)]\s*([^\n\(]+?)(?=\s*\(?[A-D]\)?[\.\)]|$)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    return {m[0].upper(): m[1].strip() for m in matches}


def has_blank(text: str) -> bool:
    """
    Check if text contains blank patterns
    
    Patterns: _____, [BLANK], ---, ...
    
    Args:
        text: Input text
    
    Returns:
        True if blank found
    """
    blank_pattern = r'_{2,}|\[BLANK\]|\<BLANK\>|---|\.\.\.'
    return bool(re.search(blank_pattern, text, re.IGNORECASE))


def count_blanks(text: str) -> int:
    """
    Count number of blanks in text
    
    Args:
        text: Input text
    
    Returns:
        Number of blanks
    """
    blank_pattern = r'_{2,}|\[BLANK\]|\<BLANK\>|---'
    return len(re.findall(blank_pattern, text, re.IGNORECASE))


def is_email_format(text: str) -> bool:
    """
    Check if text appears to be email/letter format
    
    Args:
        text: Input text
    
    Returns:
        True if email-like format
    """
    email_markers = ['To:', 'From:', 'Subject:', 'Date:', 'Dear ', 'Sincerely', 'Best regards']
    return any(marker.lower() in text.lower() for marker in email_markers)


def extract_question_numbers(text: str) -> List[int]:
    """
    Extract question numbers from text
    
    Args:
        text: Input text
    
    Returns:
        List of question numbers found
    """
    pattern = r'(\d+)\.\s*(?:What|Who|Where|When|Why|How|Which)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [int(m) for m in matches]


# ==================== VALIDATION ====================

def validate_part5_format(text: str) -> Tuple[bool, str]:
    """
    Validate Part 5 question format
    
    Requirements:
    - Has exactly one blank
    - Has 4 options (A/B/C/D)
    - Not paragraph-like
    
    Args:
        text: Input text
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    blank_count = count_blanks(text)
    option_count = count_options(text)
    
    if blank_count == 0:
        return False, "Không tìm thấy chỗ trống (___)"
    
    if blank_count > 1:
        return False, f"Có {blank_count} chỗ trống, Part 5 chỉ cần 1"
    
    if option_count < 4:
        return False, f"Chỉ có {option_count} lựa chọn, cần đủ 4 (A/B/C/D)"
    
    return True, ""


def validate_part6_format(text: str) -> Tuple[bool, str]:
    """
    Validate Part 6 question format
    
    Requirements:
    - Has multiple blanks
    - Paragraph-like structure
    
    Args:
        text: Input text
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    blank_count = count_blanks(text)
    
    if blank_count < 2:
        return False, "Part 6 cần nhiều chỗ trống trong đoạn văn"
    
    # Check for paragraph structure
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
    if len(sentences) < 3:
        return False, "Đoạn văn quá ngắn cho Part 6"
    
    return True, ""


# ==================== OUTPUT EXTRACTION ====================

def extract_answer_from_output(output: str) -> Optional[str]:
    """
    Extract answer letter from agent output
    
    Looks for patterns like:
    - "Đáp án: (A)"
    - "Answer: B"
    - "Đáp án đúng: (C)"
    
    Args:
        output: Agent output text
    
    Returns:
        Answer letter (A/B/C/D) or None
    """
    patterns = [
        r'[Đđ]áp\s*[aá]n(?:\s*đúng)?[:\s]*\(?([A-D])\)?',
        r'[Aa]nswer[:\s]*\(?([A-D])\)?',
        r'[Cc]hoose[:\s]*\(?([A-D])\)?',
        r'→\s*\(?([A-D])\)?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return None


def extract_explanation_from_output(output: str) -> Optional[str]:
    """
    Extract explanation from agent output
    
    Args:
        output: Agent output text
    
    Returns:
        Explanation text or None
    """
    patterns = [
        r'[Gg]iải\s*thích[:\s]*(.+?)(?=\n\n|\Z)',
        r'[Ee]xplanation[:\s]*(.+?)(?=\n\n|\Z)',
        r'[Ll]ý\s*do[:\s]*(.+?)(?=\n\n|\Z)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return None


# ==================== LOGGING HELPERS ====================

def truncate_for_log(text: str, max_length: int = 100) -> str:
    """
    Truncate text for logging
    
    Args:
        text: Input text
        max_length: Maximum length
    
    Returns:
        Truncated text with ellipsis if needed
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."


def format_state_for_log(state: Dict[str, Any], keys: Optional[List[str]] = None) -> str:
    """
    Format state dict for logging
    
    Args:
        state: State dictionary
        keys: Specific keys to include (None = all)
    
    Returns:
        Formatted string
    """
    if keys is None:
        keys = ["detected_part", "detection_confidence", "routing_decision", "error"]
    
    parts = []
    for key in keys:
        if key in state:
            value = state[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.2%}")
            elif isinstance(value, str) and len(value) > 50:
                parts.append(f"{key}='{value[:50]}...'")
            else:
                parts.append(f"{key}={value}")
    
    return ", ".join(parts)
