import random
import unicodedata
from typing import Optional
import re

# ==================== Heuristics / Patterns =================================
# Phrases that indicate confirmation/clarify 
# CONFIRM_PHRASES = [
#     r"\bare you sure\b", r"\bare you certain\b", r"\bbạn chắc\b", r"\bbạn có chắc\b",
#     r"\btại sao\b", r"\bwhy\b", r"\bexplain\b", r"\bgiải thích\b"
# ]

GREETINGS_SUBSTR = ["xin chào", "chào bạn", "chào", "hi", "hello", "hey"]
THANKS_SUBSTR = ["cám ơn", "cảm ơn", "thanks", "thank you"]
SMALL_TALK_SUBSTRINGS = [
    "hôm nay", "bữa nay", "bạn khỏe", "bạn thế nào", "dạo này",
    "mệt quá", "vui quá", "rảnh không", "muốn uống", "cà phê", "thời tiết",
    "how are you", "how's it going", "whats up", "what's up", "what up", "how long", "what time",
    "weather", "coffee"
]

# TOEIC markers (raw patterns to detect exam-style Qs)
TOEIC_MARKERS = [
    r"a\.", r"b\.", r"c\.", r"d\.", r"\(a\)", r"\(b\)", r"\(c\)", r"\(d\)",
    r"complete the sentence", r"fill in the blank", r"điền vào chỗ trống", r"câu thiếu", r"\b[A-D]\)"
]
TOEIC_COMPILED = re.compile("|".join(TOEIC_MARKERS), flags=re.IGNORECASE | re.UNICODE)

# Additional compiled regex list for english multi-word / apostrophes etc.
SMALL_TALK_REGEX = [
    re.compile(r"\bhow are you\b", re.IGNORECASE),
    re.compile(r"\bhow'?s it going\b", re.IGNORECASE),
    re.compile(r"\bwhat'?s up\b", re.IGNORECASE),
    re.compile(r"\bhow are you doing\b", re.IGNORECASE),
    re.compile(r"\bwhat time\b", re.IGNORECASE),
    re.compile(r"\bhow long\b", re.IGNORECASE),
    re.compile(r"\bweather\b", re.IGNORECASE),
    re.compile(r"\bthanks?\b", re.IGNORECASE),
    re.compile(r"\bthank you\b", re.IGNORECASE),
    re.compile(r"\bhi\b", re.IGNORECASE),
    re.compile(r"\bhello\b", re.IGNORECASE),
    re.compile(r"\bhey\b", re.IGNORECASE),
]

# Convenience compiled lists for contains_pattern usage if needed
GREETINGS_RE = [re.compile(re.escape(p), re.IGNORECASE) for p in GREETINGS_SUBSTR]
THANKS_RE = [re.compile(re.escape(p), re.IGNORECASE) for p in THANKS_SUBSTR]
SMALL_TALK_RE = SMALL_TALK_REGEX  
# ==================== End Heuristics / Patterns ===============================

# ==================== Helpers =================================================
def normalize_text(s: str) -> str:
    """Normalize unicode (NFC), unify apostrophes, and lowercase."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("’", "'").replace("‘", "'")
    return s.lower()

def clean_text_for_substring(s: str) -> str:
    """
    Prepare text for substring matching:
    - normalize unicode
    - remove punctuation except apostrophe (so "what's" keeps apostrophe)
    - collapse spaces
    """
    s = normalize_text(s)
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    s = re.sub(r"[_0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def contains_substring(cleaned: str, substrings) -> bool:
    """Check if any substring (normalized) appears in cleaned text."""
    for sub in substrings:
        if normalize_text(sub) in cleaned:
            return True
    return False

def contains_regex(text: str, regex_list) -> bool:
    """Check if any compiled regex matches the (raw) text."""
    for rx in regex_list:
        if rx.search(text):
            return True
    return False

def is_probable_toeic(text: str) -> bool:
    """Detect exam/toeic style markers in the raw text (fast check)."""
    if not text:
        return False
    return bool(TOEIC_COMPILED.search(text))

# ==================== Chitchat detection & reply =============================
def is_chitchat_reply(text: str) -> Optional[str]:
    """
    Return a canned Vietnamese reply if the input is chit-chat (greeting/thanks/small talk).
    Return None if not chit-chat (so host should route to agents).
    """
    if not text or not text.strip():
        return None

    # If text contains TOEIC markers anywhere -> treat as NOT chit-chat
    if is_probable_toeic(text):
        return None

    # Prepare cleaned text for substring checks
    cleaned = clean_text_for_substring(text)

    # Greetings 
    if contains_substring(cleaned, GREETINGS_SUBSTR) or contains_regex(text, [re.compile(r"\b(hi|hello|hey|xin chào|chào)\b", re.IGNORECASE)]):
        replies = [
            "Chào bạn! Rất vui được gặp bạn — mình sẵn sàng hỗ trợ bạn học TOEIC.",
            "Xin chào! Hôm nay bạn muốn luyện phần nào của TOEIC?",
            "Xin chào! Mình luôn sẵn sàng giúp bạn ôn TOEIC."
        ]
        return random.choice(replies)

    # Thanks 
    if contains_substring(cleaned, THANKS_SUBSTR) or contains_regex(text, [re.compile(r"\bthanks\b", re.IGNORECASE), re.compile(r"\bthank you\b", re.IGNORECASE)]):
        replies = [
            "Không có gì đâu, mình rất vui khi được giúp bạn!",
            "Rất hân hạnh được hỗ trợ!",
            "Cảm ơn bạn — chúc bạn học tốt!"
        ]
        return random.choice(replies)

    # Small talk 
    if contains_substring(cleaned, SMALL_TALK_SUBSTRINGS) or contains_regex(text, SMALL_TALK_RE):
        replies = [
            "Vui lòng bổ sung thông tin liên quan đến TOEIC, như những câu hỏi hoặc phần luyện thi bạn muốn.",
            "Mình chỉ có thể trả lời các câu hỏi liên quan đến TOEIC. Bạn có thể hỏi về ngữ pháp, từ vựng, hoặc các phần thi TOEIC.",
            "Vui lòng hỏi các câu hỏi liên quan đến TOEIC để mình có thể giúp bạn tốt hơn."
        ]
        return random.choice(replies)

    return None
# ==================== End Chitchat detection & reply ============================