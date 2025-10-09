import os
from PIL import Image
from typing import List, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import torch

import open_clip as clip
import warnings
import re

warnings.filterwarnings("ignore")  # tắt QuickGELU warning

# --- 1. Set API key ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyCDs_02II0VFpF_ILp4TWZe6zcxaeA085s"

# --- 2. AgentState ---
class AgentState(TypedDict):
    image_path: str
    options: List[str]
    final_answer: str
    explanation: str
    facts: str  # Thông tin trích xuất từ ảnh
    transcript: str  # Nội dung text từ audio
    question: str  # Câu hỏi liên quan đến tranh/audio

# --- 3. Khởi tạo LLM Gemini ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- 4. Khởi tạo CLIP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = clip.create_model_and_transforms("ViT-B-32", pretrained='openai')

def extract_facts(image_path: str, options: List[str]) -> str:
    """Dùng CLIP để lấy thông tin quan trọng từ ảnh và gợi ý option"""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)

    similarities = []
    for opt in options:
        text = clip.tokenize([opt]).to(device)
        text_features = clip_model.encode_text(text)
        sim = torch.cosine_similarity(image_features, text_features)
        similarities.append(sim.item())

    best_idx = similarities.index(max(similarities))
    facts = f"Option có khả năng đúng nhất theo ảnh: {options[best_idx]}"
    return facts

# --- 5. Prompt template hoàn toàn tiếng Việt ---
listening_part1 = """
Bạn là giáo viên TOEIC Listening. Nhiệm vụ của bạn là phân tích câu hỏi dựa trên bối cảnh từ tranh và transcript audio.
Hãy làm theo các bước sau:
- Xem xét bối cảnh được trích xuất từ ảnh và đoạn transcript audio.
- Xác định option đúng và loại bỏ các option sai dựa trên bối cảnh và transcript.
- Chỉ xuất 2 dòng: Đáp án và Giải thích.

Quy tắc & Ràng buộc:
- Chỉ dựa vào bối cảnh từ ảnh và transcript, không thêm thông tin ngoài.
- Giải thích phải rõ ràng, dễ hiểu, phù hợp cho người học TOEIC.
- Trình bày câu trả lời theo đúng định dạng yêu cầu.

Định dạng đầu ra:
- Đáp án (tiếng Việt): [Option A/B/C/D]
- Giải thích (tiếng Việt): [Lý do option đúng, loại bỏ option sai]
- Lưu ý: Hãy xác định hành động của con người và các vật trong tranh, chọn option phù hợp với bối cảnh.

Ví dụ:
Facts: Người đàn ông cầm búa và chuẩn bị đóng đinh.
Transcript: Người đàn ông đang đóng một cái đinh vào tường.
Các lựa chọn:
(A) Người đàn ông đang dùng tua vít để vặn đinh.
(B) Người đàn ông đang đóng một vật gì đó vào khung xây dựng.
(C) Người đàn ông đang làm khung bằng tay.
(D) Người đàn ông đang đeo kính bảo hộ.
Câu hỏi: Người đàn ông đang làm gì?

Đáp án (tiếng Việt): B
Giải thích (tiếng Việt): Option B khớp với facts: người đàn ông cầm búa và hành động “đóng một cái đinh vào tường”. A, C, D không phù hợp.

Bây giờ hãy phân tích câu hỏi sau:
Câu hỏi: {question}
"""

def make_prompt(state: AgentState) -> str:
    options_text = "\n".join(state['options'])
    return listening_part1.format(
        question=state['question']
    ).replace("{facts}", state['facts']).replace("{transcript}", state['transcript']).replace("{options_text}", options_text)

# --- 6. Hàm phân tích ---
def analyze_part1(state: AgentState) -> AgentState:
    # Trích xuất facts từ ảnh
    state['facts'] = extract_facts(state['image_path'], state['options'])

    prompt = make_prompt(state)
    response = llm([HumanMessage(content=prompt)])
    llm_output = response.content.strip()

    # Tách 2 dòng Đáp án và Giải thích
    final_answer, explanation = "", ""
    for line in llm_output.splitlines():
        line = line.strip()
        if line.startswith("Đáp án") and not final_answer:
            m = re.search(r'\b([A-D])\b', line)
            final_answer = f"Đáp án: {m.group(1)}" if m else line
        elif line.startswith("Giải thích") and not explanation:
            explanation = line

    # Fallback
    if not final_answer:
        final_answer = f"Đáp án: {state['options'][0][1]}"  # lấy chữ A/B/C/D đầu tiên
    if not explanation:
        explanation = "Giải thích: Không có thông tin chi tiết từ ảnh hoặc transcript"

    return {**state, "final_answer": final_answer, "explanation": explanation}

# --- 7. Test 4 ảnh ---
images_paths = [
    "mo_ta_tranh1.png",
    "mo_ta_tranh2.png",
    "mo_ta_tranh3.png",
    "mo_ta_tranh4.png"
]

options_list = [
    ["(A) Người đàn ông đang dùng tua vít để vặn đinh.",
     "(B) Người đàn ông đang đóng một vật gì đó vào khung xây dựng.",
     "(C) Người đàn ông đang làm khung bằng tay.",
     "(D) Người đàn ông đang đeo kính bảo hộ."],
    ["(A) Người phụ nữ đang nói chuyện điện thoại.",
     "(B) Người phụ nữ đang sử dụng điện thoại di động.",
     "(C) Người phụ nữ đang gõ trên máy tính xách tay.",
     "(D) Người phụ nữ đang viết vào sổ tay."],
    ["(A) Có một số bàn ghế ngoài trời.",
     "(B) Có một số người đang ngồi tại các bàn.",
     "(C) Có ô nhựa trên bàn.",
     "(D) Có nhiều hoa trong vườn."],
    ["(A) Người đàn ông đang cầm hải sản.",
     "(B) Người phụ nữ đang nướng cua.",
"(C) Họ đang sợ con cua.",
     "(D) Gia đình đang đi mua đồ ăn sáng."]
]

# Đây là transcript text giả lập từ voice
transcripts = [
    "Người đàn ông đang đóng một vật gì đó vào khung xây dựng.",
    "Người phụ nữ đang nói chuyện điện thoại.",
    "Người ta đang ngồi ngoài trời tại các bàn với ghế và ô.",
    "Người đàn ông đang cầm hải sản."
]

questions = [
    "Người đàn ông đang làm gì?",
    "Người phụ nữ đang làm gì?",
    "Người ta đang làm gì?",
    "Người đàn ông đang cầm gì?"
]

for img_path, opts, transcript, question in zip(images_paths, options_list, transcripts, questions):
    if not os.path.exists(img_path):
        Image.new("RGB", (224, 224)).save(img_path)

    state: AgentState = {
        "image_path": img_path,
        "options": opts,
        "final_answer": "",
        "explanation": "",
        "facts": "",
        "transcript": transcript,
        "question": question
    }
    result = analyze_part1(state)

    print(f"\nPhân tích ảnh: {img_path}")
    print(result['final_answer'])
    print(result['explanation'])
    print("=============================")