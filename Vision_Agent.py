# --- 1. Imports ---
from typing import List, TypedDict
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util
import warnings
import os

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings("ignore")

# --- 2. AgentState ---
class AgentState(TypedDict):
    image_path: str
    options: List[str]
    reasoning: str
    final_answer: str

# --- 3. Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- 4. Load Model ---
# Chỉ tải duy nhất model CLIP
MODEL_NAME = "clip-ViT-B-32"
print(f"Loading CLIP model: {MODEL_NAME}...")
clip_model = SentenceTransformer(MODEL_NAME, device=device)
print("Model loaded successfully.")

# --- 5. Hàm phân tích (chỉ dùng CLIP) ---
def analyze_task_node(state: AgentState) -> AgentState:
    image_path = state['image_path']
    options = state['options']
    
    print(f"\nAnalyzing image: {image_path}...")
    
    try:
        image = Image.open(image_path).convert("RGB")
        
        # --- Phân tích chọn đáp án bằng CLIP ---
        cleaned_options = [opt[4:] for opt in options]
        
        # Mã hóa ảnh và các lựa chọn văn bản
        image_embedding = clip_model.encode(image, convert_to_tensor=True, device=device)
        text_embeddings = clip_model.encode(cleaned_options, convert_to_tensor=True, device=device)
        
        # Tính toán độ tương đồng
        similarities = util.cos_sim(image_embedding, text_embeddings)[0]
        
        # Tìm đáp án có điểm cao nhất
        best_option_idx = torch.argmax(similarities).item()
        final_answer = options[best_option_idx][:3]
        
        # --- Tạo chuỗi giải thích ---
        reasoning_parts = ["--- CLIP Score Analysis ---"]
        for i, opt in enumerate(options):
            reasoning_parts.append(
                f"- Option {opt[:3]}: '{cleaned_options[i]}'\n"
                f"  - Similarity Score: {similarities[i].item():.4f}"
            )
        reasoning_parts.append(f"\n✅ Conclusion: Option {options[best_option_idx][:3]} has the highest similarity score.")
        
        reasoning = "\n".join(reasoning_parts)
        
        return {"reasoning": reasoning, "final_answer": final_answer.strip()}

    except Exception as e:
        return {"reasoning": f"Error processing image: {e}", "final_answer": "Error"}

# --- 6. Vòng lặp test ---
images = ["mo_ta_tranh1.jpg", "mo_ta_tranh2.jpg", "mo_ta_tranh3.jpg", "mo_ta_tranh4.jpg"]
options_list = [
    ["(A) The man is using a screwdriver to screw a nail into the building frame.",
     "(B) The man is hammering something into a building frame.",
     "(C) The man is making the frame with his hand.",
     "(D) The man is wearing protective glasses."],
    ["(A) The woman is talking on the phone.",
     "(B) The woman is using her cell phone.",
     "(C) The woman is typing on the laptop.",
     "(D) The woman is writing in her notebook."],
    ["(A) There are some tables and chairs outdoors.",
     "(B) There are some people sitting at the tables.",
     "(C) There are plastic umbrellas on the tables.",
     "(D) There are many flowers in the garden."],
    # ĐÃ SỬA LỖI CÚ PHÁP Ở ĐÂY
    ["(A) The man is holding some seafood.",
     "(B) The woman is baking a crab.",
     "(C) They are scared of the crab.",
     "(D) The family is shopping for breakfast."
    ]
]

for idx, (img_path, opts) in enumerate(zip(images, options_list), 1):
    # Hỗ trợ cả đuôi .jpg và .png
    if img_path.endswith('.jpg'):
        png_path = img_path.replace('.jpg', '.png')
        if os.path.exists(png_path):
            img_path = png_path

    if not os.path.exists(img_path):
        print(f"\nLỖI: Không tìm thấy file ảnh '{img_path}'.")
        continue

    state = {"image_path": img_path, "options": opts}
    result = analyze_task_node(state)
    print(f"\n================ KẾT QUẢ ẢNH {idx} =================")
    print(result['reasoning'])
    print(f"\n=> ĐÁP ÁN CUỐI CÙNG: {result['final_answer']}")
    print("===================================================")