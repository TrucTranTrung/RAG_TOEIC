from paddleocr import PaddleOCR
from fastapi import FastAPI, UploadFile, File
import tempfile, os, re

app = FastAPI(title="PaddleOCR API")

ocr = PaddleOCR(
    use_angle_cls=False,
    text_rec_score_thresh=0.0,
    lang="en",          # hoặc "ch"
    det_model_dir=None, # dùng model mặc định (nhẹ)
    rec_model_dir=None
)

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        img_path = tmp.name

    result = ocr.ocr(img_path)

    res = result[0]
    texts = [t.strip() for t in res["rec_texts"] if t.strip()]

    question_lines = []
    options = []

    for t in texts:
        # dòng rất ngắn → option
        if len(t.split()) <= 2 and len(t) <= 15:
            options.append(t)
        else:
            question_lines.append(t)

    # phục hồi blank
    question = " ".join(question_lines)
    question = re.sub(r'\.\s*\w+$', ' _____', question)

    # gán A/B/C/D theo thứ tự
    labels = ["A", "B", "C", "D"]
    labeled_options = [
        f"{labels[i]}. {opt}"
        for i, opt in enumerate(options[:4])
    ]

    final_text = question + "\n\n" + "\n".join(labeled_options)

    os.remove(img_path)

    return {"text": final_text}