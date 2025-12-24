from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCRVL
import tempfile
import os

app = FastAPI(title="PaddleOCR-VL API")

# load model 1 lần
pipeline = PaddleOCRVL()

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    # lưu ảnh tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        img_path = tmp.name

    results = pipeline.predict(img_path)

    texts = []
    for res in results:
        # lấy text thuần
        texts.append(res.to_dict().get("text", ""))

    os.remove(img_path)

    return {
        "text": "\n".join(texts)
    }