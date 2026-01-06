import base64
import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv(dotenv_path="config/.env") 

api_key = os.environ["OCR_KEY"]
client = Mistral(api_key=api_key)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def perform_ocr(image_path):
    """
    Performs OCR on the given image path using Mistral OCR.
    Returns the markdown text from the first page.
    """
    base64_image = encode_image(image_path)
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_image}" 
        },
        include_image_base64=True
    )
    return ocr_response.pages[0].markdown