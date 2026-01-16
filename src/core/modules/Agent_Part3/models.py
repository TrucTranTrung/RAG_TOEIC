from langchain_mistralai import ChatMistralAI
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path="config/.env")

llm_mistral = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
    api_key=os.environ["OCR_KEY"],
)

# tok = T5Tokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")