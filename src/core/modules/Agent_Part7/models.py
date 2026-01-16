from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from transformers import T5Tokenizer, T5ForConditionalGeneration
import spacy
import os

nlp = spacy.load("en_core_web_sm")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,
)

# llm_mistral = ChatMistralAI(
#     model="mistral-small-latest",
#     temperature=0,
#     api_key=os.environ["OCR_KEY"],
# )

# tok = T5Tokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")