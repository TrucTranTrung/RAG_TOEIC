from langchain_mistralai import ChatMistralAI
import load_dotenv
import os
load_dotenv.load_dotenv(dotenv_path="config/.env")

llm_mistral = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
    api_key=os.environ["OCR_KEY"],
)