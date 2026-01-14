from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="config/.env")

google_api_key = os.environ.get("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    api_key=google_api_key,
    temperature=0.0 
)