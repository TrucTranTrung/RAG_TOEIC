# models.py cho Agent_Part3
# Model được inject từ Host Agent

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="config/.env")

# Default model - sẽ được override bởi Host Agent  
google_api_key = os.environ.get("GOOGLE_API_KEY2", os.environ.get("GOOGLE_API_KEY"))

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=google_api_key,
) if google_api_key else None

# Alias for backward compatibility
llm_mistral = llm_gemini
