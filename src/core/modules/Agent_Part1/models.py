# models.py cho Agent_Part1
# Model được inject từ Host Agent, không cần hardcode ở đây

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="config/.env")

# Default model - sẽ được override bởi Host Agent
google_api_key = os.environ.get("GOOGLE_API_KEY1", os.environ.get("GOOGLE_API_KEY"))

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    api_key=google_api_key,
    temperature=0.0 
) if google_api_key else None
