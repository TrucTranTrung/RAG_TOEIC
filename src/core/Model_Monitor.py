from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import requests
import audio
import os
from dotenv import load_dotenv
from prompt import *

class TOEICAgentHub:
    def __init__(self):
        """Khởi tạo tất cả model cố định ở đây"""
        # Host agent: chịu trách nhiệm routing
        self.router = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
        # Goi con grog voi open la vi du thoi, tu tao function cho tung model nha ca nha
        self.models = {
            "listening_part1": ChatGroq(model="mixtral-8x7b-32768", temperature=0.3),
            "listening_part234": ChatGroq(model="mixtral-8x7b-32768", temperature=0.3),
            "Reading_part5": ChatGroq(model="mixtral-8x7b-32768", temperature=0.3),
            "Reading_part67": ChatGroq(model="mixtral-8x7b-32768", temperature=0.3),
            "translation": ChatOpenAI(model="gpt-4o", temperature=0.2),
        }


        # Prompt cho host agent
        self.routing_prompt = ChatPromptTemplate.from_template("""
        Bạn là host agent cho bài TOEIC.
        Dựa vào input của user, hãy chọn 1 trong các task sau:
        - reading
        - listening
        - translation

        Trả lời duy nhất bằng tên task.
        User input: {input}
        """)

    def BotLis234(self, user_input: str) -> str:
        api_url = os.getenv("API_GEMINI_ENTITIES")
        prompt = listening_part234.format(
                subject="about TEOIC Listening part 2,3,4", 
                question=user_input
            )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        # Gửi yêu cầu POST đến API
        response = requests.post(
            api_url, 
            headers={"Content-Type": "application/json"}, 
            json=payload
        )
        response.raise_for_status() 

        # Xử lý response
        response_json = response.json()
        # Trích xuất text 
        text = response_json.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        # Xử lý và chuyển đổi text thành list
        output_list = [i.strip().lower() for i in text.replace('[','').replace(']','').replace('"','').split(',') if i.strip()]
        # Chuyển list thành một chuỗi duy nhất
        output_string = ", ".join(output_list)
        
        return output_string

    def route_task(self, user_input: str) -> str:
        """Host agent quyết định gọi task nào"""
        chain = self.routing_prompt | self.router
        decision = chain.invoke({"input": user_input})
        return decision.content.strip().lower()

    def run(self, task: str, **kwargs):
        """Chạy 1 task với model và prompt đã định nghĩa"""
        if task not in self.models:
            raise ValueError(f"Task '{task}' chưa được hỗ trợ")

        model = self.models[task]
        prompt = self.prompts.get(task)

        if prompt:
            chain = prompt | model
            return chain.invoke(kwargs)
        else:
            return model.invoke(kwargs["input"])

