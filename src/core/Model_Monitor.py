from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

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

        # Prompt template cho từng task (nếu có thi hay viet ben file promt va goi qua day, lam on)
        self.prompts = {
            "reading": ChatPromptTemplate.from_template(
                "Bạn là trợ lý TOEIC Reading. Hãy giải thích và chọn đáp án: {question}"
            ),
            "listening": ChatPromptTemplate.from_template(
                "Bạn là trợ lý TOEIC Listening. Hãy phân tích hội thoại và chọn đáp án: {question}"
            ),
            "translation": ChatPromptTemplate.from_template(
                "Dịch chính xác câu sau sang tiếng Việt: {sentence}"
            ),
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

# --- Demo sử dụng ---
hub = TOEICAgentHub()

res1 = hub.run("reading", question="She ____ to the market yesterday. (A) go (B) went (C) goes)")
res2 = hub.run("translation", sentence="I will take the TOEIC test next month.")

print("Reading:", res1.content)
print("Translation:", res2.content)
