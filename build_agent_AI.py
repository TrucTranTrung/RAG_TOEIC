import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Set API key (NÊN export trong environment variables thay vì hard-code)
# Lưu ý: Nếu bạn đang chạy code này bên ngoài môi trường đã cấu hình sẵn,
# bạn nên thay thế giá trị API key bằng biến môi trường để bảo mật hơn.
os.environ["GOOGLE_API_KEY"] = "AIzaSyCDs_02II0VFpF_ILp4TWZe6zcxaeA085s"

# 2. Khởi tạo LLM (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 3. Reading Agent (Phân tích ngữ pháp/cấu trúc câu)
reading_prompt = ChatPromptTemplate.from_template(
    """Bạn là một trợ lý ngôn ngữ chuyên nghiệp và tỉ mỉ.
Phân tích câu hỏi sau để tìm ra lỗi ngữ pháp hoặc cấu trúc,
sau đó đưa ra đáp án chính xác và giải thích ngắn gọn lý do vì sao đáp án đó là đúng và các đáp án khác là sai.

Câu hỏi: {question}

---
Định dạng câu trả lời:
- Đáp án đúng: [Đáp án]
- Giải thích: [Lý do chi tiết]"""
)

reading_chain = reading_prompt | llm | StrOutputParser()

# 4. Writing Agent (Viết luận IELTS theo số lượng từ động)
writing_prompt = ChatPromptTemplate.from_template(
    """Bạn là một chuyên gia viết luận IELTS.
Dựa vào chủ đề dưới đây, hãy viết một bài luận hoàn chỉnh theo chuẩn Task 2.

- Bài viết cần có cấu trúc 4 đoạn: Mở bài, 2 đoạn thân bài và Kết bài.
- Mở bài cần paraphrase đề bài và nêu rõ quan điểm.
- Mỗi đoạn thân bài phải có một luận điểm chính và các ví dụ, giải thích hỗ trợ.
- Kết bài cần tóm tắt lại các ý đã nêu và khẳng định lại quan điểm.
- Sử dụng ngôn ngữ học thuật, từ vựng đa dạng và cấu trúc câu phức tạp.
- Độ dài khoảng {word_count} từ.

Chủ đề: {topic}"""
)

writing_chain = writing_prompt | llm | StrOutputParser()


# 5. Summarizing Agent (Tóm tắt đoạn văn tiếng Anh)
summarizing_prompt = ChatPromptTemplate.from_template(
    """You are an expert summarizer.
Please summarize the following English text into a concise, well-structured paragraph, highlighting the main points.
The summary must be in English.

Text to summarize:
---
{text_content}
---"""
)

summarizing_chain = summarizing_prompt | llm | StrOutputParser()


# 6. Hàm host (Điều phối các tác vụ)
def host(task_type, content, word_count=300):
    """
    Điều phối tác vụ đến các Agent khác nhau dựa trên task_type.

    :param task_type: Loại tác vụ ("reading", "writing", "summarizing")
    :param content: Nội dung đầu vào (câu hỏi, chủ đề, hoặc đoạn văn)
    :param word_count: Số lượng từ mong muốn cho bài luận (chỉ dùng cho task_type="writing")
    :return: Kết quả từ Agent tương ứng
    """
    if task_type == "reading":
        return reading_chain.invoke({"question": content})
    elif task_type == "writing":
        return writing_chain.invoke({"topic": content, "word_count": word_count})
    elif task_type == "summarizing":
        return summarizing_chain.invoke({"text_content": content})
    else:
        return "Task không hợp lệ!"


# 7. Demo
if __name__ == "__main__":
    print("=== Reading ===")
    print(host("reading", "The company has a _______ reputation for excellent customer service. (A) solid (B) heavy (C) thick (D) strong"))
    print("\n" + "="*20 + "\n")
    print(host("reading", "This is a reminder about our annual company picnic... What is the purpose of the email? (A) To inform employees about a new policy. (B) To announce a special anniversary. (C) To invite employees to a company event. (D) To request employees to sign up for a class."))
    print("\n" + "="*20 + "\n")
    print(host("reading", "I DOESN'T CARE?"))
    print("\n" + "="*20 + "\n")
    print(host("reading", "Câu có 2 đoạn, thì hiện tại đơn + tương lai đơn, dùng để miêu tả 1 hành động có thể xảy ra, là câu gì?"))


    print("\n=== Writing ===")
    print(host("writing", "Some people think the best way to reduce crime is to give longer prison sentences. To what extent do you agree or disagree?", word_count=500))

    print("\n\n=== Summarizing ===")
    sample_text = """Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. AI can be categorized as narrow AI, general AI, and super AI. Narrow AI is programmed to perform a single task, while general AI can successfully perform any intellectual task that a human being can. Super AI, currently hypothetical, would surpass human intelligence and is the subject of much debate regarding its future impact and ethical considerations."""
    print(f"--- Original Text (approx. 100 words) ---\n{sample_text}\n")
    print("--- Summary Result ---")
    print(host("summarizing", sample_text))

    print("\n\n=== Invalid Task Demo ===")
    print(host("analyze", "some text"))