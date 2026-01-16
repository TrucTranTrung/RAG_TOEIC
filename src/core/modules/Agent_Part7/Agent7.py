import os
import asyncio
import logging
import requests
from typing import List
from dotenv import load_dotenv

from langchain_core.tools import BaseTool, tool
from langchain_core.prompts import PromptTemplate, BasePromptTemplate

load_dotenv(dotenv_path="config/.env")

from .models import nlp, model
from .utils import pre_validate_part7_context, extract_text, format_answer_only
from .prompts import prompt_string
from ..Agent_Base.Agent import BaseAgent

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ================= TOOLS =================

@tool
def vocab_search(word: str) -> dict:
    """Tra nghĩa từ vựng."""
    logger.info("Agent is calling vocab_search")
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()[0]
        meanings = []

        for m in data.get("meanings", [])[:2]:
            d = m.get("definitions", [{}])[0]
            meanings.append({
                "partOfSpeech": m.get("partOfSpeech"),
                "definition": d.get("definition"),
                "example": d.get("example")
            })

        return {"word": word, "meanings": meanings}
    except Exception:
        return {"word": word, "meanings": [], "error": "Not found"}


@tool
def entity_finder_spacy(context: str, question: str) -> dict:
    """Bắt mọi đối tượng: Tên riêng, Danh từ sự vật, Con số."""
    if not nlp: return {"error": "spaCy model not available"}

    doc = nlp(question)
    
    # 1. TÊN RIÊNG 
    ents = [ent.text for ent in doc.ents]
    
    # 2. CỤM DANH TỪ TỔNG QUÁT (Vật, Sự việc: free registration offer, seminar)
    chunks = [chunk.text for chunk in doc.noun_chunks if len(chunk.text) > 2]
    
    # 3. DANH TỪ ĐƠN LẺ QUAN TRỌNG 
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2]

    # Gộp tất cả lại thành ĐỐI TƯỢNG TỔNG QUÁT
    targets = list(set(ents + chunks + keywords))
    print(f"\n[DEBUG] ĐỐI TƯỢNG TỔNG QUÁT TÌM ĐƯỢC: {targets}")

    # Tìm khoanh vùng (Evidence)
    evidence = []
    for line in context.split("\n"):
        if any(t.lower() in line.lower() for t in targets):
            evidence.append(line.strip())
    
    unique_evidence = list(dict.fromkeys(evidence))
    print(f"[DEBUG] CÁC DÒNG KHOANH VÙNG: {unique_evidence[:3]}...")
    return {"targets": targets, "evidence": unique_evidence[:10]}

# --- ĐỊNH NGHĨA CÁC CLASS AGENT (SỬ DỤNG GEMINI API) ---
class TOEICPart7Agent(BaseAgent):
    """Agent này dùng để sử lí part7 để chọn từ thích hợp điền vào chỗ trống trong đoạn văn"""
    def _get_tools(self) -> List[BaseTool]:
        """
        Cung cấp danh sách các tools CHUYÊN BIỆT cho Reading Part 7.
        """
        logger.info("ReadingAgent: Cung cấp tools [vocab_search]")
        return [vocab_search, entity_finder_spacy]


    def _get_prompt(self) -> BasePromptTemplate:
        """
        Cung cấp một prompt template CỤ THỂ cho ReadingAgent Part 6.
        """
        logger.info("ReadingAgent: Cung cấp prompt chuyên về Reading")

        return PromptTemplate.from_template(prompt_string)


# --- CHẠY DEMO ĐA TOOL/AGENT (TÍCH HỢP) ---
async def main():
    toeic_agent = TOEICPart7Agent(model=model)

    sample_Single_Passage = """
    To: employees@simnetsolutions.com
    From: management@simnetsolutions.com
    Subject: Seminar Opportunity
    Date: February 5
    Dear Female Employees,
    Only one week remains until registration will be closed for the Women's Leadership Seminar. This seminar is offered free of charge to all of our female employees at Simnet Solutions. To accommodate our female employees' busy schedules, identical seminars will be held on two different dates—February 21 and February 23. In order to register for this specially designed seminar, you must e-mail James Taylor in human resources by 5:00 PM. on February 12. This seminar will teach our female employees about how to communicate with confidence and credibility in the workplace. The Simnet Solutions Management Team
    Question: When will the free registration offer end?
    A. On February 5
    B. On February 12
    C. On February 21
    D. On February 23
"""
    sample_Double_Passage = """
    Sponsored by the Department of City Planning at Wurnster University
    The Department of City Planning is excited to announce a summer lecture series that will be focusing on budgeting issues that concern local residents and municipalities. Financial management is one of the most important duties of local government's operations. We hope to improve the status of budgeting at the local government level across the nation through community involvement and participation. All lectures will be held in the Hayston Building on the Wurnster campus.
    > Monday, February 1, 6:00 P.M., Room 401
    Speaker: Tim Powell, Professor of Policy Analysis at Wurnster University
    Strategic Planning—Learn how to develop budgets in order to monitor progress toward community goals and successful outcomes.
    > Wednesday, February 3, 7:00 P.M., Room 305
    Speaker: Melissa Simmons, Kennedy Institute for Policy Making
    Focusing on Our Children—Studies show that building playgrounds and sports facilities for children helps make better communities.
    > Monday, February 8, 6:00 P.M., Room 202
    Speaker: Hank Ross, Michigan Municipal League
    Managing Our County's Parks—Learn how to preserve our local parks as a valuable community resource.
    > Wednesday, February 10, 5:30 P.M., Room 404
    Speaker: Scott Watson, Executive Director, Local Government Academy
    Economic Opportunities and Local Ecology—Economic opportunity is often accompanied by potential risks to the surrounding ecosystem, and balancing the two can be difficult.
    Please contact Patricia Flores at pflores@wurnster.edu for additional information.
    To: Patricia Flores (pflores@wurnster.edu)
    From: Jake Patterson (jpatterson@wurnster.edu)
    Subject: Lecture Series
    Date: January 24
    Dear Ms. Flores,
    I work for Facilities Management here at Wurnster University. It was recently brought to my attention that there is a scheduling conflict concerning one of your lecture dates. Room 305 has been reserved for every Wednesday this semester by the Wurnster Debate Club. Therefore, I'm sorry to inform you that you will need to move the location or the time of this talk. You can visit the facilities management website in order to check the availability of other room locations and reschedule the talk.
    Jake Patterson
    
    Question: According to Mr. Patterson, whose lecture must be rescheduled?
    A. Mr. Watson's
    B. Mr. Ross's
    C. Ms. Simmons's
    D. Mr. Powell's
"""
    sample_Triple_Passage = """
    Important Notice
    Dear Castelli customers,
    Our quality assurance team has revealed that five hundred jars of Castelli's Classic Spaghetti Sauce do not meet our high standards of product quality.
    The defect has been caused by an improper seal on the lid of the jar and may have resulted in the contents spoiling due to contact with air. We are currently warning customers not to eat this product.
    What you should do: If you have already purchased a jar of Castelli's Classic Spaghetti Sauce, please send an e-mail to our Customer Service Department at cs@castellifood.com. One of our employees will provide you with a product replacement voucher. Please include your name, full address, phone number, and the product's serial number in the e-mail. Customers will receive a $12 voucher for each jar purchased. Please do not try to get a refund for this product at a retailer.
    Please remember that no other Castelli food products are affected. We encourage you to continue purchasing our products.

    To: cs@castellifood.com
    From: Tony Hester (tonyhester21@webzit.com)
    Date: March 29
    Subject: Replacement Voucher

    To whom it may concern,
    My name is Tony Hester and I appreciate the precautionary step. Around two weeks ago I purchased two jars of Castelli's Classic Spaghetti Sauce from an Ace grocery store in Hermantown, Minnesota. A week later, I purchased one more jar of it at the same place.

    I have attached the image file of both receipts to this e-mail. I would like to receive a product replacement voucher for these defective products. My address is:
    Tony Hester
    27 Bloom Street
    Hermantown, MN 55811

    I look forward to receiving a reply soon.
    Tony Hester

    To: Tony Hester (tonyhester21@webzit.com)
    From: cs@castellifood.com
    Date: March 30
    Subject: Voucher

    Dear Mr. Hester,
    Thank you very much for contacting Castelli Foods. We are committed to ensuring that our customers can continue to rely on the Castelli line of quality foods for all their dining needs. As such, we are happy to provide you with three vouchers for the cans of Classic Spaghetti Sauce you recently purchased. Please find the vouchers enclosed.

    In addition to the vouchers for the Classic Spaghetti Sauce, we would like to offer you vouchers for our new line of linguini and spaghetti pasta, Pasta Prima. Please accept these as another way for us to say that we are sorry, and we hope that you continue to turn to us for delicious Italian flavors.

    Sincerely,
    Jan Olson
    Customer Care Specialist

    Question: According to the notice, what is NOT mentioned as advice for customers?
    A. Avoiding consuming the product
    B. Reporting on the product
    C. Returning the product to a store
    D. Purchasing other Castelli products
"""

    scenarios = [
        ("KỊCH BẢN 1 (Single Passage)", sample_Single_Passage),
        ("KỊCH BẢN 2 (Double Passage)", sample_Double_Passage),
        ("KỊCH BẢN 3 (Triple Passage)", sample_Triple_Passage)
    ]

    for name, transcript in scenarios:
        logger.info("\n" + "=" * 80)
        logger.info(f"--- ĐANG CHẠY: {name} ---")

        validate_result = pre_validate_part7_context(transcript)
        if validate_result != "1":
            print(f"\n--- KẾT QUẢ {name} ---")
            print(validate_result)
            continue

        input_data = f"Hãy phân tích bài đọc hiểu sau đây và chọn đáp án chính xác nhất:\n{transcript}"
        
        try:
            res = await toeic_agent.ainvoke({"input": input_data})
            output = format_answer_only(extract_text(res))
            
            print(f"\n--- KẾT QUẢ {name} ---")
            print(output if output else "Agent không trả về nội dung. Kiểm tra API hoặc Prompt.")
            
        except Exception as e:
            logger.error(f"Lỗi thực thi {name}: {e}")

# Chạy hàm main bất đồng bộ
if __name__ == "__main__":
    asyncio.run(main())
    os._exit(0)