import logging
import asyncio

from typing import List
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate, BasePromptTemplate

# Import từ module base
from ..Agent_Base import BaseAgent
from langchain_community.llms import LlamaCpp
from ..Agent_Base.llm_wrapper import LlamaCppChatWrapper
from pathlib import Path
# from .models import llm_mistral

# Mock BaseAgent cho demo (trong thực tế sẽ import từ Agent_Base)
# from abc import ABC, abstractmethod
# from langchain_core.runnables import Runnable
# from langchain_core.language_models.chat_models import BaseChatModel
# from langchain.agents import create_agent

# Import tools và utils
from .Agent_Part5_tools import vocab_search, grammar_pattern_lookup, collocation_check, verb_form_analyzer
from .Agent_Part5_utils import clean_part5_question, extract_multiple_part5_questions, extract_output_text
from .Agent_Part5_prompts import prompt_string


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="config/.env")


# --- ĐỊNH NGHĨA AGENT PART 5 ---
class LanguageAgentPart5(BaseAgent):
    """
    Agent này dùng để xử lý TOEIC Part 5 - Grammar Questions.
    Chuyên về: verb forms, word choice, prepositions, conjunctions, etc.
    """

    def _get_tools(self) -> List[BaseTool]:
        """
        Cung cấp danh sách các tools CHUYÊN BIỆT cho Reading Part 5.
        """
        logger.info(
            "LanguageAgentPart5: Cung cấp tools [vocab_search, grammar_pattern_lookup, collocation_check, verb_form_analyzer]")
        return [
            vocab_search,
            grammar_pattern_lookup,
            collocation_check,
            verb_form_analyzer
        ]

    def _get_prompt(self) -> BasePromptTemplate:
        """
        Cung cấp một prompt template CỤ THỂ cho LanguageAgentPart5.
        """
        logger.info(
            "LanguageAgentPart5: Cung cấp prompt chuyên về Grammar Part 5")
        return PromptTemplate.from_template(prompt_string)


# --- CHẠY DEMO ---
async def main():
    """Hàm chạy chính (bất đồng bộ) để test agent."""
    logger.info("="*80)
    logger.info("🚀 Khởi tạo Model cho Agent Part 5")
    logger.info("="*80)
    model = None
    # model_name = "Local LlamaCpp"
    try:

        # Try multiple possible paths
        possible_paths = [
            "./models/toeic_qwen_q4.gguf",
            "../../../../models/toeic_qwen_q4.gguf",
            "models/toeic_qwen_q4.gguf"
        ]

        model_path = None
        for path in possible_paths:
            if Path(path).exists():
                model_path = path
                break

        if model_path:
            logger.info(f"\n📂 Found model: {model_path}")
            logger.info("🔄 Loading LlamaCpp...")

            llama_llm = LlamaCpp(
                model_path=model_path,
                n_ctx=2048,
                n_threads=4,
                temperature=0.3,
                max_tokens=2000,
                verbose=False,
            )

            model = LlamaCppChatWrapper(llm=llama_llm)
            logger.info("✅ LlamaCpp loaded successfully!")

        else:
            logger.info("\n⚠️  GGUF model not found in:")
            for p in possible_paths:
                logger.info(f"   - {p}")
            logger.info("\n📥 To download:")
            logger.info("   mkdir -p models")
            logger.info("   cd models")
            logger.info(
                "   wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf")
            logger.info(
                "   mv qwen2.5-0.5b-instruct-q4_k_m.gguf toeic_qwen_q4.gguf")
    except ImportError as e:
        logger.info(f"\n⚠️  llama-cpp-python not installed: {e}")
        logger.info("📦 Install with: pip install llama-cpp-python")
    except Exception as e:
        logger.info(f"\n⚠️  Error loading LlamaCpp: {e}")

    # --- Initialize LanguageAgentPart5 with Model ---
    print("\n--- Khởi tạo LanguageAgentPart5 với Model ---")

    try:
        grammar_agent = LanguageAgentPart5(model=model)
    except Exception as e:
        logger.error(f"Không thể khởi tạo agent: {e}")
        return

    print("\n--- Bắt đầu chạy LanguageAgentPart5 (Async) ---")

    # --- DỮ LIỆU VÍ DỤ PART 5 ---
    PART_5_EXAMPLES = """
    101. The manager _______ the report yesterday.
    (A) submit
    (B) submits
    (C) submitted
    (D) submitting

    102. We are looking forward to _______ from you soon.
    (A) hear
    (B) hearing
    (C) heard
    (D) hears

    103. The project was completed _______ schedule.
    (A) on
    (B) in
    (C) at
    (D) by
    """

    # Test với một câu đơn
    SINGLE_QUESTION = """
    The annual departmental meeting will be held next month. 
    All employees are encouraged to _______ this important event.
    (A) attendance
    (B) attend
    (C) attending
    (D) attends
    """

    # Clean input
    cleaned_input = clean_part5_question(SINGLE_QUESTION)
    print(f"\n--- Input đã clean ---")
    print(cleaned_input)

    try:
        # Sử dụng .ainvoke() (bất đồng bộ)
        result = await grammar_agent.ainvoke({
            "input": cleaned_input
        })

        print("\n" + "="*80)
        print("--- KẾT QUẢ TỪ LANGUAGEAGENTPART5 ---")
        print("="*80)
        print(extract_output_text(result['output']))
        print("="*80)

    except Exception as e:
        logger.error(f"Lỗi khi chạy agent: {e}")
        import traceback
        traceback.print_exc()

    # Test với nhiều câu
    print("\n\n" + "="*80)
    print("--- TEST VỚI NHIỀU CÂU ---")
    print("="*80)

    try:
        # Extract multiple questions
        questions = extract_multiple_part5_questions(PART_5_EXAMPLES)
        print(f"\nĐã trích xuất {len(questions)} câu hỏi:")

        for i, q in enumerate(questions, 1):
            print(f"\n--- Câu {i} ---")
            print(q)

        # Process all questions at once
        all_questions_text = "\n\n".join(questions)

        result_multiple = await grammar_agent.ainvoke({
            "input": all_questions_text
        })

        print("\n" + "="*80)
        print("--- KẾT QUẢ CHO NHIỀU CÂU ---")
        print("="*80)
        print(extract_output_text(result_multiple['output']))
        print("="*80)

    except Exception as e:
        logger.error(f"Lỗi khi chạy agent với nhiều câu: {e}")
        import traceback
        traceback.print_exc()


# Chạy hàm main bất đồng bộ
if __name__ == "__main__":
    # Test utils trước
    print("="*80)
    print("TESTING UTILS")
    print("="*80)

    test_input = """
    The manager _______ the report yesterday.
    (A) submit
    (B) submits
    (C) submitted
    (D) submitting
    """

    cleaned = clean_part5_question(test_input)
    print("Original:")
    print(test_input)
    print("\nCleaned:")
    print(cleaned)

    print("\n" + "="*80)
    print("RUNNING AGENT")
    print("="*80)

    # Run async main
    asyncio.run(main())
