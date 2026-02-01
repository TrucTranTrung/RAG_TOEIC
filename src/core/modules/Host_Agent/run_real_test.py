# run_real_test.py
# -*- coding: utf-8 -*-
"""
Real Integration Test for Host Agent

File này test Host Agent với TẤT CẢ các agent con đã implement trong project.
Sử dụng Gemini model cho tất cả agents (thay vì Mistral).

Cách chạy:
    cd src/core/modules/Host_Agent
    python run_real_test.py
    
    # Hoặc với debug mode
    python run_real_test.py --debug
    
    # Interactive mode
    python run_real_test.py -i
    
    # Test specific part
    python run_real_test.py --part part5

Yêu cầu:
    - GOOGLE_API_KEY trong config/.env
    - Các agent modules: Agent_Part1, Agent_Part2, Agent_Part3, Agent_Part5, Agent_Part6
"""

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# ==================== PATH SETUP ====================
# Thêm root project vào path để import các modules
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent  # src/core/modules/Host_Agent -> project root

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("RealTest")

# Reduce noise from other loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)


# ==================== TEST CASES ====================
@dataclass
class TestCase:
    name: str
    part: str
    input_text: str
    expected_keywords: list  # Keywords that should appear in response
    has_image: bool = False
    has_audio: bool = False
    image_path: Optional[str] = None
    audio_path: Optional[str] = None


TEST_CASES = [
    # ========== PART 5 - Grammar ==========
    TestCase(
        name="Part5_PastTense",
        part="part5",
        input_text="""
The manager _______ the report yesterday.
(A) submit
(B) submits  
(C) submitted
(D) submitting
""",
        expected_keywords=["(C)", "submitted", "quá khứ", "yesterday"]
    ),
    
    TestCase(
        name="Part5_Adverb",
        part="part5",
        input_text="""
The new policy has been _______ implemented across all departments.
(A) success
(B) successful
(C) successfully
(D) succeed
""",
        expected_keywords=["(C)", "successfully", "trạng từ", "adverb"]
    ),
    
    TestCase(
        name="Part5_Preposition",
        part="part5",
        input_text="""
All employees are encouraged _______ attend the training session.
(A) for
(B) to
(C) at
(D) with
""",
        expected_keywords=["(B)", "to", "encourage", "to-infinitive"]
    ),
    
    # ========== PART 6 - Text Completion ==========
    TestCase(
        name="Part6_Email",
        part="part6",
        input_text="""
To: All Staff
From: HR Department
Subject: Office Renovation

Dear colleagues,

We would like to inform you that the office renovation will _______ next Monday.
All employees are requested to work from home during this period.

(A) begin
(B) beginning
(C) began
(D) begun

Thank you for your understanding.
""",
        expected_keywords=["(A)", "begin", "will", "tương lai"]
    ),
    
    # ========== PART 2 - Question Response ==========
    TestCase(
        name="Part2_WhereQuestion",
        part="part2",
        input_text="""
Where is the meeting room?
(A) At 3 o'clock.
(B) On the second floor.
(C) Yes, I attended the meeting.
""",
        expected_keywords=["(B)", "second floor", "địa điểm", "where"]
    ),
    
    TestCase(
        name="Part2_WhoQuestion",
        part="part2",
        input_text="""
Who is the new manager?
(A) Yes, he is.
(B) It's on the second floor.
(C) Mr. Johnson from the sales department.
""",
        expected_keywords=["(C)", "Mr. Johnson", "người"]
    ),
    
    # ========== CHITCHAT ==========
    TestCase(
        name="Chitchat_Greeting",
        part="chitchat",
        input_text="Xin chào bạn!",
        expected_keywords=["chào", "TOEIC", "giúp"]
    ),
    
    TestCase(
        name="Chitchat_Thanks",
        part="chitchat",
        input_text="Cảm ơn bạn nhiều!",
        expected_keywords=["không có gì", "vui", "giúp"]
    ),
    
    # ========== AMBIGUOUS ==========
    TestCase(
        name="Ambiguous_NoOptions",
        part="unknown",
        input_text="What is machine learning?",
        expected_keywords=["Part", "không chắc"]
    ),
]

# Test cases với file (Part 1, Part 2, Part 3/4) - cần có file thực
TEST_CASES_WITH_FILES = [
    TestCase(
        name="Part1_Image",
        part="part1",
        input_text="""
(A) The man is using a screwdriver.
(B) The man is hammering something.
(C) The man is making the frame by hand.
(D) The man is wearing protective glasses.
""",
        expected_keywords=["Đáp án"],
        has_image=True,
        image_path="src/core/modules/data_test/mo_ta_tranh1.png"
    ),
    
    # Part 2: Audio + 3 options
    TestCase(
        name="Part2_Audio_WhereQuestion",
        part="part2",
        input_text="""
Where is the meeting room?
(A) At 3 o'clock.
(B) On the second floor.
(C) Yes, I attended the meeting.
""",
        expected_keywords=["(B)", "second floor", "địa điểm"],
        has_audio=True,
        audio_path="src/core/modules/data_test/4luSC.mp3"
    ),
    
    TestCase(
        name="Part2_Audio_WhoQuestion",
        part="part2",
        input_text="""
Who is responsible for the project?
(A) Yes, they are.
(B) Mr. Smith from accounting.
(C) Next Monday.
""",
        expected_keywords=["(B)", "Mr. Smith", "người"],
        has_audio=True,
        audio_path="src/core/modules/data_test/6KXxh.mp3"
    ),
    
    TestCase(
        name="Part3_Conversation",
        part="part3",
        input_text="""
What does the woman say about her phone service?
(A) She is unhappy with Z Mobile's service.
(B) She pays a monthly fee of 70 dollars.
(C) She gets 700 unlimited minutes with everyone.
(D) She recently moved to Canada for free calls.
""",
        expected_keywords=["Đáp án"],
        has_audio=True,
        audio_path="src/core/modules/data_test/Ld1lt.mp3"
    ),
]


# ==================== HOST AGENT SETUP ====================

def load_env():
    """Load environment variables from config/.env"""
    from dotenv import load_dotenv
    
    env_paths = [
        PROJECT_ROOT / "config" / ".env",
        Path("config/.env"),
        Path(".env"),
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"✅ Loaded env from: {env_path}")
            
            # Setup LangSmith tracing
            from .langsmith_setup import setup_langsmith
            if setup_langsmith():
                logger.info("✅ LangSmith tracing enabled")
            
            return True
    
    logger.warning("⚠️ No .env file found")
    return False


def create_gemini_models():
    """Create 3 Gemini model instances with different API keys"""
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    models = {}
    
    # Key 1 for Part 1, Part 2
    api_key1 = os.environ.get("GOOGLE_API_KEY1")
    if api_key1:
        models["gemini1"] = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=api_key1,
            temperature=0.0,
            max_tokens=4096
        )
        logger.info("✅ Created Gemini model 1 (Part 1, 2)")
    else:
        raise ValueError("GOOGLE_API_KEY1 not found in environment")
    
    # Key 2 for Part 3, Part 4
    api_key2 = os.environ.get("GOOGLE_API_KEY2")
    if api_key2:
        models["gemini2"] = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=api_key2,
            temperature=0.0,
            max_tokens=4096
        )
        logger.info("✅ Created Gemini model 2 (Part 3, 4)")
    else:
        raise ValueError("GOOGLE_API_KEY2 not found in environment")
    
    # Key 3 for Part 5, Part 6
    api_key3 = os.environ.get("GOOGLE_API_KEY3")
    if api_key3:
        models["gemini3"] = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=api_key3,
            temperature=0.0,
            max_tokens=4096
        )
        logger.info("✅ Created Gemini model 3 (Part 5, 6)")
    else:
        raise ValueError("GOOGLE_API_KEY3 not found in environment")
    
    return models


def create_host_agent_with_gemini():
    """
    Create Host Agent với 3 API keys phân bổ cho 6 agents
    
    Distribution:
    - gemini1: Part 1, Part 2
    - gemini2: Part 3, Part 4  
    - gemini3: Part 5, Part 6
    
    Returns:
        HostAgent instance
    """
    from src.core.modules.Host_Agent.registry import AgentRegistry, AgentConfig
    from src.core.modules.Host_Agent.graph import HostAgent
    
    # Import agent classes
    from src.core.modules.Agent_Part1.Agent1 import TOEICPart1Agent
    from src.core.modules.Agent_Part2.Agent2 import TOEICPart2Agent
    from src.core.modules.Agent_Part3.Agent3 import TOEICListeningAgent
    from src.core.modules.Agent_Part5.Agent_Part5 import LanguageAgentPart5
    from src.core.modules.Agent_Part6.Part6 import LanguageAgentPart6
    
    # Create 3 Gemini models with different keys
    models = create_gemini_models()
    
    # Create registry
    registry = AgentRegistry()
    
    # Register agents with distributed keys
    agent_configs = {
        # Key 1: Part 1, Part 2
        "part1": AgentConfig(
            name="TOEIC Part 1 - Image Description",
            agent_class=TOEICPart1Agent,
            model_key="gemini1",
            description="Phân tích hình ảnh với 4 lựa chọn",
            enabled=True
        ),
        "part2": AgentConfig(
            name="TOEIC Part 2 - Question-Response",
            agent_class=TOEICPart2Agent,
            model_key="gemini1",
            description="Câu hỏi - Đáp án với 3 lựa chọn",
            enabled=True
        ),
        # Key 2: Part 3, Part 4
        "part3": AgentConfig(
            name="TOEIC Part 3 - Conversations",
            agent_class=TOEICListeningAgent,
            model_key="gemini2",
            description="Nghe hội thoại",
            enabled=True
        ),
        "part4": AgentConfig(
            name="TOEIC Part 4 - Talks",
            agent_class=TOEICListeningAgent,
            model_key="gemini2",
            description="Nghe bài nói",
            enabled=True
        ),
        # Key 3: Part 5, Part 6
        "part5": AgentConfig(
            name="TOEIC Part 5 - Grammar",
            agent_class=LanguageAgentPart5,
            model_key="gemini3",
            description="Ngữ pháp - Điền từ",
            enabled=True
        ),
        "part6": AgentConfig(
            name="TOEIC Part 6 - Text Completion",
            agent_class=LanguageAgentPart6,
            model_key="gemini3",
            description="Hoàn thành đoạn văn",
            enabled=True
        ),
    }
    
    # Register all
    registry.register_many(agent_configs)
    
    # Set models (3 different keys)
    registry.set_models(models)
    
    # Create host agent
    host = HostAgent(registry=registry)
    
    logger.info(f"✅ Created Host Agent with {registry.agent_count} agents")
    logger.info(f"   Enabled: {registry.list_enabled()}")
    logger.info(f"   Models: {list(models.keys())}")
    
    return host


# ==================== TEST RUNNER ====================

async def run_test_case(host_agent, test_case: TestCase, debug: bool = False) -> dict:
    """
    Run a single test case
    
    Returns:
        dict with: success, response, error, duration
    """
    import time
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 Test: {test_case.name} (Expected: {test_case.part})")
    logger.info(f"{'='*60}")
    
    # Prepare input
    input_text = test_case.input_text
    
    if test_case.has_image and test_case.image_path:
        input_text = f"Image: {test_case.image_path}\n\n{input_text}"
        logger.info(f"📷 Image: {test_case.image_path}")
    
    if test_case.has_audio and test_case.audio_path:
        input_text = f"Audio: {test_case.audio_path}\n\n{input_text}"
        logger.info(f"🎧 Audio: {test_case.audio_path}")
    
    logger.info(f"📝 Input:\n{input_text.strip()[:200]}...")
    
    start_time = time.time()
    
    try:
        response = await host_agent.ainvoke(input_text, debug=debug)
        duration = time.time() - start_time
        
        logger.info(f"\n📤 Response ({duration:.2f}s):\n{response[:500]}...")
        
        # Check expected keywords
        response_lower = response.lower()
        found_keywords = [kw for kw in test_case.expected_keywords if kw.lower() in response_lower]
        missing_keywords = [kw for kw in test_case.expected_keywords if kw.lower() not in response_lower]
        
        success = len(missing_keywords) == 0
        
        if success:
            logger.info(f"✅ PASSED - Found all expected keywords")
        else:
            logger.warning(f"⚠️ PARTIAL - Missing keywords: {missing_keywords}")
        
        return {
            "success": success,
            "response": response,
            "duration": duration,
            "found_keywords": found_keywords,
            "missing_keywords": missing_keywords,
            "error": None
        }
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ FAILED - Error: {e}")
        return {
            "success": False,
            "response": None,
            "duration": duration,
            "error": str(e)
        }


async def run_all_tests(host_agent, include_file_tests: bool = False, debug: bool = False):
    """Run all test cases"""
    
    logger.info("\n" + "🚀" * 20)
    logger.info("RUNNING ALL TESTS")
    logger.info("🚀" * 20)
    
    test_cases = TEST_CASES.copy()
    
    if include_file_tests:
        # Check if files exist
        for tc in TEST_CASES_WITH_FILES:
            file_path = tc.image_path or tc.audio_path
            if file_path and Path(file_path).exists():
                test_cases.append(tc)
            else:
                logger.warning(f"⚠️ Skipping {tc.name} - file not found: {file_path}")
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        result = await run_test_case(host_agent, test_case, debug)
        result["test_name"] = test_case.name
        result["expected_part"] = test_case.part
        results.append(result)
        
        # Rate limiting: wait 60s between tests (except last one)
        if i < len(test_cases) - 1:
            logger.info(f"⏳ Waiting 60s to avoid rate limit... ({i+1}/{len(test_cases)} done)")
            await asyncio.sleep(60)
    
    # Summary
    print_summary(results)
    
    return results


def print_summary(results: list):
    """Print test summary"""
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed
    
    for r in results:
        status = "✅" if r["success"] else "❌"
        duration = f"{r['duration']:.2f}s" if r["duration"] else "N/A"
        logger.info(f"{status} {r['test_name']:<30} ({duration})")
        if r.get("missing_keywords"):
            logger.info(f"   Missing: {r['missing_keywords']}")
        if r.get("error"):
            logger.info(f"   Error: {r['error'][:50]}")
    
    logger.info("-" * 60)
    logger.info(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    logger.info(f"Success Rate: {passed/len(results)*100:.1f}%")
    logger.info("=" * 60)


async def run_interactive(host_agent, debug: bool = False):
    """Interactive test mode"""
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 TOEIC Host Agent - Interactive Mode")
    logger.info("=" * 60)
    logger.info("Commands:")
    logger.info("  /quit - Exit")
    logger.info("  /debug - Toggle debug mode")
    logger.info("  /list - List test cases")
    logger.info("  /run <name> - Run specific test")
    logger.info("  /all - Run all tests")
    logger.info("=" * 60)
    
    while True:
        try:
            user_input = input("\n📝 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["/quit", "/exit", "/q"]:
                logger.info("👋 Goodbye!")
                break
            
            if user_input.lower() == "/debug":
                debug = not debug
                logger.info(f"🔧 Debug mode: {'ON' if debug else 'OFF'}")
                continue
            
            if user_input.lower() == "/list":
                logger.info("📋 Available tests:")
                for tc in TEST_CASES:
                    logger.info(f"  - {tc.name} ({tc.part})")
                continue
            
            if user_input.lower() == "/all":
                await run_all_tests(host_agent, debug=debug)
                continue
            
            if user_input.lower().startswith("/run "):
                test_name = user_input[5:].strip()
                test_case = next((tc for tc in TEST_CASES if tc.name.lower() == test_name.lower()), None)
                if test_case:
                    await run_test_case(host_agent, test_case, debug)
                else:
                    logger.warning(f"Test not found: {test_name}")
                continue
            
            # Normal input
            logger.info("🤖 Processing...")
            response = await host_agent.ainvoke(user_input, debug=debug)
            logger.info(f"\n🤖 Response:\n{response}")
            
        except KeyboardInterrupt:
            logger.info("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"❌ Error: {e}")


async def run_single_part(host_agent, part: str, debug: bool = False):
    """Run all tests for a specific part"""
    
    test_cases = [tc for tc in TEST_CASES if tc.part == part]
    
    if not test_cases:
        logger.warning(f"No tests found for part: {part}")
        return
    
    logger.info(f"\n🎯 Running {len(test_cases)} tests for {part}")
    
    for tc in test_cases:
        await run_test_case(host_agent, tc, debug)
        await asyncio.sleep(1)


# ==================== MAIN ====================

async def main():
    parser = argparse.ArgumentParser(description="Real Integration Test for Host Agent")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--part", "-p", type=str, help="Test specific part (part1-part6)")
    parser.add_argument("--with-files", action="store_true", help="Include tests with image/audio files")
    
    args = parser.parse_args()
    
    # Load environment
    load_env()
    
    try:
        # Create host agent
        logger.info("\n🔧 Setting up Host Agent...")
        host_agent = create_host_agent_with_gemini()
        
        # Run based on mode
        if args.interactive:
            await run_interactive(host_agent, args.debug)
        elif args.part:
            await run_single_part(host_agent, args.part, args.debug)
        else:
            await run_all_tests(host_agent, args.with_files, args.debug)
            
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        logger.info("💡 Make sure GOOGLE_API_KEY is set in config/.env")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("💡 Make sure you're running from the correct directory")
        logger.info(f"   Current: {Path.cwd()}")
        logger.info(f"   Expected: {PROJECT_ROOT}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())