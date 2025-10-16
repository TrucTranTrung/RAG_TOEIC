import asyncio
import os
import pytest

from src.core.langchain_root_agent import get_root_agent, TOEICRequest, InputType


async def _run_once():
    agent = get_root_agent()
    ok = await agent.initialize()
    assert ok is True

    req = TOEICRequest(
        request_id="ut_trace_001",
        input_data="Choose the correct answer to complete the sentence.",
        input_type=InputType.TEXT,
        session_id="ut_session"
    )
    resp = await agent.process_request(req)
    assert resp.request_id == req.request_id
    # In trạng thái và full trace để quan sát
    agent.print_status()
    agent.print_full_trace()


@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY missing - Gemini LLM is mandatory")
def test_root_agent_trace():
    asyncio.run(_run_once())


