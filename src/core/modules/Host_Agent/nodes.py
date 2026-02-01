# nodes.py
# -*- coding: utf-8 -*-
"""
Graph Nodes for Host Agent

Each node is a pure function that:
- Takes HostAgentState as input
- Returns updated HostAgentState (immutable update pattern)
- Has single responsibility
"""

import re
import random
import logging
from typing import Dict, Any

from .state import HostAgentState
from .detection import MultiLevelDetector
from .registry import AgentRegistry

logger = logging.getLogger(__name__)

# Initialize detector (singleton)
detector = MultiLevelDetector()


# ========== NODE 1: INPUT PREPROCESSING ==========
def preprocess_input_node(state: HostAgentState) -> HostAgentState:
    """
    Extract and normalize input
    
    Responsibilities:
    - Parse file paths from message (image/audio)
    - Clean text (remove file path lines)
    - Initialize retry counters
    
    Args:
        state: Current graph state
    
    Returns:
        Updated state with:
        - raw_input: cleaned text
        - image_path: extracted image path (if any)
        - audio_path: extracted audio path (if any)
        - retry_count: initialized to 0
    """
    
    messages = state.get("messages", [])
    if not messages:
        return {
            **state,
            "error": "No input messages",
            "final_answer": "Không có tin nhắn đầu vào."
        }
    
    # Handle both message object (.content) and dict format (["content"])
    last_message = messages[-1]
    if hasattr(last_message, 'content'):
        raw_input = last_message.content
    elif isinstance(last_message, dict):
        raw_input = last_message.get("content", str(last_message))
    else:
        raw_input = str(last_message)
    
    # Ensure raw_input is string (could be list for multi-part messages)
    if isinstance(raw_input, list):
        # Extract text from list of content parts
        text_parts = []
        for part in raw_input:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                text_parts.append(part.get("text", part.get("content", str(part))))
            elif hasattr(part, 'text'):
                text_parts.append(part.text)
            else:
                text_parts.append(str(part))
        raw_input = "\n".join(text_parts)
    elif not isinstance(raw_input, str):
        raw_input = str(raw_input)
    
    # Extract file paths with multiple patterns
    # Pattern 1: "image: path.jpg" or "ảnh: path.png"
    image_patterns = [
        r'(?:image|ảnh|hình)[:\s]+([^\s]+\.(?:jpg|png|jpeg|gif|webp))',
        r'([^\s]+\.(?:jpg|png|jpeg|gif|webp))'  # Fallback: any image file
    ]
    
    audio_patterns = [
        r'(?:audio|âm thanh|file âm thanh)[:\s]+([^\s]+\.(?:mp3|wav|m4a|ogg))',
        r'([^\s]+\.(?:mp3|wav|m4a|ogg))'  # Fallback: any audio file
    ]
    
    image_match = None
    audio_match = None
    
    for pattern in image_patterns:
        image_match = re.search(pattern, raw_input, re.IGNORECASE)
        if image_match:
            break
    
    for pattern in audio_patterns:
        audio_match = re.search(pattern, raw_input, re.IGNORECASE)
        if audio_match:
            break
    
    # Clean text (remove file path lines)
    clean_input = raw_input
    if image_match:
        clean_input = clean_input.replace(image_match.group(0), '')
    if audio_match:
        clean_input = clean_input.replace(audio_match.group(0), '')
    
    clean_input = clean_input.strip()
    
    # Logging
    logger.info(f"[Preprocess] Input length: {len(clean_input)} chars")
    if image_match:
        logger.info(f"[Preprocess] Image detected: {image_match.group(1)}")
    if audio_match:
        logger.info(f"[Preprocess] Audio detected: {audio_match.group(1)}")
    
    return {
        **state,
        "raw_input": clean_input,
        "image_path": image_match.group(1) if image_match else None,
        "audio_path": audio_match.group(1) if audio_match else None,
        "retry_count": 0,
        "max_retries": state.get("max_retries", 2),
        "metadata": {
            **state.get("metadata", {}),
            "preprocessing_complete": True
        }
    }


# ========== NODE 2: CHITCHAT DETECTION ==========
def detect_chitchat_node(state: HostAgentState) -> HostAgentState:
    """
    Quick chitchat detection using pattern matching
    
    Detects:
    - Greetings (xin chào, hello, hi...)
    - Thanks (cảm ơn, thank you...)
    - Small talk (hôm nay thế nào, how are you...)
    
    Early exit if TOEIC markers found (A/B/C/D options, blanks)
    
    Args:
        state: Current graph state
    
    Returns:
        Updated state with:
        - is_chitchat: bool
        - chitchat_response: canned response (if chitchat)
        - final_answer: set if chitchat (to skip further processing)
    """
    
    raw_input = state.get("raw_input", "").lower()
    
    # Early exit: Check for TOEIC markers (if found, definitely not chitchat)
    toeic_markers = r'\([A-D]\)|_{2,}|\[BLANK\]|---'
    if re.search(toeic_markers, raw_input, re.IGNORECASE):
        logger.info("[Chitchat] TOEIC markers found → Not chitchat")
        return {**state, "is_chitchat": False}
    
    # ========== GREETING DETECTION ==========
    greetings = [
        "xin chào", "chào bạn", "chào", "hello", "hi ", "hey",
        "good morning", "good afternoon", "good evening"
    ]
    
    if any(g in raw_input for g in greetings):
        responses = [
            "Xin chào! Tôi là trợ lý TOEIC. Bạn cần giúp gì hôm nay?",
            "Chào bạn! Tôi có thể giúp bạn luyện TOEIC Part 1-7. Bạn muốn luyện phần nào?",
            "Hello! Tôi sẵn sàng hỗ trợ bạn ôn thi TOEIC. Hãy gửi câu hỏi nhé!"
        ]
        response = random.choice(responses)
        
        logger.info("[Chitchat] Greeting detected")
        return {
            **state,
            "is_chitchat": True,
            "chitchat_response": response,
            "final_answer": response
        }
    
    # ========== THANKS DETECTION ==========
    thanks = ["cảm ơn", "cám ơn", "thanks", "thank you", "thankss", "tks"]
    
    if any(t in raw_input for t in thanks):
        responses = [
            "Không có gì! Chúc bạn học tốt!",
            "Rất vui được giúp đỡ bạn!",
            "Không có chi! Nếu cần hỗ trợ thêm, cứ hỏi nhé!"
        ]
        response = random.choice(responses)
        
        logger.info("[Chitchat] Thanks detected")
        return {
            **state,
            "is_chitchat": True,
            "chitchat_response": response,
            "final_answer": response
        }
    
    # ========== SMALL TALK DETECTION ==========
    small_talk = [
        "khỏe không", "thế nào", "how are you", "what's up",
        "bạn là ai", "who are you", "giới thiệu"
    ]
    
    if any(s in raw_input for s in small_talk):
        responses = [
            "Tôi là trợ lý TOEIC của bạn! Hãy gửi câu hỏi Part 1-7 để tôi giúp nhé.",
            "Tôi khỏe! Sẵn sàng giúp bạn luyện TOEIC. Bạn muốn bắt đầu với phần nào?"
        ]
        response = random.choice(responses)
        
        logger.info("[Chitchat] Small talk detected")
        return {
            **state,
            "is_chitchat": True,
            "chitchat_response": response,
            "final_answer": response
        }
    
    logger.info("[Chitchat] Not chitchat → Continue to detection")
    return {**state, "is_chitchat": False}


# ========== NODE 3: PART DETECTION ==========
def detect_part_node(state: HostAgentState) -> HostAgentState:
    """
    Classify input into TOEIC parts using multi-level detector
    
    Detection priority:
    1. Media-based (image → Part 1, audio → Part 2/3/4)
    2. Format-based (blanks, paragraphs, etc.)
    3. Keyword-based (fallback)
    
    Args:
        state: Current graph state
    
    Returns:
        Updated state with:
        - detected_part: "part1", "part2", ..., "unknown"
        - detection_confidence: 0.0 - 1.0
        - detection_reasoning: explanation
        - detection_method: "media" / "format" / "keyword"
    """
    
    logger.info("[Detection] Running multi-level detection...")
    
    result = detector.detect(state)
    
    logger.info(f"[Detection] Result: {result.part} (confidence: {result.confidence:.2%})")
    logger.info(f"[Detection] Method: {result.method}")
    logger.info(f"[Detection] Reasoning: {result.reasoning}")
    
    return {
        **state,
        "detected_part": result.part,
        "detection_confidence": result.confidence,
        "detection_reasoning": result.reasoning,
        "detection_method": result.method
    }


# ========== NODE 4: ROUTING DECISION ==========
def route_decision_node(state: HostAgentState, registry: AgentRegistry) -> HostAgentState:
    """
    Decide routing action based on detection result
    
    Decision logic:
    - confidence < 0.60 or unknown → ask user for clarification
    - agent not found → error message
    - agent disabled → unavailable message
    - success → route to agent
    
    Args:
        state: Current graph state
        registry: AgentRegistry instance (injected via partial)
    
    Returns:
        Updated state with:
        - routing_decision: "execute" / "ask_user" / "error"
        - routed_agent_name: agent key (if executing)
        - final_answer: set if not executing
    """
    
    detected_part = state.get("detected_part")
    confidence = state.get("detection_confidence", 0.0)
    
    # ========== LOW CONFIDENCE / UNKNOWN ==========
    if detected_part == "unknown" or confidence < 0.60:
        logger.warning(f"[Routing] Low confidence ({confidence:.2%}) or unknown part")
        return {
            **state,
            "routing_decision": "ask_user",
            "final_answer": (
                f"Tôi không chắc chắn đây là phần nào của TOEIC "
                f"(độ tin cậy: {confidence:.0%}). "
                f"Bạn có thể cho biết rõ đây là Part mấy không? (Part 1-7)"
            )
        }
    
    # ========== CHECK AGENT CONFIG ==========
    agent_config = registry.get_config(detected_part)
    
    if not agent_config:
        logger.error(f"[Routing] No agent config for '{detected_part}'")
        return {
            **state,
            "routing_decision": "error",
            "error": f"No agent registered for {detected_part}",
            "final_answer": f"Xin lỗi, hệ thống chưa hỗ trợ {detected_part.upper()}."
        }
    
    if not agent_config.enabled:
        logger.warning(f"[Routing] Agent '{detected_part}' is disabled")
        return {
            **state,
            "routing_decision": "error",
            "error": f"Agent {detected_part} is disabled",
            "final_answer": f"Xin lỗi, {agent_config.name} tạm thời không khả dụng."
        }
    
    # ========== SUCCESS: ROUTE TO AGENT ==========
    logger.info(f"[Routing] Routing to: {detected_part} ({agent_config.name})")
    return {
        **state,
        "routed_agent_name": detected_part,
        "routing_decision": "execute"
    }


# ========== NODE 5: EXECUTE CHILD AGENT ==========
async def execute_agent_node(state: HostAgentState, registry: AgentRegistry) -> HostAgentState:
    """
    Execute the routed child agent
    
    Responsibilities:
    - Get agent instance from registry (lazy loading)
    - Prepare input (add file paths if present)
    - Execute agent (async)
    - Handle errors with retry mechanism
    
    Args:
        state: Current graph state
        registry: AgentRegistry instance (injected via partial)
    
    Returns:
        Updated state with:
        - agent_output: raw output from agent
        - final_answer: extracted text answer
        - error: error message (if failed)
        - retry_count: incremented on retry
    """
    
    agent_name = state.get("routed_agent_name")
    if not agent_name:
        logger.warning("[Execute] No agent name in state")
        return state
    
    # ========== GET AGENT INSTANCE ==========
    agent = registry.get_agent(agent_name)
    
    if not agent:
        logger.error(f"[Execute] Failed to get agent instance: {agent_name}")
        return {
            **state,
            "error": f"Failed to initialize {agent_name}",
            "final_answer": "Lỗi hệ thống khi khởi tạo agent."
        }
    
    # ========== PREPARE INPUT ==========
    agent_input_text = state.get("raw_input", "")
    
    # Add file paths if present
    if state.get("image_path"):
        agent_input_text = f"Image: {state['image_path']}\n\n{agent_input_text}"
    if state.get("audio_path"):
        agent_input_text = f"Audio: {state['audio_path']}\n\n{agent_input_text}"
    
    agent_input = {"input": agent_input_text}
    
    logger.info(f"[Execute] Calling {agent_name}...")
    logger.debug(f"[Execute] Input: {agent_input_text[:100]}...")
    
    # ========== EXECUTE AGENT ==========
    try:
        # Execute agent (async)
        result = await agent.ainvoke(agent_input)
        
        logger.info(f"[Execute] {agent_name} completed successfully")
        
        # Extract output text
        output_text = _extract_output_text(result)
        
        # Log output details for debugging
        logger.info(f"[Execute] Output length: {len(output_text)} chars")
        logger.info(f"[Execute] Output preview: {output_text[:300]}...")
        
        return {
            **state,
            "agent_output": result,
            "final_answer": output_text,
            "error": None
        }
        
    except Exception as e:
        return _handle_execution_error(state, agent_name, e)


def _extract_output_text(result: Dict[str, Any]) -> str:
    """
    Extract text from agent result
    
    Handles various output formats:
    - {"output": "text"}
    - {"output": AIMessage(...)}
    - {"output": [{"text": "..."}]}
    - {"output": {"content": "..."}}
    - Direct string
    """
    logger.debug(f"[Extract] Result type: {type(result)}, keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
    
    output = result.get("output", "")
    
    logger.debug(f"[Extract] Output type: {type(output)}")
    
    # Handle string output
    if isinstance(output, str):
        return output.strip()
    
    # Handle AIMessage or BaseMessage objects
    if hasattr(output, 'content'):
        logger.debug(f"[Extract] Found AIMessage with content length: {len(str(output.content))}")
        return str(output.content).strip()
    
    # Handle list output
    if isinstance(output, list) and output:
        first_item = output[0]
        if hasattr(first_item, 'content'):
            return str(first_item.content).strip()
        if isinstance(first_item, dict):
            return first_item.get("text", first_item.get("content", str(first_item))).strip()
        return str(first_item).strip()
    
    # Handle dict output
    if isinstance(output, dict):
        # Try common keys
        for key in ["text", "content", "answer", "response", "result"]:
            if key in output:
                return str(output[key]).strip()
        return str(output).strip()
    
    return str(output).strip()


def _handle_execution_error(state: HostAgentState, agent_name: str, error: Exception) -> HostAgentState:
    """Handle agent execution error with retry logic"""
    
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    
    logger.error(f"[Execute] Error in {agent_name}: {str(error)}")
    
    # Increment retry count first
    new_retry_count = retry_count + 1
    
    # Check if we can still retry (BEFORE incrementing would exceed max)
    if new_retry_count < max_retries:
        logger.info(f"[Execute] Retrying... ({new_retry_count}/{max_retries})")
        return {
            **state,
            "retry_count": new_retry_count,
            "error": f"Retry {new_retry_count}: {str(error)}"
        }
    else:
        # Max retries reached - set final_answer for error
        logger.error(f"[Execute] Max retries reached for {agent_name}")
        return {
            **state,
            "retry_count": new_retry_count,
            "error": f"Agent failed after {max_retries} retries: {str(error)}",
            "final_answer": "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại sau."
        }


# ========== NODE 6: FORMAT RESPONSE ==========
def format_response_node(state: HostAgentState) -> HostAgentState:
    """
    Format final response with optional debug metadata
    
    Adds debug footer if debug_mode is enabled in metadata.
    Also appends AIMessage to messages for LangGraph Studio compatibility.
    
    Args:
        state: Current graph state
    
    Returns:
        Updated state with:
        - final_answer: formatted response
        - messages: with AIMessage appended
        - metadata: updated with formatting_complete flag
    """
    from langchain_core.messages import AIMessage
    
    final_answer = state.get("final_answer", "")
    
    # Add debug metadata if enabled
    add_debug_info = state.get("metadata", {}).get("debug_mode", False)
    
    if add_debug_info and state.get("detected_part"):
        debug_footer = (
            f"\n\n{'='*50}\n"
            f"🔍 Debug Info:\n"
            f"- Detected Part: {state['detected_part']}\n"
            f"- Confidence: {state.get('detection_confidence', 0):.0%}\n"
            f"- Method: {state.get('detection_method', 'N/A')}\n"
            f"- Reasoning: {state.get('detection_reasoning', 'N/A')}\n"
            f"{'='*50}"
        )
        final_answer += debug_footer
    
    # Append AIMessage to messages for LangGraph Studio
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=final_answer))
    
    logger.info("[Format] Response formatted")
    
    return {
        **state,
        "messages": messages,
        "final_answer": final_answer,
        "metadata": {
            **state.get("metadata", {}),
            "formatting_complete": True
        }
    }
