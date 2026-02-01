# state.py
# -*- coding: utf-8 -*-
"""
State Definition for Host Agent Graph

Defines the shared state structure used across all graph nodes.
Uses TypedDict for type safety and IDE support.
"""

from typing import TypedDict, Optional, List, Dict, Any, Literal
from langchain_core.messages import BaseMessage


class HostAgentState(TypedDict):
    """
    Shared state across all graph nodes
    
    Design principles:
    - Immutable updates (return new dict, never mutate)
    - All fields Optional except required ones
    - Clear separation: INPUT → DETECTION → ROUTING → EXECUTION → OUTPUT
    
    Usage in nodes:
        def my_node(state: HostAgentState) -> HostAgentState:
            # Read from state
            raw_input = state.get("raw_input", "")
            
            # Return updated state (immutable)
            return {
                **state,
                "new_field": "new_value"
            }
    """
    
    # ==================== INPUT LAYER ====================
    messages: List[BaseMessage]           # Conversation history (LangChain messages)
    raw_input: str                        # Original user input (cleaned text)
    
    # File attachments (extracted from input during preprocessing)
    image_path: Optional[str]             # Path to image file (Part 1)
    audio_path: Optional[str]             # Path to audio file (Part 2/3/4)
    
    # ==================== DETECTION LAYER ====================
    is_chitchat: bool                     # Quick chitchat flag (greeting/thanks)
    chitchat_response: Optional[str]      # Canned response for chitchat
    
    detected_part: Optional[Literal[
        "part1", "part2", "part3", "part4",
        "part5", "part6", "part7", "unknown"
    ]]                                    # Detected TOEIC part
    
    detection_confidence: float           # Confidence score 0.0 - 1.0
    detection_reasoning: str              # Why this classification?
    detection_method: Optional[str]       # Detection method used:
                                          # "media" / "format" / "keyword" / "none"
    
    # ==================== ROUTING LAYER ====================
    routed_agent_name: Optional[str]      # Agent key from registry (e.g., "part5")
    routing_decision: Optional[str]       # Decision made:
                                          # "execute" / "ask_user" / "error"
    
    # ==================== EXECUTION LAYER ====================
    agent_input: Optional[Dict[str, Any]]   # Input prepared for child agent
    agent_output: Optional[Dict[str, Any]]  # Raw output from child agent
    
    # ==================== ERROR HANDLING ====================
    error: Optional[str]                  # Error message (if any)
    retry_count: int                      # Current retry attempt count
    max_retries: int                      # Maximum retries allowed (configurable)
    
    # ==================== OUTPUT LAYER ====================
    final_answer: str                     # Formatted response to return to user
    
    # ==================== METADATA ====================
    metadata: Dict[str, Any]              # Additional info:
                                          # - debug_mode: bool
                                          # - preprocessing_complete: bool
                                          # - formatting_complete: bool
                                          # - timestamps, etc.
