# config.py
# -*- coding: utf-8 -*-
"""
Agent Configuration Module

Defines configurations for all TOEIC agents.
Uses 3 different API keys to distribute load and avoid rate limits.

API Key Distribution:
- Key 1 (gemini1): Part 1, Part 2 (Image + Question-Response)
- Key 2 (gemini2): Part 3, Part 4 (Listening)
- Key 3 (gemini3): Part 5, Part 6 (Grammar + Text Completion)
"""

from typing import Dict, Any

from .registry import AgentConfig

# Lazy imports to avoid circular dependencies
# Agent classes are imported when needed, not at module load time


def get_agent_configs() -> Dict[str, AgentConfig]:
    """
    Factory function to create agent configurations
    
    Uses lazy imports to avoid circular dependencies and
    allow for easier testing/mocking.
    
    Returns:
        Dict mapping part keys to AgentConfig instances
    """
    
    # Lazy imports - only import when called
    from ..Agent_Part1.Agent1 import TOEICPart1Agent
    from ..Agent_Part2.Agent2 import TOEICPart2Agent
    from ..Agent_Part3.Agent3 import TOEICListeningAgent
    from ..Agent_Part5.Agent_Part5 import LanguageAgentPart5
    from ..Agent_Part6.Part6 import LanguageAgentPart6
    
    return {
        # ========== API Key 1: Part 1 + Part 2 ==========
        "part1": AgentConfig(
            name="TOEIC Part 1 - Image Description",
            agent_class=TOEICPart1Agent,
            model_key="gemini1",  # Key 1
            description="Phân tích hình ảnh với 4 lựa chọn đáp án (A/B/C/D)",
            enabled=True
        ),
        
        "part2": AgentConfig(
            name="TOEIC Part 2 - Question-Response",
            agent_class=TOEICPart2Agent,
            model_key="gemini1",  # Key 1
            description="Câu hỏi - Đáp án ngắn với 3 lựa chọn (A/B/C)",
            enabled=True
        ),
        
        # ========== API Key 2: Part 3 + Part 4 ==========
        "part3": AgentConfig(
            name="TOEIC Part 3 - Conversations",
            agent_class=TOEICListeningAgent,
            model_key="gemini2",  # Key 2
            description="Nghe hội thoại và trả lời câu hỏi",
            enabled=True
        ),
        
        "part4": AgentConfig(
            name="TOEIC Part 4 - Talks",
            agent_class=TOEICListeningAgent,  # Reuse same agent
            model_key="gemini2",  # Key 2
            description="Nghe bài nói và trả lời câu hỏi",
            enabled=True
        ),
        
        # ========== API Key 3: Part 5 + Part 6 ==========
        "part5": AgentConfig(
            name="TOEIC Part 5 - Incomplete Sentences",
            agent_class=LanguageAgentPart5,
            model_key="gemini3",  # Key 3
            description="Điền từ vào câu - Ngữ pháp và từ vựng",
            enabled=True
        ),
        
        "part6": AgentConfig(
            name="TOEIC Part 6 - Text Completion",
            agent_class=LanguageAgentPart6,
            model_key="gemini3",  # Key 3
            description="Hoàn thành đoạn văn - Email, thư từ",
            enabled=True
        ),
    }


# Static configuration for quick access (if imports don't fail)
# This will raise ImportError if agent modules are not available
try:
    AGENT_CONFIGS = get_agent_configs()
except ImportError as e:
    # Fallback to empty configs if agents not available
    # This allows the module to load even if agents are missing
    import logging
    logging.warning(f"Could not load agent configs: {e}")
    AGENT_CONFIGS = {}


# ==================== MODEL CONFIGURATION ====================

# Model keys and their distribution
MODEL_REQUIREMENTS = {
    "gemini1": {
        "description": "Google Gemini for Part 1, Part 2",
        "used_by": ["part1", "part2"],
        "env_key": "GOOGLE_API_KEY1",
        "recommended": "ChatGoogleGenerativeAI"
    },
    "gemini2": {
        "description": "Google Gemini for Part 3, Part 4",
        "used_by": ["part3", "part4"],
        "env_key": "GOOGLE_API_KEY2",
        "recommended": "ChatGoogleGenerativeAI"
    },
    "gemini3": {
        "description": "Google Gemini for Part 5, Part 6",
        "used_by": ["part5", "part6"],
        "env_key": "GOOGLE_API_KEY3",
        "recommended": "ChatGoogleGenerativeAI"
    }
}


def validate_models(models: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate that all required models are provided
    
    Args:
        models: Dict of model instances
    
    Returns:
        Tuple of (is_valid, missing_keys)
    """
    required_keys = set()
    for config in AGENT_CONFIGS.values():
        if config.enabled:
            required_keys.add(config.model_key)
    
    provided_keys = set(models.keys())
    missing = required_keys - provided_keys
    
    return len(missing) == 0, list(missing)


def create_models_from_env() -> Dict[str, Any]:
    """
    Create all Gemini model instances from environment variables
    
    Returns:
        Dict of model instances keyed by model name
    """
    import os
    from dotenv import load_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Load .env
    load_dotenv(dotenv_path="config/.env")
    
    models = {}
    
    for model_key, config in MODEL_REQUIREMENTS.items():
        env_key = config["env_key"]
        api_key = os.environ.get(env_key)
        
        if api_key:
            models[model_key] = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=api_key,
                temperature=0.0,
                max_tokens=4096
            )
        else:
            import logging
            logging.warning(f"Missing API key for {model_key}: {env_key}")
    
    return models
