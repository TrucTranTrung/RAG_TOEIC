# __init__.py
# -*- coding: utf-8 -*-
"""
Host Agent Module

Multi-Agent Router System for TOEIC using LangGraph.

This module provides:
- HostAgent: Main orchestrator using LangGraph
- AgentRegistry: Central registry for managing child agents
- create_host_agent: Factory function for easy setup

Example usage:
    from host_agent import create_host_agent
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_mistralai import ChatMistralAI
    
    # Setup models
    models = {
        "gemini": ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
        "mistral": ChatMistralAI(model="mistral-small-latest")
    }
    
    # Create host agent
    host = create_host_agent(models)
    
    # Use it
    result = await host.ainvoke("What is the answer? (A) ... (B) ... (C) ... (D) ...")
    print(result)
"""

from typing import Dict, Any, Optional
import logging

from .graph import HostAgent
from .registry import AgentRegistry, AgentConfig
from .state import HostAgentState
from .detection import MultiLevelDetector, DetectionResult

logger = logging.getLogger(__name__)


def create_host_agent(
    models: Dict[str, Any],
    custom_configs: Optional[Dict[str, AgentConfig]] = None,
    debug: bool = False
) -> HostAgent:
    """
    Factory function to create a fully configured Host Agent
    
    This is the recommended way to create a HostAgent instance.
    It handles:
    - Registry creation and configuration
    - Agent registration
    - Model injection
    - Validation
    
    Args:
        models: Dict of model instances, keyed by model name
                Example: {
                    "gemini": ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                }
        custom_configs: Optional dict of custom AgentConfig to override defaults
        debug: Enable debug logging
    
    Returns:
        Configured HostAgent instance ready to use
    
    Raises:
        ValueError: If required models are missing
    
    Example:
        >>> from host_agent import create_host_agent
        >>> models = {"gemini": ...}
        >>> host = create_host_agent(models)
        >>> result = await host.ainvoke("TOEIC question here")
    """
    
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    logger.info("[Factory] Creating Host Agent...")
    
    # ========== LOAD CONFIGURATIONS ==========
    from .config import get_agent_configs, validate_models
    
    # Get default configs
    try:
        agent_configs = get_agent_configs()
    except ImportError as e:
        logger.error(f"[Factory] Failed to load agent configs: {e}")
        raise RuntimeError(
            "Could not load agent configurations. "
            "Make sure all agent modules are available."
        ) from e
    
    # Apply custom configs if provided
    if custom_configs:
        agent_configs.update(custom_configs)
        logger.info(f"[Factory] Applied {len(custom_configs)} custom configs")
    
    # ========== VALIDATE MODELS ==========
    is_valid, missing = validate_models(models)
    
    if not is_valid:
        logger.error(f"[Factory] Missing required models: {missing}")
        raise ValueError(
            f"Missing required model(s): {missing}. "
            f"Please provide models for: {', '.join(missing)}"
        )
    
    logger.info(f"[Factory] Models validated: {list(models.keys())}")
    
    # ========== CREATE REGISTRY ==========
    registry = AgentRegistry()
    
    # Register all agents
    for key, config in agent_configs.items():
        registry.register(key, config)
    
    logger.info(f"[Factory] Registered {registry.agent_count} agents")
    
    # Set models
    registry.set_models(models)
    
    # ========== CREATE HOST AGENT ==========
    host = HostAgent(registry=registry)
    
    logger.info("[Factory] Host Agent created successfully")
    logger.info(f"[Factory] Enabled agents: {registry.list_enabled()}")
    
    return host


def create_host_agent_simple(
    gemini_api_key: str,
    **kwargs
) -> HostAgent:
    """
    Simplified factory that creates models from API keys
    
    Convenience function for quick setup without manually
    creating model instances.
    
    Args:
        gemini_api_key: Google API key for Gemini
        **kwargs: Additional arguments passed to create_host_agent
    
    Returns:
        Configured HostAgent instance
    
    Example:
        >>> host = create_host_agent_simple(
        ...     gemini_api_key="your-gemini-key"
        ... )
    """
    
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    models = {
        "gemini": ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=gemini_api_key,
            temperature=0.0
        )
    }
    
    return create_host_agent(models, **kwargs)


# ==================== EXPORTS ====================

__all__ = [
    # Main classes
    "HostAgent",
    "AgentRegistry",
    "AgentConfig",
    "HostAgentState",
    
    # Detection
    "MultiLevelDetector",
    "DetectionResult",
    
    # Factory functions
    "create_host_agent",
    "create_host_agent_simple",
]

__version__ = "1.0.0"
