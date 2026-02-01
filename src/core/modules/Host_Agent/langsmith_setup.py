# langsmith_setup.py
# -*- coding: utf-8 -*-
"""
LangSmith Integration for Host Agent

Provides tracing and monitoring capabilities via LangSmith.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def setup_langsmith(project_name: Optional[str] = None) -> bool:
    """
    Setup LangSmith tracing from environment variables
    
    Required environment variables:
        - LANGCHAIN_TRACING_V2: "true" to enable
        - LANGCHAIN_API_KEY: Your LangSmith API key
        - LANGCHAIN_ENDPOINT: LangSmith endpoint (optional)
        - LANGCHAIN_PROJECT: Project name (optional)
    
    Args:
        project_name: Override project name from env
    
    Returns:
        True if LangSmith is configured, False otherwise
    """
    
    # Check if tracing is enabled
    tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "").lower().strip() == "true"
    api_key = os.environ.get("LANGCHAIN_API_KEY", "").strip()
    
    if not tracing_enabled:
        logger.info("[LangSmith] Tracing is disabled (LANGCHAIN_TRACING_V2 != 'true')")
        return False
    
    if not api_key:
        logger.warning("[LangSmith] No API key found (LANGCHAIN_API_KEY not set)")
        return False
    
    # Set environment variables (LangChain reads these automatically)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    
    # Optional: endpoint
    endpoint = os.environ.get("LANGCHAIN_ENDPOINT", "").strip()
    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    else:
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
    # Optional: project name
    if project_name:
        os.environ["LANGCHAIN_PROJECT"] = project_name
    elif not os.environ.get("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = "TOEIC-Host-Agent"
    
    final_project = os.environ.get("LANGCHAIN_PROJECT")
    final_endpoint = os.environ.get("LANGCHAIN_ENDPOINT")
    
    logger.info(f"[LangSmith] ✅ Tracing enabled")
    logger.info(f"[LangSmith]    Project: {final_project}")
    logger.info(f"[LangSmith]    Endpoint: {final_endpoint}")
    
    return True


def get_langsmith_url(run_id: str = None) -> str:
    """
    Get LangSmith dashboard URL
    
    Args:
        run_id: Optional specific run ID
    
    Returns:
        URL to LangSmith dashboard
    """
    project = os.environ.get("LANGCHAIN_PROJECT", "default")
    base_url = "https://smith.langchain.com"
    
    if run_id:
        return f"{base_url}/public/{run_id}/r"
    
    return f"{base_url}/o/default-org/projects/p/{project}"


def is_langsmith_enabled() -> bool:
    """Check if LangSmith tracing is currently enabled"""
    return os.environ.get("LANGCHAIN_TRACING_V2", "").lower().strip() == "true"
