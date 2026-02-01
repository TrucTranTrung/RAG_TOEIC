# server.py
# -*- coding: utf-8 -*-
"""
LangGraph Server Entry Point

This module exposes the Host Agent graph for LangGraph Platform deployment.
Run with: langgraph up
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
from dotenv import load_dotenv
env_path = PROJECT_ROOT / "config" / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Setup LangSmith (if enabled)
from src.core.modules.Host_Agent.langsmith_setup import setup_langsmith
setup_langsmith()

# Import agent classes
from langchain_google_genai import ChatGoogleGenerativeAI
from src.core.modules.Host_Agent.config import get_agent_configs
from src.core.modules.Host_Agent.registry import AgentRegistry
from src.core.modules.Host_Agent.graph import HostAgent


def create_host_agent() -> HostAgent:
    """
    Create and configure the Host Agent for server deployment
    
    Returns:
        Configured HostAgent instance
    """
    # Create Gemini models with different API keys
    models = {}
    
    api_key1 = os.environ.get("GOOGLE_API_KEY1")
    api_key2 = os.environ.get("GOOGLE_API_KEY2") 
    api_key3 = os.environ.get("GOOGLE_API_KEY3")
    
    if api_key1:
        models["gemini1"] = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=api_key1,
            temperature=0.3
        )
    
    if api_key2:
        models["gemini2"] = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=api_key2,
            temperature=0.3
        )
    
    if api_key3:
        models["gemini3"] = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=api_key3,
            temperature=0.3
        )
    
    # Fallback to single key
    if not models:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            models["gemini1"] = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=api_key,
                temperature=0.3
            )
    
    # Create registry and register agents
    registry = AgentRegistry()
    agent_configs = get_agent_configs()
    
    for key, config in agent_configs.items():
        registry.register(key, config)
    
    # Set models
    registry.set_models(models)
    
    # Create Host Agent
    host_agent = HostAgent(registry)
    
    return host_agent


# Create the graph for LangGraph Platform
# This is what langgraph.json references
host_agent = create_host_agent()
graph = host_agent.graph
