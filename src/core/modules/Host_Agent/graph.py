# graph.py
# -*- coding: utf-8 -*-
"""
LangGraph Workflow Definition

Defines the Host Agent graph structure using LangGraph.
Orchestrates the flow between preprocessing, detection, routing, and execution.
"""

from langgraph.graph import StateGraph, END
from functools import partial
from typing import Optional, Dict, Any
import logging

from .state import HostAgentState
from .nodes import (
    preprocess_input_node,
    detect_chitchat_node,
    detect_part_node,
    route_decision_node,
    execute_agent_node,
    format_response_node
)
from .registry import AgentRegistry

logger = logging.getLogger(__name__)


class HostAgent:
    """
    LangGraph-based Host Agent for routing TOEIC questions
    
    This is the main orchestrator that:
    1. Preprocesses user input (extract files, clean text)
    2. Detects chitchat (greetings, thanks)
    3. Classifies TOEIC part (1-7) using multi-level detection
    4. Routes to appropriate child agent
    5. Executes child agent with retry mechanism
    6. Formats and returns response
    
    Usage:
        from host_agent import create_host_agent
        
        host = create_host_agent(models)
        result = await host.ainvoke("TOEIC question here")
    
    Graph Structure:
        ┌─────────────┐
        │  preprocess │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │detect_chitchat│──────────┐ (if chitchat)
        └──────┬──────┘            │
               │                   │
        ┌──────▼──────┐            │
        │ detect_part │            │
        └──────┬──────┘            │
               │                   │
        ┌──────▼──────┐            │
        │route_decision│───────────┤ (if error/ask_user)
        └──────┬──────┘            │
               │                   │
        ┌──────▼──────┐            │
        │execute_agent│◄───────────┤ (retry loop)
        └──────┬──────┘            │
               │                   │
        ┌──────▼──────┐◄───────────┘
        │format_response│
        └──────┬──────┘
               │
              END
    """
    
    def __init__(self, registry: AgentRegistry):
        """
        Initialize Host Agent
        
        Args:
            registry: AgentRegistry instance with registered agents and models
        """
        self.registry = registry
        self.graph = self._build_graph()
        
        logger.info(f"[HostAgent] Initialized with {registry.agent_count} agents")
    
    def _build_graph(self) -> StateGraph:
        """
        Build and compile the LangGraph workflow
        
        Returns:
            Compiled StateGraph ready for execution
        """
        
        logger.info("[HostAgent] Building graph...")
        
        # Create graph with state schema
        workflow = StateGraph(HostAgentState)
        
        # ==================== ADD NODES ====================
        
        # Input processing
        workflow.add_node("preprocess", preprocess_input_node)
        
        # Chitchat detection (early exit for greetings/thanks)
        workflow.add_node("detect_chitchat", detect_chitchat_node)
        
        # TOEIC part detection
        workflow.add_node("detect_part", detect_part_node)
        
        # Routing decision (inject registry via partial)
        workflow.add_node(
            "route_decision",
            partial(route_decision_node, registry=self.registry)
        )
        
        # Agent execution (inject registry via partial)
        workflow.add_node(
            "execute_agent",
            partial(execute_agent_node, registry=self.registry)
        )
        
        # Response formatting
        workflow.add_node("format_response", format_response_node)
        
        # ==================== SET ENTRY POINT ====================
        
        workflow.set_entry_point("preprocess")
        
        # ==================== ADD EDGES ====================
        
        # Linear: preprocess → detect_chitchat
        workflow.add_edge("preprocess", "detect_chitchat")
        
        # Conditional: chitchat → format_response OR detect_part
        workflow.add_conditional_edges(
            "detect_chitchat",
            _should_skip_to_response,
            {
                "detect_part": "detect_part",
                "format_response": "format_response"
            }
        )
        
        # Linear: detect_part → route_decision
        workflow.add_edge("detect_part", "route_decision")
        
        # Conditional: route_decision → execute_agent OR format_response
        workflow.add_conditional_edges(
            "route_decision",
            _should_execute_agent,
            {
                "execute_agent": "execute_agent",
                "format_response": "format_response"
            }
        )
        
        # Conditional: execute_agent → retry OR format_response
        workflow.add_conditional_edges(
            "execute_agent",
            _should_retry_execution,
            {
                "execute_agent": "execute_agent",  # Retry loop
                "format_response": "format_response"
            }
        )
        
        # Linear: format_response → END
        workflow.add_edge("format_response", END)
        
        logger.info("[HostAgent] Graph built successfully")
        
        # Compile and return
        return workflow.compile()
    
    # ==================== PUBLIC METHODS ====================
    
    async def ainvoke(
        self,
        user_input: str,
        debug: bool = False,
        max_retries: int = 2
    ) -> str:
        """
        Process user input asynchronously
        
        This is the main entry point for using the Host Agent.
        
        Args:
            user_input: User's question/input text
            debug: Enable debug mode (adds metadata to response)
            max_retries: Maximum retry attempts on failure
        
        Returns:
            Final answer string
        
        Example:
            >>> result = await host.ainvoke(
            ...     "The manager _____ the report yesterday. (A) submit (B) submits (C) submitted (D) submitting"
            ... )
            >>> print(result)
        """
        
        from langchain_core.messages import HumanMessage
        
        # Initialize state
        initial_state: Dict[str, Any] = {
            "messages": [HumanMessage(content=user_input)],
            "raw_input": "",
            "is_chitchat": False,
            "retry_count": 0,
            "max_retries": max_retries,
            "final_answer": "",
            "metadata": {"debug_mode": debug}
        }
        
        logger.info(f"[HostAgent] Processing: {user_input[:80]}...")
        
        # Setup config with LangSmith tracing
        import os
        config = {}
        if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true":
            from langchain_core.tracers import LangChainTracer
            project_name = os.environ.get("LANGCHAIN_PROJECT", "TOEIC-Host-Agent")
            tracer = LangChainTracer(project_name=project_name)
            config["callbacks"] = [tracer]
            config["run_name"] = f"HostAgent: {user_input[:50]}..."
        
        # Execute graph
        result = await self.graph.ainvoke(initial_state, config=config)
        
        logger.info("[HostAgent] Processing complete")
        
        return result.get("final_answer", "")
    
    def invoke(
        self,
        user_input: str,
        debug: bool = False,
        max_retries: int = 2
    ) -> str:
        """
        Process user input synchronously
        
        Wrapper around ainvoke for synchronous contexts.
        
        Args:
            user_input: User's question/input text
            debug: Enable debug mode
            max_retries: Maximum retry attempts
        
        Returns:
            Final answer string
        """
        import asyncio
        
        return asyncio.run(self.ainvoke(user_input, debug, max_retries))
    
    def get_state_schema(self) -> type:
        """Get the state schema class"""
        return HostAgentState
    
    def visualize(self, output_path: Optional[str] = None):
        """
        Visualize graph structure
        
        Args:
            output_path: Optional path to save PNG image
        
        Returns:
            Mermaid diagram string (if IPython not available)
        """
        try:
            from IPython.display import Image, display
            png_data = self.graph.get_graph().draw_mermaid_png()
            
            if output_path:
                with open(output_path, "wb") as f:
                    f.write(png_data)
                logger.info(f"[HostAgent] Graph saved to {output_path}")
            
            display(Image(png_data))
            return None
            
        except ImportError:
            # Fallback to mermaid text
            mermaid = self.graph.get_graph().draw_mermaid()
            if output_path:
                with open(output_path.replace(".png", ".md"), "w") as f:
                    f.write(f"```mermaid\n{mermaid}\n```")
            return mermaid
    
    def __repr__(self) -> str:
        return f"HostAgent(agents={self.registry.agent_count})"


# ==================== CONDITIONAL EDGE FUNCTIONS ====================

def _should_skip_to_response(state: HostAgentState) -> str:
    """
    Conditional: After chitchat detection
    
    Returns:
        "format_response" if chitchat, else "detect_part"
    """
    if state.get("is_chitchat"):
        return "format_response"
    return "detect_part"


def _should_execute_agent(state: HostAgentState) -> str:
    """
    Conditional: After routing decision
    
    Returns:
        "execute_agent" if routing_decision is "execute", else "format_response"
    """
    if state.get("routing_decision") == "execute":
        return "execute_agent"
    return "format_response"


def _should_retry_execution(state: HostAgentState) -> str:
    """
    Conditional: After agent execution
    
    Returns:
        "execute_agent" if error and retries remaining, else "format_response"
    """
    error = state.get("error")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    final_answer = state.get("final_answer", "")
    
    # If we already have a final_answer (success or max retries error), go to format
    if final_answer:
        return "format_response"
    
    # Retry if there's an error and we haven't exceeded max retries
    if error and retry_count < max_retries:
        return "execute_agent"
    
    return "format_response"
