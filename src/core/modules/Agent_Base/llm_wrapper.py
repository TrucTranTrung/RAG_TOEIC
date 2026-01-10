"""
LlamaCpp ChatModel Wrapper with Tool Binding Support
Converts BaseLLM (LlamaCpp) to ChatModel interface for LangChain 1.0.1+
"""

from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool
from langchain_community.llms import LlamaCpp
from typing import List, Optional, Any, Sequence, Union
from pydantic import Field
import logging

logger = logging.getLogger(__name__)


class LlamaCppChatWrapper(SimpleChatModel):
    """
    Full-featured wrapper to convert LlamaCpp (BaseLLM) into ChatModel.

    Supports:
    - Message-based interface
    - Tool binding (for agents)
    - Compatible with create_agent in LangChain 1.0.1+

    Usage:
        >>> llama_llm = LlamaCpp(model_path="model.gguf", n_ctx=2048)
        >>> chat_model = LlamaCppChatWrapper(llm=llama_llm)
        >>> agent = BaseAgent(model=chat_model)
    """

    # Pydantic v2 field declarations
    llm: LlamaCpp = Field(description="Underlying LlamaCpp model")
    bound_tools: List[BaseTool] = Field(
        default_factory=list, description="Tools bound to this model")

    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "llama_cpp_chat_wrapper"

    def bind_tools(
        self,
        tools: Sequence[Union[dict, type, BaseTool]],
        **kwargs: Any,
    ) -> "LlamaCppChatWrapper":
        """
        Bind tools to the model (required for agent support).

        This creates a new instance with tools attached.
        The tools are included in the system prompt so the model knows about them.

        Args:
            tools: List of tools to bind
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            New instance with tools bound
        """
        # Convert tools to BaseTool if needed
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                formatted_tools.append(tool)
            elif isinstance(tool, dict):
                # Tool dict format - convert to description
                formatted_tools.append(tool)
            else:
                # Type or other format
                formatted_tools.append(tool)

        # Create new instance with tools
        return self.__class__(
            llm=self.llm,
            bound_tools=formatted_tools,
        )

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """
        Convert messages to prompt and call LlamaCpp.

        Args:
            messages: List of chat messages
            stop: Stop sequences
            run_manager: Callback manager
            **kwargs: Additional arguments

        Returns:
            str: Model response
        """
        try:
            # Convert messages to text prompt
            prompt = self._format_messages(messages)

            # Call underlying LlamaCpp
            response = self.llm.invoke(prompt, stop=stop, **kwargs)

            return response

        except Exception as e:
            logger.error(f"Error in LlamaCpp wrapper: {e}")
            raise

    def _format_messages(self, messages: List[BaseMessage]) -> str:
        """
        Convert LangChain message format to text prompt.

        Includes tool descriptions if tools are bound.

        Args:
            messages: List of BaseMessage objects

        Returns:
            str: Formatted text prompt
        """
        lines = []

        # Add tool descriptions if tools are bound
        if self.bound_tools:
            lines.append("You have access to the following tools:")
            lines.append("")

            for tool in self.bound_tools:
                if isinstance(tool, BaseTool):
                    lines.append(f"- {tool.name}: {tool.description}")
                elif isinstance(tool, dict):
                    lines.append(
                        f"- {tool.get('name', 'unknown')}: {tool.get('description', '')}")

            lines.append("")
            lines.append("To use a tool, respond with:")
            lines.append("Action: [tool_name]")
            lines.append("Action Input: [input]")
            lines.append("")
            lines.append(
                "If you can answer directly without tools, just provide the answer.")
            lines.append("")

        # Add conversation messages
        for msg in messages:
            if isinstance(msg, SystemMessage):
                lines.append(f"System: {msg.content}")

            elif isinstance(msg, HumanMessage):
                lines.append(f"Human: {msg.content}")

            elif isinstance(msg, AIMessage):
                lines.append(f"Assistant: {msg.content}")

            else:
                # Generic message
                lines.append(str(msg.content))

        # Add assistant prefix for completion
        lines.append("Assistant:")

        return "\n".join(lines)

    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters."""
        return {
            "llm_type": self._llm_type,
            "model_path": getattr(self.llm, "model_path", "unknown"),
            "n_ctx": getattr(self.llm, "n_ctx", "unknown"),
            "num_tools": len(self.bound_tools),
        }
