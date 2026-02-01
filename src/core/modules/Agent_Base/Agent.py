# Setup logging
from langchain_core.runnables import Runnable
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.prompts import BasePromptTemplate
from langchain.agents import create_agent
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Base interface for all agents using LangChain 1.0.1 and LangGraph 1.0.1.

    This class automatically:
    - Creates an agent executor with LangGraph
    - Provides .invoke() and .ainvoke() methods for running agents

    Note: Uses LangChain 1.0.1 conventions with 'model' instead of 'llm'.
    """

    executor: Runnable  # LangGraph agent executor
    model: BaseChatModel  # Language model (renamed from llm)
    tools: List[BaseTool]
    prompt: BasePromptTemplate

    def __init__(
        self,
        model: BaseChatModel,
        custom_callbacks: Optional[List[BaseCallbackHandler]] = None,
        agent_executor_options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize BaseAgent with LangChain 1.0.1 conventions.

        Calls abstract methods _get_tools and _get_prompt from subclass.

        Args:
            model: A chat model instance (e.g., ChatGoogleGenerativeAI).
            custom_callbacks: List of callback handlers (optional).
            agent_executor_options: Additional options (optional).
        """
        self.model = model

        logger.info(f"Khởi tạo agent {self.__class__.__name__}...")

        # 1. Lấy thông tin cần thiết từ class con
        self.tools = self._get_tools()
        self.prompt = self._get_prompt()

        if not self.tools or not self.prompt:
            raise ValueError(
                f"{self.__class__.__name__} must implement _get_tools and _get_prompt")

        # 2. Convert prompt template to system message for LangGraph
        # LangGraph uses message-based interface, not prompt templates
        # We extract the template string and use it as system message
        try:
            # Get the template string from the prompt
            if hasattr(self.prompt, 'template'):
                system_message = self.prompt.template
            else:
                # Fallback: format without variables
                system_message = "You are a helpful AI assistant."
                logger.warning(
                    "Could not extract template from prompt, using default system message")
        except Exception as e:
            logger.error(f"Error extracting system message: {e}")
            system_message = "You are a helpful AI assistant."

        # 3. Create LangGraph agent with system message (using LangChain 1.0.1)
        logger.info(f"Creating LangGraph agent with create_agent...")

        try:
            self.executor = create_agent(
                model=self.model,
                tools=self.tools,
                system_prompt=system_message
            )
            logger.info(
                f"Agent {self.__class__.__name__} built successfully with LangGraph.")
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise RuntimeError(f"Agent creation failed: {e}") from e

    @abstractmethod
    def _get_tools(self) -> List[BaseTool]:
        """
        Class con phải trả về một danh sách các 'Tool' mà agent này sẽ sử dụng.
        """
        pass

    @abstractmethod
    def _get_prompt(self) -> BasePromptTemplate:
        """
        Class con phải trả về 'PromptTemplate' (hướng dẫn) cho agent này.
        """
        pass

    def invoke(self, input_data: dict) -> dict:
        """
        Phương thức để gọi agent (đồng bộ).
        'input_data' là một dict, ví dụ: {'input': 'Câu hỏi của bạn'}
        """
        logger.info(f"--- Gọi Agent: {self.__class__.__name__} ---")

        if "input" in input_data:
            user_message = input_data["input"]
            messages = [("human", user_message)]
            langgraph_input = {"messages": messages}
        else:

            langgraph_input = input_data

        result = self.executor.invoke(langgraph_input)

        # LangGraph returns {"messages": [...]}
        if "messages" in result:
            last_message = result["messages"][-1]
            # Extract content from last message
            if hasattr(last_message, 'content'):
                content = last_message.content
                # Handle content that could be list or string
                if isinstance(content, list):
                    # Extract text from list of content parts
                    text_parts = []
                    for part in content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif isinstance(part, dict):
                            text_parts.append(part.get("text", part.get("content", str(part))))
                        elif hasattr(part, 'text'):
                            text_parts.append(part.text)
                        else:
                            text_parts.append(str(part))
                    output = "\n".join(text_parts)
                else:
                    output = str(content)
            else:
                output = str(last_message)

            return {
                "input": input_data.get("input", ""),
                "output": output
            }
        return result

    async def ainvoke(self, input_data: dict) -> dict:
        """
        Phương thức để gọi agent (bất đồng bộ).
        'input_data' là một dict, ví dụ: {'input': 'Câu hỏi của bạn'}
        """
        logger.info(
            f"--- Gọi Agent (Bất đồng bộ): {self.__class__.__name__} ---")

        if "input" in input_data:
            user_message = input_data["input"]
            messages = [("human", user_message)]
            langgraph_input = {"messages": messages}
        else:
            langgraph_input = input_data

        # Invoke LangGraph agent
        result = await self.executor.ainvoke(langgraph_input)

        # LangGraph returns {"messages": [...]}
        if "messages" in result:
            last_message = result["messages"][-1]
            # Extract content from last message
            if hasattr(last_message, 'content'):
                content = last_message.content
                # Handle content that could be list or string
                if isinstance(content, list):
                    # Extract text from list of content parts
                    text_parts = []
                    for part in content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif isinstance(part, dict):
                            text_parts.append(part.get("text", part.get("content", str(part))))
                        elif hasattr(part, 'text'):
                            text_parts.append(part.text)
                        else:
                            text_parts.append(str(part))
                    output = "\n".join(text_parts)
                else:
                    output = str(content)
            else:
                output = str(last_message)

            return {
                "input": input_data.get("input", ""),
                "output": output
            }
        return result