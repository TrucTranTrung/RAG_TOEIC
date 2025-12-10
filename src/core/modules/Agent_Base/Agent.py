# Setup logging
from langchain_core.runnables import Runnable
from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms.base import BaseLLM
from langchain.tools import BaseTool
from langchain.prompts import BasePromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Đây là (Interface) cho tất cả các agent.

    Class này sẽ TỰ ĐỘNG làm các phần :
    - Tạo một "agent runnable" (bộ não)
    - Tạo một "agent executor" (vòng lặp chạy)
    - Cung cấp phương thức .invoke() và .ainvoke() để chạy agent.
    """

    executor: AgentExecutor
    agent_runnable: Runnable
    llm: BaseLLM
    tools: List[BaseTool]
    prompt: BasePromptTemplate

    def __init__(
        self,
        llm: BaseLLM,
        custom_callbacks: Optional[List[BaseCallbackHandler]] = None,
        agent_executor_options: Optional[Dict[str, Any]] = None
    ):
        """
        Hàm khởi tạo của BaseAgent.
        Nó sẽ gọi các phương thức abstract _get_tools và _get_prompt
        để check thông tin từ class con.

        Args:
            llm: Một instance của BaseLLM (ví dụ: ChatGoogleGenerativeAI).
            custom_callbacks: Danh sách các callback handler.
            agent_executor_options: Các tùy chọn cho AgentExecutor (ví dụ: max_iterations).
        """
        self.llm = llm

        logger.info(f"Khởi tạo agent {self.__class__.__name__}...")

        # 1. Lấy thông tin cần thiết từ class con
        self.tools = self._get_tools()
        self.prompt = self._get_prompt()

        if not self.tools or not self.prompt:
            raise ValueError(
                f"{self.__class__.__name__} must implement _get_tools and _get_prompt")

        # 2. Khởi tạo agent (chuẩn ReAct)
        self.agent_runnable = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # 3. Tự động xây dựng "trình thực thi" agent

        # Thiết lập các tùy chọn mặc định và ghi đè
        default_exec_options = {
            "verbose": False,
            "handle_parsing_errors": True,
            "max_iterations": 5
        }
        if agent_executor_options:
            default_exec_options.update(agent_executor_options)

        logger.info(
            f"Tạo AgentExecutor với các tùy chọn: {default_exec_options}")

        self.executor = AgentExecutor(
            agent=self.agent_runnable,
            tools=self.tools,
            callbacks=custom_callbacks,
            **default_exec_options
        )
        logger.info(f"Agent {self.__class__.__name__} built successfully.")

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
        return self.executor.invoke(input_data)

    async def ainvoke(self, input_data: dict) -> dict:
        """
        Phương thức để gọi agent (bất đồng bộ).
        """
        logger.info(
            f"--- Gọi Agent (Bất đồng bộ): {self.__class__.__name__} ---")
        return await self.executor.ainvoke(input_data)