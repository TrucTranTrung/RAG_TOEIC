"""
Root Agent (Orchestrator) sử dụng LangChain và React Agent Pattern
Điều phối chính của hệ thống AI Agent TOEIC với class interface
"""

import asyncio
import os
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Type
from uuid import uuid4

# LangChain imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import BaseTool
from langchain.llms.base import BaseLLM
from pydantic import Field, PrivateAttr
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.agent import AgentAction, AgentFinish

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputType(Enum):
    """Loại input có thể xử lý"""
    IMAGE = "image"
    AUDIO = "audio"
    TEXT = "text"
    UNKNOWN = "unknown"


class TOEICPart(Enum):
    """Các phần của bài thi TOEIC"""
    PART_1 = "part_1"  # Vision - Image description
    PART_2 = "part_2"  # Speech - Question/Response
    PART_3 = "part_3"  # Speech - Conversation
    PART_4 = "part_4"  # Speech - Monologue
    PART_5 = "part_5"  # Reading - Grammar/Vocabulary
    PART_6 = "part_6"  # Reading - Text completion
    PART_7 = "part_7"  # Reading - Reading comprehension


class ComplexityLevel(Enum):
    """Mức độ phức tạp của câu hỏi"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PipelineType(Enum):
    """Loại pipeline"""
    VISION = "vision"
    SPEECH = "speech"
    READING = "reading"
    RAG = "rag"


@dataclass
class TOEICRequest:
    """Request object cho hệ thống TOEIC"""
    request_id: str
    input_data: Any
    input_type: InputType
    toeic_part: Optional[TOEICPart] = None
    complexity: Optional[ComplexityLevel] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class TOEICResponse:
    """Response object cho hệ thống TOEIC"""
    request_id: str
    answer: str
    explanation: str
    confidence_score: float
    processing_time: float
    pipeline_used: PipelineType
    metadata: Optional[Dict[str, Any]] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class TOEICCallbackHandler(BaseCallbackHandler):
    """Custom callback handler cho TOEIC Agent"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TOEICCallbackHandler")
        self.actions = []
        self.observations = []
        self.llm_prompts = []  # danh sách các lời gọi LLM (prompts)
        self.llm_outputs = []  # danh sách các output từ LLM
        self.tool_traces = []  # danh sách trace cho tool (name, input, output)
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Callback khi agent thực hiện action"""
        self.logger.info(f"🎯 Agent Action: {action.tool} - {action.tool_input}")
        self.actions.append(action)
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Callback khi agent hoàn thành"""
        self.logger.info(f"✅ Agent Finish: {finish.return_values}")
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Ghi nhận mỗi lần LLM được gọi"""
        for p in prompts:
            self.llm_prompts.append(p)
        self.logger.info(f"🧠 LLM Start - {len(prompts)} prompt(s)")

    def on_llm_end(self, response, **kwargs) -> None:
        """Ghi nhận kết quả trả về từ LLM"""
        try:
            # LangChain trả về object có .generations
            texts = []
            if hasattr(response, "generations"):
                for gen in response.generations:
                    if gen and len(gen) > 0 and hasattr(gen[0], "text"):
                        texts.append(gen[0].text)
            if texts:
                for t in texts:
                    self.llm_outputs.append(t)
                self.logger.info(f"🧠 LLM End - captured {len(texts)} generation(s)")
        except Exception as e:
            self.logger.warning(f"Không thể parse LLM response: {e}")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Callback khi tool bắt đầu"""
        self.logger.info(f"🔧 Tool Start: {serialized.get('name', 'Unknown')}")
        self.tool_traces.append({
            "event": "start",
            "tool": serialized.get("name", "Unknown"),
            "input": input_str,
            "output": None
        })
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Callback khi tool kết thúc"""
        self.logger.info(f"✅ Tool End: {output[:100]}...")
        self.observations.append(output)
        # cập nhật tool trace cuối cùng nếu còn dangling
        for i in range(len(self.tool_traces) - 1, -1, -1):
            if self.tool_traces[i]["event"] == "start" and self.tool_traces[i]["output"] is None:
                self.tool_traces[i]["output"] = output
                self.tool_traces[i]["event"] = "end"
                break


class RequestClassificationTool(BaseTool):
    """Tool để phân loại request"""
    
    name: str = "classify_request"
    description: str = "Phân loại loại input và xác định TOEIC part"
    
    _logger: logging.Logger = PrivateAttr()

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._logger = logging.getLogger(f"{__name__}.RequestClassificationTool")
    
    def _run(self, input_data: str) -> str:
        """Phân loại input data"""
        self._logger.info(f"🔍 Phân loại input: {input_data}")
        
        # Logic phân loại
        if input_data.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            input_type = InputType.IMAGE
            toeic_part = TOEICPart.PART_1
        elif input_data.lower().endswith(('.wav', '.mp3', '.m4a', '.aac')):
            input_type = InputType.AUDIO
            toeic_part = TOEICPart.PART_2
        else:
            input_type = InputType.TEXT
            if len(input_data.split()) < 10:
                toeic_part = TOEICPart.PART_5
            elif len(input_data.split()) < 50:
                toeic_part = TOEICPart.PART_6
            else:
                toeic_part = TOEICPart.PART_7
        
        result = f"Input Type: {input_type.value}, TOEIC Part: {toeic_part.value}"
        self._logger.info(f"✅ Phân loại kết quả: {result}")
        return result
    
    async def _arun(self, input_data: str) -> str:
        """Async version"""
        return self._run(input_data)


class PipelineRoutingTool(BaseTool):
    """Tool để định tuyến pipeline"""
    
    name: str = "route_pipeline"
    description: str = "Định tuyến request đến pipeline phù hợp"
    
    _logger: logging.Logger = PrivateAttr()

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._logger = logging.getLogger(f"{__name__}.PipelineRoutingTool")
    
    def _run(self, toeic_part: str) -> str:
        """Định tuyến pipeline"""
        self._logger.info(f"🎯 Định tuyến pipeline cho: {toeic_part}")
        
        # Logic định tuyến
        if toeic_part == "part_1":
            pipeline_type = PipelineType.VISION
        elif toeic_part in ["part_2", "part_3", "part_4"]:
            pipeline_type = PipelineType.SPEECH
        elif toeic_part in ["part_5", "part_6", "part_7"]:
            pipeline_type = PipelineType.READING
        else:
            pipeline_type = PipelineType.RAG
        
        result = f"Pipeline Type: {pipeline_type.value}"
        self._logger.info(f"✅ Pipeline được chọn: {pipeline_type.value}")
        return result
    
    async def _arun(self, toeic_part: str) -> str:
        """Async version"""
        return self._run(toeic_part)


class ContextManagementTool(BaseTool):
    """Tool để quản lý context"""
    
    name: str = "manage_context"
    description: str = "Quản lý session và context"
    
    _logger: logging.Logger = PrivateAttr()
    sessions: Dict[str, Any] = Field(default_factory=dict)
    context_history: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._logger = logging.getLogger(f"{__name__}.ContextManagementTool")
    
    def _run(self, action: str, session_id: str = None, data: str = None) -> str:
        """Quản lý context"""
        self._logger.info(f"📝 Context action: {action}")
        
        # Chuẩn hoá alias
        normalized_action = action
        if action == "create":
            normalized_action = "create_session"
        elif action == "update":
            normalized_action = "update_context"
        elif action == "get":
            normalized_action = "get_context"

        if normalized_action == "create_session":
            session_id = str(uuid4())
            self.sessions[session_id] = {
                "created_at": time.time(),
                "request_count": 0,
                "learning_progress": {}
            }
            self.context_history[session_id] = []
            result = f"Session created: {session_id}"
        
        elif normalized_action == "update_context":
            if session_id in self.sessions:
                self.sessions[session_id]["request_count"] += 1
                self.context_history[session_id].append({
                    "timestamp": time.time(),
                    "data": data
                })
                result = f"Context updated for session: {session_id}"
            else:
                result = f"Session not found: {session_id}"
        
        elif normalized_action == "get_context":
            if session_id in self.sessions:
                context = self.sessions[session_id]
                result = f"Context for {session_id}: {context}"
            else:
                result = f"Session not found: {session_id}"
        
        else:
            result = f"Unknown action: {action}"
        
        self._logger.info(f"✅ Context result: {result}")
        return result
    
    async def _arun(self, action: str, session_id: str = None, data: str = None) -> str:
        """Async version"""
        return self._run(action, session_id, data)


class ErrorHandlingTool(BaseTool):
    """Tool để xử lý lỗi"""
    
    name: str = "handle_error"
    description: str = "Xử lý lỗi và recovery"
    
    _logger: logging.Logger = PrivateAttr()
    error_history: List[Dict[str, Any]] = Field(default_factory=list)

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._logger = logging.getLogger(f"{__name__}.ErrorHandlingTool")
    
    def _run(self, error_type: str, error_message: str) -> str:
        """Xử lý lỗi"""
        self._logger.info(f"❌ Xử lý lỗi: {error_type} - {error_message}")
        
        error_id = str(uuid4())
        error_info = {
            "error_id": error_id,
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": time.time(),
            "recovery_strategy": self._determine_recovery_strategy(error_type)
        }
        
        self.error_history.append(error_info)
        
        result = f"Error handled: {error_id}, Strategy: {error_info['recovery_strategy']}"
        self._logger.info(f"✅ Error handling result: {result}")
        return result
    
    def _determine_recovery_strategy(self, error_type: str) -> str:
        """Xác định chiến lược recovery"""
        if "timeout" in error_type.lower():
            return "retry_with_timeout"
        elif "connection" in error_type.lower():
            return "fallback_pipeline"
        elif "value" in error_type.lower():
            return "input_validation"
        else:
            return "generic_fallback"
    
    async def _arun(self, error_type: str, error_message: str) -> str:
        """Async version"""
        return self._run(error_type, error_message)


## MockLLM đã bị xoá. RootAgent yêu cầu Gemini thật qua GOOGLE_API_KEY.


class TOEICPipeline(ABC):
    """Abstract base class cho các TOEIC Pipeline"""
    
    def __init__(self, pipeline_type: PipelineType):
        self.pipeline_type = pipeline_type
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.uptime = time.time()
    
    @abstractmethod
    async def process(self, request: TOEICRequest) -> TOEICResponse:
        """Process request và trả về response"""
        pass


class VisionPipeline(TOEICPipeline):
    """Vision Pipeline cho Part 1"""
    
    def __init__(self):
        super().__init__(PipelineType.VISION)
    
    async def process(self, request: TOEICRequest) -> TOEICResponse:
        """Process vision request"""
        self.logger.info("👁️ Vision Pipeline đang xử lý request")
        
        # Simulate processing
        await asyncio.sleep(0.1)
        
        return TOEICResponse(
            request_id=request.request_id,
            answer="A) People are sitting at a table",
            explanation="Dựa trên hình ảnh, có thể thấy những người đang ngồi quanh bàn.",
            confidence_score=0.85,
            processing_time=0.1,
            pipeline_used=self.pipeline_type,
            metadata={"pipeline": "vision", "part": "1"}
        )


class SpeechPipeline(TOEICPipeline):
    """Speech Pipeline cho Part 2,3,4"""
    
    def __init__(self):
        super().__init__(PipelineType.SPEECH)
    
    async def process(self, request: TOEICRequest) -> TOEICResponse:
        """Process speech request"""
        self.logger.info("🎤 Speech Pipeline đang xử lý request")
        
        # Simulate processing
        await asyncio.sleep(0.2)
        
        return TOEICResponse(
            request_id=request.request_id,
            answer="B) Yes, I'd be happy to help",
            explanation="Dựa trên audio, câu trả lời phù hợp nhất là B.",
            confidence_score=0.80,
            processing_time=0.2,
            pipeline_used=self.pipeline_type,
            metadata={"pipeline": "speech", "part": "2,3,4"}
        )


class ReadingPipeline(TOEICPipeline):
    """Reading Pipeline cho Part 5,6,7"""
    
    def __init__(self):
        super().__init__(PipelineType.READING)
    
    async def process(self, request: TOEICRequest) -> TOEICResponse:
        """Process reading request"""
        self.logger.info("📖 Reading Pipeline đang xử lý request")
        
        # Simulate processing
        await asyncio.sleep(0.05)
        
        return TOEICResponse(
            request_id=request.request_id,
            answer="C) The meeting has been postponed",
            explanation="Dựa trên ngữ cảnh và ngữ pháp, đáp án C là phù hợp nhất.",
            confidence_score=0.90,
            processing_time=0.05,
            pipeline_used=self.pipeline_type,
            metadata={"pipeline": "reading", "part": "5,6,7"}
        )


class RAGPipeline(TOEICPipeline):
    """RAG Pipeline cho tất cả parts"""
    
    def __init__(self):
        super().__init__(PipelineType.RAG)
    
    async def process(self, request: TOEICRequest) -> TOEICResponse:
        """Process RAG request"""
        self.logger.info("🔍 RAG Pipeline đang xử lý request")
        
        # Simulate processing
        await asyncio.sleep(0.15)
        
        return TOEICResponse(
            request_id=request.request_id,
            answer="D) Based on the context and retrieved information",
            explanation="Sử dụng RAG để tìm kiếm thông tin liên quan và tạo câu trả lời.",
            confidence_score=0.75,
            processing_time=0.15,
            pipeline_used=self.pipeline_type,
            metadata={"pipeline": "rag", "rag": True}
        )


class PipelineManager:
    """Quản lý các pipeline"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PipelineManager")
        self.pipelines = {}
        self._register_pipelines()
    
    def _register_pipelines(self):
        """Đăng ký các pipeline"""
        self.logger.info("🔧 Đang đăng ký pipelines...")
        
        self.pipelines[PipelineType.VISION] = VisionPipeline()
        self.pipelines[PipelineType.SPEECH] = SpeechPipeline()
        self.pipelines[PipelineType.READING] = ReadingPipeline()
        self.pipelines[PipelineType.RAG] = RAGPipeline()
        
        self.logger.info(f"✅ Đã đăng ký {len(self.pipelines)} pipelines")
    
    def get_pipeline(self, toeic_part: TOEICPart) -> PipelineType:
        """Lấy pipeline phù hợp"""
        self.logger.info(f"🎯 Đang lấy pipeline cho {toeic_part.value}")
        
        if toeic_part == TOEICPart.PART_1:
            pipeline_type = PipelineType.VISION
        elif toeic_part in [TOEICPart.PART_2, TOEICPart.PART_3, TOEICPart.PART_4]:
            pipeline_type = PipelineType.SPEECH
        elif toeic_part in [TOEICPart.PART_5, TOEICPart.PART_6, TOEICPart.PART_7]:
            pipeline_type = PipelineType.READING
        else:
            pipeline_type = PipelineType.RAG
        
        self.logger.info(f"✅ Pipeline được chọn: {pipeline_type.value}")
        return pipeline_type
    
    async def execute_pipeline(self, pipeline_type: PipelineType, request: TOEICRequest) -> TOEICResponse:
        """Thực thi pipeline"""
        self.logger.info(f"🚀 Đang thực thi pipeline {pipeline_type.value}")
        
        if pipeline_type not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_type.value} không tồn tại")
        
        pipeline = self.pipelines[pipeline_type]
        start_time = time.time()
        
        try:
            response = await pipeline.process(request)
            processing_time = time.time() - start_time
            
            self.logger.info(f"✅ Pipeline {pipeline_type.value} hoàn thành trong {processing_time:.2f}s")
            
            response.processing_time = processing_time
            response.pipeline_used = pipeline_type
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi thực thi pipeline {pipeline_type.value}: {str(e)}")
            raise


class RootAgent:
    """Root Agent sử dụng LangChain và React Agent Pattern"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(RootAgent, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Khởi tạo Root Agent"""
        if not self._initialized:
            self.logger = logging.getLogger(f"{__name__}.RootAgent")
            self.logger.info("🚀 Khởi tạo Root Agent với LangChain...")
            
            # Initialize components
            self.llm = self._build_llm()
            self.pipeline_manager = PipelineManager()
            self.callback_handler = TOEICCallbackHandler()
            
            # Initialize tools
            self.tools = [
                RequestClassificationTool(),
                PipelineRoutingTool(),
                ContextManagementTool(),
                ErrorHandlingTool()
            ]
            
            # Create React Agent
            self._create_react_agent()
            
            # Performance metrics
            self.metrics = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_processing_time": 0.0,
                "start_time": time.time()
            }
            
            RootAgent._initialized = True
            self.logger.info("✅ Root Agent đã được khởi tạo thành công")
    
    def _build_llm(self) -> BaseLLM:
        """Khởi tạo LLM thật (Gemini). Bắt buộc có GOOGLE_API_KEY, không có sẽ raise."""
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        if not google_api_key:
            raise RuntimeError("GOOGLE_API_KEY chưa được thiết lập. Vui lòng cung cấp để chạy LLM thật (Gemini).")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.logger.info(f"🧠 Dùng Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(model=model_name, api_key=google_api_key, convert_system_message_to_human=True)
        except Exception as e:
            raise RuntimeError(f"Không thể khởi tạo Gemini LLM: {e}")
    
    def _create_react_agent(self):
        """Tạo React Agent"""
        self.logger.info("🤖 Đang tạo React Agent...")
        
        # React Agent prompt template
        prompt_template = """
        Bạn là một AI Agent chuyên xử lý bài thi TOEIC. Nhiệm vụ của bạn là:
        1. Phân loại request và xác định TOEIC part
        2. Định tuyến request đến pipeline phù hợp
        3. Quản lý context và session
        4. Xử lý lỗi và recovery
        
        Bạn có các tools sau:
        {tools}
        (Tên tools: {tool_names})
        
        Hãy sử dụng các tools một cách có hệ thống để xử lý request.
        
        Question: {input}
        
        Thought: {agent_scratchpad}
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )
        
        # Create React Agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create Agent Executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            callbacks=[self.callback_handler],
            max_iterations=5,
            handle_parsing_errors=True
        )
        
        self.logger.info("✅ React Agent đã được tạo thành công")
    
    async def initialize(self) -> bool:
        """Khởi tạo hệ thống"""
        self.logger.info("🔧 Đang khởi tạo hệ thống...")
        
        try:
            # Test agent với simple request
            test_result = await self.agent_executor.ainvoke({
                "input": "Test initialization"
            })
            
            self.logger.info("✅ Hệ thống đã được khởi tạo thành công")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi khởi tạo hệ thống: {str(e)}")
            return False
    
    async def process_request(self, request: TOEICRequest) -> TOEICResponse:
        """Xử lý request chính"""
        start_time = time.time()
        self.logger.info(f"🎯 Bắt đầu xử lý request {request.request_id}")
        
        # Update metrics
        self.metrics["total_requests"] += 1
        
        try:
            # Tạo input cho agent
            agent_input = f"""
            Request ID: {request.request_id}
            Input Data: {request.input_data}
            Input Type: {request.input_type.value}
            Session ID: {request.session_id}
            
            Hãy xử lý request này theo các bước:
            1. Phân loại request
            2. Định tuyến pipeline
            3. Quản lý context
            4. Trả về kết quả
            """
            
            # Chạy agent
            agent_result = await self.agent_executor.ainvoke({
                "input": agent_input
            })
            
            # Extract thông tin từ agent result
            answer = agent_result.get("output", "No answer generated")
            
            # Tạo response
            processing_time = time.time() - start_time
            response = TOEICResponse(
                request_id=request.request_id,
                answer=answer,
                explanation=f"Processed by React Agent with LangChain. Actions: {len(self.callback_handler.actions)}",
                confidence_score=0.85,
                processing_time=processing_time,
                pipeline_used=PipelineType.RAG,  # Default
                metadata={
                    "agent_type": "react",
                    "langchain": True,
                    "actions_count": len(self.callback_handler.actions),
                    "observations_count": len(self.callback_handler.observations)
                }
            )
            
            # Update metrics
            self.metrics["successful_requests"] += 1
            self.metrics["average_processing_time"] = (
                (self.metrics["average_processing_time"] * (self.metrics["successful_requests"] - 1) + processing_time) 
                / self.metrics["successful_requests"]
            )
            
            self.logger.info(f"✅ Request {request.request_id} hoàn thành trong {processing_time:.2f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi xử lý request {request.request_id}: {str(e)}")
            
            # Handle error với agent
            try:
                error_result = await self.agent_executor.ainvoke({
                    "input": f"Handle error: {str(e)} for request {request.request_id}"
                })
                
                # Tạo fallback response
                processing_time = time.time() - start_time
                response = TOEICResponse(
                    request_id=request.request_id,
                    answer="Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau.",
                    explanation=f"Error handled by React Agent: {error_result.get('output', 'Unknown error')}",
                    confidence_score=0.0,
                    processing_time=processing_time,
                    pipeline_used=PipelineType.RAG,
                    metadata={"error": True, "agent_handled": True}
                )
                
            except Exception as agent_error:
                self.logger.error(f"❌ Agent cũng gặp lỗi: {str(agent_error)}")
                
                # Tạo basic fallback response
                processing_time = time.time() - start_time
                response = TOEICResponse(
                    request_id=request.request_id,
                    answer="Xin lỗi, hệ thống đang gặp sự cố nghiêm trọng.",
                    explanation="System error - Agent không thể xử lý",
                    confidence_score=0.0,
                    processing_time=processing_time,
                    pipeline_used=PipelineType.RAG,
                    metadata={"error": True, "agent_failed": True}
                )
            
            # Update metrics
            self.metrics["failed_requests"] += 1
            
            return response
    
    def get_metrics(self) -> Dict[str, Any]:
        """Lấy performance metrics"""
        uptime = time.time() - self.metrics["start_time"]
        
        return {
            **self.metrics,
            "uptime": uptime,
            "success_rate": (
                self.metrics["successful_requests"] / self.metrics["total_requests"] 
                if self.metrics["total_requests"] > 0 else 0
            ),
            "requests_per_minute": (
                self.metrics["total_requests"] / (uptime / 60) 
                if uptime > 0 else 0
            ),
            "agent_actions": len(self.callback_handler.actions),
            "agent_observations": len(self.callback_handler.observations)
        }
    
    def print_status(self):
        """In ra trạng thái của Root Agent"""
        print("\n" + "="*60)
        print("🤖 ROOT AGENT STATUS (LangChain + React)")
        print("="*60)
        
        metrics = self.get_metrics()
        print(f"📊 Total Requests: {metrics['total_requests']}")
        print(f"✅ Successful: {metrics['successful_requests']}")
        print(f"❌ Failed: {metrics['failed_requests']}")
        print(f"📈 Success Rate: {metrics['success_rate']:.2%}")
        print(f"⏱️ Avg Processing Time: {metrics['average_processing_time']:.2f}s")
        print(f"🚀 Requests/Minute: {metrics['requests_per_minute']:.1f}")
        print(f"⏰ Uptime: {metrics['uptime']:.1f}s")
        print(f"🎯 Agent Actions: {metrics['agent_actions']}")
        print(f"👁️ Agent Observations: {metrics['agent_observations']}")
        
        # Pipeline status
        print("\n🔧 PIPELINE STATUS:")
        for pipeline_type in PipelineType:
            print(f"  {pipeline_type.value}: available")
        
        # Tools status
        print("\n🛠️ TOOLS STATUS:")
        for tool in self.tools:
            print(f"  {tool.name}: available")
        
        print("="*60)

    def print_full_trace(self):
        """In ra toàn bộ messages, prompts/output của LLM và trace tool calls"""
        print("\n" + "-"*60)
        print("🧩 FULL AGENT TRACE")
        print("-"*60)
        # LLM prompts
        if self.callback_handler.llm_prompts:
            print("\n🧠 LLM PROMPTS:")
            for i, p in enumerate(self.callback_handler.llm_prompts, 1):
                print(f"  [{i}] Prompt:\n{p}\n")
        # LLM outputs
        if self.callback_handler.llm_outputs:
            print("\n🧠 LLM OUTPUTS:")
            for i, t in enumerate(self.callback_handler.llm_outputs, 1):
                print(f"  [{i}] Output:\n{t}\n")
        # Agent actions
        if self.callback_handler.actions:
            print("\n🎯 AGENT ACTIONS:")
            for i, a in enumerate(self.callback_handler.actions, 1):
                print(f"  [{i}] tool={a.tool} input={a.tool_input}")
        # Tool traces
        if self.callback_handler.tool_traces:
            print("\n🔧 TOOL TRACES:")
            for i, t in enumerate(self.callback_handler.tool_traces, 1):
                print(f"  [{i}] {t['event']} - tool={t['tool']}\n      input={t['input']}\n      output={(t['output'][:200] + '...') if isinstance(t['output'], str) and len(t['output'])>200 else t['output']}")
        # Observations
        if self.callback_handler.observations:
            print("\n👁️ OBSERVATIONS:")
            for i, o in enumerate(self.callback_handler.observations, 1):
                short = o if len(o) <= 200 else o[:200] + "..."
                print(f"  [{i}] {short}")
        print("-"*60)


# Factory function
def get_root_agent() -> RootAgent:
    """Factory function để lấy Root Agent instance"""
    return RootAgent()


# Example usage
if __name__ == "__main__":
    async def main():
        # Tạo Root Agent
        agent = get_root_agent()
        
        # Khởi tạo hệ thống
        await agent.initialize()
        
        # Tạo test request
        test_request = TOEICRequest(
            request_id="test_001",
            input_data="This is a test question about grammar.",
            input_type=InputType.TEXT,
            session_id="test_session"
        )
        
        # Xử lý request
        response = await agent.process_request(test_request)
        
        # In kết quả
        print(f"\n📝 Request ID: {response.request_id}")
        print(f"💡 Answer: {response.answer}")
        print(f"📖 Explanation: {response.explanation}")
        print(f"🎯 Confidence: {response.confidence_score:.2%}")
        print(f"⏱️ Processing Time: {response.processing_time:.2f}s")
        print(f"🔧 Pipeline Used: {response.pipeline_used.value}")
        print(f"📊 Metadata: {response.metadata}")
        
        # In status
        agent.print_status()
        # In full trace (messages, state, tool calls)
        agent.print_full_trace()
    
    # Chạy example
    asyncio.run(main())
