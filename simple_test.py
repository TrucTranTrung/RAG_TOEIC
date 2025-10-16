#!/usr/bin/env python3
"""
Simple test runner cho Root Agent
Chạy tất cả test cases mà không cần pytest
"""

import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.root_agent import (
    RootAgent, RequestClassifier, PipelineManager, ContextManager, ErrorHandler,
    TOEICRequest, TOEICResponse, InputType, TOEICPart, ComplexityLevel, PipelineType,
    MockVisionPipeline, MockSpeechPipeline, MockReadingPipeline, MockRAGPipeline,
    get_root_agent
)


def test_request_classifier():
    """Test RequestClassifier"""
    print("\n📋 TESTING REQUEST CLASSIFIER")
    print("-"*40)
    
    classifier = RequestClassifier()
    
    # Test classify input type
    print("✅ Test classify input type:")
    assert classifier.classify_input_type("test_image.jpg") == InputType.IMAGE
    assert classifier.classify_input_type("test_audio.wav") == InputType.AUDIO
    assert classifier.classify_input_type("test text") == InputType.TEXT
    print("   - Image file classification: PASS")
    print("   - Audio file classification: PASS")
    print("   - Text classification: PASS")
    
    # Test detect TOEIC part
    print("✅ Test detect TOEIC part:")
    assert classifier.detect_toeic_part("test_image.jpg", InputType.IMAGE) == TOEICPart.PART_1
    assert classifier.detect_toeic_part("test_audio.wav", InputType.AUDIO) == TOEICPart.PART_2
    assert classifier.detect_toeic_part("short text", InputType.TEXT) == TOEICPart.PART_5
    print("   - Part 1 detection: PASS")
    print("   - Part 2 detection: PASS")
    print("   - Part 5 detection: PASS")
    
    # Test analyze complexity
    print("✅ Test analyze complexity:")
    assert classifier.analyze_complexity("test", TOEICPart.PART_1) == ComplexityLevel.MEDIUM
    assert classifier.analyze_complexity("test", TOEICPart.PART_5) == ComplexityLevel.LOW
    print("   - Part 1 complexity: PASS")
    print("   - Part 5 complexity: PASS")


def test_pipeline_manager():
    """Test PipelineManager"""
    print("\n🔧 TESTING PIPELINE MANAGER")
    print("-"*40)
    
    pipeline_manager = PipelineManager()
    
    # Test get pipeline
    print("✅ Test get pipeline:")
    assert pipeline_manager.get_pipeline(TOEICPart.PART_1) == PipelineType.VISION
    assert pipeline_manager.get_pipeline(TOEICPart.PART_2) == PipelineType.SPEECH
    assert pipeline_manager.get_pipeline(TOEICPart.PART_5) == PipelineType.READING
    print("   - Part 1 -> Vision: PASS")
    print("   - Part 2 -> Speech: PASS")
    print("   - Part 5 -> Reading: PASS")
    
    # Test monitor pipeline
    print("✅ Test monitor pipeline:")
    status = pipeline_manager.monitor_pipeline(PipelineType.VISION)
    assert status["status"] == "healthy"
    print("   - Pipeline monitoring: PASS")


def test_context_manager():
    """Test ContextManager"""
    print("\n📝 TESTING CONTEXT MANAGER")
    print("-"*40)
    
    context_manager = ContextManager()
    
    # Test create session
    print("✅ Test create session:")
    session_id = context_manager.create_session("test_user")
    assert isinstance(session_id, str)
    assert len(session_id) > 0
    print("   - Session creation: PASS")
    
    # Test update context
    print("✅ Test update context:")
    request = TOEICRequest(
        request_id="test_001",
        input_data="test question",
        input_type=InputType.TEXT,
        toeic_part=TOEICPart.PART_5
    )
    
    response = TOEICResponse(
        request_id="test_001",
        answer="Test answer",
        explanation="Test explanation",
        confidence_score=0.85,
        processing_time=0.1,
        pipeline_used=PipelineType.READING
    )
    
    context_manager.update_context(session_id, request, response)
    assert context_manager.sessions[session_id]["request_count"] == 1
    print("   - Context update: PASS")
    
    # Test get context
    print("✅ Test get context:")
    context = context_manager.get_context(session_id)
    assert "session_info" in context
    assert context["total_requests"] == 1
    print("   - Context retrieval: PASS")


def test_error_handler():
    """Test ErrorHandler"""
    print("\n❌ TESTING ERROR HANDLER")
    print("-"*40)
    
    error_handler = ErrorHandler()
    
    # Test handle error
    print("✅ Test handle error:")
    error = TimeoutError("Request timeout")
    context = {"request_id": "test_001"}
    error_info = error_handler.handle_error(error, context)
    assert "error_id" in error_info
    assert error_info["recovery_strategy"] == "retry_with_timeout"
    print("   - Error handling: PASS")
    
    # Test recovery
    print("✅ Test recovery:")
    recovery_success = error_handler.recover_from_error(error_info)
    assert recovery_success is True
    print("   - Error recovery: PASS")
    
    # Test fallback
    print("✅ Test fallback:")
    request = TOEICRequest(
        request_id="test_001",
        input_data="test data",
        input_type=InputType.TEXT
    )
    fallback_response = error_handler.fallback_strategy(request)
    assert isinstance(fallback_response, TOEICResponse)
    assert fallback_response.metadata["fallback"] is True
    print("   - Fallback strategy: PASS")


def test_root_agent():
    """Test RootAgent"""
    print("\n🤖 TESTING ROOT AGENT")
    print("-"*40)
    
    # Reset singleton
    RootAgent._instance = None
    RootAgent._initialized = False
    agent = RootAgent()
    
    # Test singleton
    print("✅ Test singleton pattern:")
    agent1 = RootAgent()
    agent2 = RootAgent()
    assert agent1 is agent2
    print("   - Singleton pattern: PASS")
    
    # Test route request
    print("✅ Test route request:")
    request = TOEICRequest(
        request_id="test_001",
        input_data="test_image.jpg",
        input_type=InputType.IMAGE,
        toeic_part=TOEICPart.PART_1
    )
    pipeline_type = agent.route_request(request)
    assert pipeline_type == PipelineType.VISION
    print("   - Request routing: PASS")
    
    # Test aggregate response
    print("✅ Test aggregate response:")
    responses = [
        TOEICResponse(
            request_id="test_001",
            answer="Answer A",
            explanation="Explanation A",
            confidence_score=0.8,
            processing_time=0.1,
            pipeline_used=PipelineType.VISION
        ),
        TOEICResponse(
            request_id="test_002",
            answer="Answer B",
            explanation="Explanation B",
            confidence_score=0.9,
            processing_time=0.2,
            pipeline_used=PipelineType.READING
        )
    ]
    aggregated = agent.aggregate_response(responses)
    assert aggregated.metadata["aggregated"] is True
    assert aggregated.metadata["response_count"] == 2
    print("   - Response aggregation: PASS")
    
    # Test get metrics
    print("✅ Test get metrics:")
    metrics = agent.get_metrics()
    assert "total_requests" in metrics
    assert "success_rate" in metrics
    print("   - Metrics retrieval: PASS")


async def test_async_functions():
    """Test async functions"""
    print("\n🔄 TESTING ASYNC FUNCTIONS")
    print("-"*40)
    
    # Reset singleton
    RootAgent._instance = None
    RootAgent._initialized = False
    agent = RootAgent()
    
    # Test initialize
    print("✅ Test initialize:")
    success = await agent.initialize()
    assert success is True
    print("   - System initialization: PASS")
    
    # Test process request
    print("✅ Test process request:")
    request = TOEICRequest(
        request_id="test_001",
        input_data="This is a grammar question",
        input_type=InputType.TEXT,
        session_id=agent.context_manager.create_session("test_user")
    )
    response = await agent.process_request(request)
    assert isinstance(response, TOEICResponse)
    assert response.request_id == request.request_id
    assert response.confidence_score > 0
    print("   - Request processing: PASS")
    
    # Test pipeline execution
    print("✅ Test pipeline execution:")
    pipeline_manager = PipelineManager()
    request = TOEICRequest(
        request_id="test_002",
        input_data="test_image.jpg",
        input_type=InputType.IMAGE,
        toeic_part=TOEICPart.PART_1
    )
    response = await pipeline_manager.execute_pipeline(PipelineType.VISION, request)
    assert isinstance(response, TOEICResponse)
    assert response.pipeline_used == PipelineType.VISION
    print("   - Pipeline execution: PASS")


async def test_mock_pipelines():
    """Test mock pipelines"""
    print("\n🎭 TESTING MOCK PIPELINES")
    print("-"*40)
    
    # Test Vision Pipeline
    print("✅ Test Vision Pipeline:")
    vision_pipeline = MockVisionPipeline()
    request = TOEICRequest(
        request_id="test_001",
        input_data="test_image.jpg",
        input_type=InputType.IMAGE,
        toeic_part=TOEICPart.PART_1
    )
    response = await vision_pipeline.process(request)
    assert response.pipeline_used == PipelineType.VISION
    print("   - Vision Pipeline: PASS")
    
    # Test Speech Pipeline
    print("✅ Test Speech Pipeline:")
    speech_pipeline = MockSpeechPipeline()
    request = TOEICRequest(
        request_id="test_002",
        input_data="test_audio.wav",
        input_type=InputType.AUDIO,
        toeic_part=TOEICPart.PART_2
    )
    response = await speech_pipeline.process(request)
    assert response.pipeline_used == PipelineType.SPEECH
    print("   - Speech Pipeline: PASS")
    
    # Test Reading Pipeline
    print("✅ Test Reading Pipeline:")
    reading_pipeline = MockReadingPipeline()
    request = TOEICRequest(
        request_id="test_003",
        input_data="This is a grammar question",
        input_type=InputType.TEXT,
        toeic_part=TOEICPart.PART_5
    )
    response = await reading_pipeline.process(request)
    assert response.pipeline_used == PipelineType.READING
    print("   - Reading Pipeline: PASS")
    
    # Test RAG Pipeline
    print("✅ Test RAG Pipeline:")
    rag_pipeline = MockRAGPipeline()
    request = TOEICRequest(
        request_id="test_004",
        input_data="Complex question requiring RAG",
        input_type=InputType.TEXT,
        toeic_part=TOEICPart.PART_7
    )
    response = await rag_pipeline.process(request)
    assert response.pipeline_used == PipelineType.RAG
    print("   - RAG Pipeline: PASS")


async def test_integration():
    """Test integration scenarios"""
    print("\n🔗 TESTING INTEGRATION")
    print("-"*40)
    
    # Reset singleton
    RootAgent._instance = None
    RootAgent._initialized = False
    agent = RootAgent()
    
    # Test full workflow
    print("✅ Test full workflow:")
    await agent.initialize()
    session_id = agent.context_manager.create_session("test_user")
    
    request = TOEICRequest(
        request_id="integration_test_001",
        input_data="test_image.jpg",
        input_type=InputType.IMAGE,
        session_id=session_id
    )
    
    response = await agent.process_request(request)
    assert isinstance(response, TOEICResponse)
    assert response.pipeline_used == PipelineType.VISION
    
    # Check context was updated
    context = agent.context_manager.get_context(session_id)
    assert context["total_requests"] == 1
    print("   - Full workflow: PASS")
    
    # Test multiple requests
    print("✅ Test multiple requests:")
    requests = [
        TOEICRequest(
            request_id=f"multi_test_{i:03d}",
            input_data=f"Question {i}",
            input_type=InputType.TEXT,
            session_id=session_id
        )
        for i in range(1, 4)
    ]
    
    responses = []
    for request in requests:
        response = await agent.process_request(request)
        responses.append(response)
    
    assert len(responses) == 3
    context = agent.context_manager.get_context(session_id)
    assert context["total_requests"] == 4  # 1 from previous test + 3 new
    print("   - Multiple requests: PASS")


async def main():
    """Main test runner"""
    print("🚀 CHẠY TEST CASES CHO ROOT AGENT")
    print("="*80)
    
    try:
        # Run sync tests
        test_request_classifier()
        test_pipeline_manager()
        test_context_manager()
        test_error_handler()
        test_root_agent()
        
        # Run async tests
        await test_async_functions()
        await test_mock_pipelines()
        await test_integration()
        
        print("\n✅ TẤT CẢ TEST CASES ĐÃ HOÀN THÀNH!")
        print("="*80)
        print("🎉 Root Agent và các components đã được test đầy đủ.")
        print("📊 Tất cả chức năng đều hoạt động chính xác.")
        
    except Exception as e:
        print(f"\n❌ Lỗi trong test: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
