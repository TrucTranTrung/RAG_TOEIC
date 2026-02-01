# detection.py
# -*- coding: utf-8 -*-
"""
Multi-Level Detection Module

Provides intelligent classification of TOEIC questions into parts (1-7)
using a cascading detection pipeline with confidence scoring.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import re
import logging

from .state import HostAgentState

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """
    Structured detection result
    
    Attributes:
        part: Detected TOEIC part ("part1" - "part7" or "unknown")
        confidence: Confidence score (0.0 - 1.0)
        reasoning: Human-readable explanation
        method: Detection method used ("media" / "format" / "keyword" / "none")
    """
    part: str
    confidence: float
    reasoning: str
    method: str
    
    def __repr__(self) -> str:
        return f"DetectionResult({self.part}, {self.confidence:.2%}, method={self.method})"
    
    @property
    def is_confident(self) -> bool:
        """Check if detection is confident enough (>= 60%)"""
        return self.confidence >= 0.60
    
    @property
    def is_unknown(self) -> bool:
        """Check if detection failed"""
        return self.part == "unknown"


class MultiLevelDetector:
    """
    Multi-level detection pipeline for TOEIC part classification
    
    Priority cascade (highest to lowest confidence):
    1. Media-based detection (image/audio presence)
    2. Format-based detection (text structure analysis)
    3. Keyword-based detection (fallback pattern matching)
    
    Usage:
        detector = MultiLevelDetector()
        result = detector.detect(state)
        
        if result.is_confident:
            print(f"Detected: {result.part} ({result.confidence:.0%})")
    """
    
    # Confidence thresholds for each level
    MEDIA_THRESHOLD = 0.85
    FORMAT_THRESHOLD = 0.75
    KEYWORD_THRESHOLD = 0.60
    
    def detect(self, state: HostAgentState) -> DetectionResult:
        """
        Run the full detection pipeline
        
        Args:
            state: Current graph state containing raw_input and file paths
        
        Returns:
            DetectionResult with part, confidence, reasoning, and method
        """
        
        logger.debug("[Detector] Starting detection pipeline...")
        
        # Level 1: Media-based detection (highest priority)
        result = self._detect_by_media(state)
        if result.confidence >= self.MEDIA_THRESHOLD:
            logger.debug(f"[Detector] Media detection: {result}")
            return result
        
        # Level 2: Format-based detection
        result = self._detect_by_format(state)
        if result.confidence >= self.FORMAT_THRESHOLD:
            logger.debug(f"[Detector] Format detection: {result}")
            return result
        
        # Level 3: Keyword-based detection (fallback)
        result = self._detect_by_keywords(state)
        if result.confidence >= self.KEYWORD_THRESHOLD:
            logger.debug(f"[Detector] Keyword detection: {result}")
            return result
        
        # Unknown - no confident detection
        logger.debug("[Detector] No confident detection")
        return DetectionResult(
            part="unknown",
            confidence=0.0,
            reasoning="Không tìm thấy đặc điểm rõ ràng để phân loại",
            method="none"
        )
    
    # ==================== LEVEL 1: MEDIA DETECTION ====================
    
    def _detect_by_media(self, state: HostAgentState) -> DetectionResult:
        """
        Level 1: Detect based on media attachments
        
        Rules:
        - Image + 4 options → Part 1 (95%)
        - Audio + 3 options → Part 2 (90%)
        - Audio + 4 options → Part 3/4 (85%)
        """
        
        raw_input = state.get("raw_input", "").lower()
        image_path = state.get("image_path")
        audio_path = state.get("audio_path")
        
        # Part 1: Image + 4 options
        if image_path:
            option_count = self._count_options(raw_input)
            if option_count == 4:
                return DetectionResult(
                    part="part1",
                    confidence=0.95,
                    reasoning=f"Có file ảnh + 4 lựa chọn (A/B/C/D)",
                    method="media"
                )
            elif option_count > 0:
                return DetectionResult(
                    part="part1",
                    confidence=0.80,
                    reasoning=f"Có file ảnh + {option_count} lựa chọn",
                    method="media"
                )
        
        # Part 2/3/4: Audio-based
        if audio_path:
            option_count = self._count_options(raw_input)
            
            # Part 2: Audio + 3 options (A/B/C only)
            if option_count == 3:
                return DetectionResult(
                    part="part2",
                    confidence=0.90,
                    reasoning="Có file audio + 3 lựa chọn (A/B/C) → Question-Response",
                    method="media"
                )
            
            # Part 3/4: Audio + 4 options
            elif option_count == 4:
                # Try to distinguish Part 3 vs Part 4 by keywords
                if self._has_conversation_keywords(raw_input):
                    return DetectionResult(
                        part="part3",
                        confidence=0.88,
                        reasoning="Có file audio + 4 lựa chọn + từ khóa hội thoại",
                        method="media"
                    )
                elif self._has_announcement_keywords(raw_input):
                    return DetectionResult(
                        part="part4",
                        confidence=0.88,
                        reasoning="Có file audio + 4 lựa chọn + từ khóa thông báo",
                        method="media"
                    )
                else:
                    # Default to Part 3
                    return DetectionResult(
                        part="part3",
                        confidence=0.85,
                        reasoning="Có file audio + 4 lựa chọn → Listening comprehension",
                        method="media"
                    )
        
        return DetectionResult("unknown", 0.0, "", "media")
    
    # ==================== LEVEL 2: FORMAT DETECTION ====================
    
    def _detect_by_format(self, state: HostAgentState) -> DetectionResult:
        """
        Level 2: Detect based on text format/structure
        
        Rules:
        - Single sentence + 1 blank + 4 options → Part 5
        - Paragraph + blanks (email format) → Part 6
        - Long passage + multiple questions → Part 7
        """
        
        raw_input = state.get("raw_input", "")
        
        # Part 5: Single sentence with blank
        if self._has_single_blank(raw_input) and not self._has_paragraph(raw_input):
            if self._count_options(raw_input) == 4:
                return DetectionResult(
                    part="part5",
                    confidence=0.85,
                    reasoning="Câu đơn với 1 chỗ trống + 4 lựa chọn → Grammar",
                    method="format"
                )
        
        # Part 6: Paragraph with blanks (email/letter format)
        if self._has_paragraph(raw_input):
            if self._has_blanks(raw_input):
                blank_count = self._count_blanks(raw_input)
                return DetectionResult(
                    part="part6",
                    confidence=0.80,
                    reasoning=f"Đoạn văn với {blank_count} chỗ trống → Text Completion",
                    method="format"
                )
            
            # Part 7: Long passage with multiple questions
            if self._has_multiple_questions(raw_input):
                question_count = self._count_questions(raw_input)
                return DetectionResult(
                    part="part7",
                    confidence=0.80,
                    reasoning=f"Đoạn văn dài với {question_count} câu hỏi → Reading Comprehension",
                    method="format"
                )
        
        return DetectionResult("unknown", 0.0, "", "format")
    
    # ==================== LEVEL 3: KEYWORD DETECTION ====================
    
    def _detect_by_keywords(self, state: HostAgentState) -> DetectionResult:
        """
        Level 3: Detect based on keyword matching (fallback)
        
        Lower confidence as this is less reliable.
        """
        
        raw_input = state.get("raw_input", "").lower()
        
        # Keyword mapping with priority order
        keyword_map = {
            "part1": [
                "photograph", "picture", "image", "describe the picture",
                "mô tả hình", "mô tả ảnh", "hình ảnh"
            ],
            "part2": [
                "question-response", "short answer", "trả lời ngắn"
            ],
            "part3": [
                "conversation", "dialogue", "speakers talking",
                "hội thoại", "cuộc nói chuyện", "man and woman"
            ],
            "part4": [
                "announcement", "talk", "speech", "lecture", "message",
                "thông báo", "bài nói", "tin nhắn thoại"
            ],
            "part5": [
                "grammar", "choose the correct word", "incomplete sentence",
                "ngữ pháp", "chọn từ đúng", "điền từ"
            ],
            "part6": [
                "text completion", "fill in the blanks", "email", "letter",
                "hoàn thành đoạn văn", "thư", "email"
            ],
            "part7": [
                "reading comprehension", "passage", "article", "according to",
                "đọc hiểu", "theo bài đọc", "bài viết"
            ]
        }
        
        for part, keywords in keyword_map.items():
            for keyword in keywords:
                if keyword in raw_input:
                    return DetectionResult(
                        part=part,
                        confidence=0.65,
                        reasoning=f"Phát hiện từ khóa: '{keyword}'",
                        method="keyword"
                    )
        
        return DetectionResult("unknown", 0.0, "", "keyword")
    
    # ==================== HELPER METHODS ====================
    
    def _count_options(self, text: str) -> int:
        """Count unique (A), (B), (C), (D) patterns"""
        pattern = r'\([A-D]\)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return len(set(m.upper() for m in matches))
    
    def _has_blanks(self, text: str) -> bool:
        """Check for any blank patterns"""
        blank_pattern = r'_{2,}|\[BLANK\]|\<BLANK\>|---|\.\.\.'
        return bool(re.search(blank_pattern, text, re.IGNORECASE))
    
    def _count_blanks(self, text: str) -> int:
        """Count number of blanks"""
        blank_pattern = r'_{2,}|\[BLANK\]|\<BLANK\>|---'
        return len(re.findall(blank_pattern, text, re.IGNORECASE))
    
    def _has_single_blank(self, text: str) -> bool:
        """Check for exactly one blank"""
        return self._count_blanks(text) == 1
    
    def _has_paragraph(self, text: str) -> bool:
        """
        Check if text is paragraph-like
        
        Indicators:
        - Email headers (To:, From:, Subject:)
        - Multiple sentences (>= 3 sentences with >20 chars each)
        """
        # Email headers
        if re.search(r'(To:|From:|Subject:|Date:|Dear\s)', text, re.IGNORECASE):
            return True
        
        # Multiple meaningful sentences
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        return len(sentences) >= 3
    
    def _has_multiple_questions(self, text: str) -> bool:
        """Check for multiple numbered questions"""
        return self._count_questions(text) > 1
    
    def _count_questions(self, text: str) -> int:
        """Count numbered questions"""
        question_pattern = r'\d+\.\s*(?:What|Who|Where|When|Why|How|Which)'
        return len(re.findall(question_pattern, text, re.IGNORECASE))
    
    def _has_conversation_keywords(self, text: str) -> bool:
        """Check for conversation-related keywords (Part 3)"""
        keywords = ["man", "woman", "conversation", "dialogue", "speakers"]
        return any(k in text for k in keywords)
    
    def _has_announcement_keywords(self, text: str) -> bool:
        """Check for announcement-related keywords (Part 4)"""
        keywords = ["announcement", "message", "talk", "speech", "lecture", "report"]
        return any(k in text for k in keywords)
