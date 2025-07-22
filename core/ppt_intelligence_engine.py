#!/usr/bin/env python3
"""
PPT 지능형 분석 엔진
강연자의 프레젠테이션 슬라이드를 깊이 이해하고 
음성과 결합하여 완전한 메시지를 복원하는 시스템
"""

import os
import re
import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from dataclasses import dataclass
from collections import defaultdict

try:
    import easyocr
    easyocr_available = True
except ImportError:
    easyocr_available = False

try:
    from PIL import Image, ImageEnhance, ImageFilter
    pillow_available = True
except ImportError:
    pillow_available = False

@dataclass
class SlideElement:
    """슬라이드 요소 정보"""
    element_type: str  # title, bullet, image, chart, table
    content: str
    position: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    hierarchy_level: int = 0
    
@dataclass 
class SlideStructure:
    """슬라이드 구조 정보"""
    slide_number: int
    title: str
    main_content: List[SlideElement]
    slide_type: str  # title_slide, content_slide, transition_slide, summary_slide
    layout_type: str  # text_only, text_image, chart_focused, etc.

class PPTIntelligenceEngine:
    """PPT 지능형 분석 엔진"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # OCR 엔진 초기화
        self.ocr_reader = None
        
        # 슬라이드 분석 규칙
        self.slide_patterns = self._build_slide_patterns()
        self.layout_templates = self._build_layout_templates()
        self.content_classifiers = self._build_content_classifiers()
        
        # PPT 특화 전처리 설정
        self.preprocessing_config = self._build_preprocessing_config()
        
        self.logger.info("🎨 PPT 지능형 분석 엔진 초기화 완료")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _build_slide_patterns(self) -> Dict[str, Any]:
        """슬라이드 패턴 분석 규칙"""
        return {
            "title_indicators": [
                # 한국어 제목 패턴
                r'^[0-9]+\.\s*[가-힣]',  # "1. 제목"
                r'^[가-힣]{2,20}$',      # 단순 한글 제목
                r'^[A-Z][a-zA-Z\s]{5,30}$',  # 영어 제목
                # 특수 문자로 강조된 제목
                r'^[\-\*\•]\s*[가-힣]',
                r'^【[가-힣]+】',
            ],
            "bullet_indicators": [
                r'^[\-\*\•]\s*',         # 일반적인 불릿
                r'^[0-9]+\)\s*',         # 숫자 불릿
                r'^[가-힣]\.\s*',        # 한글 불릿
                r'^[①-⑩]\s*',           # 원 숫자
                r'^\([0-9]+\)\s*',       # 괄호 숫자
            ],
            "emphasis_patterns": [
                r'\*\*([^*]+)\*\*',      # **강조**
                r'__([^_]+)__',          # __강조__
                r'『([^』]+)』',          # 『강조』
                r'「([^」]+)」',          # 「강조」
            ],
            "section_separators": [
                r'^={3,}',               # ===
                r'^-{3,}',               # ---
                r'^[▶▷]',                # 화살표
                r'^[■□]',                # 사각형
            ]
        }
    
    def _build_layout_templates(self) -> Dict[str, Any]:
        """레이아웃 템플릿 정의"""
        return {
            "title_slide": {
                "characteristics": ["large_text_top", "minimal_content", "center_aligned"],
                "expected_elements": ["main_title", "subtitle", "presenter_info"]
            },
            "agenda_slide": {
                "characteristics": ["numbered_list", "multiple_bullets", "overview_keywords"],
                "expected_elements": ["title", "agenda_items", "timeline"]
            },
            "content_slide": {
                "characteristics": ["title_and_body", "bullet_points", "detailed_content"],
                "expected_elements": ["slide_title", "main_content", "supporting_details"]
            },
            "chart_slide": {
                "characteristics": ["visual_dominant", "data_visualization", "minimal_text"],
                "expected_elements": ["chart_title", "chart_elements", "data_labels"]
            },
            "image_slide": {
                "characteristics": ["image_dominant", "caption_text", "visual_content"],
                "expected_elements": ["image_title", "image_caption", "image_content"]
            },
            "transition_slide": {
                "characteristics": ["minimal_text", "section_indicator", "navigation_element"],
                "expected_elements": ["section_title", "progress_indicator"]
            },
            "summary_slide": {
                "characteristics": ["conclusion_keywords", "recap_elements", "action_items"],
                "expected_elements": ["summary_title", "key_points", "next_steps"]
            }
        }
    
    def _build_content_classifiers(self) -> Dict[str, Any]:
        """컨텐츠 분류기"""
        return {
            "slide_types": {
                "title": ["제목", "타이틀", "주제", "발표", "소개", "Title", "Introduction"],
                "agenda": ["목차", "차례", "아젠다", "순서", "개요", "Agenda", "Contents", "Outline"],
                "problem": ["문제", "이슈", "과제", "도전", "Problem", "Issue", "Challenge"],
                "solution": ["해결", "방안", "답", "솔루션", "Solution", "Answer", "Approach"],
                "process": ["과정", "프로세스", "단계", "절차", "Process", "Steps", "Procedure"],
                "result": ["결과", "성과", "효과", "Result", "Outcome", "Achievement"],
                "conclusion": ["결론", "요약", "마무리", "Conclusion", "Summary", "Wrap-up"],
                "action": ["실행", "액션", "행동", "Action", "Implementation", "Next Steps"]
            },
            "content_importance": {
                "high": ["핵심", "중요", "반드시", "필수", "Key", "Important", "Critical", "Essential"],
                "medium": ["주요", "기본", "일반", "Main", "Basic", "General"],
                "low": ["참고", "부가", "추가", "Reference", "Additional", "Optional"]
            },
            "data_types": {
                "statistics": ["%", "percent", "통계", "데이터", "수치", "Statistics", "Data"],
                "timeline": ["년", "월", "일", "시간", "기간", "Year", "Month", "Timeline"],
                "comparison": ["vs", "대비", "비교", "차이", "Compare", "Versus", "Difference"],
                "process_flow": ["→", "▶", "다음", "그리고", "이어서", "Flow", "Next", "Then"]
            }
        }
    
    def _build_preprocessing_config(self) -> Dict[str, Any]:
        """전처리 설정"""
        return {
            "image_enhancement": {
                "contrast_factor": 1.5,
                "brightness_factor": 1.2,
                "sharpness_factor": 2.0,
                "denoise": True
            },
            "ocr_optimization": {
                "text_threshold": 0.7,
                "link_threshold": 0.4,
                "low_text": 0.4,
                "width_ths": 0.7,
                "height_ths": 0.7
            },
            "layout_detection": {
                "min_text_size": 10,
                "title_size_ratio": 1.5,
                "bullet_detection": True,
                "table_detection": True
            }
        }
    
    def analyze_ppt_intelligence(self, image_path: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """PPT 지능형 분석 수행"""
        self.logger.info(f"🎨 PPT 지능형 분석 시작: {os.path.basename(image_path)}")
        
        start_time = time.time()
        
        try:
            # 1. 이미지 전처리 및 최적화
            enhanced_image = self._enhance_slide_image(image_path)
            
            # 2. 레이아웃 분석
            layout_analysis = self._analyze_slide_layout(enhanced_image)
            
            # 3. 고정밀 OCR 수행
            ocr_results = self._perform_enhanced_ocr(enhanced_image, layout_analysis)
            
            # 4. 슬라이드 구조 파싱
            slide_structure = self._parse_slide_structure(ocr_results, layout_analysis)
            
            # 5. 컨텐츠 의미 분석
            content_analysis = self._analyze_content_meaning(slide_structure, context)
            
            # 6. 프레젠테이션 플로우 이해
            presentation_flow = self._understand_presentation_flow(slide_structure, content_analysis)
            
            # 7. 강연자 의도 추론
            speaker_intent = self._infer_speaker_intent(slide_structure, content_analysis, context)
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "ppt_intelligence": {
                    "slide_structure": slide_structure.__dict__ if hasattr(slide_structure, '__dict__') else slide_structure,
                    "content_analysis": content_analysis,
                    "presentation_flow": presentation_flow,
                    "speaker_intent": speaker_intent,
                    "layout_analysis": layout_analysis
                },
                "enhanced_understanding": {
                    "key_messages": self._extract_key_messages(slide_structure, content_analysis),
                    "visual_hierarchy": self._analyze_visual_hierarchy(slide_structure),
                    "presenter_emphasis": self._detect_presenter_emphasis(slide_structure),
                    "audience_guidance": self._analyze_audience_guidance(slide_structure, speaker_intent)
                },
                "technical_details": {
                    "processing_time": round(processing_time, 2),
                    "ocr_elements_detected": len(ocr_results) if ocr_results else 0,
                    "confidence_score": self._calculate_analysis_confidence(slide_structure, content_analysis),
                    "enhancement_applied": True
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ PPT 분석 실패: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _enhance_slide_image(self, image_path: str) -> np.ndarray:
        """슬라이드 이미지 향상 처리"""
        if not pillow_available:
            # PIL 없으면 OpenCV로 기본 처리
            image = cv2.imread(image_path)
            return cv2.convertScaleAbs(image, alpha=1.2, beta=30)
        
        # PIL을 사용한 고급 이미지 처리
        with Image.open(image_path) as img:
            # RGB 변환
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 대비 향상
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.preprocessing_config["image_enhancement"]["contrast_factor"])
            
            # 밝기 조정
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(self.preprocessing_config["image_enhancement"]["brightness_factor"])
            
            # 선명도 향상
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(self.preprocessing_config["image_enhancement"]["sharpness_factor"])
            
            # 노이즈 제거 (필요시)
            if self.preprocessing_config["image_enhancement"]["denoise"]:
                img = img.filter(ImageFilter.MedianFilter(size=3))
            
            # NumPy 배열로 변환
            return np.array(img)
    
    def _analyze_slide_layout(self, image: np.ndarray) -> Dict[str, Any]:
        """슬라이드 레이아웃 분석"""
        height, width = image.shape[:2]
        
        # 이미지를 그리드로 분할하여 텍스트 분포 분석
        grid_size = 8
        grid_h, grid_w = height // grid_size, width // grid_size
        
        text_density_map = np.zeros((grid_size, grid_size))
        
        # 간단한 텍스트 밀도 추정 (실제로는 더 정교한 방법 사용)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * grid_h, (i + 1) * grid_h
                x1, x2 = j * grid_w, (j + 1) * grid_w
                grid_region = gray[y1:y2, x1:x2]
                
                # 텍스트 밀도 추정 (에지 기반)
                edges = cv2.Canny(grid_region, 50, 150)
                text_density_map[i, j] = np.sum(edges) / (grid_h * grid_w)
        
        # 레이아웃 패턴 감지
        layout_pattern = self._detect_layout_pattern(text_density_map)
        
        return {
            "image_dimensions": {"width": width, "height": height},
            "text_density_map": text_density_map.tolist(),
            "layout_pattern": layout_pattern,
            "detected_regions": self._identify_content_regions(text_density_map, width, height)
        }
    
    def _perform_enhanced_ocr(self, image: np.ndarray, layout_analysis: Dict) -> List[Dict[str, Any]]:
        """향상된 OCR 수행"""
        if not easyocr_available:
            return []
        
        # OCR 리더 초기화 (지연 로딩)
        if self.ocr_reader is None:
            self.ocr_reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
        
        # OCR 설정 최적화
        ocr_config = self.preprocessing_config["ocr_optimization"]
        
        # EasyOCR 실행
        results = self.ocr_reader.readtext(
            image,
            detail=1,
            paragraph=False,
            width_ths=ocr_config["width_ths"],
            height_ths=ocr_config["height_ths"],
            text_threshold=ocr_config["text_threshold"],
            low_text=ocr_config["low_text"],
            link_threshold=ocr_config["link_threshold"]
        )
        
        # 결과 정리 및 향상
        enhanced_results = []
        for bbox, text, confidence in results:
            if confidence > 0.5:  # 최소 신뢰도 필터링
                # 바운딩 박스 정보 정리
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                element = {
                    "text": text.strip(),
                    "confidence": float(confidence),
                    "bbox": {
                        "x": int(min(x_coords)),
                        "y": int(min(y_coords)),
                        "width": int(max(x_coords) - min(x_coords)),
                        "height": int(max(y_coords) - min(y_coords))
                    },
                    "center": {
                        "x": int(sum(x_coords) / len(x_coords)),
                        "y": int(sum(y_coords) / len(y_coords))
                    }
                }
                
                # 텍스트 타입 분류
                element["text_type"] = self._classify_text_type(text)
                element["hierarchy_level"] = self._determine_hierarchy_level(element, layout_analysis)
                
                enhanced_results.append(element)
        
        # 위치 기반 정렬 (위에서 아래로, 왼쪽에서 오른쪽으로)
        enhanced_results.sort(key=lambda x: (x["bbox"]["y"], x["bbox"]["x"]))
        
        return enhanced_results
    
    def _parse_slide_structure(self, ocr_results: List[Dict], layout_analysis: Dict) -> SlideStructure:
        """슬라이드 구조 파싱"""
        if not ocr_results:
            return SlideStructure(
                slide_number=1,
                title="",
                main_content=[],
                slide_type="unknown",
                layout_type="unknown"
            )
        
        # 제목 찾기
        title_candidates = []
        content_elements = []
        
        for element in ocr_results:
            text = element["text"]
            
            # 제목 후보 판별
            if (element["hierarchy_level"] == 0 or  # 최상위 레벨
                element["text_type"] == "title" or
                self._is_title_text(text)):
                title_candidates.append(element)
            else:
                content_elements.append(element)
        
        # 가장 적절한 제목 선택
        slide_title = ""
        if title_candidates:
            # 위치와 크기를 고려하여 가장 적절한 제목 선택
            best_title = max(title_candidates, 
                           key=lambda x: (x["hierarchy_level"] == 0) * 2 + x["confidence"])
            slide_title = best_title["text"]
        
        # 컨텐츠 요소들을 SlideElement로 변환
        slide_elements = []
        for element in content_elements:
            slide_element = SlideElement(
                element_type=element["text_type"],
                content=element["text"],
                position=(element["bbox"]["x"], element["bbox"]["y"], 
                         element["bbox"]["width"], element["bbox"]["height"]),
                confidence=element["confidence"],
                hierarchy_level=element["hierarchy_level"]
            )
            slide_elements.append(slide_element)
        
        # 슬라이드 타입 결정
        slide_type = self._determine_slide_type(slide_title, slide_elements)
        layout_type = layout_analysis.get("layout_pattern", "unknown")
        
        return SlideStructure(
            slide_number=1,  # 단일 슬라이드 분석시 기본값
            title=slide_title,
            main_content=slide_elements,
            slide_type=slide_type,
            layout_type=layout_type
        )
    
    def _analyze_content_meaning(self, slide_structure: SlideStructure, context: Dict = None) -> Dict[str, Any]:
        """컨텐츠 의미 분석"""
        analysis = {
            "main_topics": [],
            "key_concepts": [],
            "data_points": [],
            "action_items": [],
            "emphasis_elements": []
        }
        
        all_text = slide_structure.title + " " + " ".join([elem.content for elem in slide_structure.main_content])
        
        # 주요 토픽 추출
        for topic_type, keywords in self.content_classifiers["slide_types"].items():
            if any(keyword in all_text for keyword in keywords):
                analysis["main_topics"].append(topic_type)
        
        # 핵심 개념 추출
        for element in slide_structure.main_content:
            text = element.content
            
            # 강조 요소 감지
            if any(pattern in text for pattern in ["핵심", "중요", "반드시", "Key", "Important"]):
                analysis["emphasis_elements"].append(text)
            
            # 데이터 포인트 감지
            if re.search(r'\d+%|\d+명|\d+개|\d+원', text):
                analysis["data_points"].append(text)
            
            # 액션 아이템 감지
            if any(pattern in text for pattern in ["해야", "필요", "권장", "실행", "should", "must"]):
                analysis["action_items"].append(text)
        
        # 컨텍스트 기반 향상
        if context:
            analysis = self._enhance_with_context(analysis, context)
        
        return analysis
    
    def _understand_presentation_flow(self, slide_structure: SlideStructure, content_analysis: Dict) -> Dict[str, Any]:
        """프레젠테이션 플로우 이해"""
        flow_analysis = {
            "slide_position": "unknown",  # beginning, middle, end
            "transition_type": "none",    # intro, content, conclusion
            "information_density": "medium",
            "complexity_level": "medium"
        }
        
        # 슬라이드 위치 추정
        slide_type = slide_structure.slide_type
        if slide_type in ["title", "agenda"]:
            flow_analysis["slide_position"] = "beginning"
        elif slide_type in ["conclusion", "summary", "action"]:
            flow_analysis["slide_position"] = "end"
        else:
            flow_analysis["slide_position"] = "middle"
        
        # 정보 밀도 계산
        total_elements = len(slide_structure.main_content)
        if total_elements <= 3:
            flow_analysis["information_density"] = "low"
        elif total_elements <= 7:
            flow_analysis["information_density"] = "medium"
        else:
            flow_analysis["information_density"] = "high"
        
        # 복잡도 계산
        complexity_indicators = [
            len(content_analysis["data_points"]) > 2,
            len(content_analysis["key_concepts"]) > 3,
            any("차트" in elem.content or "그래프" in elem.content for elem in slide_structure.main_content)
        ]
        
        complexity_score = sum(complexity_indicators)
        if complexity_score >= 2:
            flow_analysis["complexity_level"] = "high"
        elif complexity_score == 1:
            flow_analysis["complexity_level"] = "medium"
        else:
            flow_analysis["complexity_level"] = "low"
        
        return flow_analysis
    
    def _infer_speaker_intent(self, slide_structure: SlideStructure, content_analysis: Dict, context: Dict = None) -> Dict[str, Any]:
        """강연자 의도 추론"""
        intent_analysis = {
            "primary_intent": "inform",
            "secondary_intents": [],
            "audience_level": "general",
            "interaction_expectation": "passive",
            "emphasis_strategy": "text_based"
        }
        
        # 주요 의도 분석
        if slide_structure.slide_type in ["problem", "solution"]:
            intent_analysis["primary_intent"] = "persuade"
        elif slide_structure.slide_type in ["process", "action"]:
            intent_analysis["primary_intent"] = "instruct"
        elif len(content_analysis["data_points"]) > 2:
            intent_analysis["primary_intent"] = "analyze"
        elif slide_structure.slide_type in ["conclusion", "summary"]:
            intent_analysis["primary_intent"] = "summarize"
        
        # 청중 수준 추정
        technical_terms = sum(1 for elem in slide_structure.main_content 
                            if any(char in elem.content for char in "()[]{}%"))
        if technical_terms > 3:
            intent_analysis["audience_level"] = "expert"
        elif technical_terms > 1:
            intent_analysis["audience_level"] = "intermediate"
        
        # 상호작용 기대도
        if any("질문" in elem.content or "?" in elem.content for elem in slide_structure.main_content):
            intent_analysis["interaction_expectation"] = "active"
        
        return intent_analysis
    
    def _extract_key_messages(self, slide_structure: SlideStructure, content_analysis: Dict) -> List[str]:
        """핵심 메시지 추출"""
        key_messages = []
        
        # 제목을 핵심 메시지로
        if slide_structure.title:
            key_messages.append(slide_structure.title)
        
        # 강조 요소들
        key_messages.extend(content_analysis["emphasis_elements"][:3])
        
        # 액션 아이템들
        key_messages.extend(content_analysis["action_items"][:2])
        
        # 중요한 데이터 포인트들
        key_messages.extend(content_analysis["data_points"][:2])
        
        return key_messages[:5]  # 최대 5개
    
    def _analyze_visual_hierarchy(self, slide_structure: SlideStructure) -> Dict[str, Any]:
        """시각적 계층 구조 분석"""
        hierarchy = {
            "title_level": [],
            "primary_level": [],
            "secondary_level": [],
            "detail_level": []
        }
        
        # 제목
        if slide_structure.title:
            hierarchy["title_level"].append(slide_structure.title)
        
        # 컨텐츠 요소들을 계층별로 분류
        for element in slide_structure.main_content:
            if element.hierarchy_level == 0:
                hierarchy["primary_level"].append(element.content)
            elif element.hierarchy_level == 1:
                hierarchy["secondary_level"].append(element.content)
            else:
                hierarchy["detail_level"].append(element.content)
        
        return hierarchy
    
    def _detect_presenter_emphasis(self, slide_structure: SlideStructure) -> List[Dict[str, Any]]:
        """발표자 강조 요소 감지"""
        emphasis_elements = []
        
        for element in slide_structure.main_content:
            text = element.content
            emphasis_score = 0
            emphasis_types = []
            
            # 텍스트 기반 강조
            if any(pattern in text for pattern in ["중요", "핵심", "반드시"]):
                emphasis_score += 2
                emphasis_types.append("text_emphasis")
            
            # 위치 기반 강조 (상단, 중앙)
            if element.position[1] < 200:  # 상단
                emphasis_score += 1
                emphasis_types.append("position_emphasis")
            
            # 크기 기반 강조 (큰 텍스트)
            if element.position[3] > 50:  # 높이가 큰 텍스트
                emphasis_score += 1
                emphasis_types.append("size_emphasis")
            
            if emphasis_score > 0:
                emphasis_elements.append({
                    "content": text,
                    "emphasis_score": emphasis_score,
                    "emphasis_types": emphasis_types
                })
        
        # 강조 점수 기준 정렬
        emphasis_elements.sort(key=lambda x: x["emphasis_score"], reverse=True)
        
        return emphasis_elements[:5]
    
    def _analyze_audience_guidance(self, slide_structure: SlideStructure, speaker_intent: Dict) -> Dict[str, Any]:
        """청중 가이던스 분석"""
        guidance = {
            "reading_flow": "top_to_bottom",
            "attention_points": [],
            "cognitive_load": "medium",
            "interaction_cues": []
        }
        
        # 읽기 흐름 분석
        if len(slide_structure.main_content) > 5:
            guidance["reading_flow"] = "guided_sequence"
        
        # 주의 집중 포인트
        for element in slide_structure.main_content:
            if element.hierarchy_level == 0:
                guidance["attention_points"].append(element.content)
        
        # 인지 부하 계산
        total_elements = len(slide_structure.main_content)
        if total_elements > 7:
            guidance["cognitive_load"] = "high"
        elif total_elements < 3:
            guidance["cognitive_load"] = "low"
        
        # 상호작용 단서
        for element in slide_structure.main_content:
            if "?" in element.content or "질문" in element.content:
                guidance["interaction_cues"].append(element.content)
        
        return guidance
    
    # Helper methods
    def _detect_layout_pattern(self, density_map: np.ndarray) -> str:
        """레이아웃 패턴 감지"""
        # 간단한 패턴 감지 로직
        max_density_pos = np.unravel_index(np.argmax(density_map), density_map.shape)
        
        if max_density_pos[0] < 2:  # 상단에 밀도 높음
            return "title_focused"
        elif max_density_pos[0] > 5:  # 하단에 밀도 높음
            return "content_heavy"
        else:
            return "balanced"
    
    def _identify_content_regions(self, density_map: np.ndarray, width: int, height: int) -> List[Dict]:
        """컨텐츠 영역 식별"""
        regions = []
        grid_size = density_map.shape[0]
        
        for i in range(grid_size):
            for j in range(grid_size):
                if density_map[i, j] > 0.1:  # 임계값 이상
                    region = {
                        "type": "text_region",
                        "bounds": {
                            "x": j * (width // grid_size),
                            "y": i * (height // grid_size),
                            "width": width // grid_size,
                            "height": height // grid_size
                        },
                        "density": float(density_map[i, j])
                    }
                    regions.append(region)
        
        return regions
    
    def _classify_text_type(self, text: str) -> str:
        """텍스트 타입 분류"""
        # 제목 패턴 체크
        for pattern in self.slide_patterns["title_indicators"]:
            if re.match(pattern, text):
                return "title"
        
        # 불릿 포인트 체크
        for pattern in self.slide_patterns["bullet_indicators"]:
            if re.match(pattern, text):
                return "bullet"
        
        # 숫자/데이터 체크
        if re.search(r'\d+%|\d+명|\d+개', text):
            return "data"
        
        return "content"
    
    def _determine_hierarchy_level(self, element: Dict, layout_analysis: Dict) -> int:
        """계층 레벨 결정"""
        # 위치 기반 (상단일수록 높은 레벨)
        y_position = element["bbox"]["y"]
        height = layout_analysis["image_dimensions"]["height"]
        
        if y_position < height * 0.2:  # 상위 20%
            return 0
        elif y_position < height * 0.5:  # 상위 50%
            return 1
        else:
            return 2
    
    def _is_title_text(self, text: str) -> bool:
        """제목 텍스트 여부 판별"""
        # 간단한 휴리스틱
        return (len(text) < 50 and 
                not text.startswith(('-', '•', '*')) and
                not re.search(r'^\d+\)', text))
    
    def _determine_slide_type(self, title: str, elements: List[SlideElement]) -> str:
        """슬라이드 타입 결정"""
        title_lower = title.lower()
        all_content = title + " " + " ".join([elem.content for elem in elements])
        
        for slide_type, keywords in self.content_classifiers["slide_types"].items():
            if any(keyword.lower() in all_content.lower() for keyword in keywords):
                return slide_type
        
        return "content"
    
    def _enhance_with_context(self, analysis: Dict, context: Dict) -> Dict:
        """컨텍스트로 분석 향상"""
        if context.get('topic_keywords'):
            keywords = context['topic_keywords'].split(',')
            for keyword in keywords:
                if keyword.strip() not in analysis["key_concepts"]:
                    analysis["key_concepts"].append(keyword.strip())
        
        return analysis
    
    def _calculate_analysis_confidence(self, slide_structure: SlideStructure, content_analysis: Dict) -> float:
        """분석 신뢰도 계산"""
        factors = [
            bool(slide_structure.title),
            len(slide_structure.main_content) > 0,
            len(content_analysis["main_topics"]) > 0,
            slide_structure.slide_type != "unknown"
        ]
        
        return sum(factors) / len(factors)

# 전역 PPT 분석 엔진
global_ppt_engine = PPTIntelligenceEngine()

def analyze_ppt_slide(image_path: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """PPT 슬라이드 분석 통합 함수"""
    return global_ppt_engine.analyze_ppt_intelligence(image_path, context)

if __name__ == "__main__":
    # 테스트 실행
    print("🎨 PPT 지능형 분석 엔진 테스트")
    
    # 테스트용 이미지 경로 (실제 파일이 있어야 함)
    test_image = "/path/to/test/slide.png"
    test_context = {
        "topic_keywords": "디지털 전환, 혁신",
        "presentation_type": "business",
        "audience": "executives"
    }
    
    if os.path.exists(test_image):
        engine = PPTIntelligenceEngine()
        result = engine.analyze_ppt_intelligence(test_image, test_context)
        
        if result["status"] == "success":
            ppt_intel = result["ppt_intelligence"]
            print(f"슬라이드 제목: {ppt_intel['slide_structure']['title']}")
            print(f"슬라이드 타입: {ppt_intel['slide_structure']['slide_type']}")
            print(f"핵심 메시지: {result['enhanced_understanding']['key_messages']}")
        else:
            print(f"분석 실패: {result['error']}")
    else:
        print("테스트 이미지 파일이 없습니다.")