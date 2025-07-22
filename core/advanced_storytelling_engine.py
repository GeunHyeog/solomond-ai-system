#!/usr/bin/env python3
"""
고급 스토리텔링 엔진 - GPT-4 API 통합
다중 소스 분석 결과를 하나의 일관된 한국어 이야기로 조합
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False

try:
    import anthropic
    claude_available = True
except ImportError:
    claude_available = False

# 한국어 처리 라이브러리
try:
    from kss import split_sentences
    kss_available = True
except ImportError:
    kss_available = False

try:
    from konlpy.tag import Okt
    konlpy_available = True
except ImportError:
    konlpy_available = False

class AdvancedStorytellingEngine:
    """
    다중 소스 분석 결과를 하나의 일관된 한국어 스토리로 변환하는 고급 엔진
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        
        # AI 클라이언트 초기화
        self.gpt4_client = None
        self.claude_client = None
        self.korean_analyzer = None
        
        self._initialize_ai_clients()
        self._initialize_korean_nlp()
        
        # 스토리 템플릿
        self.story_templates = self._load_story_templates()
        
        self.logger.info("🎭 고급 스토리텔링 엔진 초기화 완료")
    
    def _setup_logger(self):
        """로거 설정"""
        logger = logging.getLogger(f"{__name__}.AdvancedStorytellingEngine")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_ai_clients(self):
        """AI 클라이언트 초기화"""
        
        # OpenAI GPT-4 클라이언트
        if openai_available:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                try:
                    self.gpt4_client = OpenAI(api_key=api_key)
                    self.logger.info("✅ GPT-4 클라이언트 초기화 성공")
                except Exception as e:
                    self.logger.warning(f"⚠️ GPT-4 초기화 실패: {e}")
            else:
                self.logger.info("ℹ️ OPENAI_API_KEY 환경변수가 설정되지 않음")
        
        # Anthropic Claude 클라이언트
        if claude_available:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                try:
                    self.claude_client = anthropic.Anthropic(api_key=api_key)
                    self.logger.info("✅ Claude 클라이언트 초기화 성공")
                except Exception as e:
                    self.logger.warning(f"⚠️ Claude 초기화 실패: {e}")
            else:
                self.logger.info("ℹ️ ANTHROPIC_API_KEY 환경변수가 설정되지 않음")
    
    def _initialize_korean_nlp(self):
        """한국어 NLP 도구 초기화"""
        if konlpy_available:
            try:
                self.korean_analyzer = Okt()
                self.logger.info("✅ 한국어 형태소 분석기 초기화 성공")
            except Exception as e:
                self.logger.warning(f"⚠️ 한국어 분석기 초기화 실패: {e}")
    
    def _load_story_templates(self):
        """스토리 템플릿 로드"""
        return {
            "general": {
                "intro": "다음은 분석된 내용을 바탕으로 구성한 종합적인 이야기입니다.",
                "sections": ["상황 개요", "주요 대화 내용", "핵심 메시지", "결론 및 인사이트"],
                "style": "자연스럽고 이해하기 쉬운 한국어로 작성"
            },
            "consultation": {
                "intro": "고객 상담 내용을 분석한 결과입니다.",
                "sections": ["고객 요청사항", "상담 진행과정", "제공된 정보", "고객 반응", "후속 조치"],
                "style": "비즈니스 상황에 적합한 정중하고 명확한 한국어"
            },
            "meeting": {
                "intro": "회의 내용을 종합 분석한 결과입니다.",
                "sections": ["회의 목적", "주요 안건", "논의 사항", "결정 사항", "액션 아이템"],
                "style": "비즈니스 맥락에서 사용하는 정확하고 간결한 한국어"
            },
            "multimedia": {
                "intro": "여러 매체를 종합 분석하여 하나의 스토리로 재구성했습니다.",
                "sections": ["시간순 전개", "핵심 내용", "등장인물/화자", "주요 메시지", "종합 평가"],
                "style": "서사적이고 흥미로운 한국어 스토리텔링"
            }
        }
    
    def create_comprehensive_story(self, analysis_results: Dict[str, Any], 
                                   story_type: str = "general") -> Dict[str, Any]:
        """
        다중 소스 분석 결과를 하나의 일관된 한국어 스토리로 변환
        
        Args:
            analysis_results: 다중 소스 분석 결과
            story_type: 스토리 유형 (general, consultation, meeting, multimedia)
            
        Returns:
            Dict containing the generated Korean story
        """
        
        try:
            self.logger.info(f"🎭 종합 스토리 생성 시작 - 유형: {story_type}")
            
            # 1. 데이터 전처리 및 구조화
            structured_data = self._structure_analysis_data(analysis_results)
            
            # 2. 시간순 정렬 및 맥락 분석
            temporal_context = self._analyze_temporal_context(structured_data)
            
            # 3. 핵심 메시지 추출
            key_messages = self._extract_key_messages(structured_data)
            
            # 4. AI 기반 스토리 생성
            generated_story = self._generate_ai_story(
                structured_data, temporal_context, key_messages, story_type
            )
            
            # 5. 한국어 문장 다듬기
            polished_story = self._polish_korean_text(generated_story)
            
            # 6. 결과 구성
            story_result = {
                "status": "success",
                "story_type": story_type,
                "generated_at": datetime.now().isoformat(),
                "story": polished_story,
                "metadata": {
                    "source_count": len(analysis_results.get("sources", [])),
                    "total_content_length": len(str(structured_data)),
                    "key_messages_count": len(key_messages),
                    "ai_engine_used": self._get_available_ai_engine()
                }
            }
            
            self.logger.info("✅ 종합 스토리 생성 완료")
            return story_result
            
        except Exception as e:
            self.logger.error(f"❌ 스토리 생성 실패: {e}")
            return {
                "status": "error",
                "error": str(e),
                "story_type": story_type,
                "fallback_story": self._create_fallback_story(analysis_results)
            }
    
    def _structure_analysis_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과를 구조화된 데이터로 변환"""
        
        structured = {
            "audio_content": [],
            "visual_content": [], 
            "document_content": [],
            "metadata": {
                "sources": [],
                "timestamps": [],
                "confidence_scores": []
            }
        }
        
        sources = analysis_results.get("sources", [])
        
        for source in sources:
            source_type = source.get("type", "unknown")
            content = source.get("analysis_result", {})
            
            if source_type == "audio":
                if content.get("transcription"):
                    structured["audio_content"].append({
                        "text": content["transcription"],
                        "confidence": content.get("confidence", 0.0),
                        "timestamp": source.get("timestamp"),
                        "source_name": source.get("name", "Unknown")
                    })
            
            elif source_type == "image":
                if content.get("text"):
                    structured["visual_content"].append({
                        "text": content["text"],
                        "confidence": content.get("confidence", 0.0),
                        "source_name": source.get("name", "Unknown")
                    })
            
            elif source_type == "document":
                if content.get("text"):
                    structured["document_content"].append({
                        "text": content["text"],
                        "source_name": source.get("name", "Unknown")
                    })
            
            # 메타데이터 수집
            structured["metadata"]["sources"].append(source.get("name", "Unknown"))
            if source.get("timestamp"):
                structured["metadata"]["timestamps"].append(source["timestamp"])
        
        return structured
    
    def _analyze_temporal_context(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """시간적 맥락 분석"""
        
        timestamps = structured_data["metadata"]["timestamps"]
        
        if not timestamps:
            return {"has_temporal_data": False}
        
        # 시간순 정렬
        sorted_timestamps = sorted(timestamps)
        
        return {
            "has_temporal_data": True,
            "start_time": sorted_timestamps[0] if sorted_timestamps else None,
            "end_time": sorted_timestamps[-1] if sorted_timestamps else None,
            "duration": len(sorted_timestamps),
            "temporal_flow": "sequential" if len(set(sorted_timestamps)) > 1 else "single_moment"
        }
    
    def _extract_key_messages(self, structured_data: Dict[str, Any]) -> List[str]:
        """핵심 메시지 추출"""
        
        key_messages = []
        
        # 오디오 내용에서 핵심 메시지
        for audio in structured_data["audio_content"]:
            text = audio["text"]
            if len(text) > 50:  # 충분한 길이의 텍스트만
                # 간단한 키워드 추출 (추후 고도화 필요)
                if any(keyword in text for keyword in ["가격", "구매", "문의", "주문"]):
                    key_messages.append(f"구매 관련: {text[:100]}...")
                elif any(keyword in text for keyword in ["문제", "오류", "수리"]):
                    key_messages.append(f"문제 해결: {text[:100]}...")
                else:
                    key_messages.append(f"일반 대화: {text[:100]}...")
        
        # 시각적 내용에서 핵심 정보
        for visual in structured_data["visual_content"]:
            text = visual["text"]
            if len(text) > 20:
                key_messages.append(f"화면 정보: {text[:80]}...")
        
        return key_messages[:10]  # 최대 10개 핵심 메시지
    
    def _generate_ai_story(self, structured_data: Dict[str, Any], 
                          temporal_context: Dict[str, Any],
                          key_messages: List[str], 
                          story_type: str) -> str:
        """AI를 활용한 스토리 생성"""
        
        template = self.story_templates.get(story_type, self.story_templates["general"])
        
        # 프롬프트 구성
        prompt = self._create_story_prompt(structured_data, temporal_context, key_messages, template)
        
        # AI 엔진 선택 및 스토리 생성
        if self.gpt4_client:
            return self._generate_with_gpt4(prompt)
        elif self.claude_client:
            return self._generate_with_claude(prompt)
        else:
            return self._generate_with_local_model(structured_data, key_messages)
    
    def _create_story_prompt(self, structured_data: Dict[str, Any],
                           temporal_context: Dict[str, Any], 
                           key_messages: List[str],
                           template: Dict[str, Any]) -> str:
        """스토리 생성용 프롬프트 작성"""
        
        audio_texts = [item["text"] for item in structured_data["audio_content"]]
        visual_texts = [item["text"] for item in structured_data["visual_content"]]
        document_texts = [item["text"] for item in structured_data["document_content"]]
        
        prompt = f"""
다음의 다각도 분석 결과를 바탕으로 일관된 한국어 이야기를 구성해주세요.

{template["intro"]}

=== 분석 데이터 ===

**음성/대화 내용:**
{chr(10).join(audio_texts[:5]) if audio_texts else "음성 데이터 없음"}

**화면/이미지 텍스트:**
{chr(10).join(visual_texts[:3]) if visual_texts else "이미지 데이터 없음"}

**문서 내용:**
{chr(10).join(document_texts[:3]) if document_texts else "문서 데이터 없음"}

**핵심 메시지:**
{chr(10).join(key_messages[:5])}

**시간적 맥락:**
{json.dumps(temporal_context, ensure_ascii=False, indent=2)}

=== 요청사항 ===

다음 구조로 {template["style"]}을 사용해서 종합적인 이야기를 만들어주세요:

{chr(10).join([f"{i+1}. {section}" for i, section in enumerate(template["sections"])])}

**중요한 요구사항:**
1. 모든 소스의 정보를 종합적으로 활용하세요
2. 시간순이나 논리적 순서로 내용을 구성하세요  
3. "누가 무엇을 말했는지" 명확하게 파악할 수 있도록 작성하세요
4. 실행 가능한 인사이트나 결론을 포함하세요
5. 자연스럽고 읽기 쉬운 한국어로 작성하세요

최종 결과는 완전한 하나의 이야기 형태로 제공해주세요.
"""
        
        return prompt
    
    def _generate_with_gpt4(self, prompt: str) -> str:
        """GPT-4를 사용한 스토리 생성"""
        
        try:
            response = self.gpt4_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 다중 소스 정보를 하나의 일관된 한국어 스토리로 구성하는 전문 작가입니다. 정확하고 자연스러운 한국어를 사용해서 읽기 쉬운 이야기를 만드세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"GPT-4 스토리 생성 실패: {e}")
            raise
    
    def _generate_with_claude(self, prompt: str) -> str:
        """Claude를 사용한 스토리 생성"""
        
        try:
            response = self.claude_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            self.logger.error(f"Claude 스토리 생성 실패: {e}")
            raise
    
    def _generate_with_local_model(self, structured_data: Dict[str, Any], 
                                  key_messages: List[str]) -> str:
        """로컬 모델을 사용한 기본 스토리 생성"""
        
        # 간단한 템플릿 기반 스토리 생성 (AI API가 없을 때 폴백)
        story_parts = []
        
        story_parts.append("## 종합 분석 결과")
        story_parts.append("")
        
        if structured_data["audio_content"]:
            story_parts.append("### 대화 내용")
            for i, audio in enumerate(structured_data["audio_content"][:3], 1):
                story_parts.append(f"{i}. {audio['text'][:200]}...")
            story_parts.append("")
        
        if structured_data["visual_content"]:
            story_parts.append("### 화면/문서 정보")  
            for i, visual in enumerate(structured_data["visual_content"][:3], 1):
                story_parts.append(f"{i}. {visual['text'][:150]}...")
            story_parts.append("")
        
        if key_messages:
            story_parts.append("### 핵심 메시지")
            for i, message in enumerate(key_messages[:5], 1):
                story_parts.append(f"{i}. {message}")
            story_parts.append("")
        
        story_parts.append("### 종합 의견")
        story_parts.append("위의 내용을 종합해보면, 주요 대화나 상호작용이 이루어졌으며, 구체적인 내용과 맥락을 파악할 수 있습니다.")
        
        return "\n".join(story_parts)
    
    def _polish_korean_text(self, text: str) -> str:
        """한국어 텍스트 다듬기"""
        
        if not text:
            return text
            
        # 기본적인 정제 작업
        polished = text.strip()
        
        # 한국어 문장 분리 (kss 사용 가능할 때)
        if kss_available:
            try:
                sentences = split_sentences(polished)
                polished = "\n\n".join(sentences) if len(sentences) > 1 else polished
            except:
                pass
        
        # 기본적인 문장 정제
        polished = polished.replace("  ", " ")  # 이중 공백 제거
        polished = polished.replace("\n\n\n", "\n\n")  # 과도한 줄바꿈 제거
        
        return polished
    
    def _get_available_ai_engine(self) -> str:
        """사용 가능한 AI 엔진 반환"""
        if self.gpt4_client:
            return "GPT-4"
        elif self.claude_client:
            return "Claude"
        else:
            return "Local Template"
    
    def _create_fallback_story(self, analysis_results: Dict[str, Any]) -> str:
        """폴백 스토리 생성"""
        return f"""
## 분석 결과 요약

총 {len(analysis_results.get('sources', []))}개의 소스가 분석되었습니다.

분석된 내용을 통해 다양한 정보와 대화가 포함되어 있음을 확인했습니다.
더 상세한 분석을 위해서는 AI 엔진 설정이 필요합니다.

**참고**: OpenAI API Key 또는 Anthropic API Key를 환경변수에 설정하면 
더욱 상세하고 일관된 스토리 생성이 가능합니다.
"""

# 전역 스토리텔링 엔진 인스턴스
global_storytelling_engine = AdvancedStorytellingEngine()

def create_comprehensive_korean_story(analysis_results: Dict[str, Any], 
                                     story_type: str = "general") -> Dict[str, Any]:
    """
    다중 소스 분석 결과를 하나의 한국어 스토리로 변환하는 편의 함수
    
    Args:
        analysis_results: 분석 결과 딕셔너리
        story_type: 스토리 유형 (general, consultation, meeting, multimedia)
    
    Returns:
        생성된 한국어 스토리 딕셔너리
    """
    return global_storytelling_engine.create_comprehensive_story(analysis_results, story_type)