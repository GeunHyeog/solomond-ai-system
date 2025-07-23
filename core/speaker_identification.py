#!/usr/bin/env python3
"""
화자 구분 시스템 - 솔로몬드 AI v2.3
"강연자는 누구였고 그 사람들은 뭐라고 말한거야?" 해결
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

class SpeakerIdentificationSystem:
    """화자 구분 및 식별 시스템"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 화자 식별 패턴들
        self.speaker_patterns = {
            "명시적_호명": [
                r"(안녕하세요|반갑습니다)[\s,]*저는?\s+([가-힣]+(?:\s*[가-힣]+)*)\s*(?:입니다|이에요|예요)",
                r"제\s*이름은\s+([가-힣]+(?:\s*[가-힣]+)*)",
                r"([가-힣]+(?:\s*[가-힣]+)*)\s*(?:강사|선생님|대표|교수|박사|님)입니다",
                r"저는\s+([가-힣]+(?:\s*[가-힣]+)*)\s*(?:라고 합니다|라고 불러주세요)"
            ],
            "직책_패턴": [
                r"([가-힣]+(?:\s*[가-힣]+)*)\s*(대표|교수|박사|강사|선생님|팀장|부장|실장)",
                r"(대표|교수|박사|강사|선생님|팀장|부장|실장)\s+([가-힣]+(?:\s*[가-힣]+)*)",
                r"([가-힣]+)\s*회사\s*(대표|CEO|CTO)"
            ],
            "자기소개_패턴": [
                r"오늘\s+강의를?\s+담당할\s+([가-힣]+(?:\s*[가-힣]+)*)",
                r"([가-힣]+(?:\s*[가-힣]+)*)\s*이\s+오늘\s+말씀드릴",
                r"발표자\s+([가-힣]+(?:\s*[가-힣]+)*)"
            ]
        }
        
        # 화자 전환 신호들
        self.speaker_transition_signals = [
            "질문이 있습니다", "답변드리겠습니다", "의견을 말씀드리면",
            "제가 생각하기로는", "저는 반대로", "다른 관점에서",
            "질문자:", "답변자:", "사회자:"
        ]
        
        # 주얼리 업계 전문가 식별 키워드
        self.jewelry_expert_keywords = {
            "보석감정사": ["감정", "등급", "4C", "캐럿", "컬러", "투명도", "커팅"],
            "디자이너": ["디자인", "스케치", "컨셉", "트렌드", "스타일"],
            "세공사": ["세팅", "가공", "마운팅", "세공", "제작"],
            "영업전문가": ["가격", "할인", "프로모션", "상담", "고객"],
            "브랜드매니저": ["브랜드", "마케팅", "런칭", "캠페인"]
        }
        
        self.logger.info("🎭 화자 구분 시스템 초기화 완료")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.SpeakerIdentificationSystem')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def analyze_speakers(self, text: str, segments: List[Dict] = None, context: Dict = None) -> Dict[str, Any]:
        """전체 화자 분석"""
        
        if not text or len(text.strip()) < 20:
            return self._create_empty_speaker_result()
        
        # 1. 명시적 화자 식별
        identified_speakers = self._identify_explicit_speakers(text)
        
        # 2. 암시적 화자 분석 (대화 패턴 기반)
        implicit_speakers = self._analyze_implicit_speakers(text, segments)
        
        # 3. 전문가 역할 추정
        expert_roles = self._identify_expert_roles(text, identified_speakers)
        
        # 4. 발언 내용 화자별 분류
        speaker_statements = self._categorize_statements_by_speaker(text, identified_speakers, implicit_speakers)
        
        # 5. 주요 발언 추출
        key_statements = self._extract_key_statements_per_speaker(speaker_statements)
        
        # 6. 사용자 친화적 요약 생성
        user_summary = self._create_speaker_summary(identified_speakers, expert_roles, key_statements, context)
        
        return {
            "status": "success",
            "identified_speakers": identified_speakers,
            "speaker_count": len(identified_speakers) + len(implicit_speakers),
            "expert_roles": expert_roles,
            "speaker_statements": speaker_statements,
            "key_statements": key_statements,
            "user_summary": user_summary,
            "analysis_confidence": self._calculate_confidence(identified_speakers, implicit_speakers),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _identify_explicit_speakers(self, text: str) -> List[Dict[str, Any]]:
        """명시적으로 언급된 화자들 식별"""
        
        speakers = []
        found_names = set()
        
        for pattern_type, patterns in self.speaker_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    # 이름 추출 (패턴에 따라 그룹 위치가 다름)
                    groups = match.groups()
                    name = None
                    title = None
                    
                    if pattern_type == "명시적_호명":
                        name = groups[1] if len(groups) > 1 else groups[0]
                    elif pattern_type == "직책_패턴":
                        if "대표|교수" in pattern:
                            name = groups[0]
                            title = groups[1]
                        else:
                            title = groups[0]
                            name = groups[1]
                    else:
                        name = groups[0]
                    
                    if name and name not in found_names:
                        speakers.append({
                            "name": name.strip(),
                            "title": title.strip() if title else None,
                            "identification_type": pattern_type,
                            "confidence": 0.9 if pattern_type == "명시적_호명" else 0.7,
                            "context": match.group(0),
                            "position_in_text": match.start()
                        })
                        found_names.add(name)
        
        return speakers
    
    def _analyze_implicit_speakers(self, text: str, segments: List[Dict] = None) -> List[Dict[str, Any]]:
        """암시적 화자 분석 (대화 패턴 기반)"""
        
        implicit_speakers = []
        
        # 화자 전환 신호 탐지
        transitions = []
        for signal in self.speaker_transition_signals:
            if signal in text:
                transitions.append(signal)
        
        # 1인칭 표현 패턴 분석
        first_person_patterns = [
            r"제가\s+([^.!?]*[.!?])",
            r"저는\s+([^.!?]*[.!?])",
            r"개인적으로\s+([^.!?]*[.!?])"
        ]
        
        speaker_segments = []
        for pattern in first_person_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                speaker_segments.append({
                    "content": match.group(0),
                    "position": match.start(),
                    "type": "first_person"
                })
        
        # 최소 2명 이상의 화자가 감지되는 경우
        if len(transitions) >= 2 or len(speaker_segments) >= 3:
            implicit_speakers.append({
                "type": "multiple_speakers_detected",
                "evidence": {
                    "transition_signals": transitions,
                    "first_person_segments": len(speaker_segments)
                },
                "confidence": 0.6
            })
        
        return implicit_speakers
    
    def _identify_expert_roles(self, text: str, speakers: List[Dict]) -> Dict[str, List[str]]:
        """전문가 역할 식별"""
        
        expert_roles = {}
        
        for speaker in speakers:
            name = speaker["name"]
            roles = []
            
            # 화자별 발언 내용에서 전문 영역 추정
            speaker_context = self._extract_speaker_context(text, name)
            
            for role, keywords in self.jewelry_expert_keywords.items():
                keyword_count = sum(1 for keyword in keywords if keyword in speaker_context)
                if keyword_count >= 2:  # 2개 이상 키워드 매치시 해당 역할로 추정
                    roles.append(role)
            
            if roles:
                expert_roles[name] = roles
        
        return expert_roles
    
    def _extract_speaker_context(self, text: str, speaker_name: str) -> str:
        """특정 화자의 발언 맥락 추출"""
        
        # 화자 이름 주변 텍스트 추출
        pattern = rf"{re.escape(speaker_name)}.{{0,500}}"
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        return " ".join(matches)
    
    def _categorize_statements_by_speaker(self, text: str, explicit_speakers: List[Dict], 
                                        implicit_speakers: List[Dict]) -> Dict[str, List[str]]:
        """화자별 발언 분류"""
        
        statements = {}
        
        # 명시적 화자들의 발언 추출
        for speaker in explicit_speakers:
            name = speaker["name"]
            speaker_statements = []
            
            # 화자 이름 이후의 발언들 추출
            name_positions = [m.start() for m in re.finditer(re.escape(name), text, re.IGNORECASE)]
            
            for pos in name_positions:
                # 해당 위치 이후 문장들 추출
                remaining_text = text[pos:pos+1000]  # 1000자 범위
                sentences = re.split(r'[.!?]', remaining_text)
                
                for sentence in sentences[:3]:  # 최대 3문장
                    if len(sentence.strip()) > 10:
                        speaker_statements.append(sentence.strip())
            
            if speaker_statements:
                statements[name] = speaker_statements
        
        return statements
    
    def _extract_key_statements_per_speaker(self, speaker_statements: Dict[str, List[str]]) -> Dict[str, List[Dict]]:
        """화자별 핵심 발언 추출"""
        
        key_statements = {}
        
        for speaker, statements in speaker_statements.items():
            speaker_key_statements = []
            
            for statement in statements:
                # 중요도 점수 계산
                importance_score = self._calculate_statement_importance(statement)
                
                if importance_score > 0.5:  # 임계값 이상인 발언만
                    speaker_key_statements.append({
                        "content": statement,
                        "importance_score": importance_score,
                        "keywords": self._extract_keywords_from_statement(statement)
                    })
            
            # 중요도 순으로 정렬
            speaker_key_statements.sort(key=lambda x: x["importance_score"], reverse=True)
            key_statements[speaker] = speaker_key_statements[:5]  # 상위 5개만
        
        return key_statements
    
    def _calculate_statement_importance(self, statement: str) -> float:
        """발언 중요도 계산"""
        
        score = 0.0
        
        # 주얼리 관련 키워드 보너스
        jewelry_keywords = ["다이아몬드", "금", "은", "반지", "목걸이", "가격", "품질", "디자인"]
        for keyword in jewelry_keywords:
            if keyword in statement:
                score += 0.2
        
        # 감정 표현 보너스
        emotion_words = ["좋다", "훌륭하다", "만족", "추천", "최고", "완벽"]
        for word in emotion_words:
            if word in statement:
                score += 0.1
        
        # 구체적 수치 보너스
        if re.search(r'\d+', statement):
            score += 0.1
        
        # 문장 길이 고려 (너무 짧거나 긴 것 페널티)
        length = len(statement)
        if 20 <= length <= 200:
            score += 0.1
        
        return min(score, 1.0)  # 최대 1.0으로 제한
    
    def _extract_keywords_from_statement(self, statement: str) -> List[str]:
        """발언에서 키워드 추출"""
        
        keywords = []
        
        # 주얼리 관련 키워드 추출
        jewelry_terms = ["다이아몬드", "금", "은", "백금", "루비", "사파이어", "에메랄드", 
                        "반지", "목걸이", "귀걸이", "팔찌", "브로치", "시계"]
        
        for term in jewelry_terms:
            if term in statement:
                keywords.append(term)
        
        return keywords
    
    def _create_speaker_summary(self, speakers: List[Dict], expert_roles: Dict, 
                              key_statements: Dict, context: Dict = None) -> str:
        """사용자 친화적 화자 요약 생성"""
        
        if not speakers:
            return "🤷‍♂️ 명확한 화자를 식별할 수 없었습니다. 음성 품질이나 발언 패턴을 확인해보세요."
        
        summary_parts = []
        
        # 화자 소개
        summary_parts.append(f"🎭 **감지된 화자: {len(speakers)}명**\\n")
        
        for speaker in speakers:
            name = speaker["name"]
            title = speaker.get("title", "")
            title_str = f"({title})" if title else ""
            
            # 전문가 역할 추가
            roles = expert_roles.get(name, [])
            role_str = f" - {', '.join(roles)} 전문가" if roles else ""
            
            summary_parts.append(f"👤 **{name}** {title_str}{role_str}")
            
            # 주요 발언 추가
            if name in key_statements and key_statements[name]:
                top_statement = key_statements[name][0]["content"]
                summary_parts.append(f"   💬 주요 발언: \"{top_statement[:100]}...\"")
            
            summary_parts.append("")
        
        # 전체 대화 특성
        if len(speakers) > 1:
            summary_parts.append("🗣️ **대화 특성**: 다중 화자 간 대화로 분석됨")
        else:
            summary_parts.append("🎤 **발표 특성**: 단일 화자 발표로 분석됨")
        
        return "\\n".join(summary_parts)
    
    def _calculate_confidence(self, explicit_speakers: List[Dict], implicit_speakers: List[Dict]) -> float:
        """분석 신뢰도 계산"""
        
        if not explicit_speakers and not implicit_speakers:
            return 0.0
        
        confidence = 0.0
        
        # 명시적 화자 보너스
        for speaker in explicit_speakers:
            confidence += speaker.get("confidence", 0.5)
        
        # 암시적 화자 보너스
        confidence += len(implicit_speakers) * 0.3
        
        return min(confidence / max(len(explicit_speakers) + len(implicit_speakers), 1), 1.0)
    
    def _create_empty_speaker_result(self) -> Dict[str, Any]:
        """빈 결과 생성"""
        return {
            "status": "no_speakers",
            "identified_speakers": [],
            "speaker_count": 0,
            "expert_roles": {},
            "speaker_statements": {},
            "key_statements": {},
            "user_summary": "🤷‍♂️ 텍스트가 너무 짧거나 화자 정보를 찾을 수 없습니다.",
            "analysis_confidence": 0.0,
            "analysis_timestamp": datetime.now().isoformat()
        }


# 전역 인스턴스 생성
global_speaker_identifier = SpeakerIdentificationSystem()

def analyze_speakers_in_text(text: str, segments: List[Dict] = None, context: Dict = None) -> Dict[str, Any]:
    """화자 분석 함수"""
    return global_speaker_identifier.analyze_speakers(text, segments, context)