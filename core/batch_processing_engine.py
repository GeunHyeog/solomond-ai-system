# 다중 파일 배치 처리 엔진 v1.0
# Phase 2: 주얼리 AI 플랫폼 - 실제 현장 요구사항 반영

import asyncio
import uuid
import time
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

class FileType(Enum):
    AUDIO = "audio"
    VIDEO = "video" 
    DOCUMENT = "document"
    IMAGE = "image"

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"

@dataclass
class FileItem:
    """개별 파일 정보"""
    file_id: str
    filename: str
    file_type: FileType
    file_path: str
    size_mb: float
    quality_score: float = 0.0
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    content: str = ""
    extracted_keywords: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    error_message: str = ""

@dataclass
class SessionConfig:
    """세션 설정"""
    session_id: str
    session_name: str
    event_type: str  # "주얼리 전시회", "업계 세미나", "고객 상담", "제품 교육"
    topic: str
    participants: List[str] = field(default_factory=list)
    expected_duration: int = 0  # 분 단위
    priority_file_types: List[FileType] = field(default_factory=list)

@dataclass
class CrossValidation:
    """크로스 검증 결과"""
    common_keywords: List[str] = field(default_factory=list)
    content_overlap_percentage: float = 0.0
    confidence_score: float = 0.0
    contradictions: List[str] = field(default_factory=list)
    missing_content: List[str] = field(default_factory=list)
    verified_content: str = ""

class BatchProcessingEngine:
    """다중 파일 배치 처리 엔진"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.processing_queue = asyncio.Queue()
        self.is_processing = False
        
    async def create_session(self, config: SessionConfig) -> str:
        """새 세션 생성"""
        session_data = {
            "config": config,
            "files": [],
            "status": "created",
            "created_at": time.time(),
            "progress": 0.0,
            "cross_validation": None,
            "final_result": None
        }
        
        self.sessions[config.session_id] = session_data
        return config.session_id
    
    async def add_files_to_session(self, session_id: str, files: List[FileItem]):
        """세션에 파일 추가"""
        if session_id not in self.sessions:
            raise ValueError(f"세션 {session_id}을 찾을 수 없습니다")
        
        session = self.sessions[session_id]
        
        for file_item in files:
            # 파일 해시 생성 (중복 방지)
            file_hash = self._generate_file_hash(file_item.file_path)
            file_item.file_id = file_hash
            
            # 품질 점수 초기 계산
            file_item.quality_score = self._calculate_initial_quality(file_item)
            
            session["files"].append(file_item)
        
        session["status"] = "files_added"
        return len(files)
    
    async def start_batch_processing(self, session_id: str) -> Dict[str, Any]:
        """배치 처리 시작"""
        if session_id not in self.sessions:
            raise ValueError(f"세션 {session_id}을 찾을 수 없습니다")
        
        session = self.sessions[session_id]
        session["status"] = "processing"
        session["started_at"] = time.time()
        
        # 병렬 처리 시작
        files = session["files"]
        
        # 우선순위에 따라 파일 정렬
        sorted_files = self._sort_files_by_priority(files, session["config"])
        
        # 비동기 병렬 처리
        tasks = []
        for file_item in sorted_files:
            task = asyncio.create_task(self._process_single_file(file_item))
            tasks.append(task)
        
        # 모든 파일 처리 완료 대기
        processing_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 업데이트
        for i, result in enumerate(processing_results):
            if isinstance(result, Exception):
                sorted_files[i].processing_status = ProcessingStatus.FAILED
                sorted_files[i].error_message = str(result)
            else:
                sorted_files[i] = result
        
        # 크로스 검증 수행
        cross_validation = await self._perform_cross_validation(sorted_files)
        session["cross_validation"] = cross_validation
        
        # 최종 결과 생성
        final_result = await self._generate_final_result(sorted_files, cross_validation)
        session["final_result"] = final_result
        
        session["status"] = "completed"
        session["completed_at"] = time.time()
        session["progress"] = 100.0
        
        return {
            "session_id": session_id,
            "status": "completed",
            "files_processed": len([f for f in sorted_files if f.processing_status == ProcessingStatus.COMPLETED]),
            "cross_validation": cross_validation,
            "final_result": final_result
        }
    
    async def _process_single_file(self, file_item: FileItem) -> FileItem:
        """개별 파일 처리"""
        start_time = time.time()
        file_item.processing_status = ProcessingStatus.PROCESSING
        
        try:
            # 파일 타입별 처리
            if file_item.file_type == FileType.AUDIO:
                content = await self._process_audio_file(file_item)
            elif file_item.file_type == FileType.VIDEO:
                content = await self._process_video_file(file_item)
            elif file_item.file_type == FileType.DOCUMENT:
                content = await self._process_document_file(file_item)
            elif file_item.file_type == FileType.IMAGE:
                content = await self._process_image_file(file_item)
            else:
                raise ValueError(f"지원하지 않는 파일 타입: {file_item.file_type}")
            
            # 주얼리 특화 후처리
            enhanced_content = await self._enhance_jewelry_content(content)
            
            # 키워드 추출
            keywords = await self._extract_jewelry_keywords(enhanced_content)
            
            # 결과 업데이트
            file_item.content = enhanced_content
            file_item.extracted_keywords = keywords
            file_item.confidence_score = self._calculate_confidence(enhanced_content, keywords)
            file_item.processing_status = ProcessingStatus.COMPLETED
            file_item.processing_time = time.time() - start_time
            
        except Exception as e:
            file_item.processing_status = ProcessingStatus.FAILED
            file_item.error_message = str(e)
            file_item.processing_time = time.time() - start_time
        
        return file_item
    
    async def _process_audio_file(self, file_item: FileItem) -> str:
        """음성 파일 처리 (기존 Whisper STT 활용)"""
        # 실제 구현에서는 기존 Whisper STT 시스템 호출
        await asyncio.sleep(0.5)  # 시뮬레이션
        
        # 모의 STT 결과 (실제로는 Whisper 엔진 호출)
        mock_content = f"""
        2025년 다이아몬드 시장 전망에 대해 말씀드리겠습니다. 
        4C 등급 중에서 특히 컬러와 클래리티 등급이 가격에 미치는 영향이 클 것으로 예상됩니다.
        GIA 인증서의 중요성이 더욱 강조되고 있으며, 프린세스 컷과 라운드 브릴리언트 컷의 수요가 증가하고 있습니다.
        1캐럿 다이아몬드의 도매가격이 전년 대비 15% 상승할 것으로 전망됩니다.
        """
        
        return mock_content.strip()
    
    async def _process_video_file(self, file_item: FileItem) -> str:
        """영상 파일 처리 (음성 추출 후 STT)"""
        await asyncio.sleep(1.0)  # 영상 처리는 더 오래 걸림
        
        # 모의 영상 음성 추출 결과
        mock_content = f"""
        화면에 보시는 것처럼 사파이어의 색상 등급은 매우 중요합니다.
        로얄 블루 사파이어의 경우 캐럿당 가격이 $3,000에서 $5,000 사이입니다.
        스리랑카산과 버마산의 품질 차이를 비교해보겠습니다.
        히트 트리트먼트 여부가 가격에 미치는 영향도 고려해야 합니다.
        """
        
        return mock_content.strip()
    
    async def _process_document_file(self, file_item: FileItem) -> str:
        """문서 파일 처리"""
        await asyncio.sleep(0.3)
        
        # 모의 문서 내용
        mock_content = f"""
        주얼리 시장 동향 보고서
        
        1. 다이아몬드 시장
        - 1캐럿 D-IF 등급: $8,500 (전월 대비 3% 상승)
        - 2캐럿 E-VS1 등급: $25,000 (안정세)
        
        2. 컬러드 스톤 시장  
        - 루비: 버마산 1캐럿 $4,500 (5% 상승)
        - 에메랄드: 콜롬비아산 1캐럿 $3,200 (2% 하락)
        
        3. 트렌드 분석
        - 랩그로운 다이아몬드 수요 증가
        - 서스테이너블 주얼리 관심 확대
        """
        
        return mock_content.strip()
    
    async def _process_image_file(self, file_item: FileItem) -> str:
        """이미지 파일 처리 (OCR)"""
        await asyncio.sleep(0.4)
        
        # 모의 OCR 결과
        mock_content = f"""
        GIA Report Number: 2141234567
        Shape: Round Brilliant
        Carat Weight: 1.52
        Color Grade: F
        Clarity Grade: VS1
        Cut Grade: Excellent
        Polish: Excellent
        Symmetry: Excellent
        Fluorescence: None
        """
        
        return mock_content.strip()
    
    async def _enhance_jewelry_content(self, content: str) -> str:
        """주얼리 특화 내용 향상"""
        # 실제로는 주얼리 용어 데이터베이스를 활용한 보정
        enhanced = content
        
        # 주얼리 용어 정규화 (예시)
        jewelry_terms = {
            "다이아몬드": "Diamond",
            "사파이어": "Sapphire", 
            "루비": "Ruby",
            "에메랄드": "Emerald",
            "4씨": "4C",
            "지아": "GIA"
        }
        
        for korean, english in jewelry_terms.items():
            if korean in enhanced:
                enhanced = enhanced.replace(korean, f"{korean}({english})")
        
        return enhanced
    
    async def _extract_jewelry_keywords(self, content: str) -> List[str]:
        """주얼리 특화 키워드 추출"""
        jewelry_keywords = [
            "다이아몬드", "루비", "사파이어", "에메랄드", "4C", "GIA", 
            "캐럿", "컬러", "클래리티", "컷", "도매가", "소매가",
            "프린세스 컷", "라운드 브릴리언트", "인증서", "감정서"
        ]
        
        found_keywords = []
        content_lower = content.lower()
        
        for keyword in jewelry_keywords:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    async def _perform_cross_validation(self, files: List[FileItem]) -> CrossValidation:
        """파일 간 크로스 검증"""
        completed_files = [f for f in files if f.processing_status == ProcessingStatus.COMPLETED]
        
        if len(completed_files) < 2:
            return CrossValidation(
                confidence_score=1.0 if len(completed_files) == 1 else 0.0,
                verified_content=completed_files[0].content if completed_files else ""
            )
        
        # 공통 키워드 찾기
        all_keywords = []
        for file_item in completed_files:
            all_keywords.extend(file_item.extracted_keywords)
        
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # 2개 이상 파일에서 발견된 키워드를 공통 키워드로 인정
        common_keywords = [k for k, v in keyword_counts.items() if v >= 2]
        
        # 내용 중복도 계산 (간단한 방식)
        all_contents = [f.content for f in completed_files]
        content_overlap = self._calculate_content_overlap(all_contents)
        
        # 신뢰도 계산
        confidence_score = min(0.95, 0.5 + (len(common_keywords) * 0.05) + (content_overlap * 0.4))
        
        # 검증된 내용 생성 (가장 긴 내용을 기준으로)
        verified_content = max(all_contents, key=len)
        
        return CrossValidation(
            common_keywords=common_keywords,
            content_overlap_percentage=content_overlap * 100,
            confidence_score=confidence_score,
            verified_content=verified_content
        )
    
    async def _generate_final_result(self, files: List[FileItem], cross_validation: CrossValidation) -> Dict[str, Any]:
        """최종 통합 결과 생성"""
        completed_files = [f for f in files if f.processing_status == ProcessingStatus.COMPLETED]
        
        # 통계 계산
        total_processing_time = sum(f.processing_time for f in files)
        avg_confidence = sum(f.confidence_score for f in completed_files) / max(len(completed_files), 1)
        
        # 주얼리 특화 인사이트 추출
        jewelry_insights = self._extract_jewelry_insights(cross_validation.verified_content, cross_validation.common_keywords)
        
        return {
            "summary": {
                "total_files": len(files),
                "successfully_processed": len(completed_files),
                "failed_files": len([f for f in files if f.processing_status == ProcessingStatus.FAILED]),
                "total_processing_time": round(total_processing_time, 2),
                "average_confidence": round(avg_confidence, 3),
                "cross_validation_score": round(cross_validation.confidence_score, 3)
            },
            "content": {
                "verified_content": cross_validation.verified_content,
                "common_keywords": cross_validation.common_keywords,
                "content_overlap_percentage": round(cross_validation.content_overlap_percentage, 1)
            },
            "jewelry_insights": jewelry_insights,
            "files_detail": [
                {
                    "filename": f.filename,
                    "type": f.file_type.value,
                    "status": f.processing_status.value,
                    "confidence": round(f.confidence_score, 3),
                    "processing_time": round(f.processing_time, 2),
                    "keywords_found": len(f.extracted_keywords)
                } for f in files
            ]
        }
    
    def _extract_jewelry_insights(self, content: str, keywords: List[str]) -> Dict[str, Any]:
        """주얼리 특화 인사이트 추출"""
        insights = {
            "price_mentions": [],
            "quality_grades": [],
            "market_trends": [],
            "technical_terms": []
        }
        
        # 가격 정보 추출 (간단한 패턴 매칭)
        import re
        price_patterns = [
            r'\$[\d,]+',
            r'[\d,]+달러',
            r'[\d,]+원',
            r'캐럿당 [\d,]+'
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, content)
            insights["price_mentions"].extend(matches)
        
        # 품질 등급 추출
        grade_keywords = ["4C", "GIA", "컷", "컬러", "클래리티", "캐럿"]
        insights["quality_grades"] = [k for k in keywords if k in grade_keywords]
        
        # 시장 트렌드 키워드
        trend_keywords = ["상승", "하락", "트렌드", "전망", "수요", "공급"]
        insights["market_trends"] = [k for k in keywords if k in trend_keywords]
        
        # 기술 용어
        tech_keywords = ["히트 트리트먼트", "프린세스 컷", "라운드 브릴리언트", "인증서"]
        insights["technical_terms"] = [k for k in keywords if k in tech_keywords]
        
        return insights
    
    # 유틸리티 메서드들
    def _generate_file_hash(self, file_path: str) -> str:
        """파일 해시 생성"""
        return hashlib.md5(f"{file_path}_{time.time()}".encode()).hexdigest()[:8]
    
    def _calculate_initial_quality(self, file_item: FileItem) -> float:
        """초기 품질 점수 계산"""
        # 파일 크기 기반 점수 (큰 파일이 일반적으로 더 좋은 품질)
        size_score = min(1.0, file_item.size_mb / 100.0)  # 100MB를 최대로
        
        # 파일 타입별 기본 점수
        type_scores = {
            FileType.AUDIO: 0.8,
            FileType.VIDEO: 0.9,  # 영상이 보통 더 완전한 정보
            FileType.DOCUMENT: 0.95,  # 문서가 가장 정확
            FileType.IMAGE: 0.7
        }
        
        type_score = type_scores.get(file_item.file_type, 0.5)
        
        return (size_score * 0.3 + type_score * 0.7)
    
    def _sort_files_by_priority(self, files: List[FileItem], config: SessionConfig) -> List[FileItem]:
        """우선순위에 따른 파일 정렬"""
        def priority_score(file_item):
            score = file_item.quality_score
            
            # 설정된 우선순위 파일 타입 가산점
            if file_item.file_type in config.priority_file_types:
                score += 0.2
            
            return score
        
        return sorted(files, key=priority_score, reverse=True)
    
    def _calculate_confidence(self, content: str, keywords: List[str]) -> float:
        """신뢰도 계산"""
        if not content:
            return 0.0
        
        # 기본 점수
        base_score = 0.6
        
        # 내용 길이 기반 가산점
        length_score = min(0.2, len(content) / 1000)
        
        # 키워드 개수 기반 가산점  
        keyword_score = min(0.2, len(keywords) * 0.02)
        
        return base_score + length_score + keyword_score
    
    def _calculate_content_overlap(self, contents: List[str]) -> float:
        """내용 중복도 계산"""
        if len(contents) < 2:
            return 1.0
        
        # 간단한 단어 기반 중복도 계산
        all_words = set()
        common_words = set()
        
        word_sets = []
        for content in contents:
            words = set(content.lower().split())
            word_sets.append(words)
            all_words.update(words)
        
        # 모든 파일에 공통으로 나타나는 단어 찾기
        if word_sets:
            common_words = word_sets[0]
            for word_set in word_sets[1:]:
                common_words = common_words.intersection(word_set)
        
        if not all_words:
            return 0.0
        
        return len(common_words) / len(all_words)
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """세션 상태 조회"""
        if session_id not in self.sessions:
            return {"error": "세션을 찾을 수 없습니다"}
        
        session = self.sessions[session_id]
        
        return {
            "session_id": session_id,
            "status": session["status"],
            "progress": session.get("progress", 0),
            "files_count": len(session["files"]),
            "created_at": session["created_at"],
            "config": {
                "session_name": session["config"].session_name,
                "event_type": session["config"].event_type,
                "topic": session["config"].topic
            }
        }

# 사용 예시
async def demo_batch_processing():
    """배치 처리 데모"""
    engine = BatchProcessingEngine()
    
    # 세션 설정
    config = SessionConfig(
        session_id=str(uuid.uuid4()),
        session_name="2025 홍콩주얼리쇼 다이아몬드 세미나",
        event_type="주얼리 전시회",
        topic="2025년 다이아몬드 시장 전망",
        participants=["전근혁 대표", "업계 전문가들"],
        priority_file_types=[FileType.AUDIO, FileType.DOCUMENT]
    )
    
    # 세션 생성
    session_id = await engine.create_session(config)
    print(f"✅ 세션 생성 완료: {session_id}")
    
    # 파일 추가
    files = [
        FileItem(
            file_id="",
            filename="main_recording.mp3",
            file_type=FileType.AUDIO,
            file_path="/path/to/main_recording.mp3",
            size_mb=25.3
        ),
        FileItem(
            file_id="",
            filename="backup_recording.wav", 
            file_type=FileType.AUDIO,
            file_path="/path/to/backup_recording.wav",
            size_mb=18.7
        ),
        FileItem(
            file_id="",
            filename="presentation.mp4",
            file_type=FileType.VIDEO,
            file_path="/path/to/presentation.mp4", 
            size_mb=156.8
        ),
        FileItem(
            file_id="",
            filename="market_report.pdf",
            file_type=FileType.DOCUMENT,
            file_path="/path/to/market_report.pdf",
            size_mb=2.1
        ),
        FileItem(
            file_id="",
            filename="gia_certificate.jpg",
            file_type=FileType.IMAGE,
            file_path="/path/to/gia_certificate.jpg",
            size_mb=0.8
        )
    ]
    
    files_added = await engine.add_files_to_session(session_id, files)
    print(f"✅ 파일 추가 완료: {files_added}개")
    
    # 배치 처리 시작
    print("🚀 배치 처리 시작...")
    result = await engine.start_batch_processing(session_id)
    
    # 결과 출력
    print("\n" + "="*60)
    print("🎉 배치 처리 완료!")
    print("="*60)
    print(f"세션 ID: {result['session_id']}")
    print(f"처리 상태: {result['status']}")
    print(f"처리된 파일: {result['files_processed']}개")
    print(f"신뢰도: {result['cross_validation'].confidence_score:.1%}")
    print(f"공통 키워드: {', '.join(result['cross_validation'].common_keywords)}")
    
    print("\n📊 최종 결과:")
    final_result = result['final_result']
    print(f"- 전체 파일: {final_result['summary']['total_files']}개")
    print(f"- 성공 처리: {final_result['summary']['successfully_processed']}개") 
    print(f"- 평균 신뢰도: {final_result['summary']['average_confidence']:.1%}")
    print(f"- 처리 시간: {final_result['summary']['total_processing_time']:.1f}초")
    
    print("\n💎 주얼리 인사이트:")
    insights = final_result['jewelry_insights']
    if insights['price_mentions']:
        print(f"- 가격 정보: {', '.join(insights['price_mentions'])}")
    if insights['quality_grades']:
        print(f"- 품질 등급: {', '.join(insights['quality_grades'])}")
    
    return result

if __name__ == "__main__":
    # 데모 실행
    import asyncio
    asyncio.run(demo_batch_processing())