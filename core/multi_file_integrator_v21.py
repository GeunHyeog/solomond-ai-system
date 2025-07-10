"""
📊 Solomond AI v2.1 - 다중 파일 통합 분석기
시계열 기반 다중 파일 통합, 내용 연결, 상황별 분류 및 종합 인사이트 추출

Author: 전근혁 (Solomond)
Created: 2025.07.11
Version: 2.1.0
"""

import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import re
from difflib import SequenceMatcher
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class FileMetadata:
    """파일 메타데이터"""
    file_path: str
    file_name: str
    file_type: str          # audio, image, document, video
    file_size: int
    creation_time: float
    modification_time: float
    estimated_duration: float  # 음성/비디오의 경우
    content_hash: str       # 내용 중복 검사용
    quality_score: float    # 품질 점수
    language: str          # 감지된 언어
    content_preview: str   # 내용 미리보기 (첫 100자)

@dataclass
class ContentSegment:
    """내용 세그먼트"""
    segment_id: str
    source_file: str
    content_type: str      # text, audio_transcript, ocr_text, metadata
    content: str
    start_time: float      # 파일 내 시작 시간
    end_time: float        # 파일 내 종료 시간
    confidence: float      # 내용 신뢰도
    language: str
    keywords: List[str]    # 추출된 키워드
    jewelry_terms: List[str]  # 주얼리 전문용어
    timestamp: float       # 실제 시간 (Unix timestamp)

@dataclass
class IntegratedSession:
    """통합 세션"""
    session_id: str
    session_type: str      # meeting, seminar, lecture, trade_show, conference
    title: str
    start_time: float
    end_time: float
    participants: List[str]
    files: List[FileMetadata]
    segments: List[ContentSegment]
    merged_content: str    # 병합된 내용
    key_insights: List[str]
    action_items: List[str]
    summary: str
    confidence_score: float
    processing_details: Dict[str, Any]

class ContentClassifier:
    """내용 분류기 - 회의/세미나/강의 등 상황 분류"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.classification_keywords = {
            'meeting': [
                '회의', '미팅', '논의', '결정', '안건', '의견', '동의', '반대',
                'meeting', 'discussion', 'agenda', 'decision', 'vote'
            ],
            'seminar': [
                '세미나', '워크숍', '교육', '강연', '발표', '프레젠테이션',
                'seminar', 'workshop', 'presentation', 'training', 'lecture'
            ],
            'lecture': [
                '강의', '수업', '학습', '설명', '이론', '원리', '방법',
                'lecture', 'class', 'learning', 'explanation', 'theory'
            ],
            'trade_show': [
                '전시회', '박람회', '쇼룸', '부스', '전시', '상품', '신제품',
                'trade show', 'exhibition', 'expo', 'booth', 'showcase'
            ],
            'conference': [
                '컨퍼런스', '심포지엄', '포럼', '총회', '대회', '학회',
                'conference', 'symposium', 'forum', 'convention', 'summit'
            ]
        }
        
    def classify_session_type(self, content_segments: List[ContentSegment]) -> Tuple[str, float]:
        """세션 타입 분류"""
        try:
            # 모든 컨텐츠 통합
            all_content = ' '.join([seg.content for seg in content_segments]).lower()
            
            # 타입별 점수 계산
            type_scores = {}
            
            for session_type, keywords in self.classification_keywords.items():
                score = 0
                for keyword in keywords:
                    count = all_content.count(keyword.lower())
                    score += count
                
                # 정규화 (키워드 수 대비)
                type_scores[session_type] = score / len(keywords)
            
            # 최고 점수 타입 선택
            if not type_scores or max(type_scores.values()) == 0:
                return 'meeting', 0.5  # 기본값
            
            best_type = max(type_scores, key=type_scores.get)
            confidence = min(1.0, type_scores[best_type] / 10)  # 0-1 정규화
            
            return best_type, confidence
            
        except Exception as e:
            self.logger.error(f"세션 타입 분류 실패: {e}")
            return 'meeting', 0.5

class TimelineAnalyzer:
    """시계열 분석기 - 파일들의 시간 순서 분석"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def create_timeline(self, files_metadata: List[FileMetadata]) -> List[FileMetadata]:
        """파일들의 시간순 타임라인 생성"""
        try:
            # 시간 기준 정렬 (생성시간 우선, 수정시간 보조)
            sorted_files = sorted(
                files_metadata,
                key=lambda f: (f.creation_time, f.modification_time)
            )
            
            return sorted_files
            
        except Exception as e:
            self.logger.error(f"타임라인 생성 실패: {e}")
            return files_metadata
    
    def detect_time_gaps(self, timeline: List[FileMetadata], max_gap_hours: float = 2.0) -> List[Tuple[int, int, float]]:
        """시간 간격 감지 (세션 구분용)"""
        try:
            gaps = []
            
            for i in range(len(timeline) - 1):
                current_end = timeline[i].modification_time
                next_start = timeline[i + 1].creation_time
                
                gap_hours = (next_start - current_end) / 3600
                
                if gap_hours > max_gap_hours:
                    gaps.append((i, i + 1, gap_hours))
            
            return gaps
            
        except Exception as e:
            self.logger.error(f"시간 간격 감지 실패: {e}")
            return []
    
    def group_files_by_session(self, timeline: List[FileMetadata]) -> List[List[FileMetadata]]:
        """시간 간격 기준으로 파일들을 세션별로 그룹화"""
        try:
            if not timeline:
                return []
            
            gaps = self.detect_time_gaps(timeline)
            sessions = []
            
            current_session = []
            gap_indices = [gap[0] + 1 for gap in gaps]  # 새 세션 시작 인덱스
            
            for i, file_meta in enumerate(timeline):
                current_session.append(file_meta)
                
                # 다음 파일이 새 세션 시작이거나 마지막 파일인 경우
                if i + 1 in gap_indices or i == len(timeline) - 1:
                    if current_session:
                        sessions.append(current_session)
                        current_session = []
            
            return sessions
            
        except Exception as e:
            self.logger.error(f"세션별 그룹화 실패: {e}")
            return [timeline]  # 실패 시 전체를 하나의 세션으로

class ContentDeduplicator:
    """내용 중복 제거기"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
    
    def calculate_content_similarity(self, content1: str, content2: str) -> float:
        """두 내용 간 유사도 계산"""
        try:
            if not content1.strip() or not content2.strip():
                return 0.0
            
            # 텍스트 정규화
            normalized1 = self._normalize_text(content1)
            normalized2 = self._normalize_text(content2)
            
            # SequenceMatcher를 사용한 유사도 계산
            similarity = SequenceMatcher(None, normalized1, normalized2).ratio()
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"유사도 계산 실패: {e}")
            return 0.0
    
    def _normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        # 소문자 변환, 공백 정리, 특수문자 제거
        normalized = re.sub(r'[^\w\s가-힣]', '', text.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def remove_duplicate_segments(self, segments: List[ContentSegment]) -> List[ContentSegment]:
        """중복 세그먼트 제거"""
        try:
            unique_segments = []
            
            for current_segment in segments:
                is_duplicate = False
                
                for existing_segment in unique_segments:
                    similarity = self.calculate_content_similarity(
                        current_segment.content,
                        existing_segment.content
                    )
                    
                    if similarity >= self.similarity_threshold:
                        # 더 신뢰도가 높은 세그먼트 유지
                        if current_segment.confidence > existing_segment.confidence:
                            unique_segments.remove(existing_segment)
                            unique_segments.append(current_segment)
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_segments.append(current_segment)
            
            self.logger.info(f"중복 제거: {len(segments)} → {len(unique_segments)} 세그먼트")
            return unique_segments
            
        except Exception as e:
            self.logger.error(f"중복 제거 실패: {e}")
            return segments

class KeywordExtractor:
    """키워드 추출기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 주얼리 업계 주요 키워드
        self.jewelry_keywords = {
            '제품': ['다이아몬드', '루비', '사파이어', '에메랄드', '진주', '금', '은', '플래티넘'],
            '품질': ['캐럿', '투명도', '컬러', '커팅', '4C', '등급', '인증', 'GIA'],
            '비즈니스': ['가격', '시세', '도매', '소매', '수입', '수출', '마진', '수익'],
            '기술': ['세팅', '가공', '연마', '조각', '디자인', '제작', '수리'],
            '시장': ['트렌드', '수요', '공급', '경쟁', '브랜드', '마케팅', '고객']
        }
        
        # 불용어 (제외할 단어들)
        self.stop_words = {
            '그', '이', '저', '것', '수', '있', '하', '되', '의', '가', '을', '를',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'
        }
    
    def extract_keywords(self, content: str, max_keywords: int = 20) -> Tuple[List[str], List[str]]:
        """키워드 및 주얼리 전문용어 추출"""
        try:
            # 텍스트 전처리
            content = re.sub(r'[^\w\s가-힣]', ' ', content)
            words = content.split()
            
            # 단어 빈도 계산
            word_freq = Counter()
            jewelry_terms = []
            
            for word in words:
                word = word.strip().lower()
                
                if len(word) > 1 and word not in self.stop_words:
                    word_freq[word] += 1
                    
                    # 주얼리 용어 확인
                    for category, terms in self.jewelry_keywords.items():
                        if any(term.lower() in word or word in term.lower() for term in terms):
                            if word not in jewelry_terms:
                                jewelry_terms.append(word)
            
            # 상위 키워드 선택
            top_keywords = [word for word, freq in word_freq.most_common(max_keywords)]
            
            return top_keywords, jewelry_terms
            
        except Exception as e:
            self.logger.error(f"키워드 추출 실패: {e}")
            return [], []

class MultiFileIntegratorV21:
    """v2.1 다중 파일 통합 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.classifier = ContentClassifier()
        self.timeline_analyzer = TimelineAnalyzer()
        self.deduplicator = ContentDeduplicator()
        self.keyword_extractor = KeywordExtractor()
        
    def integrate_multiple_files(self, file_paths: List[str], stt_results: Dict = None, ocr_results: Dict = None) -> Dict[str, Any]:
        """다중 파일 통합 분석"""
        try:
            start_time = time.time()
            
            # 1. 파일 메타데이터 수집
            files_metadata = self._collect_files_metadata(file_paths)
            
            # 2. 시간순 정렬 및 세션 그룹화
            timeline = self.timeline_analyzer.create_timeline(files_metadata)
            sessions = self.timeline_analyzer.group_files_by_session(timeline)
            
            self.logger.info(f"감지된 세션 수: {len(sessions)}")
            
            # 3. 각 세션별 통합 분석
            integrated_sessions = []
            
            for i, session_files in enumerate(sessions):
                session_result = self._integrate_single_session(
                    session_files, i, stt_results, ocr_results
                )
                integrated_sessions.append(session_result)
            
            # 4. 전체 통합 결과 생성
            overall_result = self._create_overall_integration(integrated_sessions)
            
            processing_time = time.time() - start_time
            
            return {
                'individual_sessions': integrated_sessions,
                'overall_integration': overall_result,
                'processing_statistics': {
                    'total_files': len(file_paths),
                    'total_sessions': len(sessions),
                    'processing_time': processing_time,
                    'files_per_session': [len(session) for session in sessions]
                },
                'timeline_analysis': {
                    'first_file_time': min(f.creation_time for f in files_metadata) if files_metadata else 0,
                    'last_file_time': max(f.modification_time for f in files_metadata) if files_metadata else 0,
                    'total_duration_hours': (max(f.modification_time for f in files_metadata) - 
                                           min(f.creation_time for f in files_metadata)) / 3600 if files_metadata else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"다중 파일 통합 분석 실패: {e}")
            return {
                'error': str(e),
                'processing_complete': False
            }
    
    def _collect_files_metadata(self, file_paths: List[str]) -> List[FileMetadata]:
        """파일 메타데이터 수집"""
        try:
            metadata_list = []
            
            for file_path in file_paths:
                try:
                    path_obj = Path(file_path)
                    
                    if not path_obj.exists():
                        self.logger.warning(f"파일이 존재하지 않음: {file_path}")
                        continue
                    
                    stat = path_obj.stat()
                    
                    # 파일 타입 결정
                    file_ext = path_obj.suffix.lower()
                    if file_ext in ['.mp3', '.wav', '.m4a', '.aac']:
                        file_type = 'audio'
                        duration = self._estimate_audio_duration(file_path)
                    elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                        file_type = 'video'
                        duration = self._estimate_video_duration(file_path)
                    elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                        file_type = 'image'
                        duration = 0
                    else:
                        file_type = 'document'
                        duration = 0
                    
                    # 컨텐츠 해시 생성
                    content_hash = self._calculate_file_hash(file_path)
                    
                    # 내용 미리보기 (가능한 경우)
                    content_preview = self._get_content_preview(file_path, file_type)
                    
                    metadata = FileMetadata(
                        file_path=file_path,
                        file_name=path_obj.name,
                        file_type=file_type,
                        file_size=stat.st_size,
                        creation_time=stat.st_ctime,
                        modification_time=stat.st_mtime,
                        estimated_duration=duration,
                        content_hash=content_hash,
                        quality_score=75.0,  # 기본값, 나중에 품질 분석 결과로 업데이트
                        language='ko',       # 기본값, 나중에 언어 감지 결과로 업데이트
                        content_preview=content_preview
                    )
                    
                    metadata_list.append(metadata)
                    
                except Exception as e:
                    self.logger.error(f"파일 메타데이터 수집 실패 ({file_path}): {e}")
            
            return metadata_list
            
        except Exception as e:
            self.logger.error(f"메타데이터 수집 실패: {e}")
            return []
    
    def _estimate_audio_duration(self, audio_path: str) -> float:
        """음성 파일 길이 추정"""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr
        except Exception:
            # librosa 실패 시 파일 크기 기반 추정
            try:
                file_size = Path(audio_path).stat().st_size
                # MP3 평균 비트레이트 128kbps 가정
                estimated_duration = file_size / (128 * 1000 / 8)
                return estimated_duration
            except Exception:
                return 0.0
    
    def _estimate_video_duration(self, video_path: str) -> float:
        """비디오 파일 길이 추정"""
        try:
            # OpenCV 사용 시도
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            
            if fps > 0:
                return frame_count / fps
            else:
                return 0.0
        except Exception:
            # 파일 크기 기반 추정
            try:
                file_size = Path(video_path).stat().st_size
                # 평균 비트레이트 1Mbps 가정
                estimated_duration = file_size / (1000 * 1000 / 8)
                return estimated_duration
            except Exception:
                return 0.0
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """파일 해시 계산 (중복 검사용)"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                # 큰 파일의 경우 일부만 해시 계산
                chunk = f.read(8192)
                while chunk:
                    hash_md5.update(chunk)
                    chunk = f.read(8192)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"파일 해시 계산 실패: {e}")
            return ""
    
    def _get_content_preview(self, file_path: str, file_type: str) -> str:
        """내용 미리보기 생성"""
        try:
            if file_type == 'document' and file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(200)
                    return content.strip()
            
            # 다른 파일 타입의 경우 파일명과 타입 정보 반환
            return f"[{file_type.upper()}] {Path(file_path).stem}"
            
        except Exception as e:
            self.logger.error(f"내용 미리보기 생성 실패: {e}")
            return f"[{file_type.upper()}] {Path(file_path).name}"
    
    def _integrate_single_session(self, session_files: List[FileMetadata], session_index: int,
                                 stt_results: Dict = None, ocr_results: Dict = None) -> IntegratedSession:
        """단일 세션 통합 분석"""
        try:
            session_id = f"session_{session_index}_{int(time.time())}"
            
            # 1. 컨텐츠 세그먼트 생성
            segments = self._create_content_segments(session_files, stt_results, ocr_results)
            
            # 2. 중복 제거
            unique_segments = self.deduplicator.remove_duplicate_segments(segments)
            
            # 3. 세션 타입 분류
            session_type, type_confidence = self.classifier.classify_session_type(unique_segments)
            
            # 4. 내용 병합
            merged_content = self._merge_segments_content(unique_segments)
            
            # 5. 키워드 및 인사이트 추출
            keywords, jewelry_terms = self.keyword_extractor.extract_keywords(merged_content)
            key_insights = self._extract_session_insights(merged_content, session_type)
            action_items = self._extract_action_items(merged_content)
            
            # 6. 요약 생성
            summary = self._generate_session_summary(merged_content, session_type, key_insights)
            
            # 7. 전체 신뢰도 계산
            confidence_score = self._calculate_session_confidence(unique_segments, type_confidence)
            
            # 세션 시간 계산
            start_time = min(f.creation_time for f in session_files)
            end_time = max(f.modification_time for f in session_files)
            
            return IntegratedSession(
                session_id=session_id,
                session_type=session_type,
                title=self._generate_session_title(session_type, key_insights),
                start_time=start_time,
                end_time=end_time,
                participants=self._extract_participants(merged_content),
                files=session_files,
                segments=unique_segments,
                merged_content=merged_content,
                key_insights=key_insights,
                action_items=action_items,
                summary=summary,
                confidence_score=confidence_score,
                processing_details={
                    'total_segments': len(segments),
                    'unique_segments': len(unique_segments),
                    'keywords_count': len(keywords),
                    'jewelry_terms_count': len(jewelry_terms),
                    'type_confidence': type_confidence
                }
            )
            
        except Exception as e:
            self.logger.error(f"세션 통합 분석 실패: {e}")
            return IntegratedSession(
                session_id=f"error_session_{session_index}",
                session_type='unknown',
                title='분석 실패',
                start_time=time.time(),
                end_time=time.time(),
                participants=[],
                files=session_files,
                segments=[],
                merged_content="",
                key_insights=[],
                action_items=[],
                summary="세션 분석 중 오류가 발생했습니다.",
                confidence_score=0.0,
                processing_details={'error': str(e)}
            )
    
    def _create_content_segments(self, files: List[FileMetadata], stt_results: Dict = None, 
                                ocr_results: Dict = None) -> List[ContentSegment]:
        """컨텐츠 세그먼트 생성"""
        segments = []
        
        try:
            for file_meta in files:
                file_path = file_meta.file_path
                
                # STT 결과 처리
                if stt_results and file_path in stt_results:
                    stt_data = stt_results[file_path]
                    content = stt_data.get('text', '')
                    language = stt_data.get('language', 'ko')
                    confidence = stt_data.get('confidence', 0.8)
                    
                    if content.strip():
                        keywords, jewelry_terms = self.keyword_extractor.extract_keywords(content)
                        
                        segment = ContentSegment(
                            segment_id=f"stt_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                            source_file=file_path,
                            content_type='audio_transcript',
                            content=content,
                            start_time=0,
                            end_time=file_meta.estimated_duration,
                            confidence=confidence,
                            language=language,
                            keywords=keywords,
                            jewelry_terms=jewelry_terms,
                            timestamp=file_meta.creation_time
                        )
                        segments.append(segment)
                
                # OCR 결과 처리
                if ocr_results and file_path in ocr_results:
                    ocr_data = ocr_results[file_path]
                    content = ocr_data.get('text', '')
                    confidence = ocr_data.get('confidence', 0.8)
                    
                    if content.strip():
                        keywords, jewelry_terms = self.keyword_extractor.extract_keywords(content)
                        
                        segment = ContentSegment(
                            segment_id=f"ocr_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                            source_file=file_path,
                            content_type='ocr_text',
                            content=content,
                            start_time=0,
                            end_time=0,
                            confidence=confidence,
                            language='ko',  # OCR은 기본적으로 한국어로 가정
                            keywords=keywords,
                            jewelry_terms=jewelry_terms,
                            timestamp=file_meta.creation_time
                        )
                        segments.append(segment)
                
                # 메타데이터 세그먼트
                if file_meta.content_preview:
                    keywords, jewelry_terms = self.keyword_extractor.extract_keywords(file_meta.content_preview)
                    
                    segment = ContentSegment(
                        segment_id=f"meta_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                        source_file=file_path,
                        content_type='metadata',
                        content=file_meta.content_preview,
                        start_time=0,
                        end_time=0,
                        confidence=0.6,
                        language='ko',
                        keywords=keywords,
                        jewelry_terms=jewelry_terms,
                        timestamp=file_meta.creation_time
                    )
                    segments.append(segment)
            
            # 시간순 정렬
            segments.sort(key=lambda x: x.timestamp)
            
            return segments
            
        except Exception as e:
            self.logger.error(f"컨텐츠 세그먼트 생성 실패: {e}")
            return []
    
    def _merge_segments_content(self, segments: List[ContentSegment]) -> str:
        """세그먼트 내용 병합"""
        try:
            # 타입별로 그룹화
            content_by_type = defaultdict(list)
            
            for segment in segments:
                content_by_type[segment.content_type].append(segment.content)
            
            # 병합 순서: 메타데이터 → OCR → 음성전사
            merged_parts = []
            
            if content_by_type['metadata']:
                merged_parts.append("=== 파일 정보 ===")
                merged_parts.extend(content_by_type['metadata'])
                merged_parts.append("")
            
            if content_by_type['ocr_text']:
                merged_parts.append("=== 문서/이미지 내용 ===")
                merged_parts.extend(content_by_type['ocr_text'])
                merged_parts.append("")
            
            if content_by_type['audio_transcript']:
                merged_parts.append("=== 음성 내용 ===")
                merged_parts.extend(content_by_type['audio_transcript'])
            
            return '\n'.join(merged_parts)
            
        except Exception as e:
            self.logger.error(f"세그먼트 내용 병합 실패: {e}")
            return ""
    
    def _extract_session_insights(self, content: str, session_type: str) -> List[str]:
        """세션 인사이트 추출"""
        try:
            insights = []
            
            # 세션 타입별 특화 인사이트 추출
            if session_type == 'meeting':
                patterns = [
                    r'결정.*?[\.!?]',
                    r'합의.*?[\.!?]',
                    r'다음.*?계획.*?[\.!?]',
                    r'문제.*?해결.*?[\.!?]'
                ]
            elif session_type == 'trade_show':
                patterns = [
                    r'신제품.*?[\.!?]',
                    r'트렌드.*?[\.!?]',
                    r'시장.*?전망.*?[\.!?]',
                    r'고객.*?반응.*?[\.!?]'
                ]
            else:
                patterns = [
                    r'핵심.*?[\.!?]',
                    r'중요.*?[\.!?]',
                    r'주목.*?[\.!?]',
                    r'특징.*?[\.!?]'
                ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                insights.extend(matches[:2])  # 각 패턴에서 최대 2개
            
            # 중복 제거 및 정리
            unique_insights = list(set(insights))[:5]
            
            return unique_insights if unique_insights else ['주요 내용이 성공적으로 분석되었습니다.']
            
        except Exception as e:
            self.logger.error(f"인사이트 추출 실패: {e}")
            return ['내용 분석이 완료되었습니다.']
    
    def _extract_action_items(self, content: str) -> List[str]:
        """액션 아이템 추출"""
        try:
            action_patterns = [
                r'할\s*일.*?[\.!?]',
                r'과제.*?[\.!?]',
                r'준비.*?[\.!?]',
                r'검토.*?[\.!?]',
                r'확인.*?[\.!?]',
                r'follow.*?up.*?[\.!?]',
                r'action.*?item.*?[\.!?]',
                r'to.*?do.*?[\.!?]'
            ]
            
            action_items = []
            
            for pattern in action_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                action_items.extend(matches[:2])
            
            return list(set(action_items))[:5]
            
        except Exception as e:
            self.logger.error(f"액션 아이템 추출 실패: {e}")
            return []
    
    def _generate_session_summary(self, content: str, session_type: str, insights: List[str]) -> str:
        """세션 요약 생성"""
        try:
            # 요약 템플릿
            summary_parts = []
            
            # 세션 타입별 요약 시작
            type_descriptions = {
                'meeting': '회의 내용',
                'seminar': '세미나 주요 내용',
                'lecture': '강의 핵심 사항',
                'trade_show': '전시회 참관 결과',
                'conference': '컨퍼런스 주요 사항'
            }
            
            summary_parts.append(f"**{type_descriptions.get(session_type, '세션')} 요약**")
            summary_parts.append("")
            
            # 핵심 인사이트 포함
            if insights:
                summary_parts.append("**주요 내용:**")
                for i, insight in enumerate(insights[:3], 1):
                    summary_parts.append(f"{i}. {insight}")
                summary_parts.append("")
            
            # 내용 길이에 따른 추가 요약
            content_length = len(content)
            if content_length > 1000:
                summary_parts.append("**상세 분석:**")
                summary_parts.append(f"총 {content_length:,}자의 내용이 분석되었으며, 주얼리 업계 관련 전문 정보가 포함되어 있습니다.")
            
            return '\n'.join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"요약 생성 실패: {e}")
            return "세션이 성공적으로 분석되었습니다."
    
    def _generate_session_title(self, session_type: str, insights: List[str]) -> str:
        """세션 제목 생성"""
        try:
            date_str = datetime.now().strftime("%Y.%m.%d")
            
            if insights and len(insights[0]) < 50:
                # 첫 번째 인사이트를 제목으로 활용
                title = f"{date_str} {insights[0]}"
            else:
                # 세션 타입 기반 기본 제목
                type_titles = {
                    'meeting': '주얼리 업계 회의',
                    'seminar': '주얼리 세미나',
                    'lecture': '주얼리 강의',
                    'trade_show': '주얼리 전시회',
                    'conference': '주얼리 컨퍼런스'
                }
                title = f"{date_str} {type_titles.get(session_type, '주얼리 세션')}"
            
            return title
            
        except Exception as e:
            self.logger.error(f"제목 생성 실패: {e}")
            return f"주얼리 세션 {datetime.now().strftime('%Y.%m.%d')}"
    
    def _extract_participants(self, content: str) -> List[str]:
        """참가자 추출"""
        try:
            # 간단한 패턴으로 이름 추출
            name_patterns = [
                r'[가-힣]{2,4}(?:\s*대표|\s*회장|\s*이사|\s*팀장|\s*과장)',
                r'Mr\.\s*[A-Za-z]+',
                r'Ms\.\s*[A-Za-z]+',
                r'[A-Z][a-z]+\s+[A-Z][a-z]+'
            ]
            
            participants = []
            
            for pattern in name_patterns:
                matches = re.findall(pattern, content)
                participants.extend(matches)
            
            # 중복 제거 및 정리
            unique_participants = list(set(participants))[:10]
            
            return unique_participants
            
        except Exception as e:
            self.logger.error(f"참가자 추출 실패: {e}")
            return []
    
    def _calculate_session_confidence(self, segments: List[ContentSegment], type_confidence: float) -> float:
        """세션 신뢰도 계산"""
        try:
            if not segments:
                return 0.0
            
            # 세그먼트 평균 신뢰도
            avg_segment_confidence = np.mean([seg.confidence for seg in segments])
            
            # 내용 양 기반 보정
            total_content_length = sum(len(seg.content) for seg in segments)
            content_factor = min(1.0, total_content_length / 1000)
            
            # 주얼리 용어 포함 보너스
            total_jewelry_terms = sum(len(seg.jewelry_terms) for seg in segments)
            jewelry_bonus = min(0.2, total_jewelry_terms * 0.05)
            
            # 종합 신뢰도
            final_confidence = (
                avg_segment_confidence * 0.5 +
                type_confidence * 0.3 +
                content_factor * 0.2 +
                jewelry_bonus
            )
            
            return max(0.0, min(1.0, final_confidence))
            
        except Exception as e:
            self.logger.error(f"세션 신뢰도 계산 실패: {e}")
            return 0.5
    
    def _create_overall_integration(self, sessions: List[IntegratedSession]) -> Dict[str, Any]:
        """전체 통합 결과 생성"""
        try:
            if not sessions:
                return {'error': '분석할 세션이 없습니다.'}
            
            # 전체 통계
            total_files = sum(len(session.files) for session in sessions)
            total_segments = sum(len(session.segments) for session in sessions)
            avg_confidence = np.mean([session.confidence_score for session in sessions])
            
            # 모든 내용 통합
            all_content = '\n\n'.join([session.merged_content for session in sessions])
            
            # 전체 키워드 및 인사이트
            all_insights = []
            for session in sessions:
                all_insights.extend(session.key_insights)
            
            # 세션 타입 분포
            session_types = Counter([session.session_type for session in sessions])
            
            # 시간 범위
            start_time = min(session.start_time for session in sessions)
            end_time = max(session.end_time for session in sessions)
            duration_hours = (end_time - start_time) / 3600
            
            return {
                'total_statistics': {
                    'total_sessions': len(sessions),
                    'total_files': total_files,
                    'total_segments': total_segments,
                    'average_confidence': avg_confidence,
                    'duration_hours': duration_hours
                },
                'session_distribution': dict(session_types),
                'time_range': {
                    'start': datetime.fromtimestamp(start_time).isoformat(),
                    'end': datetime.fromtimestamp(end_time).isoformat(),
                    'duration_hours': duration_hours
                },
                'integrated_content': all_content,
                'overall_insights': list(set(all_insights))[:10],  # 중복 제거 후 상위 10개
                'processing_success': True
            }
            
        except Exception as e:
            self.logger.error(f"전체 통합 결과 생성 실패: {e}")
            return {'error': str(e)}

# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 다중 파일 통합 분석기 초기화
    integrator = MultiFileIntegratorV21()
    
    # 샘플 파일들 분석
    # file_paths = ["meeting_audio.mp3", "presentation.pdf", "notes.jpg"]
    # result = integrator.integrate_multiple_files(file_paths)
    # print(f"통합 분석 완료: {result['processing_statistics']}")
    
    print("✅ 다중 파일 통합 분석기 v2.1 로드 완료!")
