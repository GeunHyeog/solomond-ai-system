#!/usr/bin/env python3
"""
🚀 완전한 멀티모달 통합 분석 엔진 v3.0
- 문서+영상+음성+이미지+유튜브 완전 지원
- 유튜브 URL 자동 다운로드 및 분석
- 여러 AI엔진 조합 분석 → 최종 요약 리포트
- 주얼리 강의/회의 내용 종합 요약

사용법:
1. 분석할 파일들을 input_files/ 폴더에 넣기
2. 유튜브 URL들을 youtube_urls.txt에 입력
3. python youtube_integrated_engine.py 실행
4. 완전한 통합 분석 결과 확인
"""

import os
import sys
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    DOCUMENT = "document"
    YOUTUBE = "youtube"
    WEB = "web"

@dataclass
class AnalysisResult:
    """분석 결과 데이터 클래스"""
    file_path: str
    file_name: str
    file_size_mb: float
    analysis_type: AnalysisType
    processing_time: float
    content: str
    jewelry_keywords: List[str]
    confidence: float
    jewelry_relevance: float
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    youtube_url: Optional[str] = None

class YouTubeProcessor:
    """유튜브 영상 처리기"""
    
    def __init__(self):
        self.download_dir = Path("temp_youtube")
        self.download_dir.mkdir(exist_ok=True)
        
    async def process_youtube_url(self, url: str) -> Dict[str, Any]:
        """유튜브 URL 처리"""
        try:
            # yt-dlp를 사용하여 유튜브 영상 다운로드
            logger.info(f"📺 유튜브 영상 다운로드 중: {url}")
            
            # 안전한 파일명 생성
            safe_filename = self.get_safe_filename(url)
            output_path = self.download_dir / f"{safe_filename}.%(ext)s"
            
            # yt-dlp 명령어 구성 (audio only for faster processing)
            import subprocess
            
            # 오디오만 추출 (더 빠른 처리)
            cmd = [
                "yt-dlp",
                "--extract-audio",
                "--audio-format", "wav",
                "--audio-quality", "0",
                "--output", str(output_path),
                "--no-playlist",
                url
            ]
            
            # 서브프로세스로 다운로드 실행
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5분 타임아웃
            
            if result.returncode != 0:
                raise Exception(f"yt-dlp 실행 실패: {result.stderr}")
            
            # 다운로드된 파일 찾기
            downloaded_files = list(self.download_dir.glob(f"{safe_filename}.*"))
            if not downloaded_files:
                raise Exception("다운로드된 파일을 찾을 수 없습니다")
            
            audio_file = downloaded_files[0]
            
            # 오디오 파일을 STT로 처리
            from multimodal_analysis_engine import AudioSTTProcessor
            stt_processor = AudioSTTProcessor()
            stt_result = await stt_processor.process_audio(str(audio_file))
            
            # 메타데이터 추가
            video_info = await self.get_video_info(url)
            
            result_data = {
                "text": stt_result["text"],
                "jewelry_keywords": stt_result["jewelry_keywords"],
                "confidence": stt_result["confidence"],
                "type": "youtube_video_stt",
                "video_info": video_info,
                "downloaded_file": str(audio_file)
            }
            
            logger.info(f"✅ 유튜브 영상 처리 완료: {video_info.get('title', 'Unknown')}")
            
            return result_data
            
        except Exception as e:
            logger.error(f"❌ 유튜브 처리 오류: {str(e)}")
            return {
                "text": f"유튜브 처리 중 오류 발생: {str(e)}",
                "jewelry_keywords": [],
                "confidence": 0.0,
                "type": "youtube_error"
            }
    
    def get_safe_filename(self, url: str) -> str:
        """URL에서 안전한 파일명 생성"""
        # 유튜브 비디오 ID 추출
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
        if video_id_match:
            return f"youtube_{video_id_match.group(1)}"
        else:
            return f"youtube_{int(time.time())}"
    
    async def get_video_info(self, url: str) -> Dict[str, Any]:
        """유튜브 비디오 정보 가져오기"""
        try:
            import subprocess
            
            cmd = [
                "yt-dlp",
                "--dump-json",
                "--no-playlist",
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                video_data = json.loads(result.stdout)
                return {
                    "title": video_data.get("title", "Unknown"),
                    "duration": video_data.get("duration", 0),
                    "uploader": video_data.get("uploader", "Unknown"),
                    "view_count": video_data.get("view_count", 0),
                    "upload_date": video_data.get("upload_date", "Unknown")
                }
            else:
                return {"title": "Unknown", "error": result.stderr}
                
        except Exception as e:
            return {"title": "Unknown", "error": str(e)}
    
    def cleanup_temp_files(self):
        """임시 다운로드 파일 정리"""
        try:
            import shutil
            if self.download_dir.exists():
                shutil.rmtree(self.download_dir)
                logger.info("🧹 임시 유튜브 파일 정리 완료")
        except Exception as e:
            logger.warning(f"⚠️ 임시 파일 정리 중 오류: {str(e)}")

class AudioSTTProcessor:
    """음성 STT 처리기"""
    
    def __init__(self):
        self.model_name = "whisper-base"
        
    async def process_audio(self, file_path: str) -> Dict[str, Any]:
        """음성 파일 STT 처리"""
        try:
            import whisper
            
            logger.info(f"🎵 음성 파일 로드 중: {file_path}")
            model = whisper.load_model("base")
            
            result = model.transcribe(file_path, language="ko")
            
            # 주얼리 키워드 추출
            jewelry_keywords = self.extract_jewelry_keywords(result["text"])
            
            return {
                "text": result["text"],
                "segments": result.get("segments", []),
                "language": result.get("language", "ko"),
                "jewelry_keywords": jewelry_keywords,
                "confidence": 0.85
            }
            
        except Exception as e:
            logger.error(f"❌ 음성 처리 오류: {str(e)}")
            return {
                "text": f"음성 처리 중 오류 발생: {str(e)}",
                "jewelry_keywords": [],
                "confidence": 0.0
            }
    
    def extract_jewelry_keywords(self, text: str) -> List[str]:
        """주얼리 키워드 추출"""
        jewelry_terms = [
            "다이아몬드", "diamond", "루비", "ruby", "사파이어", "sapphire", "에메랄드", "emerald",
            "반지", "ring", "목걸이", "necklace", "귀걸이", "earring", "팔찌", "bracelet",
            "금", "gold", "은", "silver", "백금", "platinum", "캐럿", "carat",
            "컷", "cut", "투명도", "clarity", "색상", "color", "무게", "weight",
            "GIA", "감정서", "인증서", "certificate", "보석", "gem", "주얼리", "jewelry",
            "세팅", "setting", "프롱", "prong", "파베", "pave", "솔리테어", "solitaire",
            "홍콩", "홍콩박람회", "바젤월드", "baselworld", "주얼리페어", "jewelry fair"
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for term in jewelry_terms:
            if term.lower() in text_lower:
                found_keywords.append(term)
        
        return list(set(found_keywords))

# 기존 프로세서들 (multimodal_analysis_engine.py에서 가져옴)
class VideoProcessor:
    """비디오 처리기"""
    
    async def process_video(self, file_path: str) -> Dict[str, Any]:
        """비디오 파일 처리 (음성 추출 + STT)"""
        try:
            import ffmpeg
            
            logger.info(f"🎬 비디오 파일 처리 중: {file_path}")
            
            temp_audio = f"temp_audio_{int(time.time())}.wav"
            
            (
                ffmpeg
                .input(file_path)
                .output(temp_audio, acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )
            
            stt_processor = AudioSTTProcessor()
            audio_result = await stt_processor.process_audio(temp_audio)
            
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            return {
                "text": audio_result["text"],
                "jewelry_keywords": audio_result["jewelry_keywords"],
                "confidence": audio_result["confidence"],
                "type": "video_to_audio_stt"
            }
            
        except Exception as e:
            logger.error(f"❌ 비디오 처리 오류: {str(e)}")
            return {
                "text": f"비디오 처리 중 오류 발생: {str(e)}",
                "jewelry_keywords": [],
                "confidence": 0.0
            }

class ImageProcessor:
    """이미지 처리기"""
    
    async def process_image(self, file_path: str) -> Dict[str, Any]:
        """이미지 OCR 처리"""
        try:
            import pytesseract
            from PIL import Image
            
            logger.info(f"🖼️ 이미지 OCR 처리 중: {file_path}")
            
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang='kor+eng')
            
            stt_processor = AudioSTTProcessor()
            jewelry_keywords = stt_processor.extract_jewelry_keywords(text)
            
            return {
                "text": text.strip(),
                "jewelry_keywords": jewelry_keywords,
                "confidence": 0.75,
                "type": "image_ocr"
            }
            
        except Exception as e:
            logger.error(f"❌ 이미지 처리 오류: {str(e)}")
            return {
                "text": f"이미지 처리 중 오류 발생: {str(e)}",
                "jewelry_keywords": [],
                "confidence": 0.0
            }

class DocumentProcessor:
    """문서 처리기"""
    
    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """문서 파일 처리"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.pdf':
                return await self.process_pdf(file_path)
            elif file_ext == '.docx':
                return await self.process_docx(file_path)
            elif file_ext == '.txt':
                return await self.process_txt(file_path)
            else:
                raise ValueError(f"지원하지 않는 문서 형식: {file_ext}")
                
        except Exception as e:
            logger.error(f"❌ 문서 처리 오류: {str(e)}")
            return {
                "text": f"문서 처리 중 오류 발생: {str(e)}",
                "jewelry_keywords": [],
                "confidence": 0.0
            }
    
    async def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """PDF 처리"""
        try:
            import PyPDF2
            
            logger.info(f"📄 PDF 처리 중: {file_path}")
            
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            stt_processor = AudioSTTProcessor()
            jewelry_keywords = stt_processor.extract_jewelry_keywords(text)
            
            return {
                "text": text.strip(),
                "jewelry_keywords": jewelry_keywords,
                "confidence": 0.90,
                "type": "pdf_extraction"
            }
            
        except Exception as e:
            return {
                "text": f"PDF 처리 오류: {str(e)}",
                "jewelry_keywords": [],
                "confidence": 0.0
            }
    
    async def process_docx(self, file_path: str) -> Dict[str, Any]:
        """DOCX 처리"""
        try:
            from docx import Document
            
            logger.info(f"📝 DOCX 처리 중: {file_path}")
            
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            stt_processor = AudioSTTProcessor()
            jewelry_keywords = stt_processor.extract_jewelry_keywords(text)
            
            return {
                "text": text.strip(),
                "jewelry_keywords": jewelry_keywords,
                "confidence": 0.95,
                "type": "docx_extraction"
            }
            
        except Exception as e:
            return {
                "text": f"DOCX 처리 오류: {str(e)}",
                "jewelry_keywords": [],
                "confidence": 0.0
            }
    
    async def process_txt(self, file_path: str) -> Dict[str, Any]:
        """TXT 처리"""
        try:
            logger.info(f"📋 TXT 처리 중: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            stt_processor = AudioSTTProcessor()
            jewelry_keywords = stt_processor.extract_jewelry_keywords(text)
            
            return {
                "text": text.strip(),
                "jewelry_keywords": jewelry_keywords,
                "confidence": 1.0,
                "type": "txt_extraction"
            }
            
        except Exception as e:
            return {
                "text": f"TXT 처리 오류: {str(e)}",
                "jewelry_keywords": [],
                "confidence": 0.0
            }

class CompleteMultimodalEngine:
    """완전한 멀티모달 통합 분석 엔진 (유튜브 포함)"""
    
    def __init__(self):
        self.audio_processor = AudioSTTProcessor()
        self.video_processor = VideoProcessor()
        self.image_processor = ImageProcessor()
        self.document_processor = DocumentProcessor()
        self.youtube_processor = YouTubeProcessor()
        
    async def analyze_single_file(self, file_path: str) -> AnalysisResult:
        """단일 파일 분석"""
        start_time = time.time()
        file_name = Path(file_path).name
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        logger.info(f"🔍 파일 분석 시작: {file_name} ({file_size_mb:.2f}MB)")
        
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in ['.mp3', '.wav', '.m4a', '.aac']:
                analysis_type = AnalysisType.AUDIO
                result = await self.audio_processor.process_audio(file_path)
                
            elif file_ext in ['.mp4', '.mov', '.avi', '.mkv']:
                analysis_type = AnalysisType.VIDEO
                result = await self.video_processor.process_video(file_path)
                
            elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                analysis_type = AnalysisType.IMAGE
                result = await self.image_processor.process_image(file_path)
                
            elif file_ext in ['.pdf', '.docx', '.txt']:
                analysis_type = AnalysisType.DOCUMENT
                result = await self.document_processor.process_document(file_path)
                
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {file_ext}")
            
            processing_time = time.time() - start_time
            
            jewelry_relevance = self.calculate_jewelry_relevance(
                result.get("jewelry_keywords", []), 
                result.get("text", "")
            )
            
            return AnalysisResult(
                file_path=file_path,
                file_name=file_name,
                file_size_mb=file_size_mb,
                analysis_type=analysis_type,
                processing_time=processing_time,
                content=result.get("text", ""),
                jewelry_keywords=result.get("jewelry_keywords", []),
                confidence=result.get("confidence", 0.0),
                jewelry_relevance=jewelry_relevance,
                metadata=result,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ 파일 분석 실패: {file_name} - {str(e)}")
            
            return AnalysisResult(
                file_path=file_path,
                file_name=file_name,
                file_size_mb=file_size_mb,
                analysis_type=AnalysisType.DOCUMENT,
                processing_time=processing_time,
                content="",
                jewelry_keywords=[],
                confidence=0.0,
                jewelry_relevance=0.0,
                metadata={},
                success=False,
                error_message=str(e)
            )
    
    async def analyze_youtube_url(self, url: str) -> AnalysisResult:
        """유튜브 URL 분석"""
        start_time = time.time()
        
        logger.info(f"📺 유튜브 URL 분석 시작: {url}")
        
        try:
            result = await self.youtube_processor.process_youtube_url(url)
            
            processing_time = time.time() - start_time
            
            # 추정 파일 크기 (다운로드된 오디오 파일 기준)
            file_size_mb = 0.0
            if "downloaded_file" in result and os.path.exists(result["downloaded_file"]):
                file_size_mb = os.path.getsize(result["downloaded_file"]) / (1024 * 1024)
            
            jewelry_relevance = self.calculate_jewelry_relevance(
                result.get("jewelry_keywords", []), 
                result.get("text", "")
            )
            
            # 비디오 제목을 파일명으로 사용
            video_title = result.get("video_info", {}).get("title", "Unknown YouTube Video")
            
            return AnalysisResult(
                file_path=url,
                file_name=f"[YouTube] {video_title}",
                file_size_mb=file_size_mb,
                analysis_type=AnalysisType.YOUTUBE,
                processing_time=processing_time,
                content=result.get("text", ""),
                jewelry_keywords=result.get("jewelry_keywords", []),
                confidence=result.get("confidence", 0.0),
                jewelry_relevance=jewelry_relevance,
                metadata=result,
                success=True,
                youtube_url=url
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ 유튜브 분석 실패: {url} - {str(e)}")
            
            return AnalysisResult(
                file_path=url,
                file_name=f"[YouTube] {url}",
                file_size_mb=0.0,
                analysis_type=AnalysisType.YOUTUBE,
                processing_time=processing_time,
                content="",
                jewelry_keywords=[],
                confidence=0.0,
                jewelry_relevance=0.0,
                metadata={},
                success=False,
                error_message=str(e),
                youtube_url=url
            )
    
    def calculate_jewelry_relevance(self, keywords: List[str], text: str) -> float:
        """주얼리 관련성 계산"""
        if not keywords:
            return 0.0
        
        keyword_score = min(len(keywords) / 10, 1.0)
        
        text_length = len(text.split())
        if text_length > 0:
            density_score = len(keywords) / text_length * 100
            density_score = min(density_score, 1.0)
        else:
            density_score = 0.0
        
        relevance = (keyword_score * 0.7) + (density_score * 0.3)
        return round(relevance, 2)
    
    async def integrate_complete_analysis(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """완전한 멀티모달 분석 결과 통합 (유튜브 포함)"""
        logger.info("🔗 완전한 멀티모달 분석 결과 통합 중...")
        
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {
                "summary": "분석 가능한 소스가 없습니다.",
                "total_sources": len(results),
                "successful_sources": 0,
                "total_processing_time": sum(r.processing_time for r in results),
                "jewelry_relevance": 0.0
            }
        
        # 소스별 텍스트 통합
        all_text = []
        all_keywords = []
        source_breakdown = {
            "audio": 0, "video": 0, "image": 0, 
            "document": 0, "youtube": 0
        }
        
        for result in successful_results:
            if result.content.strip():
                source_label = f"[{result.analysis_type.value.upper()}] {result.file_name}"
                all_text.append(f"{source_label}\n{result.content}")
            
            all_keywords.extend(result.jewelry_keywords)
            source_breakdown[result.analysis_type.value] += 1
        
        combined_text = "\n\n" + "="*50 + "\n\n".join(all_text)
        unique_keywords = list(set(all_keywords))
        
        # 통합 분석 생성
        jewelry_summary = self.generate_comprehensive_summary(combined_text, unique_keywords, source_breakdown)
        
        # 통계 계산
        avg_confidence = sum(r.confidence for r in successful_results) / len(successful_results)
        avg_jewelry_relevance = sum(r.jewelry_relevance for r in successful_results) / len(successful_results)
        total_processing_time = sum(r.processing_time for r in results)
        
        return {
            "summary": jewelry_summary,
            "combined_text": combined_text,
            "unique_keywords": unique_keywords,
            "total_sources": len(results),
            "successful_sources": len(successful_results),
            "failed_sources": len(results) - len(successful_results),
            "average_confidence": round(avg_confidence, 2),
            "average_jewelry_relevance": round(avg_jewelry_relevance, 2),
            "total_processing_time": round(total_processing_time, 2),
            "source_breakdown": source_breakdown,
            "youtube_count": source_breakdown["youtube"]
        }
    
    def generate_comprehensive_summary(self, text: str, keywords: List[str], breakdown: Dict[str, int]) -> str:
        """종합적인 주얼리 업계 요약 생성"""
        if not text.strip():
            return "분석할 콘텐츠가 없습니다."
        
        summary_parts = []
        
        # 소스 다양성 언급
        source_types = [k for k, v in breakdown.items() if v > 0]
        if len(source_types) > 3:
            summary_parts.append(f"다양한 멀티미디어 소스({', '.join(source_types)})를 통한 종합적인 주얼리 분석입니다.")
        
        # 유튜브 콘텐츠 특별 언급
        if breakdown.get("youtube", 0) > 0:
            summary_parts.append(f"유튜브 영상 {breakdown['youtube']}개를 포함한 온라인 콘텐츠 분석이 포함되었습니다.")
        
        # 주요 키워드 언급
        if keywords:
            top_keywords = keywords[:8]
            summary_parts.append(f"핵심 주얼리 키워드: {', '.join(top_keywords)}")
        
        # 내용 분석
        text_lower = text.lower()
        
        if "교육" in text or "강의" in text or "세미나" in text:
            summary_parts.append("교육 및 학습 콘텐츠가 중심을 이룹니다.")
        
        if "시장" in text or "트렌드" in text or "동향" in text:
            summary_parts.append("시장 동향 및 업계 트렌드 분석이 포함되어 있습니다.")
        
        if "기술" in text or "제조" in text or "공정" in text:
            summary_parts.append("기술적 제조 과정 및 공정에 대한 내용이 다뤄집니다.")
        
        if any(term in text_lower for term in ["gia", "certificate", "감정", "인증"]):
            summary_parts.append("보석 감정 및 인증 관련 전문 정보가 포함되어 있습니다.")
        
        # 길이에 따른 평가
        if len(text) > 5000:
            summary_parts.append("대용량의 상세한 전문 자료로 심층적인 인사이트를 제공합니다.")
        elif len(text) > 1500:
            summary_parts.append("충분한 분량의 실무 중심 정보가 담겨 있습니다.")
        
        return " ".join(summary_parts) if summary_parts else "주얼리 업계 관련 기본 정보가 포함되어 있습니다."

def setup_directories():
    """필요한 디렉토리 설정"""
    dirs = ["input_files", "output_results"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    return dirs

def find_input_files() -> List[str]:
    """입력 파일 찾기"""
    input_dir = Path("input_files")
    
    supported_extensions = [
        "*.mp3", "*.wav", "*.m4a", "*.aac",
        "*.mp4", "*.mov", "*.avi", "*.mkv",
        "*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp",
        "*.pdf", "*.docx", "*.txt"
    ]
    
    all_files = []
    for ext in supported_extensions:
        files = list(input_dir.glob(ext))
        all_files.extend([str(f) for f in files])
    
    return all_files

def find_youtube_urls() -> List[str]:
    """유튜브 URL 찾기"""
    url_file = Path("youtube_urls.txt")
    
    if not url_file.exists():
        # 빈 파일 생성
        with open(url_file, 'w', encoding='utf-8') as f:
            f.write("# 분석할 유튜브 URL들을 여기에 입력하세요 (한 줄에 하나씩)\n")
            f.write("# 예시:\n")
            f.write("# https://www.youtube.com/watch?v=example1\n")
            f.write("# https://youtu.be/example2\n")
        return []
    
    urls = []
    with open(url_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if 'youtube.com' in line or 'youtu.be' in line:
                    urls.append(line)
    
    return urls

async def main():
    """메인 실행 함수"""
    print("🚀 솔로몬드 완전한 멀티모달 통합 분석 엔진 v3.0")
    print("📺 유튜브 지원 포함 - 문서+영상+음성+이미지+유튜브 완전 지원")
    print("=" * 80)
    
    # 디렉토리 설정
    setup_directories()
    
    # 입력 소스 확인
    input_files = find_input_files()
    youtube_urls = find_youtube_urls()
    
    total_sources = len(input_files) + len(youtube_urls)
    
    if total_sources == 0:
        print("📁 분석할 소스가 없습니다.")
        print("   1. input_files/ 폴더에 파일들을 넣어주세요")
        print("   2. youtube_urls.txt에 유튜브 URL들을 입력해주세요")
        print("🔧 지원 형식: MP3, WAV, M4A, MP4, MOV, JPG, PNG, PDF, DOCX, TXT, YouTube")
        return
    
    print(f"📋 분석할 소스 총 {total_sources}개 발견:")
    
    # 파일 목록 출력
    if input_files:
        print(f"\n📁 로컬 파일 {len(input_files)}개:")
        for i, file_path in enumerate(input_files, 1):
            file_size = os.path.getsize(file_path) / (1024*1024)
            print(f"   {i}. {Path(file_path).name} ({file_size:.2f}MB)")
    
    # 유튜브 URL 목록 출력
    if youtube_urls:
        print(f"\n📺 유튜브 영상 {len(youtube_urls)}개:")
        for i, url in enumerate(youtube_urls, 1):
            print(f"   {i}. {url}")
    
    # yt-dlp 설치 확인
    print(f"\n🔧 시스템 요구사항 확인 중...")
    try:
        import subprocess
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ yt-dlp 설치 확인: {result.stdout.strip()}")
        else:
            print("❌ yt-dlp가 설치되지 않았습니다.")
            print("   설치 방법: pip install yt-dlp")
            if youtube_urls:
                print("⚠️ 유튜브 분석을 위해서는 yt-dlp가 필요합니다.")
                return
    except FileNotFoundError:
        print("❌ yt-dlp가 설치되지 않았습니다.")
        print("   설치 방법: pip install yt-dlp")
        if youtube_urls:
            print("⚠️ 유튜브 분석을 위해서는 yt-dlp가 필요합니다.")
            return
    
    # 분석 시작 확인
    response = input(f"\n🔄 완전한 멀티모달 통합 분석을 시작하시겠습니까? (y/n): ")
    if response.lower() not in ['y', 'yes', '예', 'ㅇ']:
        print("⏸️ 분석을 취소했습니다.")
        return
    
    # 완전한 멀티모달 분석 엔진 초기화
    engine = CompleteMultimodalEngine()
    
    print(f"\n🔍 {total_sources}개 소스 완전한 멀티모달 분석 시작...")
    start_time = time.time()
    
    # 모든 소스 분석
    analysis_results = []
    
    # 로컬 파일 분석
    if input_files:
        print(f"\n📁 로컬 파일 {len(input_files)}개 분석 중...")
        for i, file_path in enumerate(input_files, 1):
            print(f"\n📊 파일 진행률: {i}/{len(input_files)}")
            result = await engine.analyze_single_file(file_path)
            analysis_results.append(result)
            
            if result.success:
                print(f"✅ 성공: {result.file_name}")
                print(f"   🎯 관련성: {result.jewelry_relevance:.2f}")
                print(f"   🔑 키워드: {', '.join(result.jewelry_keywords[:5])}")
            else:
                print(f"❌ 실패: {result.file_name} - {result.error_message}")
    
    # 유튜브 URL 분석
    if youtube_urls:
        print(f"\n📺 유튜브 영상 {len(youtube_urls)}개 분석 중...")
        for i, url in enumerate(youtube_urls, 1):
            print(f"\n📊 유튜브 진행률: {i}/{len(youtube_urls)}")
            result = await engine.analyze_youtube_url(url)
            analysis_results.append(result)
            
            if result.success:
                print(f"✅ 성공: {result.file_name}")
                print(f"   🎯 관련성: {result.jewelry_relevance:.2f}")
                print(f"   🔑 키워드: {', '.join(result.jewelry_keywords[:5])}")
                print(f"   ⏱️ 처리시간: {result.processing_time:.1f}초")
            else:
                print(f"❌ 실패: {result.file_name} - {result.error_message}")
    
    # 완전한 멀티모달 통합 분석
    print(f"\n🔗 완전한 멀티모달 통합 분석 중...")
    integrated_result = await engine.integrate_complete_analysis(analysis_results)
    
    total_time = time.time() - start_time
    
    # 임시 파일 정리
    engine.youtube_processor.cleanup_temp_files()
    
    # 결과 저장
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output_results")
    
    # 상세 결과 JSON 저장
    detailed_results = {
        "timestamp": timestamp,
        "analysis_info": {
            "engine_version": "v3.0_complete_multimodal",
            "youtube_support": True,
            "total_sources": total_sources,
            "local_files": len(input_files),
            "youtube_videos": len(youtube_urls)
        },
        "individual_analysis": [asdict(r) for r in analysis_results],
        "integrated_analysis": integrated_result,
        "performance": {
            "total_processing_time": total_time,
            "average_time_per_source": total_time / total_sources if total_sources > 0 else 0,
            "sources_per_minute": (total_sources / total_time) * 60 if total_time > 0 else 0
        }
    }
    
    json_file = output_dir / f"complete_multimodal_analysis_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    # 요약 리포트 저장
    report_file = output_dir / f"complete_analysis_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("💎 솔로몬드 완전한 멀티모달 통합 분석 리포트 v3.0\n")
        f.write("📺 유튜브 지원 포함 - 문서+영상+음성+이미지+유튜브 완전 지원\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"📊 전체 통계:\n")
        f.write(f"   총 소스 수: {integrated_result['total_sources']}\n")
        f.write(f"   - 로컬 파일: {len(input_files)}개\n")
        f.write(f"   - 유튜브 영상: {integrated_result['youtube_count']}개\n")
        f.write(f"   성공 소스: {integrated_result['successful_sources']}\n")
        f.write(f"   실패 소스: {integrated_result['failed_sources']}\n")
        f.write(f"   전체 처리시간: {integrated_result['total_processing_time']:.2f}초\n")
        f.write(f"   평균 신뢰도: {integrated_result['average_confidence']:.2f}\n")
        f.write(f"   평균 주얼리 관련성: {integrated_result['average_jewelry_relevance']:.2f}\n\n")
        
        f.write(f"🔑 발견된 주얼리 키워드 ({len(integrated_result['unique_keywords'])}개):\n")
        f.write(f"   {', '.join(integrated_result['unique_keywords'])}\n\n")
        
        f.write(f"📋 소스별 분석:\n")
        breakdown = integrated_result['source_breakdown']
        f.write(f"   음성: {breakdown['audio']}개\n")
        f.write(f"   비디오: {breakdown['video']}개\n")
        f.write(f"   이미지: {breakdown['image']}개\n")
        f.write(f"   문서: {breakdown['document']}개\n")
        f.write(f"   유튜브: {breakdown['youtube']}개\n\n")
        
        f.write(f"🎯 완전한 통합 분석 요약:\n")
        f.write(f"   {integrated_result['summary']}\n\n")
        
        if integrated_result['combined_text']:
            f.write(f"📝 전체 내용 (처음 1000자):\n")
            f.write(f"   {integrated_result['combined_text'][:1000]}...\n")
    
    # 최종 결과 출력
    print(f"\n" + "=" * 80)
    print(f"🎉 완전한 멀티모달 통합 분석 완료!")
    print(f"   📊 총 {integrated_result['total_sources']}개 소스 처리")
    print(f"      - 로컬 파일: {len(input_files)}개")
    print(f"      - 유튜브 영상: {integrated_result['youtube_count']}개")
    print(f"   ✅ 성공: {integrated_result['successful_sources']}개")
    print(f"   ❌ 실패: {integrated_result['failed_sources']}개")
    print(f"   ⏱️ 전체 시간: {integrated_result['total_processing_time']:.2f}초")
    print(f"   📈 평균 관련성: {integrated_result['average_jewelry_relevance']:.2f}")
    print(f"   🔑 키워드 수: {len(integrated_result['unique_keywords'])}개")
    print(f"\n💾 결과 저장:")
    print(f"   📄 상세 데이터: {json_file}")
    print(f"   📋 분석 리포트: {report_file}")
    print(f"\n🎯 완전한 통합 요약:")
    print(f"   {integrated_result['summary']}")
    
    print(f"\n🚀 이제 문서+영상+음성+이미지+유튜브를 모두 통합한")
    print(f"   완벽한 주얼리 강의/회의 종합 분석이 완성되었습니다!")

if __name__ == "__main__":
    asyncio.run(main())
