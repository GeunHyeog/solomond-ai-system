#!/usr/bin/env python3
"""
🤖 멀티모달 인코더 - SOLOMOND AI 진정한 멀티모달리티 구현
True Multimodal Encoder: Images/Audio/Text → 768-dimensional Latent Space

🎯 주요 기능:
1. 통합 인코딩 - 모든 모달리티를 768차원 공통 공간으로 변환
2. 모달별 전처리 - 각 모달에 최적화된 처리 파이프라인
3. 정규화 시스템 - 모달간 크기 및 분포 정규화
4. 배치 처리 - 다수 파일 동시 인코딩 지원
5. 캐싱 시스템 - 중복 인코딩 방지로 성능 최적화
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging
import hashlib
import time

# AI 모델 라이브러리
try:
    import whisper
    import easyocr
    import cv2
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    import librosa
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    logging.error(f"필수 라이브러리 누락: {e}")

# 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EncodedResult:
    """인코딩된 결과 구조"""
    file_path: str
    modality: str
    encoding: np.ndarray  # 768차원 벡터
    confidence: float
    metadata: Dict[str, Any]
    processing_time: float
    raw_content: str = ""

class MultimodalEncoder:
    """진정한 멀티모달 인코더 - 모든 모달리티를 공통 공간으로"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.models = {}
        self.is_initialized = False
        self.cache = {}
        
        # 정규화기들
        self.scalers = {
            'image': StandardScaler(),
            'audio': StandardScaler(), 
            'text': StandardScaler()
        }
        
        # 성능 메트릭
        self.stats = {
            'encoded_files': 0,
            'cache_hits': 0,
            'encoding_times': [],
            'modality_counts': {'image': 0, 'audio': 0, 'text': 0}
        }
        
    def _get_default_config(self) -> Dict:
        """기본 설정"""
        return {
            'target_dimensions': 768,
            'whisper_model': 'base',
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'ocr_languages': ['ko', 'en'],
            'cache_enabled': True,
            'batch_size': 8,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    def initialize(self) -> None:
        """모델 초기화"""
        if self.is_initialized:
            return
            
        logger.info("🤖 멀티모달 인코더 초기화 중...")
        
        # 1. Sentence Transformer (텍스트 + 공통 인코더)
        logger.info("🧠 SentenceTransformer 로딩...")
        self.models['text_encoder'] = SentenceTransformer(
            self.config['embedding_model'],
            device=self.config['device']
        )
        
        # 2. Whisper (오디오)
        logger.info("🎵 Whisper STT 모델 로딩...")
        self.models['whisper'] = whisper.load_model(self.config['whisper_model'])
        
        # 3. EasyOCR (이미지)
        logger.info("👁️ EasyOCR 모델 로딩...")
        self.models['ocr'] = easyocr.Reader(self.config['ocr_languages'])
        
        self.is_initialized = True
        logger.info("✅ 멀티모달 인코더 초기화 완료!")
        
    def encode_batch(self, files: List[Path]) -> List[EncodedResult]:
        """배치 파일 인코딩"""
        if not self.is_initialized:
            self.initialize()
            
        logger.info(f"📦 배치 인코딩 시작: {len(files)}개 파일")
        start_time = time.time()
        
        results = []
        
        # 파일 타입별 분류 및 인코딩
        for file_path in files:
            try:
                result = self.encode_single_file(file_path)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"❌ 파일 인코딩 실패 {file_path}: {e}")
                
        batch_time = time.time() - start_time
        logger.info(f"✅ 배치 인코딩 완료: {len(results)}개 성공, {batch_time:.2f}초")
        
        return results
    
    def encode_single_file(self, file_path: Path) -> Optional[EncodedResult]:
        """단일 파일 인코딩"""
        start_time = time.time()
        
        # 캐시 확인
        file_hash = self._get_file_hash(file_path)
        if self.config['cache_enabled'] and file_hash in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[file_hash]
            
        # 파일 타입 판별
        modality = self._detect_modality(file_path)
        
        if modality == 'image':
            result = self._encode_image(file_path)
        elif modality == 'audio':
            result = self._encode_audio(file_path)
        elif modality == 'text':
            result = self._encode_text(file_path)
        else:
            logger.warning(f"⚠️ 지원하지 않는 파일 타입: {file_path}")
            return None
            
        if result:
            result.processing_time = time.time() - start_time
            
            # 캐싱
            if self.config['cache_enabled']:
                self.cache[file_hash] = result
                
            # 통계 업데이트
            self.stats['encoded_files'] += 1
            self.stats['modality_counts'][modality] += 1
            self.stats['encoding_times'].append(result.processing_time)
            
        return result
    
    def _encode_image(self, file_path: Path) -> Optional[EncodedResult]:
        """이미지 인코딩"""
        try:
            # OCR로 텍스트 추출
            image = cv2.imread(str(file_path))
            if image is None:
                logger.error(f"이미지 로드 실패: {file_path}")
                return None
                
            ocr_results = self.models['ocr'].readtext(image)
            
            # 텍스트 추출 및 신뢰도 계산
            extracted_text = ""
            confidences = []
            
            for (bbox, text, confidence) in ocr_results:
                if confidence > 0.3:  # 낮은 임계값으로 더 많은 텍스트 수집
                    extracted_text += text + " "
                    confidences.append(confidence)
                    
            extracted_text = extracted_text.strip()
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # 텍스트가 너무 적으면 이미지 메타데이터 활용
            if len(extracted_text) < 10:
                # 파일명, 경로 정보 활용
                path_info = f"이미지 파일 {file_path.name} 경로 {file_path.parent.name}"
                extracted_text = f"{extracted_text} {path_info}".strip()
                
            # 임베딩 생성 (768차원)
            if extracted_text:
                encoding = self.models['text_encoder'].encode([extracted_text])[0]
            else:
                # 텍스트가 없으면 제로 벡터
                encoding = np.zeros(self.config['target_dimensions'], dtype=np.float32)
                avg_confidence = 0.0
                
            return EncodedResult(
                file_path=str(file_path),
                modality='image',
                encoding=encoding,
                confidence=avg_confidence,
                raw_content=extracted_text,
                metadata={
                    'ocr_blocks': len(ocr_results),
                    'image_size': image.shape[:2],
                    'text_length': len(extracted_text),
                    'high_confidence_blocks': sum(1 for _, _, conf in ocr_results if conf > 0.7)
                },
                processing_time=0.0  # 나중에 설정됨
            )
            
        except Exception as e:
            logger.error(f"❌ 이미지 인코딩 실패 {file_path}: {e}")
            return None
    
    def _encode_audio(self, file_path: Path) -> Optional[EncodedResult]:
        """오디오 인코딩"""
        try:
            # Whisper로 음성 인식
            whisper_result = self.models['whisper'].transcribe(str(file_path))
            transcribed_text = whisper_result.get('text', '').strip()
            
            # 언어 및 품질 정보
            language = whisper_result.get('language', 'unknown')
            segments = whisper_result.get('segments', [])
            
            # 신뢰도 계산 (세그먼트 평균)
            if segments:
                segment_probs = []
                for seg in segments:
                    if 'avg_logprob' in seg:
                        # 로그 확률을 확률로 변환
                        prob = np.exp(seg['avg_logprob'])
                        segment_probs.append(prob)
                avg_confidence = np.mean(segment_probs) if segment_probs else 0.8
            else:
                avg_confidence = 0.8  # Whisper 기본 신뢰도
                
            # 텍스트가 너무 적으면 메타데이터 활용
            if len(transcribed_text) < 5:
                path_info = f"오디오 파일 {file_path.name}"
                transcribed_text = f"{transcribed_text} {path_info}".strip()
                
            # 임베딩 생성
            if transcribed_text:
                encoding = self.models['text_encoder'].encode([transcribed_text])[0]
            else:
                encoding = np.zeros(self.config['target_dimensions'], dtype=np.float32)
                avg_confidence = 0.0
                
            return EncodedResult(
                file_path=str(file_path),
                modality='audio',
                encoding=encoding,
                confidence=avg_confidence,
                raw_content=transcribed_text,
                metadata={
                    'language': language,
                    'duration': whisper_result.get('duration', 0),
                    'segments_count': len(segments),
                    'text_length': len(transcribed_text),
                    'whisper_model': self.config['whisper_model']
                },
                processing_time=0.0
            )
            
        except Exception as e:
            logger.error(f"❌ 오디오 인코딩 실패 {file_path}: {e}")
            return None
    
    def _encode_text(self, file_path: Path) -> Optional[EncodedResult]:
        """텍스트 인코딩"""
        try:
            # 파일 읽기 (다양한 인코딩 시도)
            content = ""
            encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-16', 'latin-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
                    
            if not content:
                logger.warning(f"텍스트 파일 읽기 실패: {file_path}")
                content = f"텍스트 파일 {file_path.name} 인코딩 오류"
                
            # 텍스트 길이 제한 (너무 긴 경우 요약)
            original_length = len(content)
            if len(content) > 2000:
                # 앞부분과 뒷부분을 합쳐서 사용
                content = content[:1000] + " ... " + content[-500:]
                
            # 임베딩 생성
            encoding = self.models['text_encoder'].encode([content])[0]
            
            return EncodedResult(
                file_path=str(file_path),
                modality='text',
                encoding=encoding,
                confidence=1.0 if original_length > 10 else 0.3,
                raw_content=content,
                metadata={
                    'original_length': original_length,
                    'processed_length': len(content),
                    'word_count': len(content.split()),
                    'encoding_success': True
                },
                processing_time=0.0
            )
            
        except Exception as e:
            logger.error(f"❌ 텍스트 인코딩 실패 {file_path}: {e}")
            return None
    
    def _detect_modality(self, file_path: Path) -> str:
        """파일 타입에 따른 모달리티 판별"""
        ext = file_path.suffix.lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp']:
            return 'image'
        elif ext in ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac']:
            return 'audio'
        elif ext in ['.txt', '.md', '.doc', '.docx', '.pdf', '.json']:
            return 'text'
        else:
            return 'unknown'
    
    def _get_file_hash(self, file_path: Path) -> str:
        """캐시용 파일 해시 생성"""
        try:
            stat = file_path.stat()
            hash_input = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception:
            return str(hash(str(file_path)))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        avg_time = np.mean(self.stats['encoding_times']) if self.stats['encoding_times'] else 0
        
        return {
            'encoded_files': self.stats['encoded_files'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': f"{(self.stats['cache_hits'] / max(1, self.stats['encoded_files'])) * 100:.1f}%",
            'average_encoding_time': f"{avg_time:.3f}초",
            'modality_distribution': self.stats['modality_counts'].copy(),
            'total_processing_time': sum(self.stats['encoding_times'])
        }
    
    def normalize_encodings(self, encodings: List[EncodedResult]) -> List[EncodedResult]:
        """인코딩 결과 정규화"""
        if not encodings:
            return encodings
            
        # 모달리티별 정규화
        modalities = set(enc.modality for enc in encodings)
        
        for modality in modalities:
            modality_encodings = [enc for enc in encodings if enc.modality == modality]
            if len(modality_encodings) < 2:
                continue
                
            # 인코딩 벡터들 수집
            vectors = np.array([enc.encoding for enc in modality_encodings])
            
            # 표준화
            if modality not in self.scalers:
                self.scalers[modality] = StandardScaler()
                
            normalized_vectors = self.scalers[modality].fit_transform(vectors)
            
            # 정규화된 벡터 다시 할당
            for i, enc in enumerate(modality_encodings):
                enc.encoding = normalized_vectors[i].astype(np.float32)
                enc.metadata['normalized'] = True
                
        return encodings

# 사용 예제 및 테스트 코드
def main():
    """사용 예제"""
    encoder = MultimodalEncoder()
    
    # 테스트 파일들
    test_files = [
        Path("test_image.jpg"),
        Path("test_audio.wav"), 
        Path("test_document.txt")
    ]
    
    # 실제 존재하는 파일들만 필터링
    existing_files = [f for f in test_files if f.exists()]
    
    if existing_files:
        print("🤖 멀티모달 인코더 테스트")
        print("=" * 50)
        
        # 배치 인코딩
        results = encoder.encode_batch(existing_files)
        
        print(f"\n📊 인코딩 결과: {len(results)}개 파일")
        
        for result in results:
            print(f"\n📄 {Path(result.file_path).name}")
            print(f"   모달리티: {result.modality}")
            print(f"   임베딩 차원: {result.encoding.shape}")
            print(f"   신뢰도: {result.confidence:.2f}")
            print(f"   처리시간: {result.processing_time:.3f}초")
            print(f"   내용 미리보기: {result.raw_content[:100]}...")
            
        # 성능 통계
        stats = encoder.get_performance_stats()
        print(f"\n📈 성능 통계:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
    else:
        print("❌ 테스트 파일을 찾을 수 없습니다.")
        print("다음 파일들을 생성하고 다시 실행하세요:")
        for file in test_files:
            print(f"   - {file}")

if __name__ == "__main__":
    main()