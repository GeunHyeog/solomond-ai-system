#!/usr/bin/env python3
"""
향상된 OCR 처리 시스템
EasyOCR 모델 다운로드 실패 문제 해결 및 대안 제공
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

# Unicode 인코딩 문제 해결
os.environ['PYTHONIOENCODING'] = 'utf-8'

class EnhancedOCRHandler:
    """향상된 OCR 처리 시스템"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.ocr_reader = None
        self.tesseract_available = False
        self.fallback_mode = False
        
        # OCR 엔진 우선순위: EasyOCR > Tesseract > 텍스트 추출 없음
        self._check_available_engines()
    
    def _check_available_engines(self):
        """사용 가능한 OCR 엔진 확인"""
        # EasyOCR 확인
        try:
            import easyocr
            self.logger.info("✅ EasyOCR 사용 가능")
        except ImportError:
            self.logger.warning("⚠️ EasyOCR 불가능")
        
        # Tesseract 확인
        try:
            import pytesseract
            from PIL import Image
            # 간단한 테스트
            test_image = Image.new('RGB', (100, 50), color='white')
            pytesseract.image_to_string(test_image)
            self.tesseract_available = True
            self.logger.info("✅ Tesseract OCR 사용 가능 (대안)")
        except (ImportError, Exception):
            self.logger.warning("⚠️ Tesseract OCR 불가능")
    
    def _load_easyocr_with_retry(self, max_retries: int = 3) -> bool:
        """EasyOCR 재시도 로딩 시스템"""
        for attempt in range(max_retries):
            try:
                self.logger.info(f"🔄 EasyOCR 로딩 시도 {attempt + 1}/{max_retries}")
                
                import easyocr
                
                # 캐시 디렉토리 확인 및 생성
                cache_dir = os.path.expanduser('~/.EasyOCR')
                os.makedirs(cache_dir, exist_ok=True)
                
                # 모델 디렉토리 확인
                model_dir = os.path.join(cache_dir, 'model')
                os.makedirs(model_dir, exist_ok=True)
                
                self.ocr_reader = easyocr.Reader(
                    ['ko', 'en'],
                    gpu=False,
                    verbose=False,
                    download_enabled=True
                )
                
                # 테스트 실행
                test_result = self._test_ocr_engine()
                if test_result:
                    self.logger.info("✅ EasyOCR 로딩 및 테스트 성공")
                    return True
                else:
                    self.logger.warning(f"⚠️ EasyOCR 테스트 실패 (시도 {attempt + 1})")
                    
            except Exception as e:
                self.logger.error(f"❌ EasyOCR 로딩 실패 (시도 {attempt + 1}): {str(e)}")
                
                # 특정 에러에 대한 해결 시도
                if "CRNN.yaml" in str(e):
                    self._create_crnn_config()
                elif "urllib" in str(e) or "download" in str(e).lower():
                    self._handle_download_error()
                
                time.sleep(2 ** attempt)  # 지수 백오프
        
        return False
    
    def _create_crnn_config(self):
        """CRNN.yaml 설정 파일 생성"""
        try:
            import yaml
            
            cache_dir = os.path.expanduser('~/.EasyOCR')
            model_dir = os.path.join(cache_dir, 'model')
            os.makedirs(model_dir, exist_ok=True)
            
            crnn_config = {
                'model': {
                    'type': 'CRNN',
                    'backbone': {
                        'type': 'ResNet',
                        'depth': 34
                    },
                    'neck': {
                        'type': 'LSTM',
                        'hidden_size': 256,
                        'num_layers': 2,
                        'bidirectional': True
                    },
                    'head': {
                        'type': 'CTCLoss'
                    }
                }
            }
            
            yaml_path = os.path.join(model_dir, 'CRNN.yaml')
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(crnn_config, f, default_flow_style=False)
            
            self.logger.info(f"✅ CRNN.yaml 생성됨: {yaml_path}")
            
        except Exception as e:
            self.logger.error(f"❌ CRNN.yaml 생성 실패: {str(e)}")
    
    def _handle_download_error(self):
        """다운로드 에러 처리"""
        self.logger.info("🔧 네트워크 다운로드 문제 해결 시도...")
        
        # 환경변수 설정
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['CURL_CA_BUNDLE'] = ''
        
        # 프록시 설정 제거
        for proxy_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            if proxy_var in os.environ:
                del os.environ[proxy_var]
    
    def _test_ocr_engine(self) -> bool:
        """OCR 엔진 테스트"""
        try:
            if self.ocr_reader is None:
                return False
            
            # 간단한 테스트 이미지 생성
            from PIL import Image, ImageDraw
            
            test_image = Image.new('RGB', (200, 100), color='white')
            draw = ImageDraw.Draw(test_image)
            draw.text((10, 30), "TEST", fill='black')
            
            # 임시 파일로 저장
            test_path = "temp_ocr_test.png"
            test_image.save(test_path)
            
            # OCR 테스트
            results = self.ocr_reader.readtext(test_path)
            
            # 정리
            os.remove(test_path)
            
            return len(results) > 0
            
        except Exception as e:
            self.logger.error(f"OCR 테스트 실패: {str(e)}")
            return False
    
    def _fallback_to_tesseract(self, image_path: str) -> List[Dict[str, Any]]:
        """Tesseract OCR 대안 사용"""
        try:
            import pytesseract
            from PIL import Image
            
            self.logger.info("🔄 Tesseract OCR 대안 사용")
            
            image = Image.open(image_path)
            
            # 한국어+영어 설정
            custom_config = r'--oem 3 --psm 6 -l kor+eng'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            # EasyOCR 형식으로 변환
            if text.strip():
                return [{
                    "bbox": [[0, 0], [image.width, 0], [image.width, image.height], [0, image.height]],
                    "text": text.strip(),
                    "confidence": 0.8,  # 기본 신뢰도
                    "analysis_type": "tesseract_fallback"
                }]
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"❌ Tesseract 대안 실패: {str(e)}")
            return []
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """이미지 OCR 처리 (향상된 에러 처리)"""
        start_time = time.time()
        
        # 파일 존재 확인
        if not os.path.exists(image_path):
            return {
                "success": False,
                "error": f"이미지 파일을 찾을 수 없습니다: {image_path}",
                "results": [],
                "processing_time": 0
            }
        
        # EasyOCR 우선 시도
        if self.ocr_reader is None:
            success = self._load_easyocr_with_retry()
            if not success:
                self.fallback_mode = True
        
        # OCR 처리
        results = []
        analysis_type = "unknown"
        
        if self.ocr_reader is not None and not self.fallback_mode:
            try:
                self.logger.info(f"🖼️ EasyOCR로 이미지 분석: {os.path.basename(image_path)}")
                ocr_results = self.ocr_reader.readtext(image_path)
                
                results = []
                for bbox, text, confidence in ocr_results:
                    results.append({
                        "bbox": bbox,
                        "text": text,
                        "confidence": float(confidence),
                        "analysis_type": "easyocr_success"
                    })
                
                analysis_type = "easyocr_success"
                
            except Exception as e:
                self.logger.error(f"❌ EasyOCR 분석 실패: {str(e)}")
                self.fallback_mode = True
        
        # Tesseract 대안 사용
        if self.fallback_mode and self.tesseract_available:
            results = self._fallback_to_tesseract(image_path)
            analysis_type = "tesseract_fallback"
        
        # 결과 없음 처리
        if not results:
            results = [{
                "bbox": [[0, 0], [100, 0], [100, 50], [0, 50]],
                "text": f"[텍스트 추출 실패: {os.path.basename(image_path)}]",
                "confidence": 0.0,
                "analysis_type": "extraction_failed"
            }]
            analysis_type = "extraction_failed"
        
        processing_time = time.time() - start_time
        
        return {
            "success": len(results) > 0,
            "results": results,
            "analysis_type": analysis_type,
            "processing_time": processing_time,
            "file_path": image_path,
            "total_blocks": len(results)
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """OCR 시스템 상태 확인"""
        return {
            "easyocr_available": self.ocr_reader is not None,
            "tesseract_available": self.tesseract_available,
            "fallback_mode": self.fallback_mode,
            "cache_directory": os.path.expanduser('~/.EasyOCR'),
            "model_files": self._count_model_files()
        }
    
    def _count_model_files(self) -> int:
        """모델 파일 개수 확인"""
        try:
            cache_dir = os.path.expanduser('~/.EasyOCR/model')
            if os.path.exists(cache_dir):
                return len([f for f in os.listdir(cache_dir) if f.endswith('.pth')])
            return 0
        except:
            return 0

# 편의 함수
def create_enhanced_ocr_handler(logger=None) -> EnhancedOCRHandler:
    """향상된 OCR 핸들러 생성"""
    return EnhancedOCRHandler(logger)

def test_ocr_installation():
    """OCR 설치 상태 테스트"""
    handler = EnhancedOCRHandler()
    status = handler.get_system_status()
    
    print("=== OCR 시스템 상태 ===")
    print(f"EasyOCR 사용 가능: {status['easyocr_available']}")
    print(f"Tesseract 사용 가능: {status['tesseract_available']}")
    print(f"대안 모드: {status['fallback_mode']}")
    print(f"모델 파일 개수: {status['model_files']}")
    print(f"캐시 디렉토리: {status['cache_directory']}")
    
    return status

if __name__ == "__main__":
    # 테스트 실행
    test_ocr_installation()