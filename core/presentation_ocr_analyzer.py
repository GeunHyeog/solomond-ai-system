"""
솔로몬드 AI 시스템 - PPT 화면 특화 OCR 분석 엔진
현장에서 촬영한 PPT 화면의 OCR 정확도 분석 및 최적화 모듈
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os
from pathlib import Path
import json
from datetime import datetime
import re

# 이미지 처리 라이브러리
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# OCR 라이브러리들
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# 컴퓨터 비전
try:
    import skimage
    from skimage import feature, filters, morphology, measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

class PresentationOCRAnalyzer:
    """PPT 화면 특화 OCR 분석 클래스"""
    
    def __init__(self):
        # PPT 특성 감지 기준값들
        self.ppt_detection_thresholds = {
            "aspect_ratio_min": 1.2,  # 일반적인 PPT 가로세로 비율
            "aspect_ratio_max": 2.0,
            "min_text_area_ratio": 0.05,  # 텍스트가 차지하는 최소 면적 비율
            "slide_border_threshold": 0.8,  # 슬라이드 경계 감지 임계값
        }
        
        # OCR 품질 평가 기준
        self.quality_metrics = {
            "excellent": {"confidence": 0.9, "word_accuracy": 0.95, "layout_score": 0.9},
            "good": {"confidence": 0.8, "word_accuracy": 0.85, "layout_score": 0.8},
            "fair": {"confidence": 0.7, "word_accuracy": 0.75, "layout_score": 0.7},
            "poor": {"confidence": 0.6, "word_accuracy": 0.65, "layout_score": 0.6}
        }
        
        # PPT 레이아웃 패턴
        self.layout_patterns = {
            "title_slide": {"title_area": (0.1, 0.2, 0.9, 0.4), "content_area": (0.1, 0.5, 0.9, 0.8)},
            "bullet_slide": {"title_area": (0.1, 0.1, 0.9, 0.25), "content_area": (0.1, 0.3, 0.9, 0.9)},
            "two_column": {"left_area": (0.05, 0.25, 0.48, 0.9), "right_area": (0.52, 0.25, 0.95, 0.9)},
            "image_text": {"text_area": (0.05, 0.1, 0.6, 0.9), "image_area": (0.65, 0.2, 0.95, 0.8)}
        }
        
        # EasyOCR 초기화
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en', 'ko'], gpu=False)
            except Exception as e:
                logging.warning(f"EasyOCR 초기화 실패: {e}")
        
        # 스레드 풀 executor
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logging.info("PPT 화면 특화 OCR 분석 엔진 초기화 완료")
    
    async def analyze_presentation_image(self, 
                                       image_data: bytes, 
                                       filename: str,
                                       enhance_quality: bool = True) -> Dict:
        """
        PPT 화면 이미지 종합 분석
        
        Args:
            image_data: 이미지 바이너리 데이터
            filename: 파일명
            enhance_quality: 이미지 품질 향상 여부
            
        Returns:
            PPT OCR 분석 결과 딕셔너리
        """
        try:
            print(f"📊 PPT 화면 OCR 분석 시작: {filename}")
            
            # 이미지 로드
            image = await self._load_image_data(image_data)
            if image is None:
                return {
                    "success": False,
                    "error": "이미지 데이터 로드 실패",
                    "filename": filename
                }
            
            # PPT 슬라이드 감지
            slide_detection = await self._detect_ppt_slide(image)
            
            # 이미지 전처리 및 최적화
            if enhance_quality:
                optimized_image = await self._optimize_ppt_image(image, slide_detection)
            else:
                optimized_image = image
            
            # 병렬로 OCR 분석 수행
            ocr_tasks = [
                self._perform_ocr_analysis(optimized_image, "tesseract"),
                self._perform_ocr_analysis(optimized_image, "easyocr"),
                self._analyze_layout_structure(optimized_image),
                self._detect_tables_and_charts(optimized_image),
                self._analyze_text_regions(optimized_image)
            ]
            
            results = await asyncio.gather(*ocr_tasks, return_exceptions=True)
            
            # 결과 통합
            tesseract_result = results[0] if not isinstance(results[0], Exception) else {}
            easyocr_result = results[1] if not isinstance(results[1], Exception) else {}
            layout_result = results[2] if not isinstance(results[2], Exception) else {}
            tables_charts = results[3] if not isinstance(results[3], Exception) else {}
            text_regions = results[4] if not isinstance(results[4], Exception) else {}
            
            # 최적 OCR 결과 선택
            best_ocr_result = self._select_best_ocr_result(tesseract_result, easyocr_result)
            
            # OCR 정확도 평가
            accuracy_assessment = await self._assess_ocr_accuracy(
                best_ocr_result, layout_result, text_regions
            )
            
            # 개선 제안 생성
            recommendations = self._generate_ppt_recommendations(
                slide_detection, accuracy_assessment, layout_result, image
            )
            
            # 최종 결과 구성
            result = {
                "success": True,
                "filename": filename,
                "image_info": {
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "file_size_mb": round(len(image_data) / (1024 * 1024), 2)
                },
                "slide_detection": slide_detection,
                "ocr_analysis": {
                    "best_method": best_ocr_result.get("method", "unknown"),
                    "extracted_text": best_ocr_result.get("text", ""),
                    "confidence": best_ocr_result.get("confidence", 0.0),
                    "word_count": len(best_ocr_result.get("text", "").split()),
                    "tesseract_result": tesseract_result,
                    "easyocr_result": easyocr_result
                },
                "layout_analysis": layout_result,
                "tables_and_charts": tables_charts,
                "text_regions": text_regions,
                "accuracy_assessment": accuracy_assessment,
                "recommendations": recommendations,
                "analyzed_at": datetime.now().isoformat()
            }
            
            print(f"✅ PPT OCR 분석 완료: 정확도 {accuracy_assessment.get('overall_score', 0)}/100")
            return result
            
        except Exception as e:
            logging.error(f"PPT 화면 OCR 분석 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "filename": filename
            }
    
    async def _load_image_data(self, image_data: bytes) -> Optional[Image.Image]:
        """이미지 데이터 로드"""
        try:
            if not PIL_AVAILABLE:
                raise Exception("PIL 라이브러리가 필요합니다")
            
            image = Image.open(io.BytesIO(image_data))
            
            # RGB로 변환
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logging.error(f"이미지 데이터 로드 오류: {e}")
            return None
    
    async def _detect_ppt_slide(self, image: Image.Image) -> Dict:
        """PPT 슬라이드 자동 감지"""
        try:
            width, height = image.size
            aspect_ratio = width / height
            
            # 기본 PPT 특성 확인
            is_ppt_ratio = (self.ppt_detection_thresholds["aspect_ratio_min"] <= 
                           aspect_ratio <= self.ppt_detection_thresholds["aspect_ratio_max"])
            
            # 이미지를 numpy 배열로 변환
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # 슬라이드 경계 감지
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 가장 큰 사각형 찾기 (슬라이드 경계)
            slide_boundary = None
            max_area = 0
            
            for contour in contours:
                # 컨투어를 다각형으로 근사
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # 4개 꼭짓점을 가진 사각형인지 확인
                if len(approx) == 4:
                    area = cv2.contourArea(contour)
                    if area > max_area and area > (width * height * 0.3):  # 전체 면적의 30% 이상
                        max_area = area
                        slide_boundary = approx
            
            # 슬라이드 경계 정확도
            boundary_score = min(1.0, max_area / (width * height)) if slide_boundary is not None else 0.0
            
            # 텍스트 영역 비율 추정
            text_area_ratio = await self._estimate_text_area_ratio(gray)
            
            # PPT 감지 신뢰도 계산
            ppt_confidence = 0.0
            if is_ppt_ratio:
                ppt_confidence += 0.3
            if boundary_score > 0.5:
                ppt_confidence += 0.4
            if text_area_ratio > self.ppt_detection_thresholds["min_text_area_ratio"]:
                ppt_confidence += 0.3
            
            return {
                "is_presentation": ppt_confidence > 0.6,
                "confidence": round(ppt_confidence, 3),
                "aspect_ratio": round(aspect_ratio, 2),
                "slide_boundary": slide_boundary.tolist() if slide_boundary is not None else None,
                "boundary_score": round(boundary_score, 3),
                "text_area_ratio": round(text_area_ratio, 3),
                "image_characteristics": {
                    "width": width,
                    "height": height,
                    "is_landscape": width > height,
                    "resolution_category": self._categorize_resolution(width, height)
                }
            }
            
        except Exception as e:
            logging.error(f"PPT 슬라이드 감지 오류: {e}")
            return {"error": str(e)}
    
    async def _estimate_text_area_ratio(self, gray_image: np.ndarray) -> float:
        """텍스트 영역 비율 추정"""
        try:
            # 적응적 임계화로 텍스트 영역 추출
            adaptive_thresh = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 모폴로지 연산으로 텍스트 영역 정리
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
            
            # 텍스트 영역 비율 계산
            text_pixels = np.sum(cleaned > 0)
            total_pixels = cleaned.shape[0] * cleaned.shape[1]
            
            return text_pixels / total_pixels
            
        except Exception as e:
            logging.error(f"텍스트 영역 비율 추정 오류: {e}")
            return 0.0
    
    def _categorize_resolution(self, width: int, height: int) -> str:
        """해상도 카테고리 분류"""
        total_pixels = width * height
        
        if total_pixels >= 2073600:  # 1920x1080 이상
            return "high"
        elif total_pixels >= 921600:  # 1280x720 이상
            return "medium"
        else:
            return "low"
    
    async def _optimize_ppt_image(self, image: Image.Image, slide_detection: Dict) -> Image.Image:
        """PPT 이미지 전처리 최적화"""
        try:
            optimized = image.copy()
            
            # 해상도 최적화 (너무 작으면 업스케일링)
            width, height = optimized.size
            if width < 1200 or height < 800:
                scale_factor = max(1200/width, 800/height)
                new_size = (int(width * scale_factor), int(height * scale_factor))
                optimized = optimized.resize(new_size, Image.Resampling.LANCZOS)
            
            # 대비 향상
            enhancer = ImageEnhance.Contrast(optimized)
            optimized = enhancer.enhance(1.3)
            
            # 선명도 향상
            enhancer = ImageEnhance.Sharpness(optimized)
            optimized = enhancer.enhance(1.2)
            
            # 밝기 조정 (너무 어두우면 밝게)
            img_array = np.array(optimized)
            mean_brightness = np.mean(img_array)
            if mean_brightness < 100:  # 0-255 범위에서 100 미만이면 어두움
                enhancer = ImageEnhance.Brightness(optimized)
                optimized = enhancer.enhance(1.2)
            
            # 노이즈 제거
            optimized = optimized.filter(ImageFilter.MedianFilter(size=3))
            
            # 슬라이드 경계가 감지되었으면 크롭
            if slide_detection.get("slide_boundary") and slide_detection.get("boundary_score", 0) > 0.7:
                try:
                    boundary = np.array(slide_detection["slide_boundary"])
                    x, y, w, h = cv2.boundingRect(boundary)
                    # 약간의 여백을 두고 크롭
                    margin = 10
                    left = max(0, x - margin)
                    top = max(0, y - margin)
                    right = min(optimized.width, x + w + margin)
                    bottom = min(optimized.height, y + h + margin)
                    optimized = optimized.crop((left, top, right, bottom))
                except Exception as crop_error:
                    logging.warning(f"슬라이드 크롭 실패: {crop_error}")
            
            return optimized
            
        except Exception as e:
            logging.error(f"PPT 이미지 최적화 오류: {e}")
            return image
    
    async def _perform_ocr_analysis(self, image: Image.Image, method: str) -> Dict:
        """OCR 분석 수행"""
        try:
            if method == "tesseract" and TESSERACT_AVAILABLE:
                return await self._ocr_with_tesseract(image)
            elif method == "easyocr" and EASYOCR_AVAILABLE and self.easyocr_reader:
                return await self._ocr_with_easyocr(image)
            else:
                return {"error": f"OCR 방법 {method}을 사용할 수 없습니다"}
                
        except Exception as e:
            logging.error(f"OCR 분석 오류 ({method}): {e}")
            return {"error": str(e)}
    
    async def _ocr_with_tesseract(self, image: Image.Image) -> Dict:
        """Tesseract OCR 분석"""
        try:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                image.save(temp_file.name, 'PNG')
                temp_path = temp_file.name
            
            try:
                # 비동기로 Tesseract 실행
                loop = asyncio.get_event_loop()
                
                # PPT에 최적화된 설정
                config = '--oem 3 --psm 6 -c preserve_interword_spaces=1'
                
                text = await loop.run_in_executor(
                    self.executor,
                    lambda: pytesseract.image_to_string(temp_path, config=config, lang='kor+eng')
                )
                
                # 상세 데이터 추출
                data = await loop.run_in_executor(
                    self.executor,
                    lambda: pytesseract.image_to_data(
                        temp_path, 
                        config=config,
                        lang='kor+eng',
                        output_type=pytesseract.Output.DICT
                    )
                )
                
                # 신뢰도 계산
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                # 단어별 결과 정리
                words = []
                for i in range(len(data['text'])):
                    if int(data['conf'][i]) > 0 and data['text'][i].strip():
                        words.append({
                            "text": data['text'][i],
                            "confidence": int(data['conf'][i]),
                            "bbox": {
                                "left": data['left'][i],
                                "top": data['top'][i],
                                "width": data['width'][i],
                                "height": data['height'][i]
                            }
                        })
                
                return {
                    "method": "tesseract",
                    "text": text.strip(),
                    "confidence": round(avg_confidence / 100, 3),
                    "word_count": len(text.split()),
                    "words": words,
                    "raw_data": data
                }
                
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logging.error(f"Tesseract OCR 오류: {e}")
            return {"method": "tesseract", "error": str(e)}
    
    async def _ocr_with_easyocr(self, image: Image.Image) -> Dict:
        """EasyOCR 분석"""
        try:
            # PIL Image를 numpy array로 변환
            image_array = np.array(image)
            
            # 비동기로 EasyOCR 실행
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.easyocr_reader.readtext,
                image_array
            )
            
            # 결과 파싱
            texts = []
            confidences = []
            words = []
            
            for detection in result:
                if len(detection) >= 2:
                    bbox, text, confidence = detection[0], detection[1], detection[2] if len(detection) > 2 else 0.5
                    
                    texts.append(text)
                    confidences.append(confidence)
                    
                    # 바운딩 박스 정리
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    left, top = min(x_coords), min(y_coords)
                    width, height = max(x_coords) - left, max(y_coords) - top
                    
                    words.append({
                        "text": text,
                        "confidence": round(confidence * 100, 1),
                        "bbox": {
                            "left": int(left),
                            "top": int(top),
                            "width": int(width),
                            "height": int(height)
                        }
                    })
            
            full_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "method": "easyocr",
                "text": full_text,
                "confidence": round(avg_confidence, 3),
                "word_count": len(full_text.split()),
                "words": words,
                "raw_result": result
            }
            
        except Exception as e:
            logging.error(f"EasyOCR 오류: {e}")
            return {"method": "easyocr", "error": str(e)}
    
    def _select_best_ocr_result(self, tesseract_result: Dict, easyocr_result: Dict) -> Dict:
        """최적 OCR 결과 선택"""
        try:
            # 에러가 있는 결과 제외
            valid_results = []
            if "error" not in tesseract_result and tesseract_result.get("text"):
                valid_results.append(tesseract_result)
            if "error" not in easyocr_result and easyocr_result.get("text"):
                valid_results.append(easyocr_result)
            
            if not valid_results:
                return {"method": "none", "text": "", "confidence": 0.0}
            
            # 신뢰도가 높은 결과 선택
            best_result = max(valid_results, key=lambda x: x.get("confidence", 0))
            
            # 두 결과가 비슷하면 더 많은 텍스트를 추출한 것 선택
            if len(valid_results) == 2:
                conf_diff = abs(valid_results[0].get("confidence", 0) - valid_results[1].get("confidence", 0))
                if conf_diff < 0.1:  # 신뢰도 차이가 0.1 미만이면
                    best_result = max(valid_results, key=lambda x: x.get("word_count", 0))
            
            return best_result
            
        except Exception as e:
            logging.error(f"최적 OCR 결과 선택 오류: {e}")
            return {"method": "error", "text": "", "confidence": 0.0}
    
    async def _analyze_layout_structure(self, image: Image.Image) -> Dict:
        """레이아웃 구조 분석"""
        try:
            width, height = image.size
            
            # 이미지를 그리드로 분할하여 텍스트 밀도 분석
            grid_size = 8
            cell_width = width // grid_size
            cell_height = height // grid_size
            
            text_density_map = []
            
            for row in range(grid_size):
                row_densities = []
                for col in range(grid_size):
                    left = col * cell_width
                    top = row * cell_height
                    right = min(left + cell_width, width)
                    bottom = min(top + cell_height, height)
                    
                    # 셀 영역 크롭
                    cell = image.crop((left, top, right, bottom))
                    
                    # 간단한 텍스트 밀도 계산 (엣지 기반)
                    cell_array = np.array(cell.convert('L'))
                    edges = cv2.Canny(cell_array, 50, 150)
                    density = np.sum(edges > 0) / (cell_array.shape[0] * cell_array.shape[1])
                    
                    row_densities.append(round(density, 4))
                
                text_density_map.append(row_densities)
            
            # 레이아웃 패턴 감지
            detected_pattern = self._detect_layout_pattern(text_density_map)
            
            # 제목 영역 감지
            title_region = self._detect_title_region(text_density_map, width, height)
            
            # 컬럼 구조 감지
            column_structure = self._detect_column_structure(text_density_map)
            
            return {
                "text_density_map": text_density_map,
                "detected_pattern": detected_pattern,
                "title_region": title_region,
                "column_structure": column_structure,
                "layout_score": self._calculate_layout_score(text_density_map, detected_pattern)
            }
            
        except Exception as e:
            logging.error(f"레이아웃 구조 분석 오류: {e}")
            return {"error": str(e)}
    
    def _detect_layout_pattern(self, density_map: List[List[float]]) -> Dict:
        """레이아웃 패턴 감지"""
        try:
            grid_size = len(density_map)
            
            # 상단 영역 (제목) 텍스트 밀도
            top_density = np.mean([density_map[i] for i in range(min(2, grid_size))])
            
            # 중간 영역 텍스트 밀도
            mid_start = grid_size // 4
            mid_end = 3 * grid_size // 4
            mid_density = np.mean([density_map[i] for i in range(mid_start, mid_end)])
            
            # 좌우 균형도 계산
            left_density = np.mean([[row[j] for j in range(grid_size//2)] for row in density_map])
            right_density = np.mean([[row[j] for j in range(grid_size//2, grid_size)] for row in density_map])
            balance_ratio = min(left_density, right_density) / max(left_density, right_density) if max(left_density, right_density) > 0 else 0
            
            # 패턴 분류
            if top_density > mid_density * 1.5:
                pattern_type = "title_slide"
            elif balance_ratio < 0.6:  # 좌우 불균형
                pattern_type = "image_text"
            elif balance_ratio > 0.8 and left_density > 0.01 and right_density > 0.01:
                pattern_type = "two_column"
            else:
                pattern_type = "bullet_slide"
            
            return {
                "pattern_type": pattern_type,
                "confidence": min(1.0, max(top_density, mid_density) * 10),
                "balance_ratio": round(balance_ratio, 3),
                "top_density": round(top_density, 4),
                "mid_density": round(mid_density, 4)
            }
            
        except Exception as e:
            logging.error(f"레이아웃 패턴 감지 오류: {e}")
            return {"error": str(e)}
    
    def _detect_title_region(self, density_map: List[List[float]], width: int, height: int) -> Dict:
        """제목 영역 감지"""
        try:
            grid_size = len(density_map)
            
            # 상단 2-3행에서 제목 영역 찾기
            title_candidates = []
            for row in range(min(3, grid_size)):
                row_density = density_map[row]
                if max(row_density) > 0.02:  # 텍스트가 있는 것으로 판단
                    title_candidates.append((row, max(row_density), np.mean(row_density)))
            
            if title_candidates:
                # 가장 밀도가 높은 행을 제목으로 선택
                best_row = max(title_candidates, key=lambda x: x[1])
                row_idx = best_row[0]
                
                # 제목 영역 좌표 계산
                cell_height = height // grid_size
                title_region = {
                    "top": row_idx * cell_height,
                    "bottom": (row_idx + 1) * cell_height,
                    "left": 0,
                    "right": width,
                    "confidence": min(1.0, best_row[1] * 20)
                }
            else:
                title_region = None
            
            return {
                "region": title_region,
                "candidates": title_candidates
            }
            
        except Exception as e:
            logging.error(f"제목 영역 감지 오류: {e}")
            return {"error": str(e)}
    
    def _detect_column_structure(self, density_map: List[List[float]]) -> Dict:
        """컬럼 구조 감지"""
        try:
            grid_size = len(density_map)
            
            # 세로 방향 텍스트 밀도 합계
            col_densities = []
            for col in range(grid_size):
                col_density = sum(density_map[row][col] for row in range(grid_size))
                col_densities.append(col_density)
            
            # 컬럼 경계 감지 (밀도가 낮은 지점)
            boundaries = []
            threshold = np.mean(col_densities) * 0.3
            
            for i in range(1, grid_size - 1):
                if col_densities[i] < threshold and col_densities[i-1] > threshold and col_densities[i+1] > threshold:
                    boundaries.append(i)
            
            # 컬럼 수 추정
            if len(boundaries) == 0:
                column_count = 1
            elif len(boundaries) == 1:
                column_count = 2
            else:
                column_count = len(boundaries) + 1
            
            return {
                "column_count": column_count,
                "boundaries": boundaries,
                "column_densities": col_densities,
                "structure_confidence": min(1.0, max(col_densities) * 5)
            }
            
        except Exception as e:
            logging.error(f"컬럼 구조 감지 오류: {e}")
            return {"error": str(e)}
    
    def _calculate_layout_score(self, density_map: List[List[float]], pattern_info: Dict) -> float:
        """레이아웃 점수 계산"""
        try:
            # 전체 텍스트 밀도
            total_density = np.mean([np.mean(row) for row in density_map])
            
            # 패턴 일치도
            pattern_confidence = pattern_info.get("confidence", 0)
            
            # 균형도
            balance_ratio = pattern_info.get("balance_ratio", 0.5)
            
            # 종합 점수 (0-1)
            layout_score = (total_density * 20 + pattern_confidence + balance_ratio) / 3
            
            return round(min(1.0, layout_score), 3)
            
        except Exception as e:
            logging.error(f"레이아웃 점수 계산 오류: {e}")
            return 0.5
    
    async def _detect_tables_and_charts(self, image: Image.Image) -> Dict:
        """표와 차트 감지"""
        try:
            # 이미지를 numpy 배열로 변환
            img_array = np.array(image.convert('L'))
            
            # 가로선 감지 (표의 행 구분선)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, horizontal_kernel)
            
            # 세로선 감지 (표의 열 구분선)
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical_lines = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, vertical_kernel)
            
            # 표 구조 감지
            table_mask = cv2.add(horizontal_lines, vertical_lines)
            table_score = np.sum(table_mask > 0) / (img_array.shape[0] * img_array.shape[1])
            
            # 원형 구조 감지 (파이 차트)
            circles = cv2.HoughCircles(
                img_array, cv2.HOUGH_GRADIENT, 1, img_array.shape[0]//8,
                param1=50, param2=30, minRadius=30, maxRadius=min(img_array.shape)//4
            )
            
            pie_chart_detected = circles is not None and len(circles[0]) > 0
            
            # 직사각형 구조 감지 (막대 차트)
            contours, _ = cv2.findContours(
                cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)[1],
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            rectangles = []
            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    area = cv2.contourArea(contour)
                    if area > 1000:  # 충분히 큰 사각형만
                        rectangles.append(contour)
            
            bar_chart_detected = len(rectangles) >= 3  # 3개 이상의 막대
            
            return {
                "table_detected": table_score > 0.01,
                "table_score": round(table_score, 4),
                "pie_chart_detected": pie_chart_detected,
                "bar_chart_detected": bar_chart_detected,
                "chart_elements": {
                    "horizontal_lines": int(np.sum(horizontal_lines > 0)),
                    "vertical_lines": int(np.sum(vertical_lines > 0)),
                    "circles": len(circles[0]) if circles is not None else 0,
                    "rectangles": len(rectangles)
                }
            }
            
        except Exception as e:
            logging.error(f"표/차트 감지 오류: {e}")
            return {"error": str(e)}
    
    async def _analyze_text_regions(self, image: Image.Image) -> Dict:
        """텍스트 영역 분석"""
        try:
            # 이미지를 그레이스케일로 변환
            gray = np.array(image.convert('L'))
            
            # 적응적 임계화
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 텍스트 영역 찾기
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 텍스트 블록 분석
            text_blocks = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # 최소 크기 필터
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 텍스트 블록 특성 분석
                    aspect_ratio = w / h
                    if 0.1 <= aspect_ratio <= 10:  # 합리적인 가로세로 비율
                        text_blocks.append({
                            "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                            "area": int(area),
                            "aspect_ratio": round(aspect_ratio, 2)
                        })
            
            # 읽기 순서 추정 (위에서 아래, 왼쪽에서 오른쪽)
            text_blocks.sort(key=lambda block: (block["bbox"]["y"], block["bbox"]["x"]))
            
            # 텍스트 밀도 계산
            total_text_area = sum(block["area"] for block in text_blocks)
            image_area = image.width * image.height
            text_density = total_text_area / image_area
            
            return {
                "text_blocks": text_blocks,
                "block_count": len(text_blocks),
                "text_density": round(text_density, 4),
                "reading_order_confidence": 0.8 if len(text_blocks) > 0 else 0.0
            }
            
        except Exception as e:
            logging.error(f"텍스트 영역 분석 오류: {e}")
            return {"error": str(e)}
    
    async def _assess_ocr_accuracy(self, 
                                 ocr_result: Dict, 
                                 layout_result: Dict, 
                                 text_regions: Dict) -> Dict:
        """OCR 정확도 평가"""
        try:
            # 기본 신뢰도
            base_confidence = ocr_result.get("confidence", 0.0)
            
            # 텍스트 양 평가
            word_count = ocr_result.get("word_count", 0)
            text_quantity_score = min(1.0, word_count / 50)  # 50단어를 기준으로 정규화
            
            # 레이아웃 일치도
            layout_score = layout_result.get("layout_score", 0.5)
            
            # 텍스트 영역 커버리지
            detected_blocks = text_regions.get("block_count", 0)
            coverage_score = min(1.0, detected_blocks / 10)  # 10개 블록을 기준으로 정규화
            
            # 종합 정확도 점수
            overall_score = (
                base_confidence * 0.4 +
                text_quantity_score * 0.2 +
                layout_score * 0.2 +
                coverage_score * 0.2
            ) * 100
            
            # 품질 레벨 평가
            if overall_score >= 85:
                quality_level = "excellent"
            elif overall_score >= 70:
                quality_level = "good"
            elif overall_score >= 55:
                quality_level = "fair"
            else:
                quality_level = "poor"
            
            return {
                "overall_score": round(overall_score, 1),
                "quality_level": quality_level,
                "component_scores": {
                    "base_confidence": round(base_confidence, 3),
                    "text_quantity": round(text_quantity_score, 3),
                    "layout_quality": round(layout_score, 3),
                    "coverage": round(coverage_score, 3)
                },
                "detailed_metrics": {
                    "word_count": word_count,
                    "detected_blocks": detected_blocks,
                    "ocr_method": ocr_result.get("method", "unknown")
                }
            }
            
        except Exception as e:
            logging.error(f"OCR 정확도 평가 오류: {e}")
            return {"error": str(e)}
    
    def _generate_ppt_recommendations(self, 
                                    slide_detection: Dict,
                                    accuracy_assessment: Dict, 
                                    layout_result: Dict,
                                    original_image: Image.Image) -> List[Dict]:
        """PPT OCR 개선 제안 생성"""
        recommendations = []
        
        try:
            # 슬라이드 감지 기반 제안
            if not slide_detection.get("is_presentation", False):
                recommendations.append({
                    "type": "slide_detection",
                    "priority": "medium",
                    "issue": "PPT 슬라이드로 인식되지 않았습니다",
                    "solution": "화면 전체가 포함되도록 정면에서 촬영하세요",
                    "icon": "📊"
                })
            
            # 해상도 관련 제안
            width, height = original_image.size
            resolution_category = slide_detection.get("image_characteristics", {}).get("resolution_category", "low")
            
            if resolution_category == "low":
                recommendations.append({
                    "type": "resolution",
                    "priority": "high",
                    "issue": f"해상도가 낮습니다 ({width}x{height})",
                    "solution": "더 가까이에서 촬영하거나 더 높은 해상도로 설정하세요",
                    "icon": "📷"
                })
            
            # OCR 정확도 기반 제안
            overall_score = accuracy_assessment.get("overall_score", 0)
            
            if overall_score < 70:
                base_confidence = accuracy_assessment.get("component_scores", {}).get("base_confidence", 0)
                
                if base_confidence < 0.7:
                    recommendations.append({
                        "type": "ocr_accuracy",
                        "priority": "high",
                        "issue": f"텍스트 인식률이 낮습니다 ({base_confidence:.2f})",
                        "solution": "조명을 충분히 확보하고 화면에 그림자가 생기지 않도록 촬영하세요",
                        "icon": "💡"
                    })
            
            # 레이아웃 관련 제안
            layout_score = layout_result.get("layout_score", 0)
            
            if layout_score < 0.6:
                recommendations.append({
                    "type": "layout",
                    "priority": "medium",
                    "issue": "슬라이드 레이아웃 구조 인식이 어렵습니다",
                    "solution": "화면을 정면에서 수직으로 촬영하여 왜곡을 최소화하세요",
                    "icon": "📐"
                })
            
            # 표/차트 관련 제안
            tables_charts = layout_result.get("tables_and_charts", {})
            if tables_charts.get("table_detected") or tables_charts.get("pie_chart_detected"):
                recommendations.append({
                    "type": "tables_charts",
                    "priority": "medium",
                    "issue": "표나 차트가 감지되었습니다",
                    "solution": "표와 차트의 텍스트가 선명하게 보이도록 더 가까이에서 촬영하세요",
                    "icon": "📈"
                })
            
            # 일반적인 촬영 팁
            if len(recommendations) >= 2:
                recommendations.append({
                    "type": "general",
                    "priority": "low",
                    "issue": "전반적인 촬영 품질 개선이 필요합니다",
                    "solution": "삼각대 사용, 충분한 조명, 화면 정면 촬영을 권장합니다",
                    "icon": "🎯"
                })
            
            # 성공적인 경우의 격려
            if overall_score >= 85:
                recommendations.append({
                    "type": "success",
                    "priority": "info",
                    "issue": "우수한 PPT 화면 인식 결과입니다",
                    "solution": "현재와 같은 촬영 방식을 유지하세요",
                    "icon": "✅"
                })
            
            return recommendations[:5]  # 최대 5개 제안
            
        except Exception as e:
            logging.error(f"PPT 개선 제안 생성 오류: {e}")
            return [{
                "type": "error",
                "priority": "low",
                "issue": "개선 제안 생성 중 오류 발생",
                "solution": "수동으로 PPT 화면 품질을 확인해 주세요",
                "icon": "⚠️"
            }]

# 전역 인스턴스
_presentation_ocr_analyzer_instance = None

def get_presentation_ocr_analyzer() -> PresentationOCRAnalyzer:
    """전역 PPT OCR 분석기 인스턴스 반환"""
    global _presentation_ocr_analyzer_instance
    if _presentation_ocr_analyzer_instance is None:
        _presentation_ocr_analyzer_instance = PresentationOCRAnalyzer()
    return _presentation_ocr_analyzer_instance

# 편의 함수들
async def analyze_presentation_ocr(image_data: bytes, filename: str, **kwargs) -> Dict:
    """PPT 화면 OCR 분석 편의 함수"""
    analyzer = get_presentation_ocr_analyzer()
    return await analyzer.analyze_presentation_image(image_data, filename, **kwargs)

def check_ppt_ocr_support() -> Dict:
    """PPT OCR 분석기 지원 상태 확인"""
    return {
        "libraries": {
            "PIL": PIL_AVAILABLE,
            "tesseract": TESSERACT_AVAILABLE,
            "easyocr": EASYOCR_AVAILABLE,
            "opencv": True,  # cv2는 기본 설치
            "skimage": SKIMAGE_AVAILABLE
        },
        "features": {
            "slide_detection": True,
            "layout_analysis": True,
            "table_chart_detection": True,
            "text_region_analysis": True,
            "quality_assessment": True
        },
        "layout_patterns": list(PresentationOCRAnalyzer().layout_patterns.keys()),
        "quality_levels": ["excellent", "good", "fair", "poor"]
    }

if __name__ == "__main__":
    # 테스트 코드
    async def test_ppt_analyzer():
        print("PPT 화면 특화 OCR 분석 엔진 테스트")
        support_info = check_ppt_ocr_support()
        print(f"지원 상태: {support_info}")
    
    import asyncio
    import io
    asyncio.run(test_ppt_analyzer())
