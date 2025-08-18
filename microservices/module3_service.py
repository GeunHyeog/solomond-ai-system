#!/usr/bin/env python3
"""
💎 Module 3: 보석 분석 마이크로서비스
- 보석 이미지 분석 및 산지 판정 기능을 FastAPI로 변환
- 스마트 메모리 매니저와 완전 통합
- 포트 8003에서 실행
"""

import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import tempfile
import os

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io

# 최적화된 컴포넌트들 import
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from core.optimized_ai_loader import optimized_loader
from core.smart_memory_manager import get_memory_stats
from core.robust_file_processor import robust_processor
from config.security_config import get_system_config

logger = logging.getLogger(__name__)

# Pydantic 모델들
class GemstoneAnalysisRequest(BaseModel):
    """보석 분석 요청 모델"""
    analysis_type: str = "comprehensive"  # 'quick', 'comprehensive', 'detailed'
    color_analysis: bool = True
    clarity_analysis: bool = True
    size_estimation: bool = True
    origin_prediction: bool = True

class GemstoneResult(BaseModel):
    """보석 분석 결과 모델"""
    session_id: str
    status: str
    results: Dict[str, Any]
    processing_time: float
    timestamp: datetime

class GemstoneAnalysisService:
    """보석 분석 서비스"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}
        self.analysis_cache: Dict[str, Any] = {}
        
        # 보석 종류별 색상 기준값 (HSV)
        self.gemstone_colors = {
            'ruby': {'h_range': (0, 10), 'description': '루비 - 붉은색'},
            'emerald': {'h_range': (60, 80), 'description': '에메랄드 - 녹색'},
            'sapphire_blue': {'h_range': (100, 130), 'description': '사파이어 - 파란색'},
            'sapphire_yellow': {'h_range': (20, 40), 'description': '사파이어 - 노란색'},
            'diamond': {'h_range': (0, 360), 'description': '다이아몬드 - 무색/다양'},
            'amethyst': {'h_range': (270, 300), 'description': '자수정 - 보라색'},
            'topaz': {'h_range': (40, 60), 'description': '토파즈 - 황금색'},
            'aquamarine': {'h_range': (180, 200), 'description': '아쿠아마린 - 하늘색'}
        }
        
        # 산지별 특성 데이터베이스 (간단화)
        self.origin_database = {
            'myanmar': {'rubies': 0.8, 'sapphires': 0.3, 'description': '미얀마 - 루비의 명산지'},
            'colombia': {'emeralds': 0.9, 'description': '콜롬비아 - 에메랄드의 최고 산지'},
            'kashmir': {'sapphires': 0.9, 'description': '카시미르 - 최고급 사파이어'},
            'sri_lanka': {'sapphires': 0.7, 'rubies': 0.6, 'description': '스리랑카 - 다양한 보석'},
            'thailand': {'rubies': 0.6, 'sapphires': 0.5, 'description': '태국 - 루비/사파이어'},
            'madagascar': {'sapphires': 0.7, 'emeralds': 0.4, 'description': '마다가스카르'},
            'brazil': {'emeralds': 0.6, 'topaz': 0.8, 'description': '브라질 - 에메랄드/토파즈'},
            'afghanistan': {'emeralds': 0.7, 'description': '아프가니스탄 - 에메랄드'},
            'australia': {'sapphires': 0.6, 'description': '호주 - 사파이어'}
        }
        
    async def analyze_gemstone(self, file_data: bytes, filename: str,
                             analysis_type: str = "comprehensive",
                             color_analysis: bool = True,
                             clarity_analysis: bool = True,
                             size_estimation: bool = True,
                             origin_prediction: bool = True) -> Dict[str, Any]:
        """보석 이미지 분석 실행"""
        session_id = str(uuid.uuid4())
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 세션 정보 저장
            self.active_sessions[session_id] = {
                'start_time': start_time,
                'status': 'processing',
                'filename': filename,
                'analysis_type': analysis_type
            }
            
            # 이미지 전처리
            file_info = await robust_processor.process_file(file_data, filename)
            
            if file_info.conversion_path is None:
                raise Exception("이미지 파일 전처리 실패")
            
            # 이미지 로드
            image = cv2.imread(file_info.conversion_path)
            if image is None:
                raise Exception("이미지를 읽을 수 없습니다")
            
            results = {
                'session_id': session_id,
                'filename': filename,
                'file_info': {
                    'size_bytes': file_info.original_size,
                    'format': file_info.format
                },
                'image_properties': {},
                'color_analysis': {},
                'clarity_analysis': {},
                'size_estimation': {},
                'gemstone_identification': {},
                'origin_prediction': {},
                'quality_assessment': {}
            }
            
            # 기본 이미지 속성 분석
            results['image_properties'] = self._analyze_image_properties(image)
            
            # 색상 분석
            if color_analysis:
                results['color_analysis'] = await self._analyze_color(image)
            
            # 투명도/선명도 분석
            if clarity_analysis:
                results['clarity_analysis'] = await self._analyze_clarity(image)
            
            # 크기 추정
            if size_estimation:
                results['size_estimation'] = await self._estimate_size(image)
            
            # 보석 식별
            results['gemstone_identification'] = await self._identify_gemstone(
                image, results.get('color_analysis', {})
            )
            
            # 산지 예측
            if origin_prediction:
                results['origin_prediction'] = await self._predict_origin(
                    results['gemstone_identification'], 
                    results.get('color_analysis', {}),
                    results.get('clarity_analysis', {})
                )
            
            # 품질 평가
            results['quality_assessment'] = await self._assess_quality(results)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # 세션 완료
            self.active_sessions[session_id]['status'] = 'completed'
            self.active_sessions[session_id]['processing_time'] = processing_time
            
            # 임시 파일 정리
            robust_processor.cleanup_temp_files(file_info)
            
            return {
                'session_id': session_id,
                'status': 'success',
                'results': results,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"보석 분석 실패 {session_id}: {e}")
            
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = 'failed'
                self.active_sessions[session_id]['error'] = str(e)
            
            raise HTTPException(status_code=500, detail=f"보석 분석 실패: {str(e)}")
    
    def _analyze_image_properties(self, image: np.ndarray) -> Dict[str, Any]:
        """기본 이미지 속성 분석"""
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        return {
            'dimensions': {'width': int(width), 'height': int(height)},
            'channels': int(channels),
            'total_pixels': int(width * height),
            'aspect_ratio': round(width / height, 2)
        }
    
    async def _analyze_color(self, image: np.ndarray) -> Dict[str, Any]:
        """색상 분석"""
        try:
            # BGR을 HSV로 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 색상 히스토그램
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            # 주요 색상 추출
            dominant_h = np.argmax(hist_h)
            dominant_s = np.argmax(hist_s)
            dominant_v = np.argmax(hist_v)
            
            # RGB 평균값
            mean_color_bgr = np.mean(image, axis=(0, 1))
            mean_color_rgb = mean_color_bgr[::-1]  # BGR to RGB
            
            return {
                'dominant_hue': int(dominant_h),
                'dominant_saturation': int(dominant_s),
                'dominant_value': int(dominant_v),
                'mean_rgb': [int(c) for c in mean_color_rgb],
                'color_description': self._describe_color(dominant_h, dominant_s, dominant_v),
                'saturation_level': self._categorize_saturation(dominant_s),
                'brightness_level': self._categorize_brightness(dominant_v)
            }
            
        except Exception as e:
            logger.error(f"색상 분석 실패: {e}")
            return {'error': str(e)}
    
    def _describe_color(self, h: int, s: int, v: int) -> str:
        """색상 설명 생성"""
        if s < 30:
            return "무색/회색"
        elif 0 <= h < 10 or 170 <= h < 180:
            return "빨간색"
        elif 10 <= h < 25:
            return "주황색"
        elif 25 <= h < 35:
            return "노란색"
        elif 35 <= h < 85:
            return "녹색"
        elif 85 <= h < 130:
            return "파란색"
        elif 130 <= h < 170:
            return "보라색"
        else:
            return "혼합색"
    
    def _categorize_saturation(self, s: int) -> str:
        """채도 분류"""
        if s < 50:
            return "낮은 채도"
        elif s < 150:
            return "중간 채도"
        else:
            return "높은 채도"
    
    def _categorize_brightness(self, v: int) -> str:
        """밝기 분류"""
        if v < 85:
            return "어두움"
        elif v < 170:
            return "중간 밝기"
        else:
            return "밝음"
    
    async def _analyze_clarity(self, image: np.ndarray) -> Dict[str, Any]:
        """투명도/선명도 분석"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 라플라시안으로 선명도 측정
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 가장자리 검출
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 명암 대비
            contrast = gray.std()
            
            return {
                'sharpness_score': float(laplacian_var),
                'edge_density': float(edge_density),
                'contrast_score': float(contrast),
                'clarity_grade': self._grade_clarity(laplacian_var, edge_density, contrast),
                'transparency_estimate': self._estimate_transparency(image)
            }
            
        except Exception as e:
            logger.error(f"선명도 분석 실패: {e}")
            return {'error': str(e)}
    
    def _grade_clarity(self, sharpness: float, edge_density: float, contrast: float) -> str:
        """선명도 등급 부여"""
        combined_score = (sharpness / 1000 + edge_density * 100 + contrast / 10) / 3
        
        if combined_score > 15:
            return "매우 우수 (FL-IF)"
        elif combined_score > 10:
            return "우수 (VVS1-VVS2)"
        elif combined_score > 7:
            return "양호 (VS1-VS2)"
        elif combined_score > 5:
            return "보통 (SI1-SI2)"
        else:
            return "낮음 (I1-I3)"
    
    def _estimate_transparency(self, image: np.ndarray) -> Dict[str, Any]:
        """투명도 추정"""
        # 간단한 투명도 추정 (밝은 영역의 비율로)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_pixels = np.sum(gray > 200)
        total_pixels = gray.size
        transparency_ratio = bright_pixels / total_pixels
        
        return {
            'transparency_ratio': float(transparency_ratio),
            'transparency_level': 'high' if transparency_ratio > 0.3 else 'medium' if transparency_ratio > 0.1 else 'low'
        }
    
    async def _estimate_size(self, image: np.ndarray) -> Dict[str, Any]:
        """크기 추정 (픽셀 기반)"""
        try:
            height, width = image.shape[:2]
            
            # 간단한 크기 추정 (실제로는 참조 객체가 필요)
            estimated_mm = {
                'width_mm': round(width * 0.1, 1),  # 가정: 1픽셀 = 0.1mm
                'height_mm': round(height * 0.1, 1),
                'area_mm2': round(width * height * 0.01, 1)
            }
            
            # 캐럿 추정 (매우 대략적)
            area_mm2 = estimated_mm['area_mm2']
            estimated_carat = area_mm2 / 20  # 대략적 추정
            
            return {
                'pixel_dimensions': {'width': int(width), 'height': int(height)},
                'estimated_size_mm': estimated_mm,
                'estimated_carat': round(estimated_carat, 2),
                'size_category': self._categorize_size(estimated_carat),
                'note': '정확한 크기 측정을 위해서는 참조 객체가 필요합니다'
            }
            
        except Exception as e:
            logger.error(f"크기 추정 실패: {e}")
            return {'error': str(e)}
    
    def _categorize_size(self, carat: float) -> str:
        """크기 분류"""
        if carat < 0.5:
            return "소형 (0.5ct 미만)"
        elif carat < 1.0:
            return "중소형 (0.5-1ct)"
        elif carat < 2.0:
            return "중형 (1-2ct)"
        elif carat < 5.0:
            return "대형 (2-5ct)"
        else:
            return "특대형 (5ct 초과)"
    
    async def _identify_gemstone(self, image: np.ndarray, color_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """보석 종류 식별"""
        try:
            if not color_analysis:
                return {'error': '색상 분석 데이터가 필요합니다'}
            
            dominant_h = color_analysis.get('dominant_hue', 0)
            saturation = color_analysis.get('dominant_saturation', 0)
            
            # 색상 기반 보석 추정
            possible_gemstones = []
            
            for gemstone, color_data in self.gemstone_colors.items():
                h_range = color_data['h_range']
                
                if gemstone == 'diamond':
                    # 다이아몬드는 채도가 낮음
                    if saturation < 50:
                        possible_gemstones.append({
                            'type': gemstone,
                            'confidence': 0.7,
                            'description': color_data['description']
                        })
                else:
                    # 색상 범위 확인
                    if h_range[0] <= dominant_h <= h_range[1]:
                        confidence = 0.8 if saturation > 100 else 0.6
                        possible_gemstones.append({
                            'type': gemstone,
                            'confidence': confidence,
                            'description': color_data['description']
                        })
            
            # 신뢰도 순으로 정렬
            possible_gemstones.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'primary_identification': possible_gemstones[0] if possible_gemstones else {
                    'type': 'unknown',
                    'confidence': 0.1,
                    'description': '알 수 없는 보석'
                },
                'alternative_identifications': possible_gemstones[1:3],
                'total_candidates': len(possible_gemstones)
            }
            
        except Exception as e:
            logger.error(f"보석 식별 실패: {e}")
            return {'error': str(e)}
    
    async def _predict_origin(self, gemstone_id: Dict[str, Any], 
                            color_analysis: Dict[str, Any], 
                            clarity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """산지 예측"""
        try:
            primary_gem = gemstone_id.get('primary_identification', {})
            gem_type = primary_gem.get('type', 'unknown')
            
            if gem_type == 'unknown':
                return {'message': '보석 종류를 먼저 식별해야 합니다'}
            
            # 보석별 주요 산지 추천
            origin_predictions = []
            
            for origin, data in self.origin_database.items():
                # 해당 보석에 대한 산지의 점수가 있는 경우
                gem_key = f"{gem_type}s" if not gem_type.endswith('s') else gem_type
                
                if gem_key in data:
                    base_score = data[gem_key]
                    
                    # 품질 기반 보정
                    quality_multiplier = 1.0
                    if clarity_analysis.get('clarity_grade', '').startswith('매우 우수'):
                        quality_multiplier = 1.2
                    elif clarity_analysis.get('clarity_grade', '').startswith('우수'):
                        quality_multiplier = 1.1
                    
                    final_score = base_score * quality_multiplier
                    
                    origin_predictions.append({
                        'origin': origin,
                        'probability': min(final_score, 0.95),
                        'description': data['description'],
                        'confidence_level': 'high' if final_score > 0.8 else 'medium' if final_score > 0.6 else 'low'
                    })
            
            # 확률 순으로 정렬
            origin_predictions.sort(key=lambda x: x['probability'], reverse=True)
            
            return {
                'primary_origin': origin_predictions[0] if origin_predictions else {
                    'origin': 'unknown',
                    'probability': 0.1,
                    'description': '산지 추정 불가'
                },
                'alternative_origins': origin_predictions[1:3],
                'total_candidates': len(origin_predictions),
                'note': '산지 예측은 추정치이며, 정확한 판단을 위해서는 전문가 감정이 필요합니다'
            }
            
        except Exception as e:
            logger.error(f"산지 예측 실패: {e}")
            return {'error': str(e)}
    
    async def _assess_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """전체 품질 평가"""
        try:
            # 각 분석 결과에서 점수 추출
            color_score = self._score_color(results.get('color_analysis', {}))
            clarity_score = self._score_clarity(results.get('clarity_analysis', {}))
            size_score = self._score_size(results.get('size_estimation', {}))
            
            # 가중 평균
            overall_score = (color_score * 0.4 + clarity_score * 0.4 + size_score * 0.2)
            
            return {
                'color_score': color_score,
                'clarity_score': clarity_score,
                'size_score': size_score,
                'overall_score': round(overall_score, 1),
                'grade': self._assign_grade(overall_score),
                'market_value_estimate': self._estimate_value(overall_score, results),
                'recommendations': self._generate_recommendations(results)
            }
            
        except Exception as e:
            logger.error(f"품질 평가 실패: {e}")
            return {'error': str(e)}
    
    def _score_color(self, color_analysis: Dict[str, Any]) -> float:
        """색상 점수 계산"""
        if not color_analysis or 'error' in color_analysis:
            return 5.0
        
        saturation = color_analysis.get('dominant_saturation', 0)
        # 높은 채도일수록 높은 점수 (보석의 경우)
        return min(10.0, saturation / 25.5)
    
    def _score_clarity(self, clarity_analysis: Dict[str, Any]) -> float:
        """선명도 점수 계산"""
        if not clarity_analysis or 'error' in clarity_analysis:
            return 5.0
        
        clarity_grade = clarity_analysis.get('clarity_grade', '')
        if '매우 우수' in clarity_grade:
            return 10.0
        elif '우수' in clarity_grade:
            return 8.5
        elif '양호' in clarity_grade:
            return 7.0
        elif '보통' in clarity_grade:
            return 5.5
        else:
            return 3.0
    
    def _score_size(self, size_estimation: Dict[str, Any]) -> float:
        """크기 점수 계산"""
        if not size_estimation or 'error' in size_estimation:
            return 5.0
        
        carat = size_estimation.get('estimated_carat', 0)
        # 크기에 따른 점수 (로그 스케일)
        import math
        return min(10.0, math.log(max(carat, 0.1)) + 7)
    
    def _assign_grade(self, overall_score: float) -> str:
        """전체 등급 부여"""
        if overall_score >= 9.0:
            return "A+ (최고급)"
        elif overall_score >= 8.0:
            return "A (고급)"
        elif overall_score >= 7.0:
            return "B+ (상급)"
        elif overall_score >= 6.0:
            return "B (중급)"
        elif overall_score >= 5.0:
            return "C+ (보통)"
        else:
            return "C (낮음)"
    
    def _estimate_value(self, overall_score: float, results: Dict[str, Any]) -> Dict[str, Any]:
        """시장가치 추정 (매우 대략적)"""
        base_value = 100  # 기본값 (USD)
        
        # 점수에 따른 승수
        score_multiplier = max(0.1, overall_score / 5)
        
        # 크기에 따른 승수
        size_info = results.get('size_estimation', {})
        carat = size_info.get('estimated_carat', 0.5)
        size_multiplier = max(1.0, carat ** 1.5)
        
        estimated_value = base_value * score_multiplier * size_multiplier
        
        return {
            'estimated_value_usd': round(estimated_value, 0),
            'value_range': {
                'min_usd': round(estimated_value * 0.7, 0),
                'max_usd': round(estimated_value * 1.3, 0)
            },
            'disclaimer': '이 추정가는 대략적인 참고값이며, 실제 시장가치는 전문 감정사의 감정을 받으시기 바랍니다.'
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        clarity_score = results.get('quality_assessment', {}).get('clarity_score', 5)
        if clarity_score < 6:
            recommendations.append("선명도 개선을 위해 전문적인 연마를 고려해보세요")
        
        color_score = results.get('quality_assessment', {}).get('color_score', 5)
        if color_score > 8:
            recommendations.append("우수한 색상을 가지고 있어 보석의 가치가 높습니다")
        
        size_info = results.get('size_estimation', {})
        if size_info.get('estimated_carat', 0) > 2:
            recommendations.append("대형 보석으로 희소가치가 높습니다")
        
        recommendations.append("정확한 감정을 위해 공인 감정기관의 감정서를 받아보세요")
        recommendations.append("적절한 보관과 관리로 보석의 가치를 유지하세요")
        
        return recommendations

# FastAPI 앱 생성
app = FastAPI(
    title="Module 3: 보석 분석 서비스",
    description="보석 이미지 분석 및 산지 판정",
    version="4.0.0"
)

# 서비스 인스턴스
service = GemstoneAnalysisService()

@app.get("/health")
async def health_check():
    """헬스체크"""
    memory_stats = get_memory_stats()
    
    return {
        "status": "healthy",
        "service": "module3_gemstone",
        "version": "4.0.0",
        "memory": {
            "memory_percent": memory_stats.get('memory_info', {}).get('percent', 0)
        },
        "analysis": {
            "active_sessions": len(service.active_sessions),
            "supported_gemstones": len(service.gemstone_colors),
            "origin_database": len(service.origin_database)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze", response_model=GemstoneResult)
async def analyze_gemstone(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    analysis_type: str = "comprehensive",
    color_analysis: bool = True,
    clarity_analysis: bool = True,
    size_estimation: bool = True,
    origin_prediction: bool = True
):
    """보석 분석 API"""
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
    
    # 파일 읽기
    file_data = await file.read()
    if len(file_data) == 0:
        raise HTTPException(status_code=400, detail="빈 파일입니다")
    
    # 분석 실행
    result = await service.analyze_gemstone(
        file_data, file.filename,
        analysis_type, color_analysis, clarity_analysis, 
        size_estimation, origin_prediction
    )
    
    return result

@app.get("/gemstone-types")
async def get_gemstone_types():
    """지원되는 보석 종류 조회"""
    return {
        "supported_gemstones": service.gemstone_colors,
        "total_types": len(service.gemstone_colors),
        "analysis_features": [
            "색상 분석",
            "투명도/선명도 분석",
            "크기 추정",
            "보석 종류 식별",
            "산지 예측",
            "품질 평가"
        ]
    }

@app.get("/origins")
async def get_origin_database():
    """산지 데이터베이스 조회"""
    return {
        "origin_database": service.origin_database,
        "total_origins": len(service.origin_database),
        "note": "산지 예측은 색상, 선명도 등의 특성을 바탕으로 한 추정값입니다"
    }

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """세션 정보 조회"""
    if session_id not in service.active_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    return service.active_sessions[session_id]

@app.get("/sessions")
async def list_sessions():
    """전체 세션 목록"""
    return {
        "total_sessions": len(service.active_sessions),
        "sessions": service.active_sessions
    }

@app.get("/stats")
async def get_service_stats():
    """서비스 통계"""
    memory_stats = get_memory_stats()
    
    return {
        "service_info": {
            "name": "module3_gemstone",
            "version": "4.0.0",
            "uptime": "실행 중"
        },
        "memory": memory_stats,
        "sessions": {
            "total": len(service.active_sessions),
            "active": len([s for s in service.active_sessions.values() 
                          if s['status'] == 'processing']),
            "completed": len([s for s in service.active_sessions.values() 
                             if s['status'] == 'completed'])
        },
        "analysis_capabilities": {
            "gemstone_types": len(service.gemstone_colors),
            "origin_regions": len(service.origin_database),
            "analysis_features": 6
        }
    }

if __name__ == "__main__":
    logger.info("💎 Module 3 보석 분석 서비스 시작: http://localhost:8003")
    
    uvicorn.run(
        "module3_service:app",
        host="0.0.0.0", 
        port=8003,
        reload=True,
        log_level="info"
    )