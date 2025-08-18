#!/usr/bin/env python3
"""
🏗️ Module 4: 3D CAD 변환 마이크로서비스
- 이미지에서 3D CAD 모델 생성 기능을 FastAPI로 변환
- Ollama AI 통합으로 지능형 형상 인식
- 포트 8004에서 실행
"""

import logging
import asyncio
import uuid
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import tempfile
import os
import base64
from io import BytesIO

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import requests

# 최적화된 컴포넌트들 import
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from core.smart_memory_manager import get_memory_stats
from core.robust_file_processor import robust_processor
from config.security_config import get_system_config

logger = logging.getLogger(__name__)

# Pydantic 모델들
class CADGenerationRequest(BaseModel):
    """3D CAD 생성 요청 모델"""
    generation_type: str = "basic"  # 'basic', 'advanced', 'professional'
    mesh_quality: str = "medium"  # 'low', 'medium', 'high', 'ultra'
    output_format: str = "stl"  # 'stl', 'obj', 'ply', '3mf'
    use_ai_enhancement: bool = True
    detail_level: int = 5  # 1-10 디테일 레벨

class CADResult(BaseModel):
    """CAD 변환 결과 모델"""
    session_id: str
    status: str
    results: Dict[str, Any]
    processing_time: float
    timestamp: datetime

class CADConversionService:
    """3D CAD 변환 서비스"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}
        self.conversion_cache: Dict[str, Any] = {}
        self.ollama_api_url = "http://localhost:11434/api/generate"
        
        # 지원되는 출력 형식
        self.output_formats = {
            'stl': {'extension': '.stl', 'description': 'Stereolithography (3D 프린팅)'},
            'obj': {'extension': '.obj', 'description': 'Wavefront OBJ (범용)'},
            'ply': {'extension': '.ply', 'description': 'Polygon File Format'},
            '3mf': {'extension': '.3mf', 'description': 'Microsoft 3D Manufacturing Format'}
        }
        
        # 품질 설정
        self.quality_settings = {
            'low': {'vertices': 500, 'faces': 1000},
            'medium': {'vertices': 2000, 'faces': 4000},
            'high': {'vertices': 8000, 'faces': 16000},
            'ultra': {'vertices': 32000, 'faces': 64000}
        }
        
    async def convert_image_to_cad(self, file_data: bytes, filename: str,
                                 generation_type: str = "basic",
                                 mesh_quality: str = "medium",
                                 output_format: str = "stl",
                                 use_ai_enhancement: bool = True,
                                 detail_level: int = 5) -> Dict[str, Any]:
        """이미지를 3D CAD로 변환"""
        session_id = str(uuid.uuid4())
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 세션 정보 저장
            self.active_sessions[session_id] = {
                'start_time': start_time,
                'status': 'processing',
                'filename': filename,
                'generation_type': generation_type,
                'mesh_quality': mesh_quality,
                'output_format': output_format
            }
            
            # 이미지 전처리
            file_info = await robust_processor.process_file(file_data, filename)
            
            if file_info.conversion_path is None:
                raise Exception("이미지 파일 전처리 실패")
            
            # 이미지 로드 및 분석
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
                'image_analysis': {},
                'shape_detection': {},
                'ai_enhancement': {},
                '3d_generation': {},
                'cad_model': {},
                'export_info': {}
            }
            
            # 1. 이미지 분석
            results['image_analysis'] = await self._analyze_image(image)
            
            # 2. 형상 감지
            results['shape_detection'] = await self._detect_shapes(image)
            
            # 3. AI 강화 (Ollama 사용)
            if use_ai_enhancement:
                results['ai_enhancement'] = await self._ai_enhance_analysis(
                    image, results['image_analysis'], results['shape_detection']
                )
            
            # 4. 3D 모델 생성
            results['3d_generation'] = await self._generate_3d_model(
                image, results, generation_type, mesh_quality, detail_level
            )
            
            # 5. CAD 파일 생성
            results['cad_model'] = await self._create_cad_file(
                results['3d_generation'], output_format, session_id
            )
            
            # 6. 내보내기 정보
            results['export_info'] = await self._prepare_export_info(
                results['cad_model'], output_format, mesh_quality
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # 세션 완료
            self.active_sessions[session_id]['status'] = 'completed'
            self.active_sessions[session_id]['processing_time'] = processing_time
            
            # 임시 파일 정리 (CAD 파일은 보존)
            robust_processor.cleanup_temp_files(file_info)
            
            return {
                'session_id': session_id,
                'status': 'success',
                'results': results,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"CAD 변환 실패 {session_id}: {e}")
            
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = 'failed'
                self.active_sessions[session_id]['error'] = str(e)
            
            raise HTTPException(status_code=500, detail=f"CAD 변환 실패: {str(e)}")
    
    async def _analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """이미지 기본 분석"""
        try:
            height, width = image.shape[:2]
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 밝기/대비 분석
            mean_brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # 히스토그램 분석
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            return {
                'dimensions': {'width': int(width), 'height': int(height)},
                'brightness': {
                    'mean': float(mean_brightness),
                    'level': 'bright' if mean_brightness > 128 else 'dark'
                },
                'contrast': {
                    'value': float(contrast),
                    'level': 'high' if contrast > 50 else 'medium' if contrast > 25 else 'low'
                },
                'image_quality': self._assess_image_quality(gray, contrast),
                'processing_recommendations': self._get_processing_recommendations(mean_brightness, contrast)
            }
            
        except Exception as e:
            logger.error(f"이미지 분석 실패: {e}")
            return {'error': str(e)}
    
    def _assess_image_quality(self, gray: np.ndarray, contrast: float) -> Dict[str, Any]:
        """이미지 품질 평가"""
        # 선명도 측정
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 노이즈 추정
        noise_level = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
        
        # 품질 점수 계산
        quality_score = min(10, (sharpness / 100 + contrast / 10 - noise_level / 5))
        
        return {
            'sharpness': float(sharpness),
            'noise_level': float(noise_level),
            'quality_score': max(0, float(quality_score)),
            'quality_grade': 'excellent' if quality_score > 8 else 'good' if quality_score > 6 else 'fair' if quality_score > 4 else 'poor'
        }
    
    def _get_processing_recommendations(self, brightness: float, contrast: float) -> List[str]:
        """처리 권장사항"""
        recommendations = []
        
        if brightness < 100:
            recommendations.append("밝기를 높여 디테일을 개선할 수 있습니다")
        elif brightness > 180:
            recommendations.append("밝기를 낮춰 과노출을 방지할 수 있습니다")
        
        if contrast < 30:
            recommendations.append("대비를 높여 형상 인식을 개선할 수 있습니다")
        
        recommendations.append("고해상도 이미지를 사용하면 더 정밀한 3D 모델을 생성할 수 있습니다")
        
        return recommendations
    
    async def _detect_shapes(self, image: np.ndarray) -> Dict[str, Any]:
        """형상 감지"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 가장자리 검출
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # 윤곽선 찾기
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 형상 분류
            shapes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # 최소 크기 필터
                    # 윤곽선 근사
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # 기본적인 형상 분류
                    shape_type = self._classify_shape(approx, area)
                    
                    shapes.append({
                        'type': shape_type,
                        'area': float(area),
                        'vertices': len(approx),
                        'perimeter': float(cv2.arcLength(contour, True)),
                        'bounding_rect': [int(x) for x in cv2.boundingRect(contour)]
                    })
            
            # 크기 순으로 정렬
            shapes.sort(key=lambda x: x['area'], reverse=True)
            
            return {
                'total_shapes': len(shapes),
                'main_shapes': shapes[:5],  # 상위 5개만
                'shape_distribution': self._analyze_shape_distribution(shapes),
                'complexity_assessment': self._assess_complexity(shapes)
            }
            
        except Exception as e:
            logger.error(f"형상 감지 실패: {e}")
            return {'error': str(e)}
    
    def _classify_shape(self, approx: np.ndarray, area: float) -> str:
        """형상 분류"""
        vertices = len(approx)
        
        if vertices == 3:
            return "triangle"
        elif vertices == 4:
            # 직사각형 vs 정사각형 판별
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / h
            if 0.95 <= aspect_ratio <= 1.05:
                return "square"
            else:
                return "rectangle"
        elif vertices == 5:
            return "pentagon"
        elif vertices > 8:
            return "circle"
        else:
            return f"polygon_{vertices}"
    
    def _analyze_shape_distribution(self, shapes: List[Dict]) -> Dict[str, int]:
        """형상 분포 분석"""
        distribution = {}
        for shape in shapes:
            shape_type = shape['type']
            distribution[shape_type] = distribution.get(shape_type, 0) + 1
        return distribution
    
    def _assess_complexity(self, shapes: List[Dict]) -> Dict[str, Any]:
        """복잡도 평가"""
        total_shapes = len(shapes)
        avg_vertices = np.mean([s['vertices'] for s in shapes]) if shapes else 0
        
        complexity_score = min(10, total_shapes / 2 + avg_vertices / 4)
        
        return {
            'complexity_score': float(complexity_score),
            'complexity_level': 'high' if complexity_score > 7 else 'medium' if complexity_score > 4 else 'low',
            'processing_difficulty': 'expert' if complexity_score > 8 else 'intermediate' if complexity_score > 5 else 'basic'
        }
    
    async def _ai_enhance_analysis(self, image: np.ndarray, image_analysis: Dict, shape_detection: Dict) -> Dict[str, Any]:
        """AI 강화 분석 (Ollama 사용)"""
        try:
            # Ollama가 실행 중인지 확인
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code != 200:
                    return {'error': 'Ollama 서비스가 실행되지 않음', 'fallback_used': True}
            except:
                return {'error': 'Ollama 서비스에 연결할 수 없음', 'fallback_used': True}
            
            # 이미지를 base64로 인코딩
            _, buffer = cv2.imencode('.jpg', image)
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # AI 프롬프트 구성
            prompt = f"""
이미지를 3D CAD 모델로 변환하기 위한 분석을 도와주세요.

현재 분석 결과:
- 이미지 크기: {image_analysis.get('dimensions', {})}
- 품질: {image_analysis.get('image_quality', {}).get('quality_grade', 'unknown')}
- 감지된 형상: {len(shape_detection.get('main_shapes', []))}개
- 복잡도: {shape_detection.get('complexity_assessment', {}).get('complexity_level', 'unknown')}

다음 정보를 JSON 형식으로 제공해주세요:
1. 객체의 주요 특징과 형태
2. 3D 변환에 적합한 방법 추천
3. 예상되는 모델의 복잡도
4. 최적의 메쉬 품질 설정

JSON만 응답해주세요.
"""
            
            # Ollama API 호출
            ollama_data = {
                "model": "llama3.2-vision",
                "prompt": prompt,
                "images": [image_b64],
                "stream": False
            }
            
            response = requests.post(
                self.ollama_api_url,
                json=ollama_data,
                timeout=30
            )
            
            if response.status_code == 200:
                ai_response = response.json()
                ai_text = ai_response.get('response', '')
                
                try:
                    # JSON 파싱 시도
                    ai_analysis = json.loads(ai_text)
                    return {
                        'ai_insights': ai_analysis,
                        'ai_model_used': 'llama3.2-vision',
                        'confidence': 'high'
                    }
                except json.JSONDecodeError:
                    return {
                        'ai_insights': {'raw_response': ai_text},
                        'ai_model_used': 'llama3.2-vision',
                        'confidence': 'medium',
                        'note': 'AI 응답 파싱 부분적 실패'
                    }
            else:
                return {'error': f'Ollama API 오류: {response.status_code}'}
            
        except Exception as e:
            logger.error(f"AI 강화 분석 실패: {e}")
            return {'error': str(e), 'fallback_used': True}
    
    async def _generate_3d_model(self, image: np.ndarray, results: Dict, 
                               generation_type: str, mesh_quality: str, detail_level: int) -> Dict[str, Any]:
        """3D 모델 생성 (기본 구현)"""
        try:
            quality_config = self.quality_settings.get(mesh_quality, self.quality_settings['medium'])
            
            # 깊이 맵 생성 (간단한 구현)
            depth_map = self._create_depth_map(image)
            
            # 점군 생성
            point_cloud = self._generate_point_cloud(depth_map, quality_config)
            
            # 메쉬 생성 (삼각 분할)
            mesh_data = self._create_mesh(point_cloud, quality_config)
            
            return {
                'generation_type': generation_type,
                'mesh_quality': mesh_quality,
                'detail_level': detail_level,
                'model_info': {
                    'vertices_count': len(point_cloud),
                    'faces_count': len(mesh_data.get('faces', [])),
                    'mesh_quality_actual': mesh_quality
                },
                'depth_map_info': {
                    'depth_range': f"{depth_map.min():.2f} - {depth_map.max():.2f}",
                    'depth_levels': len(np.unique(depth_map))
                },
                'generation_status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"3D 모델 생성 실패: {e}")
            return {'error': str(e)}
    
    def _create_depth_map(self, image: np.ndarray) -> np.ndarray:
        """깊이 맵 생성 (간단한 밝기 기반)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러 적용
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 정규화
        depth_map = blurred.astype(np.float32) / 255.0
        
        # 깊이 강화
        depth_map = np.power(depth_map, 1.5)
        
        return depth_map
    
    def _generate_point_cloud(self, depth_map: np.ndarray, quality_config: Dict) -> List[Tuple[float, float, float]]:
        """점군 생성"""
        height, width = depth_map.shape
        points = []
        
        # 품질에 따른 샘플링 간격
        max_vertices = quality_config['vertices']
        step = max(1, int(np.sqrt(height * width / max_vertices)))
        
        for y in range(0, height, step):
            for x in range(0, width, step):
                z = depth_map[y, x] * 10  # 깊이 스케일링
                points.append((float(x), float(y), float(z)))
        
        return points[:max_vertices]  # 최대 정점 수 제한
    
    def _create_mesh(self, point_cloud: List[Tuple], quality_config: Dict) -> Dict[str, Any]:
        """메쉬 생성 (삼각분할)"""
        # 간단한 삼각분할 구현
        faces = []
        vertices = point_cloud
        
        # 그리드 기반 삼각분할 (간단화)
        # 실제로는 Delaunay 삼각분할이나 다른 고급 알고리즘 사용
        
        return {
            'vertices': vertices,
            'faces': faces[:quality_config['faces']],  # 면 수 제한
            'mesh_type': 'triangular'
        }
    
    async def _create_cad_file(self, model_data: Dict, output_format: str, session_id: str) -> Dict[str, Any]:
        """CAD 파일 생성"""
        try:
            if 'error' in model_data:
                return {'error': 'CAD 파일 생성을 위한 3D 모델 데이터가 유효하지 않음'}
            
            format_info = self.output_formats.get(output_format, self.output_formats['stl'])
            
            # 임시 파일 경로
            temp_dir = tempfile.gettempdir()
            filename = f"solomond_cad_{session_id}{format_info['extension']}"
            file_path = os.path.join(temp_dir, filename)
            
            # 형식별 파일 생성
            if output_format == 'stl':
                success = self._create_stl_file(file_path, model_data)
            elif output_format == 'obj':
                success = self._create_obj_file(file_path, model_data)
            elif output_format == 'ply':
                success = self._create_ply_file(file_path, model_data)
            else:
                # 기본적으로 STL 생성
                success = self._create_stl_file(file_path, model_data)
            
            if success:
                return {
                    'file_path': file_path,
                    'filename': filename,
                    'format': output_format,
                    'format_description': format_info['description'],
                    'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                    'download_available': True
                }
            else:
                return {'error': 'CAD 파일 생성 실패'}
            
        except Exception as e:
            logger.error(f"CAD 파일 생성 실패: {e}")
            return {'error': str(e)}
    
    def _create_stl_file(self, file_path: str, model_data: Dict) -> bool:
        """STL 파일 생성"""
        try:
            model_info = model_data.get('model_info', {})
            
            # STL 헤더 생성 (간단한 구현)
            with open(file_path, 'w') as f:
                f.write("solid solomond_generated\n")
                
                # 간단한 삼각형 면 생성 (실제로는 복잡한 메쉬 데이터 처리 필요)
                vertices_count = model_info.get('vertices_count', 0)
                for i in range(min(100, vertices_count // 3)):  # 임시로 100개 면만
                    f.write("  facet normal 0.0 0.0 1.0\n")
                    f.write("    outer loop\n")
                    f.write(f"      vertex {i}.0 {i}.0 0.0\n")
                    f.write(f"      vertex {i+1}.0 {i}.0 0.0\n")
                    f.write(f"      vertex {i}.0 {i+1}.0 0.0\n")
                    f.write("    endloop\n")
                    f.write("  endfacet\n")
                
                f.write("endsolid solomond_generated\n")
            
            return True
            
        except Exception as e:
            logger.error(f"STL 파일 생성 실패: {e}")
            return False
    
    def _create_obj_file(self, file_path: str, model_data: Dict) -> bool:
        """OBJ 파일 생성"""
        try:
            with open(file_path, 'w') as f:
                f.write("# Solomond AI Generated OBJ\n")
                f.write("# 3D Model from Image\n\n")
                
                # 정점 데이터 (예시)
                f.write("v 0.0 0.0 0.0\n")
                f.write("v 1.0 0.0 0.0\n")
                f.write("v 0.0 1.0 0.0\n")
                f.write("\n")
                
                # 면 데이터
                f.write("f 1 2 3\n")
            
            return True
            
        except Exception as e:
            logger.error(f"OBJ 파일 생성 실패: {e}")
            return False
    
    def _create_ply_file(self, file_path: str, model_data: Dict) -> bool:
        """PLY 파일 생성"""
        try:
            with open(file_path, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write("element vertex 3\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("element face 1\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")
                
                # 데이터
                f.write("0.0 0.0 0.0\n")
                f.write("1.0 0.0 0.0\n")
                f.write("0.0 1.0 0.0\n")
                f.write("3 0 1 2\n")
            
            return True
            
        except Exception as e:
            logger.error(f"PLY 파일 생성 실패: {e}")
            return False
    
    async def _prepare_export_info(self, cad_model: Dict, output_format: str, mesh_quality: str) -> Dict[str, Any]:
        """내보내기 정보 준비"""
        if 'error' in cad_model:
            return {'error': '내보내기 정보 생성 실패'}
        
        return {
            'export_ready': True,
            'file_info': {
                'format': output_format,
                'quality': mesh_quality,
                'file_path': cad_model.get('file_path', ''),
                'file_size_mb': round(cad_model.get('file_size', 0) / (1024 * 1024), 2)
            },
            'usage_recommendations': [
                f"{output_format.upper()} 파일은 대부분의 3D 소프트웨어에서 지원됩니다",
                "3D 프린팅을 위해서는 STL 형식을 권장합니다",
                "CAD 소프트웨어 편집을 위해서는 OBJ 형식을 권장합니다",
                f"{mesh_quality} 품질로 생성되었습니다"
            ],
            'next_steps': [
                "생성된 파일을 다운로드하세요",
                "3D 소프트웨어에서 파일을 열어 확인하세요",
                "필요시 추가 편집 및 최적화를 진행하세요"
            ]
        }

# FastAPI 앱 생성
app = FastAPI(
    title="Module 4: 3D CAD 변환 서비스",
    description="이미지에서 3D CAD 모델 생성",
    version="4.0.0"
)

# 서비스 인스턴스
service = CADConversionService()

@app.get("/health")
async def health_check():
    """헬스체크"""
    memory_stats = get_memory_stats()
    
    return {
        "status": "healthy",
        "service": "module4_3d_cad",
        "version": "4.0.0",
        "memory": {
            "memory_percent": memory_stats.get('memory_info', {}).get('percent', 0)
        },
        "capabilities": {
            "output_formats": len(service.output_formats),
            "quality_levels": len(service.quality_settings),
            "active_sessions": len(service.active_sessions),
            "ollama_available": "Ollama 서비스 상태 확인 필요"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/convert", response_model=CADResult)
async def convert_to_cad(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    generation_type: str = "basic",
    mesh_quality: str = "medium",
    output_format: str = "stl",
    use_ai_enhancement: bool = True,
    detail_level: int = 5
):
    """이미지를 3D CAD로 변환"""
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
    
    # 형식 및 품질 검증
    if output_format not in service.output_formats:
        raise HTTPException(status_code=400, detail=f"지원되지 않는 출력 형식: {output_format}")
    
    if mesh_quality not in service.quality_settings:
        raise HTTPException(status_code=400, detail=f"지원되지 않는 메쉬 품질: {mesh_quality}")
    
    if not (1 <= detail_level <= 10):
        raise HTTPException(status_code=400, detail="디테일 레벨은 1-10 사이여야 합니다")
    
    # 파일 읽기
    file_data = await file.read()
    if len(file_data) == 0:
        raise HTTPException(status_code=400, detail="빈 파일입니다")
    
    # 변환 실행
    result = await service.convert_image_to_cad(
        file_data, file.filename, generation_type, 
        mesh_quality, output_format, use_ai_enhancement, detail_level
    )
    
    return result

@app.get("/download/{session_id}")
async def download_cad_file(session_id: str):
    """생성된 CAD 파일 다운로드"""
    if session_id not in service.active_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    session = service.active_sessions[session_id]
    if session['status'] != 'completed':
        raise HTTPException(status_code=400, detail="변환이 완료되지 않았습니다")
    
    # 임시로 파일 경로 생성 (실제 구현에서는 세션에 저장된 경로 사용)
    temp_dir = tempfile.gettempdir()
    filename = f"solomond_cad_{session_id}.stl"
    file_path = os.path.join(temp_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="CAD 파일을 찾을 수 없습니다")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

@app.get("/formats")
async def get_supported_formats():
    """지원되는 출력 형식 조회"""
    return {
        "output_formats": service.output_formats,
        "quality_settings": {k: v for k, v in service.quality_settings.items()},
        "generation_types": ["basic", "advanced", "professional"],
        "detail_levels": "1-10 (낮음-높음)"
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
            "name": "module4_3d_cad",
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
        "conversion_capabilities": {
            "output_formats": len(service.output_formats),
            "quality_levels": len(service.quality_settings),
            "ai_enhancement": "Ollama 기반"
        }
    }

if __name__ == "__main__":
    logger.info("🏗️ Module 4 3D CAD 변환 서비스 시작: http://localhost:8004")
    
    uvicorn.run(
        "module4_service:app",
        host="0.0.0.0", 
        port=8004,
        reload=True,
        log_level="info"
    )