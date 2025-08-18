#!/usr/bin/env python3
"""
🏗️ 모듈 4: 이미지→3D CAD 변환 시스템
일러스트 이미지를 라이노 3D CAD 파일로 변환

주요 기능:
- 2D 일러스트 이미지 분석
- AI 기반 3D 형상 추론
- 라이노 3D 스크립트 생성
- 유색보석 주얼리 특화 (반지, 팔찌, 목걸이, 귀걸이)
- 생산 자동화 워크플로우

기술적 도전:
- 2D→3D 변환 알고리즘
- 주얼리 CAD 전문 지식
- 라이노 스크립팅 자동화
- 생산 가능한 설계 최적화
"""

import streamlit as st
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFilter
import json
import math
from typing import Dict, List, Any, Optional, Tuple
import base64
import io

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Ollama AI 통합 (v2.0 고도화)
try:
    sys.path.append(str(PROJECT_ROOT / "shared"))
    from ollama_interface import global_ollama
    # v2 고도화된 인터페이스 추가
    from ollama_interface_v2 import advanced_ollama, premium_insight, quick_summary as v2_summary
    OLLAMA_AVAILABLE = global_ollama.health_check()
    OLLAMA_V2_AVAILABLE = True
    CAD_MODEL = global_ollama.select_model("cad_conversion")
    print("✅ 3D CAD v2 Ollama 인터페이스 로드 완료!")
except ImportError as e:
    try:
        # v1 인터페이스만 시도
        from ollama_interface import global_ollama
        OLLAMA_AVAILABLE = global_ollama.health_check()
        OLLAMA_V2_AVAILABLE = False
        CAD_MODEL = global_ollama.select_model("cad_conversion")
        print("⚠️ 3D CAD v1 Ollama 인터페이스만 사용 가능")
    except ImportError:
        OLLAMA_AVAILABLE = False
        OLLAMA_V2_AVAILABLE = False
        CAD_MODEL = None
        print(f"❌ 3D CAD Ollama 인터페이스 로드 실패: {e}")

# 페이지 설정
st.set_page_config(
    page_title="🏗️ 이미지→3D CAD",
    page_icon="🏗️",
    layout="wide"
)

# 주얼리 CAD 템플릿 데이터베이스
JEWELRY_TEMPLATES = {
    "ring": {
        "base_parameters": {
            "inner_diameter": 17.0,  # mm (size 7)
            "band_width": 2.0,
            "band_thickness": 1.5,
            "setting_height": 3.0
        },
        "stone_settings": {
            "solitaire": {"prongs": 6, "height": 4.0},
            "halo": {"center_stone": True, "halo_stones": 20},
            "three_stone": {"stones": 3, "spacing": 1.5},
            "eternity": {"stones": "continuous", "stone_size": 0.1}
        },
        "rhino_commands": [
            "Circle 0,0,0 {inner_diameter/2}",
            "ExtrudeCrv {band_thickness}",
            "Cap",
            "OffsetSrf {band_width}"
        ]
    },
    "necklace": {
        "base_parameters": {
            "chain_length": 450.0,  # mm (18 inches)
            "chain_width": 1.5,
            "pendant_width": 15.0,
            "pendant_height": 20.0
        },
        "chain_types": {
            "cable": {"link_length": 3.0, "wire_thickness": 0.8},
            "curb": {"link_length": 4.0, "wire_thickness": 1.0},
            "rope": {"twist_angle": 45, "wire_count": 4},
            "snake": {"segment_count": 100, "taper": True}
        },
        "rhino_commands": [
            "Circle 0,0,0 {chain_width/2}",
            "InterpCrv",
            "Pipe {chain_width/2}",
            "ArrayPolar"
        ]
    },
    "earring": {
        "base_parameters": {
            "post_diameter": 0.8,
            "post_length": 10.0,
            "back_diameter": 6.0,
            "design_height": 15.0
        },
        "styles": {
            "stud": {"simple": True, "stone_size": 5.0},
            "drop": {"length": 25.0, "attachment": "hook"},
            "hoop": {"diameter": 20.0, "thickness": 2.0},
            "chandelier": {"tiers": 3, "complexity": "high"}
        },
        "rhino_commands": [
            "Cylinder 0,0,0 {post_diameter/2} {post_length}",
            "Sphere 0,0,{post_length} {back_diameter/2}",
            "BooleanUnion"
        ]
    },
    "bracelet": {
        "base_parameters": {
            "inner_circumference": 180.0,  # mm (7 inches)
            "band_width": 8.0,
            "band_thickness": 2.0,
            "clasp_length": 15.0
        },
        "styles": {
            "tennis": {"stones": 50, "setting": "prong"},
            "bangle": {"solid": True, "opening": 30.0},
            "chain": {"links": 20, "flexibility": True},
            "cuff": {"open": True, "adjustable": False}
        },
        "rhino_commands": [
            "Circle 0,0,0 {inner_circumference/(2*pi)}",
            "OffsetCrv {band_width/2}",
            "ExtrudeCrv {band_thickness}",
            "FilletEdge"
        ]
    }
}

class ImageToCADConverter:
    """이미지→3D CAD 변환 시스템"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_image_analysis()
    
    def initialize_session_state(self):
        """세션 상태 초기화"""
        if "uploaded_images" not in st.session_state:
            st.session_state.uploaded_images = []
        if "cad_results" not in st.session_state:
            st.session_state.cad_results = []
        if "current_project" not in st.session_state:
            st.session_state.current_project = None
    
    def setup_image_analysis(self):
        """이미지 분석 설정"""
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.svg']
        self.jewelry_types = ["ring", "necklace", "earring", "bracelet"]
    
    def analyze_jewelry_image(self, image: Image.Image) -> Dict[str, Any]:
        """주얼리 이미지 분석"""
        try:
            # 이미지 전처리
            processed_image = self.preprocess_for_analysis(image)
            
            # 기본 특성 추출
            features = self.extract_jewelry_features(processed_image)
            
            # 치수 추정
            dimensions = self.estimate_dimensions(processed_image, features)
            
            # 주얼리 타입 분류
            jewelry_type = self.classify_jewelry_type(features)
            
            return {
                "processed_image": processed_image,
                "features": features,
                "dimensions": dimensions,
                "jewelry_type": jewelry_type,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"이미지 분석 중 오류: {str(e)}"}
    
    def preprocess_for_analysis(self, image: Image.Image) -> Image.Image:
        """분석용 이미지 전처리"""
        # RGB 모드로 변환
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 크기 정규화
        max_size = 800
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # 대비 향상
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(1.3)
        
        # 선명도 향상
        sharpener = ImageEnhance.Sharpness(enhanced)
        final_image = sharpener.enhance(1.2)
        
        return final_image
    
    def extract_jewelry_features(self, image: Image.Image) -> Dict[str, Any]:
        """주얼리 특성 추출"""
        features = {}
        
        try:
            # NumPy 배열로 변환
            img_array = np.array(image)
            
            # 에지 검출
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # 윤곽선 검출
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 가장 큰 윤곽선 (주 객체)
                main_contour = max(contours, key=cv2.contourArea)
                
                # 기하학적 특성
                area = cv2.contourArea(main_contour)
                perimeter = cv2.arcLength(main_contour, True)
                
                # 경계 사각형
                x, y, w, h = cv2.boundingRect(main_contour)
                aspect_ratio = w / h if h > 0 else 1.0
                
                # 원형성 (4π*면적/둘레²)
                circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                features.update({
                    "area": float(area),
                    "perimeter": float(perimeter),
                    "aspect_ratio": float(aspect_ratio),
                    "circularity": float(circularity),
                    "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    "contour_points": len(main_contour)
                })
            
            # 색상 특성
            colors = self.analyze_colors(image)
            features["dominant_colors"] = colors
            
            # 대칭성 분석
            symmetry = self.analyze_symmetry(gray)
            features["symmetry"] = symmetry
            
            return features
            
        except Exception as e:
            return {"error": f"특성 추출 중 오류: {str(e)}"}
    
    def analyze_colors(self, image: Image.Image) -> List[Dict]:
        """색상 분석"""
        try:
            img_array = np.array(image)
            pixels = img_array.reshape(-1, 3)
            
            # 간단한 색상 클러스터링
            from collections import Counter
            
            # 색상 양자화
            quantized = (pixels // 64) * 64
            color_counts = Counter(map(tuple, quantized))
            
            dominant_colors = []
            total_pixels = len(pixels)
            
            for i, (color, count) in enumerate(color_counts.most_common(5)):
                dominant_colors.append({
                    "rgb": color,
                    "percentage": (count / total_pixels) * 100,
                    "is_metallic": self.is_metallic_color(color),
                    "is_stone_color": self.is_stone_color(color)
                })
            
            return dominant_colors
            
        except Exception as e:
            return [{"error": str(e)}]
    
    def is_metallic_color(self, rgb: Tuple[int, int, int]) -> bool:
        """금속 색상 판별"""
        r, g, b = rgb
        
        # 금색 범위
        if 180 <= r <= 255 and 150 <= g <= 200 and 0 <= b <= 100:
            return True
        
        # 은색 범위
        if abs(r - g) < 30 and abs(g - b) < 30 and 150 <= r <= 255:
            return True
        
        # 로즈골드 범위
        if 200 <= r <= 255 and 100 <= g <= 180 and 100 <= b <= 150:
            return True
        
        return False
    
    def is_stone_color(self, rgb: Tuple[int, int, int]) -> bool:
        """보석 색상 판별"""
        r, g, b = rgb
        
        # 다이아몬드 (무색-약간 노란색)
        if abs(r - g) < 20 and abs(g - b) < 20 and 200 <= r <= 255:
            return True
        
        # 유색 보석들
        color_ranges = [
            (150, 0, 0, 255, 100, 100),    # 루비 빨간색
            (0, 0, 150, 100, 100, 255),    # 사파이어 파란색
            (0, 150, 0, 100, 255, 100),    # 에메랄드 녹색
            (100, 0, 100, 255, 100, 255),  # 자수정 보라색
        ]
        
        for r_min, g_min, b_min, r_max, g_max, b_max in color_ranges:
            if r_min <= r <= r_max and g_min <= g <= g_max and b_min <= b <= b_max:
                return True
        
        return False
    
    def analyze_symmetry(self, gray_image: np.ndarray) -> Dict[str, float]:
        """대칭성 분석"""
        try:
            h, w = gray_image.shape
            
            # 수직 대칭성
            left_half = gray_image[:, :w//2]
            right_half = np.fliplr(gray_image[:, w//2:])
            
            # 크기 맞추기
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # 상관관계 계산
            vertical_symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            
            # 수평 대칭성
            top_half = gray_image[:h//2, :]
            bottom_half = np.flipud(gray_image[h//2:, :])
            
            min_height = min(top_half.shape[0], bottom_half.shape[0])
            top_half = top_half[:min_height, :]
            bottom_half = bottom_half[:min_height, :]
            
            horizontal_symmetry = np.corrcoef(top_half.flatten(), bottom_half.flatten())[0, 1]
            
            return {
                "vertical": float(vertical_symmetry) if not np.isnan(vertical_symmetry) else 0.0,
                "horizontal": float(horizontal_symmetry) if not np.isnan(horizontal_symmetry) else 0.0
            }
            
        except Exception as e:
            return {"vertical": 0.0, "horizontal": 0.0, "error": str(e)}
    
    def estimate_dimensions(self, image: Image.Image, features: Dict) -> Dict[str, float]:
        """치수 추정"""
        # 기본 가정: 이미지에서 1000픽셀 = 30mm (반지 기준)
        pixel_to_mm = 30.0 / 1000.0
        
        bbox = features.get("bounding_box", {})
        width_px = bbox.get("width", 100)
        height_px = bbox.get("height", 100)
        
        estimated_width = width_px * pixel_to_mm
        estimated_height = height_px * pixel_to_mm
        
        return {
            "width_mm": round(estimated_width, 2),
            "height_mm": round(estimated_height, 2),
            "area_mm2": round(estimated_width * estimated_height, 2),
            "scale_factor": pixel_to_mm
        }
    
    def classify_jewelry_type(self, features: Dict) -> str:
        """주얼리 타입 분류"""
        aspect_ratio = features.get("aspect_ratio", 1.0)
        circularity = features.get("circularity", 0.0)
        
        # 분류 로직
        if circularity > 0.7:
            return "ring"  # 원형에 가까우면 반지
        elif aspect_ratio > 3.0:
            return "bracelet"  # 가로로 긴 형태면 팔찌
        elif aspect_ratio > 1.5:
            return "necklace"  # 세로로 긴 형태면 목걸이
        else:
            return "earring"  # 나머지는 귀걸이
    
    def generate_cad_with_ollama(self, analysis_data: Dict, user_specs: Dict) -> Dict[str, Any]:
        """🏆 v2 고도화 Ollama AI CAD 스크립트 생성"""
        if not OLLAMA_AVAILABLE:
            return {"error": "Ollama AI를 사용할 수 없습니다."}
        
        try:
            jewelry_type = analysis_data.get("jewelry_type", "ring")
            features = analysis_data.get("features", {})
            dimensions = analysis_data.get("dimensions", {})
            
            # 템플릿 가져오기
            template = JEWELRY_TEMPLATES.get(jewelry_type, JEWELRY_TEMPLATES["ring"])
            
            # CAD 분석 정보 구성
            cad_info = f"""
주얼리 유형: {jewelry_type}
치수 정보:
- 너비: {dimensions.get('width_mm', 0)}mm
- 높이: {dimensions.get('height_mm', 0)}mm
- 면적: {dimensions.get('area_mm2', 0)}mm²

이미지 특성:
- 원형성: {features.get('circularity', 0):.3f}
- 종횡비: {features.get('aspect_ratio', 1):.3f}
- 대칭성: 수직 {features.get('symmetry', {}).get('vertical', 0):.3f}, 수평 {features.get('symmetry', {}).get('horizontal', 0):.3f}

사용자 요구사항:
- 재질: {user_specs.get('material', '18K 골드')}
- 보석: {user_specs.get('stone_type', '다이아몬드')}
- 특별 요청: {user_specs.get('special_requirements', '없음')}

기본 템플릿 명령어: {template.get('rhino_commands', [])[0] if template.get('rhino_commands') else 'Circle 0,0,0 10'}
            """
            
            if OLLAMA_V2_AVAILABLE:
                # v2 인터페이스로 다양한 레벨 CAD 생성
                v2_cad = self.process_cad_v2(cad_info, jewelry_type, user_specs)
                return {
                    "cad_analysis_v2": v2_cad,
                    "best_cad_script": v2_cad.get('best_script', '# CAD 스크립트 생성 실패'),
                    "generation_time": datetime.now().isoformat(),
                    "v2_processed": True
                }
            else:
                # 기존 v1 분석
                cad_prompt = f"""
당신은 주얼리 CAD 전문가이자 라이노 3D 스크립팅 전문가입니다. 
다음 분석 데이터를 바탕으로 라이노 3D 스크립트를 생성해주세요.

{cad_info}

다음 형식으로 응답해주세요:

1. **설계 개념**: 디자인의 핵심 아이디어
2. **기술적 사양**: 정확한 치수와 재료 사양
3. **라이노 스크립트**: 
```rhinoscript
// 주요 구조 생성 명령어들
{template.get('rhino_commands', [])[0] if template.get('rhino_commands') else 'Circle 0,0,0 10'}
// 추가 세부 작업
```
4. **생산성 고려사항**: 실제 제작 시 주의사항
5. **품질 검증 포인트**: 확인해야 할 핵심 요소들

한국어로 상세히 작성해주세요.
                """
                
                # Ollama AI 호출
                response = global_ollama.generate_response(
                    cad_prompt,
                    model=CAD_MODEL,
                    temperature=0.2,
                    max_tokens=2000
                )
                
                # 스크립트 추출
                rhino_script = self.extract_rhino_script(response)
                
                return {
                    "ai_response": response,
                    "rhino_script": rhino_script,
                    "model_used": CAD_MODEL,
                    "jewelry_type": jewelry_type,
                    "generation_time": datetime.now().isoformat(),
                    "template_used": template
                }
            
        except Exception as e:
            return {"error": f"CAD 생성 중 오류: {str(e)}"}
    
    def process_cad_v2(self, cad_info: str, jewelry_type: str, user_specs: dict) -> dict:
        """🏆 v2 고도화 CAD 생성 - 5개 모델 전략 활용"""
        
        try:
            analysis_results = {}
            
            # 🔥 PREMIUM CAD 설계 (Gemma3-27B) - 고급 CAD 설계
            try:
                premium_result = advanced_ollama.advanced_generate(
                    task_type="cad_complex",
                    content=cad_info,
                    task_goal=f"{jewelry_type} 프리미엄 3D CAD 설계 및 라이노 스크립트 생성",
                    quality_priority=True,
                    speed_priority=False
                )
                analysis_results['premium'] = {
                    'title': '🔥 프리미엄 CAD 설계 (Gemma3-27B)',
                    'content': premium_result,
                    'script': self.extract_rhino_script(premium_result),
                    'model': 'gemma3:27b',
                    'tier': 'PREMIUM'
                }
            except Exception as e:
                analysis_results['premium'] = {
                    'title': '🔥 프리미엄 CAD 설계 (오류)',
                    'content': f"프리미엄 CAD 생성 실패: {str(e)}",
                    'script': '// 프리미엄 CAD 생성 실패',
                    'tier': 'PREMIUM'
                }
            
            # ⚡ STANDARD CAD 설계 (Qwen3-8B) - 균형잡힌 CAD 생성
            try:
                standard_result = advanced_ollama.advanced_generate(
                    task_type="cad_simple",
                    content=cad_info,
                    task_goal=f"{jewelry_type} 표준 3D CAD 설계",
                    quality_priority=False,
                    speed_priority=False
                )
                analysis_results['standard'] = {
                    'title': '⚡ 표준 CAD 설계 (Qwen3-8B)',
                    'content': standard_result,
                    'script': self.extract_rhino_script(standard_result),
                    'model': 'qwen3:8b',
                    'tier': 'STANDARD'
                }
            except Exception as e:
                analysis_results['standard'] = {
                    'title': '⚡ 표준 CAD 설계 (오류)',
                    'content': f"표준 CAD 생성 실패: {str(e)}",
                    'script': '// 표준 CAD 생성 실패',
                    'tier': 'STANDARD'
                }
            
            # 🚀 FAST CAD 설계 (Gemma3-4B) - 빠른 프로토타입
            try:
                fast_result = v2_summary(f"주얼리 CAD 빠른 프로토타입: {cad_info}")
                analysis_results['fast'] = {
                    'title': '🚀 빠른 프로토타입 (Gemma3-4B)',
                    'content': fast_result,
                    'script': self.generate_simple_cad_script(jewelry_type),
                    'model': 'gemma3:4b',
                    'tier': 'FAST'
                }
            except Exception as e:
                analysis_results['fast'] = {
                    'title': '🚀 빠른 프로토타입 (오류)',
                    'content': f"빠른 CAD 생성 실패: {str(e)}",
                    'script': self.generate_simple_cad_script(jewelry_type),
                    'tier': 'FAST'
                }
            
            # 최고 품질 CAD 스크립트 선택
            best_script = self.generate_simple_cad_script(jewelry_type)  # 폴백
            best_tier = 'fallback'
            
            for tier in ['premium', 'standard', 'fast']:
                if tier in analysis_results and 'script' in analysis_results[tier]:
                    script = analysis_results[tier]['script']
                    if script and len(script) > 50 and "실패" not in script:
                        best_script = script
                        best_tier = tier
                        break
            
            analysis_results['best_script'] = best_script
            analysis_results['best_tier'] = best_tier
            analysis_results['v2_processed'] = True
            
            return analysis_results
            
        except Exception as e:
            return {
                'error': {
                    'title': '❌ v2 CAD 시스템 오류',
                    'content': f"v2 CAD 시스템 오류: {str(e)}",
                    'script': self.generate_simple_cad_script(jewelry_type),
                    'tier': 'ERROR'
                },
                'best_script': self.generate_simple_cad_script(jewelry_type),
                'v2_processed': False
            }
    
    def generate_simple_cad_script(self, jewelry_type: str) -> str:
        """간단한 CAD 스크립트 생성 (폴백용)"""
        simple_scripts = {
            "ring": "Circle 0,0,0 10\nPipe\nCap both ends",
            "necklace": "InterpCrv\nPipe 1.5\nArray Polar",
            "bracelet": "Circle 0,0,0 30\nPipe 2\nArray",
            "earring": "Sphere 0,0,0 5\nMove 0,0,10"
        }
        return simple_scripts.get(jewelry_type, simple_scripts["ring"])
    
    def extract_rhino_script(self, ai_response: str) -> str:
        """AI 응답에서 라이노 스크립트 추출"""
        try:
            import re
            
            # 코드 블록 패턴 찾기
            patterns = [
                r'```rhinoscript\n(.*?)\n```',
                r'```rhino\n(.*?)\n```',
                r'```\n(.*?)\n```'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, ai_response, re.DOTALL)
                if match:
                    return match.group(1).strip()
            
            # 패턴이 없으면 전체 응답 반환
            return ai_response
            
        except Exception as e:
            return f"// 스크립트 추출 오류: {str(e)}"
    
    def render_upload_interface(self):
        """업로드 인터페이스"""
        st.markdown("## 📤 일러스트 이미지 업로드")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "주얼리 일러스트 이미지를 업로드하세요",
                type=['jpg', 'jpeg', 'png', 'bmp', 'svg'],
                accept_multiple_files=True,
                help="지원 형식: JPG, PNG, BMP, SVG"
            )
            
            if uploaded_files:
                st.session_state.uploaded_images = uploaded_files
                st.success(f"✅ {len(uploaded_files)}개 이미지 업로드 완료")
        
        with col2:
            st.markdown("### 💡 업로드 팁")
            st.info("""
            **좋은 일러스트 조건:**
            - 정면도/측면도 포함
            - 명확한 윤곽선
            - 단순한 배경
            - 높은 해상도
            - 정확한 비율
            """)
    
    def render_specifications_interface(self):
        """사양 입력 인터페이스"""
        st.markdown("## ⚙️ 제작 사양 입력")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📏 기본 사양")
            material = st.selectbox(
                "재질 선택",
                ["18K 화이트골드", "18K 옐로골드", "18K 로즈골드", "플래티넘", "실버"]
            )
            
            jewelry_size = st.text_input("사이즈", "7 (반지 기준)")
        
        with col2:
            st.markdown("### 💎 보석 설정")
            stone_type = st.selectbox(
                "메인 스톤",
                ["다이아몬드", "루비", "사파이어", "에메랄드", "없음"]
            )
            
            stone_size = st.text_input("스톤 크기", "1.0ct")
        
        with col3:
            st.markdown("### 🎨 특별 요청")
            finish = st.selectbox(
                "표면 마감",
                ["폴리싱", "매트", "브러시", "해머"]
            )
            
            special_requirements = st.text_area("추가 요청사항", "")
        
        return {
            "material": material,
            "jewelry_size": jewelry_size,
            "stone_type": stone_type,
            "stone_size": stone_size,
            "finish": finish,
            "special_requirements": special_requirements
        }
    
    def render_conversion_interface(self):
        """변환 실행 인터페이스"""
        st.markdown("## 🏗️ 3D CAD 변환")
        
        if not st.session_state.uploaded_images:
            st.warning("먼저 이미지를 업로드해주세요.")
            return
        
        user_specs = self.render_specifications_interface()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 CAD 변환 시작", type="primary"):
                self.run_cad_conversion(user_specs)
        
        with col2:
            if st.button("🧹 결과 초기화"):
                st.session_state.cad_results = []
                st.rerun()
        
        with col3:
            st.metric("Ollama AI", "✅ 연결됨" if OLLAMA_AVAILABLE else "❌ 불가능")
    
    def run_cad_conversion(self, user_specs: Dict):
        """CAD 변환 실행"""
        if not OLLAMA_AVAILABLE:
            st.error("❌ Ollama AI를 사용할 수 없습니다.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, uploaded_file in enumerate(st.session_state.uploaded_images):
            status_text.text(f"변환 중: {uploaded_file.name}")
            
            try:
                # 이미지 로드
                image = Image.open(uploaded_file)
                
                # 이미지 분석
                analysis_data = self.analyze_jewelry_image(image)
                if "error" in analysis_data:
                    continue
                
                # CAD 생성
                cad_result = self.generate_cad_with_ollama(analysis_data, user_specs)
                
                # 결과 저장
                conversion_result = {
                    "filename": uploaded_file.name,
                    "analysis_data": analysis_data,
                    "user_specs": user_specs,
                    "cad_result": cad_result,
                    "timestamp": datetime.now()
                }
                
                results.append(conversion_result)
                
                # 진행률 업데이트
                progress_bar.progress((i + 1) / len(st.session_state.uploaded_images))
                
            except Exception as e:
                st.error(f"{uploaded_file.name} 변환 중 오류: {str(e)}")
        
        st.session_state.cad_results = results
        status_text.text("✅ 변환 완료!")
        st.success(f"총 {len(results)}개 이미지 변환 완료")
    
    def render_results(self):
        """변환 결과 표시"""
        if not st.session_state.cad_results:
            st.info("아직 변환 결과가 없습니다.")
            return
        
        st.markdown("## 📊 변환 결과")
        
        # 결과 선택
        if len(st.session_state.cad_results) > 1:
            selected_idx = st.selectbox(
                "결과 선택",
                range(len(st.session_state.cad_results)),
                format_func=lambda x: st.session_state.cad_results[x]["filename"]
            )
            current_result = st.session_state.cad_results[selected_idx]
        else:
            current_result = st.session_state.cad_results[0]
        
        # 결과 표시
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🖼️ 원본 이미지")
            analysis_data = current_result["analysis_data"]
            if "processed_image" in analysis_data:
                st.image(
                    analysis_data["processed_image"],
                    caption=current_result["filename"]
                )
            
            # 분석 정보
            st.markdown("### 📊 분석 정보")
            jewelry_type = analysis_data.get("jewelry_type", "Unknown")
            dimensions = analysis_data.get("dimensions", {})
            
            st.write(f"**유형**: {jewelry_type}")
            st.write(f"**예상 크기**: {dimensions.get('width_mm', 0):.1f}×{dimensions.get('height_mm', 0):.1f}mm")
        
        with col2:
            st.markdown("### 🏗️ CAD 변환 결과")
            cad_result = current_result["cad_result"]
            
            if "error" not in cad_result:
                ai_response = cad_result.get("ai_response", "")
                st.markdown(ai_response)
                
                # 라이노 스크립트 표시
                st.markdown("### 📝 라이노 스크립트")
                rhino_script = cad_result.get("rhino_script", "")
                st.code(rhino_script, language="rhinoscript")
                
                # 다운로드 버튼
                st.download_button(
                    "📥 스크립트 다운로드",
                    rhino_script,
                    file_name=f"jewelry_cad_{current_result['filename']}.rvb",
                    mime="text/plain"
                )
            else:
                st.error(f"변환 오류: {cad_result['error']}")
    
    def render_sidebar(self):
        """사이드바"""
        with st.sidebar:
            st.markdown("## ⚙️ 설정")
            
            st.markdown("### 🏗️ 지원 주얼리")
            st.info("""
            - 💍 반지 (Ring)
            - 📿 목걸이 (Necklace)  
            - 👂 귀걸이 (Earring)
            - 📿 팔찌 (Bracelet)
            """)
            
            st.markdown("### 🎯 특화 기능")
            st.info("""
            - 유색보석 설정 최적화
            - 생산 가능한 설계
            - 라이노 3D 스크립트 자동 생성
            - 실시간 AI 분석
            """)
            
            st.markdown("### 📊 통계")
            st.metric("변환 완료", len(st.session_state.cad_results))
            st.metric("업로드된 이미지", len(st.session_state.uploaded_images))
            
            st.markdown("### ⚠️ 주의사항")
            st.warning("""
            본 시스템은 연구 목적입니다.
            실제 생산 전에는 전문가
            검토가 필요합니다.
            """)
            
            if st.button("🏠 메인 대시보드로"):
                st.markdown("메인 대시보드: http://localhost:8505")
    
    def run(self):
        """모듈 실행"""
        st.markdown("# 🏗️ 이미지→3D CAD 변환 시스템")
        st.markdown("일러스트를 라이노 3D CAD 파일로 변환 (연구 단계)")
        
        self.render_sidebar()
        
        st.markdown("---")
        
        # 탭 구성
        tab1, tab2, tab3 = st.tabs(["📤 업로드", "🏗️ 변환", "📊 결과"])
        
        with tab1:
            self.render_upload_interface()
        
        with tab2:
            self.render_conversion_interface()
        
        with tab3:
            self.render_results()

def main():
    """메인 함수"""
    converter = ImageToCADConverter()
    converter.run()

if __name__ == "__main__":
    main()