#!/usr/bin/env python3
"""
💎 모듈 3: 보석 산지 분석 시스템
AI 기반 보석 이미지 분석 및 원산지 추정

주요 기능:
- 보석 이미지 업로드 및 전처리
- Ollama AI 기반 보석 특성 분석
- 산지별 특성 데이터베이스 매칭
- 신뢰도 평가 및 분석 리포트
- 보석학 전문 지식 활용
"""

import streamlit as st
import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
import json
from typing import Dict, List, Any, Optional, Tuple

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Ollama AI 통합 (v2.0 고도화)
try:
    sys.path.append(str(PROJECT_ROOT / "shared"))
    from ollama_interface import global_ollama
    # v2 고도화된 인터페이스 추가
    from ollama_interface_v2 import advanced_ollama, expert_gemstone, premium_insight
    OLLAMA_AVAILABLE = global_ollama.health_check()
    OLLAMA_V2_AVAILABLE = True
    GEMSTONE_MODEL = global_ollama.select_model("gemstone_analysis")
    print("✅ 보석 분석 v2 Ollama 인터페이스 로드 완료!")
except ImportError as e:
    try:
        # v1 인터페이스만 시도
        from ollama_interface import global_ollama
        OLLAMA_AVAILABLE = global_ollama.health_check()
        OLLAMA_V2_AVAILABLE = False
        GEMSTONE_MODEL = global_ollama.select_model("gemstone_analysis")
        print("⚠️ 보석 분석 v1 Ollama 인터페이스만 사용 가능")
    except ImportError:
        OLLAMA_AVAILABLE = False
        OLLAMA_V2_AVAILABLE = False
        GEMSTONE_MODEL = None
        print(f"❌ 보석 분석 Ollama 인터페이스 로드 실패: {e}")

# 페이지 설정
st.set_page_config(
    page_title="💎 보석 산지 분석",
    page_icon="💎",
    layout="wide"
)

# 보석 산지별 특성 데이터베이스
GEMSTONE_DATABASE = {
    "diamond": {
        "regions": {
            "Botswana": {
                "characteristics": ["Type IIa diamonds", "exceptional clarity", "large carat sizes"],
                "inclusions": ["metallic inclusions", "feather patterns"],
                "color_range": ["D-Z", "fancy colors rare"],
                "confidence_indicators": ["octahedral crystal form", "specific gravity 3.52"]
            },
            "Russia": {
                "characteristics": ["Type I diamonds", "industrial grade common", "cubic crystals"],
                "inclusions": ["graphite inclusions", "cloud formations"],
                "color_range": ["commercial grades", "brown tints common"],
                "confidence_indicators": ["twinned crystals", "surface trigons"]
            },
            "Australia": {
                "characteristics": ["brown diamonds", "champagne colors", "cognac shades"],
                "inclusions": ["brown radiation spots", "graining"],
                "color_range": ["brown spectrum", "rare pinks"],
                "confidence_indicators": ["Australian certificate", "specific brown hues"]
            },
            "South_Africa": {
                "characteristics": ["Cape series", "traditional white diamonds", "historical mines"],
                "inclusions": ["cape yellow", "strain patterns"],
                "color_range": ["cape to colorless", "yellow tints"],
                "confidence_indicators": ["kimberlite origins", "traditional cuts"]
            }
        }
    },
    "ruby": {
        "regions": {
            "Myanmar": {
                "characteristics": ["pigeon blood red", "silk inclusions", "exceptional color"],
                "inclusions": ["rutile silk", "calcite crystals"],
                "color_range": ["pure red", "slight pink undertones"],
                "confidence_indicators": ["Mogok origin", "specific gravity 4.0"]
            },
            "Thailand": {
                "characteristics": ["darker red", "iron content", "heat treatment common"],
                "inclusions": ["iron oxide", "healed fractures"],
                "color_range": ["brownish red", "purplish red"],
                "confidence_indicators": ["Chanthaburi origin", "specific absorption spectrum"]
            },
            "Sri_Lanka": {
                "characteristics": ["Ceylon rubies", "lighter reds", "pink rubies"],
                "inclusions": ["zircon halos", "liquid inclusions"],
                "color_range": ["pink to red", "purplish undertones"],
                "confidence_indicators": ["hexagonal zoning", "specific pleochroism"]
            }
        }
    },
    "sapphire": {
        "regions": {
            "Kashmir": {
                "characteristics": ["cornflower blue", "velvety appearance", "exceptional rarity"],
                "inclusions": ["silk inclusions", "three-phase inclusions"],
                "color_range": ["pure blue", "slight violet undertones"],
                "confidence_indicators": ["specific locality", "historical significance"]
            },
            "Ceylon": {
                "characteristics": ["Ceylon sapphires", "light to medium blue", "high clarity"],
                "inclusions": ["zircon crystals", "healed fissures"],
                "color_range": ["colorless to deep blue", "parti-colors"],
                "confidence_indicators": ["Sri Lankan certificate", "specific gravity variations"]
            },
            "Australia": {
                "characteristics": ["dark blue", "green undertones", "inky appearance"],
                "inclusions": ["needle-like inclusions", "color zoning"],
                "color_range": ["very dark blue", "green-blue"],
                "confidence_indicators": ["Australian mining", "specific iron content"]
            }
        }
    },
    "emerald": {
        "regions": {
            "Colombia": {
                "characteristics": ["pure green", "three-phase inclusions", "exceptional color"],
                "inclusions": ["jagged three-phase", "calcite crystals"],
                "color_range": ["bluish green to pure green"],
                "confidence_indicators": ["Muzo/Chivor origin", "specific gravity 2.7"]
            },
            "Zambia": {
                "characteristics": ["darker green", "bluish undertones", "iron inclusions"],
                "inclusions": ["mica plates", "amphibole needles"],
                "color_range": ["bluish green", "darker tones"],
                "confidence_indicators": ["Kafubu River", "specific pleochroism"]
            },
            "Brazil": {
                "characteristics": ["lighter green", "cleaner stones", "heat treatment"],
                "inclusions": ["two-phase inclusions", "mica platelets"],
                "color_range": ["yellowish green to green"],
                "confidence_indicators": ["Brazilian cutting", "specific origin markers"]
            }
        }
    }
}

class GemstoneSrcAnalyzer:
    """보석 산지 분석 시스템"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_image_processing()
    
    def initialize_session_state(self):
        """세션 상태 초기화"""
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = []
        if "uploaded_images" not in st.session_state:
            st.session_state.uploaded_images = []
        if "current_analysis" not in st.session_state:
            st.session_state.current_analysis = None
    
    def setup_image_processing(self):
        """이미지 처리 설정"""
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.max_image_size = (800, 600)
    
    def preprocess_image(self, image: Image.Image) -> Dict[str, Any]:
        """이미지 전처리 및 특성 추출"""
        try:
            # 이미지 크기 조정
            image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            
            # 이미지 향상
            enhancer = ImageEnhance.Contrast(image)
            enhanced_image = enhancer.enhance(1.2)
            
            # 색상 분석
            colors = self.extract_dominant_colors(image)
            
            # 텍스처 분석 (기본적인 방법)
            texture_info = self.analyze_texture(image)
            
            # 기본 메타데이터
            metadata = {
                "size": image.size,
                "mode": image.mode,
                "format": getattr(image, 'format', 'Unknown'),
                "dominant_colors": colors,
                "texture_analysis": texture_info,
                "enhancement_applied": True
            }
            
            return {
                "processed_image": enhanced_image,
                "metadata": metadata,
                "original_image": image
            }
            
        except Exception as e:
            st.error(f"이미지 전처리 중 오류: {str(e)}")
            return None
    
    def extract_dominant_colors(self, image: Image.Image, num_colors: int = 5) -> List[Dict]:
        """주요 색상 추출"""
        try:
            # RGB 모드로 변환
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # NumPy 배열로 변환
            img_array = np.array(image)
            pixels = img_array.reshape(-1, 3)
            
            # 간단한 색상 클러스터링 (K-means 대신 히스토그램 사용)
            from collections import Counter
            
            # 색상 양자화
            quantized_pixels = (pixels // 32) * 32
            color_counts = Counter(map(tuple, quantized_pixels))
            
            # 상위 색상들
            dominant_colors = []
            for i, (color, count) in enumerate(color_counts.most_common(num_colors)):
                dominant_colors.append({
                    "rank": i + 1,
                    "rgb": color,
                    "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                    "percentage": (count / len(pixels)) * 100,
                    "description": self.describe_color(color)
                })
            
            return dominant_colors
            
        except Exception as e:
            return [{"error": str(e)}]
    
    def describe_color(self, rgb: Tuple[int, int, int]) -> str:
        """색상을 보석학적 용어로 설명"""
        r, g, b = rgb
        
        # 기본적인 색상 분류
        if r > g and r > b:
            if r > 150:
                return "강한 적색계"
            else:
                return "약한 적색계"
        elif g > r and g > b:
            if g > 150:
                return "강한 녹색계"
            else:
                return "약한 녹색계"
        elif b > r and b > g:
            if b > 150:
                return "강한 청색계"
            else:
                return "약한 청색계"
        else:
            brightness = (r + g + b) / 3
            if brightness > 200:
                return "밝은 무채색"
            elif brightness > 100:
                return "중간 무채색"
            else:
                return "어두운 무채색"
    
    def analyze_texture(self, image: Image.Image) -> Dict[str, Any]:
        """기본적인 텍스처 분석"""
        try:
            # 그레이스케일 변환
            gray_image = image.convert('L')
            img_array = np.array(gray_image)
            
            # 기본 통계
            mean_brightness = np.mean(img_array)
            std_brightness = np.std(img_array)
            
            # 에지 검출 (간단한 방법)
            from scipy import ndimage
            edges = ndimage.sobel(img_array)
            edge_density = np.mean(edges > np.mean(edges))
            
            return {
                "mean_brightness": float(mean_brightness),
                "brightness_variation": float(std_brightness),
                "edge_density": float(edge_density),
                "texture_smoothness": "높음" if std_brightness < 30 else "낮음"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_with_ollama(self, image_data: Dict[str, Any], gemstone_type: str) -> Dict[str, Any]:
        """🏆 v2 고도화 Ollama AI 보석 분석"""
        if not OLLAMA_AVAILABLE:
            return {"error": "Ollama AI를 사용할 수 없습니다."}
        
        try:
            # 이미지 정보를 텍스트로 변환
            metadata = image_data["metadata"]
            
            # 상세한 이미지 정보 구성
            image_info = f"""
보석 유형: {gemstone_type}
이미지 크기: {metadata.get('size', 'Unknown')}
주요 색상 정보:
"""
            
            # 색상 정보 추가
            for color in metadata.get('dominant_colors', [])[:3]:
                image_info += f"- {color.get('description', 'Unknown')}: {color.get('percentage', 0):.1f}%\n"
            
            # 텍스처 정보 추가
            texture = metadata.get('texture_analysis', {})
            image_info += f"""
텍스처 분석:
- 평균 밝기: {texture.get('mean_brightness', 0):.1f}
- 밝기 변화: {texture.get('brightness_variation', 0):.1f}
- 질감 부드러움: {texture.get('texture_smoothness', 'Unknown')}"""
            
            if OLLAMA_V2_AVAILABLE:
                # v2 인터페이스로 다양한 레벨 분석
                v2_analysis = self.process_gemstone_v2(image_info, gemstone_type)
                return {
                    "ai_analysis_v2": v2_analysis,
                    "best_analysis": v2_analysis.get('best_analysis', '분석 결과 없음'),
                    "analysis_time": datetime.now().isoformat(),
                    "v2_processed": True
                }
            else:
                # 기존 v1 분석
                analysis_prompt = f"""
당신은 세계적인 보석학 전문가입니다. 다음 보석 이미지 정보를 바탕으로 산지를 분석해주세요.

{image_info}

분석해주세요:
1. 가능한 산지 (최대 3곳, 확률 포함)
2. 분석 근거 (색상, 내포물, 기타 특성)
3. 신뢰도 평가 (1-10점)
4. 추가 검증이 필요한 항목
5. 보석학적 권장사항

한국어로 상세히 분석해주세요."""
                
                # Ollama AI 호출
                response = global_ollama.generate_response(
                    analysis_prompt,
                    model=GEMSTONE_MODEL,
                    temperature=0.3,
                    max_tokens=1500
                )
                
                return {
                    "ai_analysis": response,
                    "model_used": GEMSTONE_MODEL,
                    "analysis_time": datetime.now().isoformat(),
                "confidence_score": self.extract_confidence_score(response)
            }
            
        except Exception as e:
            return {"error": f"AI 분석 중 오류: {str(e)}"}
    
    def process_gemstone_v2(self, image_info: str, gemstone_type: str) -> dict:
        """🏆 v2 고도화 보석 분석 - 5개 모델 전략 활용"""
        
        try:
            analysis_results = {}
            
            # 🏆 ULTIMATE 전문가 분석 (GPT-OSS-20B) - 최고 수준 보석학 분석
            try:
                ultimate_result = expert_gemstone(image_info, "research")  # 연구 수준
                analysis_results['ultimate'] = {
                    'title': '🏆 궁극 전문가 분석 (GPT-OSS-20B)',
                    'content': ultimate_result,
                    'model': 'gpt-oss:20b',
                    'tier': 'ULTIMATE',
                    'level': '연구 수준'
                }
            except Exception as e:
                analysis_results['ultimate'] = {
                    'title': '🏆 궁극 전문가 분석 (오류)',
                    'content': f"궁극 분석 실패: {str(e)}",
                    'model': 'gpt-oss:20b',
                    'tier': 'ULTIMATE'
                }
            
            # 🔥 PREMIUM 전문가 분석 (Gemma3-27B) - 고급 보석학 인사이트
            try:
                premium_result = expert_gemstone(image_info, "standard")  # 표준 전문가 수준
                analysis_results['premium'] = {
                    'title': '🔥 프리미엄 전문가 분석 (Gemma3-27B)',
                    'content': premium_result,
                    'model': 'gemma3:27b',
                    'tier': 'PREMIUM',
                    'level': '전문가 수준'
                }
            except Exception as e:
                analysis_results['premium'] = {
                    'title': '🔥 프리미엄 전문가 분석 (오류)',
                    'content': f"프리미엄 분석 실패: {str(e)}",
                    'model': 'gemma3:27b',
                    'tier': 'PREMIUM'
                }
            
            # ⚡ STANDARD 기본 분석 (Qwen3-8B) - 균형잡힌 보석 식별
            try:
                standard_result = expert_gemstone(image_info, "basic")  # 기본 식별 수준
                analysis_results['standard'] = {
                    'title': '⚡ 표준 보석 식별 (Qwen3-8B)',
                    'content': standard_result,
                    'model': 'qwen3:8b',
                    'tier': 'STANDARD',
                    'level': '기본 식별'
                }
            except Exception as e:
                analysis_results['standard'] = {
                    'title': '⚡ 표준 보석 식별 (오류)',
                    'content': f"표준 분석 실패: {str(e)}",
                    'model': 'qwen3:8b',
                    'tier': 'STANDARD'
                }
            
            # 🛡️ STABLE 안정적 분석 (Qwen2.5-7B) - 신뢰할 수 있는 기본 분석
            try:
                stable_result = advanced_ollama.advanced_generate(
                    task_type="gemstone_basic",
                    content=image_info,
                    task_goal=f"{gemstone_type} 보석 안정적 분석",
                    quality_priority=False,
                    speed_priority=False
                )
                analysis_results['stable'] = {
                    'title': '🛡️ 안정적 기본 분석 (Qwen2.5-7B)',
                    'content': stable_result,
                    'model': 'qwen2.5:7b',
                    'tier': 'STABLE',
                    'level': '기본 안정'
                }
            except Exception as e:
                analysis_results['stable'] = {
                    'title': '🛡️ 안정적 기본 분석 (오류)',
                    'content': f"안정 분석 실패: {str(e)}",
                    'model': 'qwen2.5:7b',
                    'tier': 'STABLE'
                }
            
            # 최고 품질 분석 결과 선택
            best_analysis = "분석 결과 없음"
            best_tier = 'none'
            
            for tier in ['ultimate', 'premium', 'standard', 'stable']:
                if tier in analysis_results and 'content' in analysis_results[tier]:
                    content = analysis_results[tier]['content']
                    if content and len(content) > 50 and "실패" not in content:
                        best_analysis = content
                        best_tier = tier
                        break
            
            analysis_results['best_analysis'] = best_analysis
            analysis_results['best_tier'] = best_tier
            analysis_results['v2_processed'] = True
            
            return analysis_results
            
        except Exception as e:
            return {
                'error': {
                    'title': '❌ v2 보석 분석 시스템 오류',
                    'content': f"v2 분석 시스템 오류: {str(e)}",
                    'model': 'none',
                    'tier': 'ERROR'
                },
                'best_analysis': f"v2 분석 실패: {str(e)}",
                'v2_processed': False
            }
    
    def extract_confidence_score(self, analysis_text: str) -> float:
        """AI 분석 결과에서 신뢰도 점수 추출"""
        try:
            # 간단한 패턴 매칭으로 신뢰도 추출
            import re
            
            patterns = [
                r'신뢰도[:\s]*(\d+)[/\s]*10',
                r'확신도[:\s]*(\d+)',
                r'신뢰성[:\s]*(\d+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, analysis_text)
                if match:
                    return float(match.group(1)) / 10.0
            
            # 기본값
            return 0.7
            
        except:
            return 0.5
    
    def generate_detailed_report(self, analysis_data: Dict[str, Any]) -> str:
        """상세 분석 리포트 생성"""
        report = f"""
# 💎 보석 산지 분석 리포트

## 📊 기본 정보
- **분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **보석 유형**: {analysis_data.get('gemstone_type', 'Unknown')}
- **분석 모델**: {analysis_data.get('ai_result', {}).get('model_used', 'N/A')}
- **신뢰도 점수**: {analysis_data.get('ai_result', {}).get('confidence_score', 0):.2f}/1.0

## 🎨 색상 분석
"""
        
        colors = analysis_data.get('image_data', {}).get('metadata', {}).get('dominant_colors', [])
        for i, color in enumerate(colors[:3], 1):
            report += f"**{i}순위**: {color.get('description', 'Unknown')} ({color.get('percentage', 0):.1f}%)\n"
        
        report += f"""
## 🔬 AI 분석 결과
{analysis_data.get('ai_result', {}).get('ai_analysis', 'AI 분석 결과가 없습니다.')}

## 📋 권장사항
- 정확한 감정을 위해서는 전문 감정기관의 정밀 검사를 권장합니다.
- 본 분석은 이미지 기반의 예비 분석으로, 참고용으로만 사용하세요.
- 실제 거래 시에는 공인 감정서를 반드시 확인하시기 바랍니다.

---
*본 리포트는 솔로몬드 AI 보석 분석 시스템에 의해 생성되었습니다.*
        """
        
        return report
    
    def render_upload_interface(self):
        """파일 업로드 인터페이스"""
        st.markdown("## 📤 보석 이미지 업로드")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "보석 이미지를 업로드하세요",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                accept_multiple_files=True,
                help="지원 형식: JPG, PNG, BMP, TIFF"
            )
            
            if uploaded_files:
                st.session_state.uploaded_images = uploaded_files
                st.success(f"✅ {len(uploaded_files)}개 이미지 업로드 완료")
        
        with col2:
            st.markdown("### 💡 업로드 팁")
            st.info("""
            **좋은 이미지 조건:**
            - 밝은 조명 환경
            - 선명한 초점
            - 보석이 중앙에 위치
            - 배경이 단순함
            - 고해상도 (권장)
            """)
    
    def render_gemstone_selection(self):
        """보석 유형 선택"""
        st.markdown("## 💎 보석 유형 선택")
        
        gemstone_options = {
            "diamond": "💎 다이아몬드",
            "ruby": "❤️ 루비", 
            "sapphire": "💙 사파이어",
            "emerald": "💚 에메랄드"
        }
        
        selected_type = st.selectbox(
            "분석할 보석 유형을 선택해주세요",
            options=list(gemstone_options.keys()),
            format_func=lambda x: gemstone_options[x]
        )
        
        # 선택된 보석에 대한 정보 표시
        if selected_type in GEMSTONE_DATABASE:
            regions = list(GEMSTONE_DATABASE[selected_type]["regions"].keys())
            st.info(f"**{gemstone_options[selected_type]}** 분석 가능한 산지: {', '.join(regions)}")
        
        return selected_type
    
    def render_analysis_interface(self):
        """분석 실행 인터페이스"""
        st.markdown("## 🔬 분석 실행")
        
        if not st.session_state.uploaded_images:
            st.warning("먼저 이미지를 업로드해주세요.")
            return
        
        gemstone_type = self.render_gemstone_selection()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 분석 시작", type="primary"):
                self.run_analysis(gemstone_type)
        
        with col2:
            if st.button("🧹 결과 초기화"):
                st.session_state.analysis_results = []
                st.session_state.current_analysis = None
                st.rerun()
        
        with col3:
            st.metric("Ollama AI", "✅ 연결됨" if OLLAMA_AVAILABLE else "❌ 불가능")
    
    def run_analysis(self, gemstone_type: str):
        """분석 실행"""
        if not OLLAMA_AVAILABLE:
            st.error("❌ Ollama AI를 사용할 수 없습니다.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, uploaded_file in enumerate(st.session_state.uploaded_images):
            status_text.text(f"분석 중: {uploaded_file.name}")
            
            try:
                # 이미지 로드
                image = Image.open(uploaded_file)
                
                # 이미지 전처리
                processed_data = self.preprocess_image(image)
                if not processed_data:
                    continue
                
                # AI 분석
                ai_result = self.analyze_with_ollama(processed_data, gemstone_type)
                
                # 결과 저장
                analysis_result = {
                    "filename": uploaded_file.name,
                    "gemstone_type": gemstone_type,
                    "image_data": processed_data,
                    "ai_result": ai_result,
                    "timestamp": datetime.now()
                }
                
                results.append(analysis_result)
                
                # 진행률 업데이트
                progress_bar.progress((i + 1) / len(st.session_state.uploaded_images))
                
            except Exception as e:
                st.error(f"{uploaded_file.name} 분석 중 오류: {str(e)}")
        
        st.session_state.analysis_results = results
        st.session_state.current_analysis = results[-1] if results else None
        
        status_text.text("✅ 분석 완료!")
        st.success(f"총 {len(results)}개 이미지 분석 완료")
    
    def render_results(self):
        """분석 결과 표시"""
        if not st.session_state.analysis_results:
            st.info("아직 분석 결과가 없습니다.")
            return
        
        st.markdown("## 📊 분석 결과")
        
        # 결과 선택
        if len(st.session_state.analysis_results) > 1:
            selected_idx = st.selectbox(
                "결과 선택",
                range(len(st.session_state.analysis_results)),
                format_func=lambda x: st.session_state.analysis_results[x]["filename"]
            )
            current_result = st.session_state.analysis_results[selected_idx]
        else:
            current_result = st.session_state.analysis_results[0]
        
        # 결과 표시
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🖼️ 이미지")
            if current_result["image_data"]["processed_image"]:
                st.image(
                    current_result["image_data"]["processed_image"],
                    caption=current_result["filename"]
                )
            
            # 기본 정보
            st.markdown("### 📋 기본 정보")
            st.write(f"**파일명**: {current_result['filename']}")
            st.write(f"**보석 유형**: {current_result['gemstone_type']}")
            st.write(f"**신뢰도**: {current_result['ai_result'].get('confidence_score', 0):.2f}")
        
        with col2:
            st.markdown("### 🤖 AI 분석 결과")
            ai_analysis = current_result["ai_result"].get("ai_analysis", "분석 결과가 없습니다.")
            st.markdown(ai_analysis)
            
            # 상세 리포트 생성
            if st.button("📄 상세 리포트 생성"):
                detailed_report = self.generate_detailed_report(current_result)
                st.markdown(detailed_report)
                
                # 다운로드 버튼
                st.download_button(
                    "📥 리포트 다운로드",
                    detailed_report,
                    file_name=f"gemstone_analysis_{current_result['filename']}.md",
                    mime="text/markdown"
                )
    
    def render_sidebar(self):
        """사이드바"""
        with st.sidebar:
            st.markdown("## ⚙️ 설정")
            
            st.markdown("### 💎 지원 보석")
            st.info("""
            - 💎 다이아몬드
            - ❤️ 루비
            - 💙 사파이어  
            - 💚 에메랄드
            """)
            
            st.markdown("### 🌍 분석 가능 산지")
            for gemstone, data in GEMSTONE_DATABASE.items():
                regions = list(data["regions"].keys())
                st.write(f"**{gemstone.title()}**: {', '.join(regions)}")
            
            st.markdown("### 📊 통계")
            st.metric("분석 완료", len(st.session_state.analysis_results))
            st.metric("업로드된 이미지", len(st.session_state.uploaded_images))
            
            if st.button("🏠 메인 대시보드로"):
                st.markdown("메인 대시보드: http://localhost:8505")
    
    def run(self):
        """모듈 실행"""
        st.markdown("# 💎 보석 산지 분석 시스템")
        st.markdown("AI 기반 보석 이미지 분석 및 원산지 추정")
        
        self.render_sidebar()
        
        st.markdown("---")
        
        # 탭 구성
        tab1, tab2, tab3 = st.tabs(["📤 업로드", "🔬 분석", "📊 결과"])
        
        with tab1:
            self.render_upload_interface()
        
        with tab2:
            self.render_analysis_interface()
        
        with tab3:
            self.render_results()

def main():
    """메인 함수"""
    analyzer = GemstoneSrcAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()