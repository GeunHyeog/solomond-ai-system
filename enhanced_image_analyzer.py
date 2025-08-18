# -*- coding: utf-8 -*-
"""
향상된 이미지 분석기 - OCR + AI 비전 분석 통합
기존 텍스트 추출에 이미지 내용 분석, 객체 인식, 상황 파악 추가
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64
import io

# 프로젝트 경로 추가  
sys.path.append(str(Path(__file__).parent))

class EnhancedImageAnalyzer:
    """OCR + AI 비전 분석 통합 시스템"""
    
    def __init__(self):
        self.ocr_available = False
        self.vision_models = {
            'gemini': False,
            'openai': False,
            'claude': False
        }
        
        # 사용 가능한 도구들 확인
        self._check_available_tools()
        
    def _check_available_tools(self):
        """사용 가능한 분석 도구 확인"""
        
        print("🔍 이미지 분석 도구 확인 중...")
        
        # EasyOCR 확인
        try:
            import easyocr
            self.ocr_available = True
            print("  ✅ EasyOCR (텍스트 추출)")
        except ImportError:
            print("  ❌ EasyOCR 불가능")
        
        # Gemini Vision 확인
        try:
            import google.generativeai as genai
            # API 키가 없어도 모듈은 로드 가능
            self.vision_models['gemini'] = True
            print("  ✅ Gemini Vision (이미지 분석)")
        except ImportError:
            print("  ❌ Gemini Vision 불가능")
        
        # OpenAI Vision 확인
        try:
            import openai
            self.vision_models['openai'] = True
            print("  ✅ OpenAI GPT-4V (이미지 분석)")
        except ImportError:
            print("  ❌ OpenAI Vision 불가능")
        
        # PIL 확인 (기본 이미지 처리)
        try:
            from PIL import Image
            print("  ✅ PIL (이미지 처리)")
        except ImportError:
            print("  ❌ PIL 불가능")
    
    def analyze_image_comprehensive(self, image_path: str) -> Dict[str, Any]:
        """이미지 종합 분석 - OCR + 비전 분석"""
        
        print(f"\n🖼️ 종합 이미지 분석: {os.path.basename(image_path)}")
        
        if not os.path.exists(image_path):
            return {'error': '파일이 존재하지 않습니다'}
        
        results = {
            'filename': os.path.basename(image_path),
            'file_size': os.path.getsize(image_path),
            'ocr_text': '',
            'visual_analysis': '',
            'combined_insights': '',
            'analysis_methods': []
        }
        
        # 1. OCR 텍스트 추출
        if self.ocr_available:
            print("  📝 OCR 텍스트 추출 중...")
            ocr_result = self._extract_text_ocr(image_path)
            if ocr_result:
                results['ocr_text'] = ocr_result
                results['analysis_methods'].append('OCR')
                print(f"    추출: {len(ocr_result)}글자")
        
        # 2. AI 비전 분석
        vision_result = self._analyze_with_vision_ai(image_path)
        if vision_result:
            results['visual_analysis'] = vision_result
            results['analysis_methods'].append('Vision AI')
            print(f"    분석: {len(vision_result)}글자")
        
        # 3. 결합된 인사이트 생성
        if results['ocr_text'] or results['visual_analysis']:
            combined = self._create_combined_insights(results)
            results['combined_insights'] = combined
            print(f"    통합: {len(combined)}글자")
        
        return results
    
    def _extract_text_ocr(self, image_path: str) -> Optional[str]:
        """EasyOCR로 텍스트 추출"""
        
        try:
            import easyocr
            reader = easyocr.Reader(['ko', 'en'])
            
            results = reader.readtext(image_path)
            
            if results:
                # 신뢰도 0.5 이상인 텍스트만 사용
                texts = [item[1] for item in results if item[2] > 0.5]
                return ' '.join(texts)
            
            return None
            
        except Exception as e:
            print(f"    OCR 오류: {str(e)}")
            return None
    
    def _analyze_with_vision_ai(self, image_path: str) -> Optional[str]:
        """AI 비전 모델로 이미지 분석"""
        
        # 간단한 비전 분석 (Gemini가 있다면 사용)
        if self.vision_models['gemini']:
            return self._analyze_with_gemini(image_path)
        
        # 다른 모델들도 가능하면 추가
        return self._analyze_with_basic_vision(image_path)
    
    def _analyze_with_gemini(self, image_path: str) -> Optional[str]:
        """Gemini Vision으로 이미지 분석"""
        
        try:
            import google.generativeai as genai
            from PIL import Image
            
            # API 키 확인 (환경변수에서)
            api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                print("    Gemini API 키 없음, 기본 분석 사용")
                return None
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro-vision')
            
            # 이미지 로드
            image = Image.open(image_path)
            
            # 분석 프롬프트
            prompt = """이 이미지를 분석해주세요:
1. 이미지에 무엇이 보이나요?
2. 주요 객체나 사람은 무엇인가요?
3. 전체적인 상황이나 맥락은 무엇인가요?
4. 특별히 주목할 만한 것이 있나요?

간결하고 명확하게 한국어로 답변해주세요."""
            
            response = model.generate_content([prompt, image])
            
            if response and response.text:
                return response.text.strip()
            
            return None
            
        except Exception as e:
            print(f"    Gemini 분석 오류: {str(e)}")
            return None
    
    def _analyze_with_basic_vision(self, image_path: str) -> Optional[str]:
        """기본적인 이미지 속성 분석"""
        
        try:
            from PIL import Image
            import numpy as np
            
            image = Image.open(image_path)
            
            # 기본 정보 추출
            width, height = image.size
            mode = image.mode
            
            analysis = []
            analysis.append(f"이미지 크기: {width}x{height}")
            analysis.append(f"색상 모드: {mode}")
            
            # 색상 분석
            if mode == 'RGB':
                # 평균 색상 계산
                np_image = np.array(image)
                avg_color = np.mean(np_image, axis=(0, 1))
                
                if avg_color[0] > 200 and avg_color[1] > 200 and avg_color[2] > 200:
                    analysis.append("전체적으로 밝은 이미지")
                elif avg_color[0] < 100 and avg_color[1] < 100 and avg_color[2] < 100:
                    analysis.append("전체적으로 어두운 이미지")
                else:
                    analysis.append("중간 밝기의 이미지")
            
            # 화면비 분석
            aspect_ratio = width / height
            if aspect_ratio > 1.5:
                analysis.append("가로형 이미지 (풍경/스크린샷 가능성)")
            elif aspect_ratio < 0.7:
                analysis.append("세로형 이미지 (모바일 화면 가능성)")
            else:
                analysis.append("정방형에 가까운 이미지")
            
            return ' | '.join(analysis)
            
        except Exception as e:
            print(f"    기본 분석 오류: {str(e)}")
            return None
    
    def _create_combined_insights(self, results: Dict[str, Any]) -> str:
        """OCR 텍스트와 비전 분석을 결합한 인사이트 생성"""
        
        insights = []
        
        ocr_text = results.get('ocr_text', '')
        visual_analysis = results.get('visual_analysis', '')
        
        # 기본 정보
        if ocr_text and visual_analysis:
            insights.append("📋 텍스트와 시각적 요소가 모두 포함된 이미지입니다.")
        elif ocr_text:
            insights.append("📝 주로 텍스트 기반 이미지입니다.")
        elif visual_analysis:
            insights.append("🖼️ 주로 시각적 요소 중심의 이미지입니다.")
        
        # 내용 기반 인사이트
        combined_content = f"{ocr_text} {visual_analysis}".lower()
        
        if any(keyword in combined_content for keyword in ['2025', '날짜', 'date', 'thu', 'pm']):
            insights.append("📅 날짜/시간 정보가 포함되어 있습니다.")
        
        if any(keyword in combined_content for keyword in ['rise', 'eco', '상승', '환경']):
            insights.append("📈 성장이나 환경과 관련된 내용일 수 있습니다.")
        
        if any(keyword in combined_content for keyword in ['global', '글로벌', '국제', 'cultura']):
            insights.append("🌍 국제적이거나 문화적 내용이 포함되어 있습니다.")
        
        # 길이 기반 분석
        total_length = len(ocr_text) + len(visual_analysis)
        if total_length > 500:
            insights.append("📊 상당한 양의 정보가 담긴 이미지입니다.")
        
        if not insights:
            insights.append("💭 이미지에서 다양한 정보를 추출했습니다.")
        
        return ' '.join(insights)
    
    def analyze_multiple_images(self, image_folder: str) -> List[Dict[str, Any]]:
        """폴더의 모든 이미지 일괄 분석"""
        
        print(f"\n📁 폴더 일괄 분석: {image_folder}")
        
        if not os.path.exists(image_folder):
            print("❌ 폴더가 존재하지 않습니다.")
            return []
        
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_files = []
        
        for file in os.listdir(image_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_folder, file))
        
        if not image_files:
            print("❌ 이미지 파일이 없습니다.")
            return []
        
        print(f"📊 {len(image_files)}개 이미지 발견")
        
        results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 분석 중...")
            result = self.analyze_image_comprehensive(image_path)
            results.append(result)
        
        return results

def main():
    """메인 실행"""
    
    print("=== 향상된 이미지 분석기 ===")
    print("OCR + AI 비전 분석으로 이미지를 종합적으로 분석합니다.")
    
    try:
        analyzer = EnhancedImageAnalyzer()
        
        # user_files/images 폴더 분석
        image_folder = "user_files/images"
        
        if os.path.exists(image_folder):
            results = analyzer.analyze_multiple_images(image_folder)
            
            if results:
                print(f"\n" + "="*60)
                print("📊 종합 분석 결과")
                print("="*60)
                
                total_ocr_chars = 0
                total_vision_chars = 0
                
                for result in results:
                    print(f"\n📁 {result['filename']}")
                    print(f"   크기: {result['file_size']//1024}KB")
                    print(f"   방법: {', '.join(result['analysis_methods'])}")
                    
                    if result['ocr_text']:
                        ocr_len = len(result['ocr_text'])
                        total_ocr_chars += ocr_len
                        print(f"   📝 OCR: {ocr_len}글자")
                        if ocr_len <= 100:
                            print(f"      내용: {result['ocr_text']}")
                        else:
                            print(f"      내용: {result['ocr_text'][:100]}...")
                    
                    if result['visual_analysis']:
                        vision_len = len(result['visual_analysis'])
                        total_vision_chars += vision_len
                        print(f"   🔍 비전: {vision_len}글자")
                        if vision_len <= 100:
                            print(f"      분석: {result['visual_analysis']}")
                        else:
                            print(f"      분석: {result['visual_analysis'][:100]}...")
                    
                    if result['combined_insights']:
                        print(f"   💡 인사이트: {result['combined_insights']}")
                
                print(f"\n📈 전체 통계:")
                print(f"   분석 이미지: {len(results)}개")
                print(f"   총 OCR 텍스트: {total_ocr_chars}글자")
                print(f"   총 비전 분석: {total_vision_chars}글자")
                
            else:
                print("\n❌ 분석할 이미지가 없습니다.")
        
        else:
            print(f"\n❌ {image_folder} 폴더가 없습니다.")
            print("user_files/images/ 폴더에 이미지를 넣어주세요.")
    
    except KeyboardInterrupt:
        print("\n⏹️ 사용자가 중단했습니다.")
    except Exception as e:
        print(f"\n💥 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()