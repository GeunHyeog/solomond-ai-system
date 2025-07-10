"""
올인원 주얼리 AI 분석 시스템
모든 파일 형식을 동시에 업로드하고 통합 분석하여 주얼리 강의/회의 내용을 완전히 파악

실행 방법:
streamlit run ultimate_jewelry_analyzer.py
"""

import streamlit as st
import asyncio
import os
import tempfile
import json
from pathlib import Path
import time
from datetime import datetime
import base64
import io
from typing import List, Dict, Any
import zipfile

# 필수 라이브러리 import
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    st.error("Whisper가 설치되지 않았습니다. pip install openai-whisper")

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    st.warning("OCR 기능이 제한됩니다. pip install pillow pytesseract")

try:
    import cv2
    import moviepy.editor as mp
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False
    st.warning("비디오 처리가 제한됩니다.")

# Streamlit 페이지 설정
st.set_page_config(
    page_title="💎 솔로몬드 올인원 주얼리 AI 분석",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

class UltimateJewelryAnalyzer:
    """올인원 주얼리 분석 시스템"""
    
    def __init__(self):
        self.whisper_model = None
        self.results = {}
        
        # 주얼리 키워드 데이터베이스
        self.jewelry_keywords = {
            'diamonds': ['다이아몬드', 'diamond', '다이아', 'brilliant', '브릴리언트'],
            'gemstones': ['루비', 'ruby', '사파이어', 'sapphire', '에메랄드', 'emerald'],
            '4c': ['캐럿', 'carat', '컬러', 'color', '클래리티', 'clarity', '컷', 'cut'],
            'certification': ['GIA', 'SSEF', 'Gübelin', '감정서', 'certificate'],
            'jewelry_types': ['반지', 'ring', '목걸이', 'necklace', '귀걸이', 'earring'],
            'business': ['가격', 'price', '할인', 'discount', '투자', 'investment'],
            'technical': ['형광', 'fluorescence', '인클루전', 'inclusion', '처리', 'treatment']
        }
    
    def load_whisper_model(self):
        """Whisper 모델 로드"""
        if not WHISPER_AVAILABLE:
            return False
            
        if self.whisper_model is None:
            try:
                with st.spinner("🎤 Whisper 음성 인식 모델 로딩 중..."):
                    self.whisper_model = whisper.load_model("base")
                st.success("✅ Whisper 모델 로드 완료")
                return True
            except Exception as e:
                st.error(f"Whisper 로드 실패: {e}")
                return False
        return True
    
    def analyze_audio(self, audio_file) -> Dict[str, Any]:
        """음성 파일 분석"""
        if not self.load_whisper_model():
            return {"error": "Whisper 모델을 로드할 수 없습니다"}
        
        try:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_file.read())
                temp_path = temp_file.name
            
            # Whisper로 음성 인식
            with st.spinner(f"🎤 음성 인식 중... ({audio_file.name})"):
                result = self.whisper_model.transcribe(temp_path)
            
            # 임시 파일 삭제
            os.unlink(temp_path)
            
            # 주얼리 키워드 분석
            text = result.get('text', '')
            jewelry_score = self.calculate_jewelry_relevance(text)
            
            return {
                'type': 'audio',
                'filename': audio_file.name,
                'text': text,
                'language': result.get('language', 'unknown'),
                'jewelry_score': jewelry_score,
                'keywords_found': self.extract_jewelry_keywords(text),
                'confidence': 0.9 if len(text) > 50 else 0.7
            }
            
        except Exception as e:
            return {"error": f"음성 분석 실패: {str(e)}"}
    
    def analyze_video(self, video_file) -> Dict[str, Any]:
        """비디오 파일 분석 (음성 추출 + STT)"""
        if not VIDEO_AVAILABLE:
            return {"error": "비디오 처리 라이브러리가 설치되지 않았습니다"}
        
        try:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(video_file.read())
                temp_path = temp_file.name
            
            with st.spinner(f"🎬 비디오에서 음성 추출 중... ({video_file.name})"):
                # MoviePy로 음성 추출
                video = mp.VideoFileClip(temp_path)
                audio_path = temp_path.replace('.mp4', '.wav')
                video.audio.write_audiofile(audio_path, verbose=False, logger=None)
                video.close()
            
            # 추출된 음성을 Whisper로 분석
            if self.load_whisper_model():
                with st.spinner("🎤 추출된 음성 인식 중..."):
                    result = self.whisper_model.transcribe(audio_path)
                
                text = result.get('text', '')
                jewelry_score = self.calculate_jewelry_relevance(text)
                
                # 임시 파일들 삭제
                os.unlink(temp_path)
                os.unlink(audio_path)
                
                return {
                    'type': 'video',
                    'filename': video_file.name,
                    'text': text,
                    'language': result.get('language', 'unknown'),
                    'jewelry_score': jewelry_score,
                    'keywords_found': self.extract_jewelry_keywords(text),
                    'confidence': 0.85
                }
            else:
                os.unlink(temp_path)
                return {"error": "Whisper 모델을 로드할 수 없습니다"}
                
        except Exception as e:
            return {"error": f"비디오 분석 실패: {str(e)}"}
    
    def analyze_image(self, image_file) -> Dict[str, Any]:
        """이미지 파일 분석 (OCR)"""
        if not OCR_AVAILABLE:
            return {"error": "OCR 라이브러리가 설치되지 않았습니다"}
        
        try:
            # PIL로 이미지 열기
            image = Image.open(image_file)
            
            with st.spinner(f"🖼️ 이미지 텍스트 추출 중... ({image_file.name})"):
                # OCR로 텍스트 추출
                text = pytesseract.image_to_string(image, lang='kor+eng')
            
            if not text.strip():
                return {
                    'type': 'image',
                    'filename': image_file.name,
                    'text': '',
                    'jewelry_score': 0.0,
                    'keywords_found': [],
                    'confidence': 0.3,
                    'note': '이미지에서 텍스트를 찾을 수 없습니다'
                }
            
            jewelry_score = self.calculate_jewelry_relevance(text)
            
            return {
                'type': 'image',
                'filename': image_file.name,
                'text': text.strip(),
                'jewelry_score': jewelry_score,
                'keywords_found': self.extract_jewelry_keywords(text),
                'confidence': 0.7 if len(text.strip()) > 20 else 0.5
            }
            
        except Exception as e:
            return {"error": f"이미지 분석 실패: {str(e)}"}
    
    def analyze_document(self, doc_file) -> Dict[str, Any]:
        """문서 파일 분석"""
        try:
            # 파일 확장자에 따른 처리
            file_ext = Path(doc_file.name).suffix.lower()
            
            if file_ext == '.txt':
                text = doc_file.read().decode('utf-8')
            elif file_ext == '.pdf':
                try:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(doc_file.read()))
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                except ImportError:
                    return {"error": "PDF 처리를 위해 PyPDF2를 설치해주세요"}
            else:
                text = str(doc_file.read())
            
            jewelry_score = self.calculate_jewelry_relevance(text)
            
            return {
                'type': 'document',
                'filename': doc_file.name,
                'text': text[:2000] + "..." if len(text) > 2000 else text,
                'full_text': text,
                'jewelry_score': jewelry_score,
                'keywords_found': self.extract_jewelry_keywords(text),
                'confidence': 0.9
            }
            
        except Exception as e:
            return {"error": f"문서 분석 실패: {str(e)}"}
    
    def calculate_jewelry_relevance(self, text: str) -> float:
        """주얼리 관련성 점수 계산"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        total_keywords = 0
        found_keywords = 0
        
        for category, keywords in self.jewelry_keywords.items():
            for keyword in keywords:
                total_keywords += 1
                if keyword.lower() in text_lower:
                    found_keywords += 1
        
        return min(1.0, found_keywords / max(1, total_keywords) * 5)
    
    def extract_jewelry_keywords(self, text: str) -> List[str]:
        """텍스트에서 주얼리 키워드 추출"""
        if not text:
            return []
        
        text_lower = text.lower()
        found_keywords = []
        
        for category, keywords in self.jewelry_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    found_keywords.append(keyword)
        
        return list(set(found_keywords))
    
    def generate_comprehensive_analysis(self, all_results: List[Dict]) -> Dict[str, Any]:
        """모든 분석 결과를 통합하여 종합 분석 생성"""
        
        # 유효한 결과만 필터링
        valid_results = [r for r in all_results if 'error' not in r and r.get('text')]
        
        if not valid_results:
            return {
                'status': 'error',
                'message': '분석 가능한 파일이 없습니다'
            }
        
        # 전체 텍스트 통합
        all_text = ""
        for result in valid_results:
            full_text = result.get('full_text', result.get('text', ''))
            all_text += f"\n\n=== {result['filename']} ===\n{full_text}"
        
        # 주얼리 키워드 종합
        all_keywords = []
        total_jewelry_score = 0
        
        for result in valid_results:
            all_keywords.extend(result.get('keywords_found', []))
            total_jewelry_score += result.get('jewelry_score', 0)
        
        unique_keywords = list(set(all_keywords))
        avg_jewelry_score = total_jewelry_score / len(valid_results)
        
        # 내용 분석 및 요약
        content_analysis = self.analyze_content_theme(all_text, unique_keywords)
        
        # 파일 타입별 통계
        file_types = {}
        for result in valid_results:
            file_type = result['type']
            if file_type not in file_types:
                file_types[file_type] = 0
            file_types[file_type] += 1
        
        return {
            'status': 'success',
            'summary': {
                'total_files': len(all_results),
                'analyzed_files': len(valid_results),
                'file_types': file_types,
                'jewelry_relevance': avg_jewelry_score,
                'keywords_found': unique_keywords[:20],  # 상위 20개
                'content_theme': content_analysis
            },
            'detailed_analysis': {
                'main_topic': content_analysis.get('main_topic', '주얼리 관련 내용'),
                'key_points': content_analysis.get('key_points', []),
                'session_type': content_analysis.get('session_type', '미확인'),
                'target_audience': content_analysis.get('target_audience', '일반'),
                'confidence': min(1.0, avg_jewelry_score + 0.3)
            },
            'recommendations': self.generate_recommendations(content_analysis, unique_keywords)
        }
    
    def analyze_content_theme(self, text: str, keywords: List[str]) -> Dict[str, Any]:
        """텍스트 내용 분석으로 주제 파악"""
        
        # 키워드 기반 주제 분류
        diamond_keywords = ['다이아몬드', 'diamond', '4c', 'gia', 'brilliant']
        gemstone_keywords = ['루비', 'sapphire', 'emerald', '유색보석']
        business_keywords = ['가격', 'price', '투자', 'investment', '시장']
        technical_keywords = ['감정', 'certificate', '처리', 'treatment']
        
        diamond_count = sum(1 for k in keywords if any(d in k.lower() for d in diamond_keywords))
        gemstone_count = sum(1 for k in keywords if any(g in k.lower() for g in gemstone_keywords))
        business_count = sum(1 for k in keywords if any(b in k.lower() for b in business_keywords))
        technical_count = sum(1 for k in keywords if any(t in k.lower() for t in technical_keywords))
        
        # 주제 결정
        if diamond_count > 2:
            main_topic = "다이아몬드 전문 교육"
            session_type = "다이아몬드 강의"
        elif gemstone_count > 2:
            main_topic = "유색보석 교육"
            session_type = "유색보석 세미나"
        elif business_count > 2:
            main_topic = "주얼리 비즈니스"
            session_type = "비즈니스 미팅"
        elif technical_count > 2:
            main_topic = "보석 감정 기술"
            session_type = "기술 교육"
        else:
            main_topic = "종합 주얼리 교육"
            session_type = "일반 세미나"
        
        # 키 포인트 추출 (키워드 기반)
        key_points = []
        if diamond_count > 0:
            key_points.append("다이아몬드 4C 등급 체계")
        if gemstone_count > 0:
            key_points.append("유색보석 품질 평가")
        if business_count > 0:
            key_points.append("주얼리 시장 동향")
        if technical_count > 0:
            key_points.append("보석 감정 기술")
        
        # 대상 청중 추정
        if technical_count > business_count:
            target_audience = "전문가/감정사"
        elif business_count > technical_count:
            target_audience = "업계 종사자"
        else:
            target_audience = "일반 교육생"
        
        return {
            'main_topic': main_topic,
            'session_type': session_type,
            'key_points': key_points,
            'target_audience': target_audience,
            'keyword_distribution': {
                'diamond': diamond_count,
                'gemstone': gemstone_count,
                'business': business_count,
                'technical': technical_count
            }
        }
    
    def generate_recommendations(self, content_analysis: Dict, keywords: List[str]) -> List[str]:
        """분석 결과 기반 권장사항 생성"""
        recommendations = []
        
        session_type = content_analysis.get('session_type', '')
        
        if '다이아몬드' in session_type:
            recommendations.append("💎 GIA 다이아몬드 그레이딩 심화 과정 추천")
            recommendations.append("📊 다이아몬드 4C 실습 교육 확대")
        
        if '유색보석' in session_type:
            recommendations.append("🔴 루비/사파이어 origin 판별 교육")
            recommendations.append("💚 에메랄드 처리 기술 세미나")
        
        if '비즈니스' in session_type:
            recommendations.append("📈 주얼리 시장 트렌드 분석 보고서")
            recommendations.append("💰 투자 가치 평가 방법론 교육")
        
        if '기술' in session_type:
            recommendations.append("🔬 최신 감정 장비 사용법 교육")
            recommendations.append("🎯 품질 평가 표준화 과정")
        
        if not recommendations:
            recommendations.append("📚 주얼리 기초 교육 과정 추천")
            recommendations.append("🌟 업계 네트워킹 세미나 참여")
        
        return recommendations

def main():
    st.title("💎 솔로몬드 올인원 주얼리 AI 분석 시스템")
    st.markdown("### 🚀 모든 자료를 업로드하고 통합 분석으로 주얼리 강의/회의 내용을 완전히 파악하세요!")
    
    # 분석기 초기화
    analyzer = UltimateJewelryAnalyzer()
    
    # 사이드바 - 시스템 상태
    with st.sidebar:
        st.header("🔧 시스템 상태")
        st.write(f"🎤 Whisper STT: {'✅' if WHISPER_AVAILABLE else '❌'}")
        st.write(f"🖼️ OCR: {'✅' if OCR_AVAILABLE else '❌'}")
        st.write(f"🎬 비디오: {'✅' if VIDEO_AVAILABLE else '❌'}")
        
        st.markdown("---")
        st.header("📋 지원 파일 형식")
        st.write("🎤 **음성**: MP3, WAV, M4A")
        st.write("🎬 **비디오**: MP4, MOV, AVI")
        st.write("🖼️ **이미지**: JPG, PNG, GIF")
        st.write("📄 **문서**: TXT, PDF")
        
        st.markdown("---")
        st.header("🎯 분석 기능")
        st.write("• 음성→텍스트 변환")
        st.write("• 비디오→음성 추출→텍스트")
        st.write("• 이미지→OCR→텍스트")
        st.write("• 문서 텍스트 추출")
        st.write("• 주얼리 키워드 분석")
        st.write("• **통합 결론 도출**")
    
    # 메인 영역
    st.markdown("## 📤 파일 업로드 및 분석")
    
    # 파일 업로더
    uploaded_files = st.file_uploader(
        "분석할 파일들을 모두 선택하세요 (여러 파일 동시 선택 가능)",
        type=['mp3', 'wav', 'm4a', 'mp4', 'mov', 'avi', 'jpg', 'jpeg', 'png', 'gif', 'txt', 'pdf'],
        accept_multiple_files=True,
        help="Ctrl(Cmd) + 클릭으로 여러 파일을 동시에 선택할 수 있습니다"
    )
    
    if uploaded_files:
        st.success(f"📁 {len(uploaded_files)}개 파일이 업로드되었습니다!")
        
        # 업로드된 파일 목록 표시
        with st.expander("📋 업로드된 파일 목록", expanded=True):
            for i, file in enumerate(uploaded_files, 1):
                file_size = len(file.getvalue()) / (1024*1024)  # MB
                st.write(f"{i}. **{file.name}** ({file_size:.2f}MB)")
        
        # 분석 시작 버튼
        if st.button("🚀 **통합 분석 시작**", type="primary"):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            all_results = []
            
            # 각 파일 분석
            for i, file in enumerate(uploaded_files):
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"분석 중... ({i+1}/{len(uploaded_files)}) {file.name}")
                
                # 파일 확장자에 따른 분석
                file_ext = Path(file.name).suffix.lower()
                
                if file_ext in ['.mp3', '.wav', '.m4a']:
                    result = analyzer.analyze_audio(file)
                elif file_ext in ['.mp4', '.mov', '.avi']:
                    result = analyzer.analyze_video(file)
                elif file_ext in ['.jpg', '.jpeg', '.png', '.gif']:
                    result = analyzer.analyze_image(file)
                elif file_ext in ['.txt', '.pdf']:
                    result = analyzer.analyze_document(file)
                else:
                    result = {"error": f"지원하지 않는 파일 형식: {file_ext}"}
                
                all_results.append(result)
            
            progress_bar.progress(1.0)
            status_text.text("✅ 모든 파일 분석 완료! 통합 결과 생성 중...")
            
            # 통합 분석 수행
            comprehensive_analysis = analyzer.generate_comprehensive_analysis(all_results)
            
            # 결과 표시
            st.markdown("---")
            st.markdown("## 📊 **통합 분석 결과**")
            
            if comprehensive_analysis['status'] == 'success':
                summary = comprehensive_analysis['summary']
                analysis = comprehensive_analysis['detailed_analysis']
                
                # 요약 정보
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("총 파일 수", summary['total_files'])
                with col2:
                    st.metric("분석 성공", summary['analyzed_files'])
                with col3:
                    st.metric("주얼리 관련성", f"{summary['jewelry_relevance']:.1%}")
                with col4:
                    st.metric("분석 신뢰도", f"{analysis['confidence']:.1%}")
                
                # 핵심 결론
                st.markdown("### 🎯 **핵심 결론**")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info(f"""
                    **📚 주제**: {analysis['main_topic']}
                    
                    **🎪 세션 유형**: {analysis['session_type']}
                    
                    **👥 대상 청중**: {analysis['target_audience']}
                    """)
                
                with col2:
                    st.markdown("**📝 주요 키워드**")
                    keywords_text = ", ".join(summary['keywords_found'][:10])
                    st.write(keywords_text)
                
                # 핵심 포인트
                if analysis['key_points']:
                    st.markdown("### 🔑 **핵심 포인트**")
                    for point in analysis['key_points']:
                        st.write(f"• {point}")
                
                # 권장사항
                recommendations = comprehensive_analysis['recommendations']
                if recommendations:
                    st.markdown("### 💡 **권장사항**")
                    for rec in recommendations:
                        st.write(f"• {rec}")
                
                # 개별 파일 분석 결과
                with st.expander("📄 개별 파일 분석 상세 결과", expanded=False):
                    for result in all_results:
                        if 'error' not in result:
                            st.markdown(f"**📁 {result['filename']}**")
                            st.write(f"- 타입: {result['type']}")
                            st.write(f"- 주얼리 관련성: {result.get('jewelry_score', 0):.1%}")
                            st.write(f"- 추출 텍스트: {result.get('text', '')[:200]}...")
                            st.markdown("---")
                        else:
                            st.error(f"❌ {result.get('filename', '알 수 없는 파일')}: {result['error']}")
                
                # 결과 다운로드
                result_json = json.dumps(comprehensive_analysis, ensure_ascii=False, indent=2)
                st.download_button(
                    "📥 분석 결과 다운로드 (JSON)",
                    result_json,
                    f"jewelry_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
                
            else:
                st.error(f"❌ 분석 실패: {comprehensive_analysis['message']}")
            
            status_text.text("🎉 통합 분석 완료!")
    
    else:
        st.info("👆 위에서 분석할 파일들을 선택해주세요!")
        
        # 사용법 안내
        st.markdown("### 📖 사용법")
        st.markdown("""
        1. **파일 업로드**: 음성, 영상, 이미지, 문서 파일들을 모두 선택
        2. **통합 분석**: 모든 파일을 AI가 분석하여 내용 추출
        3. **결론 도출**: 주얼리 강의/회의 내용을 종합적으로 파악
        4. **결과 활용**: 세미나 요약, 학습 포인트, 후속 조치 등에 활용
        """)
        
        st.markdown("### 🎯 **이런 분석이 가능합니다**")
        st.markdown("""
        - 🎤 **음성 강의** → 전체 내용 텍스트화 및 핵심 포인트 추출
        - 🎬 **세미나 영상** → 음성 추출 후 주요 내용 분석
        - 📸 **발표 자료 사진** → OCR로 텍스트 추출 및 내용 보완
        - 📄 **배포 자료** → 문서 내용과 강의 내용 매칭
        - 🔗 **모든 자료 통합** → 완전한 세미나/회의 내용 파악
        """)

if __name__ == "__main__":
    main()
