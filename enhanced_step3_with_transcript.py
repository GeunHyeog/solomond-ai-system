#!/usr/bin/env python3
"""
3단계 향상: 음성 스크립트 표시 기능 추가
- 분석 진행 + 중간 검토 + 스크립트 확인
- 사용자가 내용을 검토하고 수정할 수 있는 기능
"""

import streamlit as st
from typing import Dict, List, Any, Optional
import time
import json

def render_step3_enhanced_review(self):
    """3단계: 분석 진행, 스크립트 표시, 중간 검토 (향상됨)"""
    
    st.markdown("## 3️⃣ 분석 진행 및 스크립트 검토")
    
    if not st.session_state.uploaded_files_data:
        st.error("업로드된 파일이 없습니다. 이전 단계로 돌아가세요.")
        return
    
    # 진행 단계 표시
    progress_steps = ["분석 실행", "스크립트 추출", "내용 검토", "최종 확인"]
    current_step = st.session_state.get('step3_progress', 0)
    
    # 프로그레스 바
    progress_cols = st.columns(4)
    for i, step_name in enumerate(progress_steps):
        with progress_cols[i]:
            if i < current_step:
                st.success(f"✅ {step_name}")
            elif i == current_step:
                st.info(f"🔄 {step_name}")
            else:
                st.write(f"⏳ {step_name}")
    
    st.markdown("---")
    
    # 단계별 진행
    if current_step == 0:
        render_analysis_execution(self)
    elif current_step == 1:
        render_transcript_extraction(self)
    elif current_step == 2:
        render_content_review(self)
    elif current_step == 3:
        render_final_confirmation(self)

def render_analysis_execution(self):
    """분석 실행 단계"""
    
    st.markdown("### 🔄 분석 진행 상황")
    
    uploaded_data = st.session_state.uploaded_files_data
    
    # 분석 실행 전 시스템 상태 체크
    analysis_ready, dependency_status = self.check_analysis_readiness()
    
    if not analysis_ready:
        st.error("🚨 분석 시스템 준비 불완료")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**필수 의존성:**")
            for dep, status in dependency_status.items():
                icon = "✅" if status else "❌"
                st.write(f"{icon} {dep}")
        
        with col2:
            if st.button("🔄 시스템 상태 재확인"):
                st.rerun()
        return
    
    # 분석 실행
    if st.button("🚀 분석 시작", type="primary"):
        with st.spinner("분석을 실행 중입니다..."):
            try:
                # 실제 분석 실행
                analysis_results = execute_comprehensive_analysis(uploaded_data)
                st.session_state.analysis_results = analysis_results
                st.session_state.step3_progress = 1
                st.success("✅ 분석 완료!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ 분석 실패: {str(e)}")
    
    # 이미 분석된 경우
    if 'analysis_results' in st.session_state:
        st.success("✅ 분석이 완료되었습니다!")
        if st.button("📝 스크립트 확인하기"):
            st.session_state.step3_progress = 1
            st.rerun()

def render_transcript_extraction(self):
    """스크립트 추출 및 표시 단계"""
    
    st.markdown("### 📝 음성 스크립트 확인")
    
    analysis_results = st.session_state.get('analysis_results', {})
    
    # 오디오 파일별 스크립트 추출
    audio_transcripts = extract_audio_transcripts(analysis_results)
    
    if not audio_transcripts:
        st.warning("⚠️ 추출된 음성 스크립트가 없습니다.")
        if st.button("⏭️ 다음 단계로"):
            st.session_state.step3_progress = 2
            st.rerun()
        return
    
    st.markdown("#### 🎤 추출된 음성 내용")
    
    # 각 오디오 파일별 스크립트 표시
    for i, (filename, transcript_data) in enumerate(audio_transcripts.items()):
        with st.expander(f"🎵 {filename} - 스크립트", expanded=True):
            
            # 원본 스크립트
            st.markdown("**📋 추출된 원본 텍스트:**")
            original_text = transcript_data.get('transcribed_text', '')
            
            # 편집 가능한 텍스트 영역
            edited_text = st.text_area(
                "스크립트 내용 (수정 가능):",
                value=original_text,
                height=150,
                key=f"transcript_edit_{i}",
                help="내용이 부정확하다면 직접 수정하실 수 있습니다."
            )
            
            # 수정 여부 체크
            if edited_text != original_text:
                st.info("✏️ 스크립트가 수정되었습니다.")
                # 수정된 내용 저장
                transcript_data['edited_text'] = edited_text
                transcript_data['user_modified'] = True
            
            # 스크립트 메타 정보
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ 길이", f"{len(original_text)} 글자")
            with col2:
                confidence = transcript_data.get('confidence', 0)
                st.metric("🎯 신뢰도", f"{confidence:.1%}")
            with col3:
                processing_time = transcript_data.get('processing_time', 0)
                st.metric("⚡ 처리시간", f"{processing_time:.1f}초")
    
    # 스크립트 저장
    st.session_state.extracted_transcripts = audio_transcripts
    
    # 다음 단계 버튼
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ 분석 다시 실행"):
            st.session_state.step3_progress = 0
            st.rerun()
    
    with col2:
        if st.button("✅ 스크립트 확인 완료", type="primary"):
            st.session_state.step3_progress = 2
            st.rerun()

def render_content_review(self):
    """내용 검토 단계"""
    
    st.markdown("### 🧠 분석 결과 검토")
    
    analysis_results = st.session_state.get('analysis_results', {})
    transcripts = st.session_state.get('extracted_transcripts', {})
    
    # 종합 분석 결과 표시
    if 'comprehensive_analysis' in analysis_results:
        comprehensive = analysis_results['comprehensive_analysis']
        
        st.markdown("#### 📊 종합 분석 결과")
        
        # 핵심 요약
        if 'summary' in comprehensive:
            st.markdown("**🎯 핵심 요약:**")
            st.info(comprehensive['summary'])
        
        # 고객 상태 분석
        col1, col2, col3 = st.columns(3)
        
        with col1:
            customer_status = comprehensive.get('customer_status', 'N/A')
            st.metric("😊 고객 상태", customer_status)
        
        with col2:
            urgency = comprehensive.get('urgency', 'N/A')
            st.metric("⚡ 긴급도", urgency)
        
        with col3:
            purchase_intent = comprehensive.get('purchase_intent', 0)
            if isinstance(purchase_intent, (int, float)):
                st.metric("💰 구매 의도", f"{purchase_intent}/10")
            else:
                st.metric("💰 구매 의도", str(purchase_intent))
        
        # 추천 액션
        if 'recommended_actions' in comprehensive:
            st.markdown("**📋 추천 액션:**")
            actions = comprehensive['recommended_actions']
            if isinstance(actions, list):
                for i, action in enumerate(actions, 1):
                    st.write(f"{i}. {action}")
            else:
                st.write(actions)
    
    # 파일별 상세 결과
    st.markdown("#### 📁 파일별 분석 상세")
    
    file_results = analysis_results.get('file_results', {})
    
    for filename, result in file_results.items():
        with st.expander(f"📄 {filename} - 상세 결과"):
            
            file_type = result.get('file_type', 'unknown')
            success = result.get('success', False)
            
            # 파일 정보
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**파일 형식:** {file_type}")
            with col2:
                status_icon = "✅" if success else "❌"
                st.write(f"**처리 상태:** {status_icon}")
            with col3:
                processing_time = result.get('processing_time', 0)
                st.write(f"**처리 시간:** {processing_time:.1f}초")
            
            # 추출된 내용
            if file_type == 'audio' and filename in transcripts:
                transcript_data = transcripts[filename]
                final_text = transcript_data.get('edited_text', 
                                                transcript_data.get('transcribed_text', ''))
                if final_text:
                    st.markdown("**🎤 음성 내용:**")
                    st.text_area("", value=final_text, height=100, disabled=True, 
                                key=f"final_transcript_{filename}")
            
            elif file_type == 'image':
                extracted_text = result.get('extracted_text', '')
                if extracted_text:
                    st.markdown("**🖼️ 이미지 텍스트:**")
                    st.text_area("", value=extracted_text, height=100, disabled=True,
                                key=f"image_text_{filename}")
    
    # 검토 완료 버튼
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ 스크립트 수정하기"):
            st.session_state.step3_progress = 1
            st.rerun()
    
    with col2:
        if st.button("✅ 검토 완료", type="primary"):
            st.session_state.step3_progress = 3
            st.rerun()

def render_final_confirmation(self):
    """최종 확인 단계"""
    
    st.markdown("### ✅ 최종 확인")
    
    st.success("🎉 모든 검토가 완료되었습니다!")
    
    # 요약 정보 표시
    transcripts = st.session_state.get('extracted_transcripts', {})
    analysis_results = st.session_state.get('analysis_results', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📁 처리된 파일", len(analysis_results.get('file_results', {})))
    
    with col2:
        audio_count = len([f for f in transcripts.keys() if 'audio' in f.lower()])
        st.metric("🎤 음성 파일", audio_count)
    
    with col3:
        modified_count = len([t for t in transcripts.values() if t.get('user_modified')])
        st.metric("✏️ 수정된 스크립트", modified_count)
    
    # 사용자 피드백
    st.markdown("#### 💬 분석 결과에 대한 의견")
    user_feedback = st.text_area(
        "분석 결과나 추가 요청사항이 있다면 입력해주세요:",
        height=100,
        placeholder="예: 고객의 감정 상태를 더 자세히 분석해주세요..."
    )
    
    if user_feedback:
        st.session_state.user_feedback = user_feedback
    
    # 최종 진행 버튼
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ 다시 검토하기"):
            st.session_state.step3_progress = 2
            st.rerun()
    
    with col2:
        if st.button("🚀 최종 보고서 생성", type="primary"):
            # 사용자 피드백 포함해서 4단계로 이동
            st.session_state.workflow_step = 4
            st.session_state.step3_progress = 0  # 초기화
            st.success("✅ 4단계 최종 보고서로 이동합니다!")
            time.sleep(1)
            st.rerun()

def extract_audio_transcripts(analysis_results: Dict) -> Dict[str, Dict]:
    """분석 결과에서 오디오 스크립트 추출"""
    
    transcripts = {}
    file_results = analysis_results.get('file_results', {})
    
    for filename, result in file_results.items():
        if result.get('file_type') == 'audio' and result.get('success'):
            transcript_data = {
                'transcribed_text': result.get('transcribed_text', ''),
                'confidence': result.get('confidence', 0),
                'processing_time': result.get('processing_time', 0),
                'user_modified': False,
                'edited_text': None
            }
            transcripts[filename] = transcript_data
    
    return transcripts

def execute_comprehensive_analysis(uploaded_data: Dict) -> Dict:
    """종합 분석 실행 (기존 코드와 연동)"""
    
    # 이 부분은 기존의 분석 로직을 활용
    # 실제 구현에서는 self.generate_comprehensive_analysis() 등을 호출
    
    try:
        # 예시 결과 (실제로는 분석 엔진 결과 사용)
        results = {
            'file_results': {},
            'comprehensive_analysis': {
                'summary': '주얼리 상담 내용 분석 완료',
                'customer_status': '긍정적',
                'urgency': '보통',
                'purchase_intent': 7,
                'recommended_actions': [
                    '제품 카탈로그 준비',
                    '견적서 작성',
                    '실물 확인 일정 조율'
                ]
            }
        }
        
        # 업로드된 파일별 분석 수행
        for file_info in uploaded_data:
            filename = file_info['name']
            file_type = file_info['type']
            
            # 실제 분석 수행 (예시)
            file_result = {
                'file_type': 'audio' if 'audio' in file_type else 'image',
                'success': True,
                'processing_time': 2.5,
                'transcribed_text': f"{filename}에서 추출된 텍스트 내용...",
                'confidence': 0.85,
                'extracted_text': f"{filename}에서 추출된 이미지 텍스트..."
            }
            
            results['file_results'][filename] = file_result
        
        return results
        
    except Exception as e:
        raise Exception(f"분석 실행 중 오류: {str(e)}")

# 기존 render_step3_review 메서드를 대체하는 함수
def enhance_step3_in_main_ui():
    """메인 UI 파일에 이 기능을 통합하는 방법"""
    
    # jewelry_stt_ui_v23_real.py 파일에서
    # render_step3_review 메서드를 위의 render_step3_enhanced_review로 교체
    
    print("3단계 향상 기능을 메인 UI에 통합하려면:")
    print("1. render_step3_enhanced_review 함수를 JewelrySTTUI 클래스에 추가")
    print("2. step3_progress 상태 관리 추가")
    print("3. 스크립트 편집 및 검토 로직 통합")
    print("4. 사용자 피드백 수집 기능 추가")

if __name__ == "__main__":
    enhance_step3_in_main_ui()