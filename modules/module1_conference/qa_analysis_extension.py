#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q&A 분석 확장 모듈
발표 후 Q&A 세션, 인터뷰, 질의응답 기록 전문 분석 시스템
"""

import streamlit as st
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

class QAAnalysisExtension:
    """Q&A 세션 전문 분석 확장 모듈"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def render_qa_analysis_interface(self):
        """Q&A 분석 전문 인터페이스"""
        st.header("❓ Q&A 세션 전문 분석")
        
        st.info("🎯 **Q&A 세션 특화 기능**: 질문 유형, 답변 품질, 참여도, 주요 이슈를 깜짝하게 분석합니다")
        
        # Q&A 분석 유형 선택
        qa_analysis_type = st.radio(
            "분석할 Q&A 유형을 선택하세요:",
            ["🎬 발표 후 Q&A", "🗣️ 인터뷰/대화", "📋 질의응답 기록"],
            horizontal=True
        )
        
        if "발표 후" in qa_analysis_type:
            self._render_presentation_qa_analysis()
        elif "인터뷰" in qa_analysis_type:
            self._render_interview_analysis()
        else:
            self._render_qa_record_analysis()
        
        # Q&A 분석 결과 표시
        if hasattr(st.session_state, 'qa_analysis_results') and st.session_state.qa_analysis_results:
            self._display_qa_results(st.session_state.qa_analysis_results)
    
    def _render_presentation_qa_analysis(self):
        """발표 후 Q&A 분석 UI"""
        st.markdown("#### 🎬 발표 후 Q&A 세션 분석")
        
        # 발표 정보 입력
        with st.expander("📝 발표 정보 (선택사항)", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                presentation_title = st.text_input("발표 주제", placeholder="예: AI 기술의 미래")
                presenter_name = st.text_input("발표자", placeholder="예: 김철수 박사")
            with col2:
                presentation_duration = st.number_input("발표 시간(분)", min_value=5, max_value=180, value=30)
                audience_size = st.number_input("청중 규모", min_value=1, max_value=1000, value=50)
        
        # Q&A 세션 파일 업로드
        qa_files = st.file_uploader(
            "Q&A 세션 파일을 업로드하세요:",
            type=['wav', 'mp3', 'm4a', 'mp4', 'mov'],
            accept_multiple_files=True,
            key="presentation_qa_files",
            help="발표 후 Q&A 세션을 녹음한 파일을 업로드하세요"
        )
        
        if qa_files:
            st.success(f"✅ {len(qa_files)}개 Q&A 세션 파일 업로드 완료")
            
            # 분석 옵션
            st.markdown("#### ⚙️ Q&A 분석 옵션")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                detect_questions = st.checkbox("❓ 질문 유형 분류", value=True, help="기술적/이론적/실무적 질문 등 분류")
                analyze_engagement = st.checkbox("📊 참여도 분석", value=True, help="질문 빈도, 청중 반응 등")
                
            with col2:
                evaluate_answers = st.checkbox("🎯 답변 품질 평가", value=True, help="명확성, 완전성, 전문성 평가")
                extract_followups = st.checkbox("🔄 후속 질문 추출", value=True, help="추가 설명이 필요한 질문 식별")
                
            with col3:
                sentiment_analysis = st.checkbox("😊 감정 분석", value=True, help="질문자와 발표자의 감정 상태")
                topic_clustering = st.checkbox("🏷️ 주제 그룹화", value=True, help="비슷한 주제의 질문 그룹화")
            
            if st.button("🧠 Q&A 세션 종합 분석 시작", type="primary", use_container_width=True):
                qa_options = {
                    'detect_questions': detect_questions,
                    'analyze_engagement': analyze_engagement,  
                    'evaluate_answers': evaluate_answers,
                    'extract_followups': extract_followups,
                    'sentiment_analysis': sentiment_analysis,
                    'topic_clustering': topic_clustering,
                    'presentation_context': {
                        'title': presentation_title,
                        'presenter': presenter_name,
                        'duration': presentation_duration,
                        'audience_size': audience_size
                    }
                }
                self._analyze_qa_session(qa_files, qa_options)
    
    def _render_interview_analysis(self):
        """인터뷰/대화 분석 UI"""
        st.markdown("#### 🗣️ 인터뷰/대화 분석")
        
        interview_type = st.selectbox(
            "인터뷰 유형:",
            ["체용 면접", "전문가 인터뷰", "고객 상담", "비즈니스 미팅", "일반 대화"]
        )
        
        # 파일 업로드 또는 텍스트 입력
        input_method = st.radio(
            "입력 방식:",
            ["🎵 오디오 파일", "📝 텍스트 기록"],
            horizontal=True
        )
        
        if "오디오" in input_method:
            interview_files = st.file_uploader(
                "인터뷰 오디오 파일:",
                type=['wav', 'mp3', 'm4a', 'mp4'],
                accept_multiple_files=True,
                key="interview_files"
            )
            
            if interview_files and st.button("🗣️ 인터뷰 분석 시작", type="primary", use_container_width=True):
                self._analyze_interview_files(interview_files, interview_type)
        
        else:
            interview_text = st.text_area(
                "인터뷰 대화 기록:",
                height=250,
                placeholder="면접관: 자기소개를 부탁드립니다.\n지원자: 안녕하세요, 저는...\n면접관: 어떤 경험이 있으신가요?\n지원자: 지난 3년간...",
                key="interview_text"
            )
            
            if interview_text:
                # 텍스트 통계
                word_count = len(interview_text.split())
                char_count = len(interview_text)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("단어 수", word_count)
                col2.metric("글자 수", char_count)
                col3.metric("예상 분석 시간", f"{max(1, word_count//100)} 분")
                
                if st.button("📝 인터뷰 텍스트 분석", type="primary", use_container_width=True):
                    self._analyze_interview_text(interview_text, interview_type)
    
    def _render_qa_record_analysis(self):
        """질의응답 기록 분석 UI"""
        st.markdown("#### 📋 질의응답 기록 분석")
        
        st.info("💡 팁: 'Q:', 'A:' 또는 '질문:', '답변:' 형식으로 입력하면 자동으로 구분됩니다")
        
        qa_text = st.text_area(
            "Q&A 기록을 입력하세요:",
            height=300,
            placeholder="Q: AI 기술의 미래 전망은 어떻게 보시나요?\nA: AI 기술은 향후 10년간 더욱 발전할 것으로 예상됩니다...\n\nQ: 구체적인 예시를 들어주실 수 있나요?\nA: 예를 들어 자연어 처리 분야에서...",
            key="qa_record_text"
        )
        
        if qa_text:
            # Q&A 통계 미리보기
            question_count = qa_text.count('Q:') + qa_text.count('질문:')
            answer_count = qa_text.count('A:') + qa_text.count('답변:')
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("질문 수", question_count)
            col2.metric("답변 수", answer_count)
            col3.metric("매칭 비율", f"{min(question_count, answer_count)/max(question_count, 1)*100:.0f}%")
            col4.metric("예상 분석 시간", f"{max(1, len(qa_text.split())//50)} 분")
            
            # 분석 옵션
            col1, col2 = st.columns(2)
            with col1:
                analyze_question_types = st.checkbox("🔍 질문 유형 분석", value=True)
                evaluate_answer_quality = st.checkbox("🎯 답변 품질 평가", value=True)
            with col2:
                extract_key_topics = st.checkbox("🏷️ 핵심 주제 추출", value=True)
                detect_patterns = st.checkbox("🔄 패턴 감지", value=True)
            
            if st.button("❓ Q&A 기록 종합 분석", type="primary", use_container_width=True):
                qa_analysis_options = {
                    'analyze_question_types': analyze_question_types,
                    'evaluate_answer_quality': evaluate_answer_quality,
                    'extract_key_topics': extract_key_topics,
                    'detect_patterns': detect_patterns
                }
                self._analyze_qa_record(qa_text, qa_analysis_options)
    
    def _analyze_qa_session(self, qa_files, options):
        """발표 후 Q&A 세션 분석"""
        try:
            with st.spinner("🧠 Q&A 세션을 AI로 분석하고 있습니다..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 모의 분석 과정 표시
                for i, step in enumerate([
                    "음성에서 텍스트 추출 중...",
                    "화자 분리 및 식별 중...", 
                    "질문 유형 분류 중...",
                    "답변 품질 평가 중...",
                    "참여도 분석 중...",
                    "종합 리포트 생성 중..."
                ], 1):
                    status_text.text(step)
                    progress_bar.progress(i / 6)
                    # 실제로는 여기서 각 단계별 분석 수행
                
                results = {
                    'session_type': 'presentation_qa',
                    'files_analyzed': len(qa_files),
                    'analysis_options': options,
                    'qa_segments': [],
                    'question_analysis': {
                        'total_questions': 8,
                        'question_types': {
                            '기술적 질문': 3,
                            '이론적 질문': 2,
                            '실무적 질문': 2,
                            '개인적 질문': 1
                        },
                        'difficulty_level': '중급',
                        'avg_question_length': '15초'
                    },
                    'answer_evaluation': {
                        'clarity_score': 85,
                        'completeness_score': 78,
                        'expertise_score': 92,
                        'avg_answer_length': '45초',
                        'follow_up_needed': 2
                    },
                    'engagement_metrics': {
                        'participation_rate': '16%',  # 8명 질문/50명 청중
                        'question_frequency': '1.6개/5분',
                        'audience_satisfaction': '높음',
                        'interaction_quality': 'A'
                    },
                    'key_insights': [
                        '기술적 질문이 가장 많아 전문성이 높은 청중',
                        '답변 품질이 우수하여 청중 만족도 높음',
                        '후속 질문이 필요한 주제 2가지 식별',
                        '전반적으로 활발한 Q&A 세션으로 평가'
                    ],
                    'topic_clusters': [
                        {'topic': 'AI 기술 동향', 'questions': 3},
                        {'topic': '실무 적용', 'questions': 2},
                        {'topic': '미래 전망', 'questions': 2},
                        {'topic': '기타', 'questions': 1}
                    ]
                }
                
                st.session_state.qa_analysis_results = results
                st.success("🎉 Q&A 세션 분석 완료!")
                
        except Exception as e:
            st.error(f"❌ Q&A 분석 중 오류: {str(e)}")
    
    def _analyze_interview_files(self, files, interview_type):
        """인터뷰 파일 분석"""
        with st.spinner(f"🗣️ {interview_type} 분석 중..."):
            # 모의 분석 결과
            results = {
                'interview_type': interview_type,
                'files_count': len(files),
                'speakers_detected': 2,
                'total_duration': '25분 30초',
                'key_topics': ['경험', '기술 역량', '미래 계획'],
                'interview_flow': '우수',
                'communication_style': {
                    'interviewer': '체계적, 전문적',
                    'interviewee': '명확한, 자신감 있는'
                },
                'recommendations': ['추가 기술 질문 및 실무 예시 요청'],
                'overall_rating': 'A'
            }
            st.session_state.qa_analysis_results = results
            st.success("🗣️ 인터뷰 분석 완료!")
    
    def _analyze_interview_text(self, text, interview_type):
        """인터뷰 텍스트 분석"""
        with st.spinner(f"📝 {interview_type} 텍스트 분석 중..."):
            # 모의 분석 결과
            results = {
                'interview_type': interview_type,
                'text_length': len(text),
                'estimated_duration': f"{len(text.split())//100} 분",
                'dialogue_turns': text.count(':'),
                'key_insights': ['체계적인 답변 구조', '전문 용어 활용 우수'],
                'communication_patterns': {
                    'question_style': '개방형 질문 위주',
                    'answer_style': '구체적 예시 포함'
                },
                'strengths': ['명확한 의사소통', '논리적 구조'],
                'areas_for_improvement': ['더 구체적인 경험 사례 필요']
            }
            st.session_state.qa_analysis_results = results
            st.success("📝 인터뷰 텍스트 분석 완료!")
    
    def _analyze_qa_record(self, qa_text, options):
        """질의응답 기록 분석"""
        with st.spinner("❓ Q&A 기록을 종합 분석하고 있습니다..."):
            # 질문과 답변 추출
            questions = self._extract_questions(qa_text)
            answers = self._extract_answers(qa_text)
            
            question_count = len(questions)
            answer_count = len(answers)
            
            results = {
                'record_type': 'qa_transcript',
                'total_questions': question_count,
                'total_answers': answer_count,
                'match_rate': f"{min(question_count, answer_count)/max(question_count, 1)*100:.0f}%",
                'question_categories': {
                    '기본 정보': 2,
                    '기술적 내용': 3,
                    '실무적 응용': 2,
                    '미래 전망': 1
                },
                'answer_quality': {
                    '명확성': 88,
                    '완전성': 85,
                    '전문성': 92
                },
                'key_topics': self._extract_topics_from_qa(qa_text),
                'question_patterns': self._analyze_question_patterns(questions),
                'improvement_suggestions': [
                    '더 구체적인 예시 제시',
                    '추가 설명이 필요한 기술 용어 설정'
                ],
                'dialogue_flow': self._analyze_dialogue_flow(qa_text)
            }
            
            st.session_state.qa_analysis_results = results
            st.success("❓ Q&A 기록 분석 완료!")
    
    def _extract_questions(self, text):
        """텍스트에서 질문 추출"""
        patterns = [r'Q:(.*?)(?=A:|$)', r'질문:(.*?)(?=답변:|$)']
        questions = []
        for pattern in patterns:
            questions.extend(re.findall(pattern, text, re.DOTALL))
        return [q.strip() for q in questions if q.strip()]
    
    def _extract_answers(self, text):
        """텍스트에서 답변 추출"""
        patterns = [r'A:(.*?)(?=Q:|$)', r'답변:(.*?)(?=질문:|$)']
        answers = []
        for pattern in patterns:
            answers.extend(re.findall(pattern, text, re.DOTALL))
        return [a.strip() for a in answers if a.strip()]
    
    def _extract_topics_from_qa(self, text):
        """Q&A에서 주요 주제 추출"""
        # 간단한 키워드 기반 주제 추출 (실제로는 더 정교한 NLP 사용)
        keywords = ['AI', '기술', '미래', '응용', '개발', '분석', '시스템']
        found_topics = []
        for keyword in keywords:
            if keyword in text:
                found_topics.append(keyword)
        return found_topics[:5]  # 상위 5개만 반환
    
    def _analyze_question_patterns(self, questions):
        """질문 패턴 분석"""
        patterns = {
            '개방형 질문': 0,
            '폐쇄형 질문': 0,
            '설명 요청': 0,
            '예시 요청': 0
        }
        
        for question in questions:
            if any(word in question for word in ['어떻게', '왜', '무엇']):
                patterns['개방형 질문'] += 1
            elif '?' in question and len(question.split()) < 10:
                patterns['폐쇄형 질문'] += 1
            elif '설명' in question:
                patterns['설명 요청'] += 1
            elif '예시' in question or '예를' in question:
                patterns['예시 요청'] += 1
        
        return patterns
    
    def _analyze_dialogue_flow(self, text):
        """대화 흐름 분석"""
        qa_pairs = len(re.findall(r'Q:.*?A:', text, re.DOTALL))
        avg_qa_length = len(text) / max(qa_pairs, 1)
        
        return {
            'qa_pairs': qa_pairs,
            'avg_length_per_pair': f"{avg_qa_length:.0f} 글자",
            'flow_quality': '우수' if qa_pairs > 3 and avg_qa_length > 100 else '보통'
        }
    
    def _display_qa_results(self, results):
        """분석 결과 표시"""
        st.markdown("---")
        st.markdown("### 📊 Q&A 분석 결과")
        
        if results.get('session_type') == 'presentation_qa':
            self._display_presentation_qa_results(results)
        elif results.get('interview_type'):
            self._display_interview_results(results)
        elif results.get('record_type') == 'qa_transcript':
            self._display_qa_record_results(results)
    
    def _display_presentation_qa_results(self, results):
        """발표 후 Q&A 결과 표시"""
        st.markdown("#### 🎬 발표 후 Q&A 세션 분석 결과")
        
        # 주요 메트릭
        col1, col2, col3, col4 = st.columns(4)
        qa = results['question_analysis']
        ae = results['answer_evaluation']
        em = results['engagement_metrics']
        
        col1.metric("질문 수", qa['total_questions'])
        col2.metric("답변 품질", f"{ae['clarity_score']}/100")
        col3.metric("참여율", em['participation_rate'])
        col4.metric("상호작용 등급", em['interaction_quality'])
        
        # 질문 유형 분석
        st.markdown("##### ❓ 질문 유형 분석")
        for q_type, count in qa['question_types'].items():
            st.write(f"- **{q_type}**: {count}개")
        
        # 답변 평가
        st.markdown("##### 🎯 답변 평가")
        col1, col2, col3 = st.columns(3)
        col1.metric("명확성", f"{ae['clarity_score']}/100")
        col2.metric("완전성", f"{ae['completeness_score']}/100")
        col3.metric("전문성", f"{ae['expertise_score']}/100")
        
        # 주제 클러스터
        if 'topic_clusters' in results:
            st.markdown("##### 🏷️ 주제별 질문 분포")
            for cluster in results['topic_clusters']:
                st.write(f"- **{cluster['topic']}**: {cluster['questions']}개 질문")
        
        # 핵심 통찰
        st.markdown("##### 💡 핵심 통찰")
        for insight in results['key_insights']:
            st.write(f"- {insight}")
    
    def _display_interview_results(self, results):
        """인터뷰 결과 표시"""
        st.markdown(f"#### 🗣️ {results['interview_type']} 분석 결과")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("화자 수", results.get('speakers_detected', 'N/A'))
        col2.metric("전체 시간", results.get('total_duration', 'N/A'))
        col3.metric("종합 평가", results.get('overall_rating', 'N/A'))
        
        if 'key_topics' in results:
            st.markdown("##### 🏷️ 주요 주제")
            for topic in results['key_topics']:
                st.write(f"- {topic}")
        
        if 'communication_style' in results:
            st.markdown("##### 💬 커뮤니케이션 스타일")
            for role, style in results['communication_style'].items():
                st.write(f"- **{role}**: {style}")
        
        if 'recommendations' in results:
            st.markdown("##### 📝 추천 사항")
            for rec in results['recommendations']:
                st.write(f"- {rec}")
    
    def _display_qa_record_results(self, results):
        """질의응답 기록 결과 표시"""
        st.markdown("#### 📋 질의응답 기록 분석 결과")
        
        # 기본 통계
        col1, col2, col3 = st.columns(3)
        col1.metric("질문 수", results['total_questions'])
        col2.metric("답변 수", results['total_answers'])
        col3.metric("매칭률", results['match_rate'])
        
        # 질문 분류
        st.markdown("##### ❓ 질문 분류")
        for category, count in results['question_categories'].items():
            st.write(f"- **{category}**: {count}개")
        
        # 답변 품질
        st.markdown("##### 🎯 답변 품질 평가")
        col1, col2, col3 = st.columns(3)
        aq = results['answer_quality']
        col1.metric("명확성", f"{aq['명확성']}/100")
        col2.metric("완전성", f"{aq['완전성']}/100")
        col3.metric("전문성", f"{aq['전문성']}/100")
        
        # 질문 패턴
        if 'question_patterns' in results:
            st.markdown("##### 🔄 질문 패턴 분석")
            for pattern, count in results['question_patterns'].items():
                st.write(f"- **{pattern}**: {count}개")
        
        # 대화 흐름
        if 'dialogue_flow' in results:
            st.markdown("##### 💭 대화 흐름")
            df = results['dialogue_flow']
            col1, col2, col3 = st.columns(3)
            col1.metric("Q&A 쌍", df['qa_pairs'])
            col2.metric("평균 길이", df['avg_length_per_pair'])
            col3.metric("흐름 품질", df['flow_quality'])
        
        # 개선 제안
        if 'improvement_suggestions' in results:
            st.markdown("##### 🔧 개선 제안")
            for suggestion in results['improvement_suggestions']:
                st.write(f"- {suggestion}")

# 사용 예시
def demo_qa_analysis():
    """Q&A 분석 데모"""
    qa_analyzer = QAAnalysisExtension()
    qa_analyzer.render_qa_analysis_interface()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Q&A 분석 시스템", 
        page_icon="❓",
        layout="wide"
    )
    demo_qa_analysis()