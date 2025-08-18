#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit 통합 종합 상황 분석기
"""
import streamlit as st
import os
import time
import json
from pathlib import Path
from datetime import datetime
import tempfile

# 최적화 설정
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'

def init_session_state():
    """세션 상태 초기화"""
    if 'comprehensive_analysis_complete' not in st.session_state:
        st.session_state.comprehensive_analysis_complete = False
    if 'comprehensive_results' not in st.session_state:
        st.session_state.comprehensive_results = None
    if 'analysis_progress' not in st.session_state:
        st.session_state.analysis_progress = 0

@st.cache_resource
def load_ai_models():
    """AI 모델 캐시된 로딩"""
    models = {}
    
    try:
        import whisper
        models['whisper'] = whisper.load_model("tiny", device="cpu")
        st.success("✅ Whisper tiny 모델 로드 완료")
    except Exception as e:
        st.error(f"❌ Whisper 로딩 실패: {e}")
        models['whisper'] = None
    
    try:
        import easyocr
        models['ocr'] = easyocr.Reader(['ko', 'en'], gpu=False)
        st.success("✅ EasyOCR 모델 로드 완료")
    except Exception as e:
        st.error(f"❌ EasyOCR 로딩 실패: {e}")
        models['ocr'] = None
    
    return models

def discover_user_files():
    """사용자 파일 발견"""
    user_files = Path("user_files")
    all_files = []
    
    if user_files.exists():
        for file_path in user_files.rglob("*"):
            if file_path.is_file() and file_path.name != "README.md":
                try:
                    stat = file_path.stat()
                    file_info = {
                        'path': str(file_path),
                        'name': file_path.name,
                        'size_mb': stat.st_size / 1024 / 1024,
                        'modified_time': datetime.fromtimestamp(stat.st_mtime),
                        'ext': file_path.suffix.lower()
                    }
                    all_files.append(file_info)
                except Exception:
                    continue
    
    # 시간순 정렬
    all_files.sort(key=lambda x: x['modified_time'])
    return all_files

def analyze_file_batch(files, models, progress_placeholder):
    """파일 배치 분석"""
    results = {
        'audio_results': [],
        'image_results': [],
        'video_results': [],
        'timeline': []
    }
    
    total_files = len(files)
    
    for i, file_info in enumerate(files):
        progress = (i + 1) / total_files
        progress_placeholder.progress(progress, f"분석 중: {file_info['name']} ({i+1}/{total_files})")
        
        ext = file_info['ext']
        
        try:
            if ext in ['.m4a', '.wav', '.mp3'] and models['whisper']:
                # 오디오 분석 (크기 제한)
                if file_info['size_mb'] < 20:
                    result = models['whisper'].transcribe(file_info['path'])
                    transcript = result.get('text', '').strip()
                    
                    if transcript:
                        audio_data = {
                            'file': file_info['name'],
                            'transcript': transcript,
                            'timestamp': file_info['modified_time'].isoformat(),
                            'size_mb': file_info['size_mb']
                        }
                        results['audio_results'].append(audio_data)
            
            elif ext in ['.jpg', '.jpeg', '.png'] and models['ocr']:
                # 이미지 분석 (크기 제한)
                if file_info['size_mb'] < 10:
                    ocr_results = models['ocr'].readtext(file_info['path'])
                    texts = [text for (bbox, text, conf) in ocr_results if conf > 0.5]
                    
                    if texts:
                        combined_text = ' '.join(texts)
                        image_data = {
                            'file': file_info['name'],
                            'extracted_text': combined_text,
                            'timestamp': file_info['modified_time'].isoformat(),
                            'text_blocks': len(texts)
                        }
                        results['image_results'].append(image_data)
            
            elif ext in ['.mov', '.mp4']:
                # 비디오 메타데이터
                video_data = {
                    'file': file_info['name'],
                    'size_mb': file_info['size_mb'],
                    'timestamp': file_info['modified_time'].isoformat(),
                    'type': 'video_metadata'
                }
                results['video_results'].append(video_data)
            
            # 타임라인에 추가
            results['timeline'].append({
                'timestamp': file_info['modified_time'].isoformat(),
                'file': file_info['name'],
                'type': ext[1:],
                'processed': True
            })
            
        except Exception as e:
            # 오류 처리
            results['timeline'].append({
                'timestamp': file_info['modified_time'].isoformat(),
                'file': file_info['name'],
                'type': ext[1:],
                'processed': False,
                'error': str(e)[:100]
            })
    
    return results

def generate_comprehensive_story(results):
    """종합 스토리 생성"""
    story_parts = []
    
    # 시간순 정렬
    timeline = sorted(results['timeline'], key=lambda x: x['timestamp'])
    
    for event in timeline:
        if not event['processed']:
            continue
            
        file_name = event['file']
        file_type = event['type']
        
        # 내용 찾기
        content = ""
        if file_type in ['m4a', 'wav', 'mp3']:
            for audio in results['audio_results']:
                if audio['file'] == file_name:
                    content = audio['transcript'][:300]
                    break
        elif file_type in ['jpg', 'jpeg', 'png']:
            for image in results['image_results']:
                if image['file'] == file_name:
                    content = image['extracted_text'][:300]
                    break
        elif file_type in ['mov', 'mp4']:
            for video in results['video_results']:
                if video['file'] == file_name:
                    content = f"비디오 파일 ({video['size_mb']:.1f}MB)"
                    break
        
        if content:
            story_parts.append({
                'timestamp': event['timestamp'],
                'file': file_name,
                'type': file_type,
                'content': content
            })
    
    return story_parts

def main():
    """메인 Streamlit 앱"""
    st.set_page_config(
        page_title="솔로몬드 AI - 종합 상황 분석",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 솔로몬드 AI - 종합 상황 분석")
    st.markdown("**실제 상황의 모든 파일들을 하나로 통합 분석**")
    
    init_session_state()
    
    # 사이드바
    with st.sidebar:
        st.header("🔧 분석 설정")
        
        auto_optimize = st.checkbox("자동 최적화", value=True, help="파일 크기에 따른 자동 최적화")
        include_videos = st.checkbox("비디오 포함", value=True, help="비디오 메타데이터 포함")
        max_file_size = st.slider("최대 파일 크기 (MB)", 1, 50, 20, help="처리할 최대 파일 크기")
    
    # 파일 발견
    st.header("📁 상황 파일 발견")
    
    if st.button("🔍 파일 탐색", type="primary"):
        with st.spinner("파일 탐색 중..."):
            files = discover_user_files()
        
        if files:
            st.success(f"✅ {len(files)}개 파일 발견")
            
            # 파일 타입별 분류
            audio_files = [f for f in files if f['ext'] in ['.m4a', '.wav', '.mp3']]
            image_files = [f for f in files if f['ext'] in ['.jpg', '.jpeg', '.png']]
            video_files = [f for f in files if f['ext'] in ['.mov', '.mp4']]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎵 오디오", len(audio_files))
            with col2:
                st.metric("🖼️ 이미지", len(image_files))
            with col3:
                st.metric("🎬 비디오", len(video_files))
            
            # 파일 목록 표시
            with st.expander("📋 발견된 파일 목록"):
                for file_info in files:
                    st.write(f"- **{file_info['name']}** ({file_info['size_mb']:.1f}MB) - {file_info['modified_time'].strftime('%Y-%m-%d %H:%M')}")
            
            st.session_state.discovered_files = files
        else:
            st.warning("⚠️ user_files 폴더에서 파일을 찾을 수 없습니다.")
    
    # 종합 분석 실행
    if 'discovered_files' in st.session_state:
        st.header("🎯 종합 상황 분석")
        
        if st.button("🚀 종합 분석 시작", type="primary"):
            with st.spinner("AI 모델 로딩 중..."):
                models = load_ai_models()
            
            if models['whisper'] or models['ocr']:
                st.info("📊 파일 분석을 시작합니다...")
                
                # 진행률 표시
                progress_placeholder = st.empty()
                
                # 배치 분석
                with st.spinner("종합 분석 중..."):
                    results = analyze_file_batch(
                        st.session_state.discovered_files, 
                        models, 
                        progress_placeholder
                    )
                
                st.session_state.comprehensive_results = results
                st.session_state.comprehensive_analysis_complete = True
                
                st.success("✅ 종합 분석 완료!")
                st.rerun()
            else:
                st.error("❌ AI 모델 로딩에 실패했습니다.")
    
    # 결과 표시
    if st.session_state.comprehensive_analysis_complete and st.session_state.comprehensive_results:
        st.header("📊 종합 분석 결과")
        
        results = st.session_state.comprehensive_results
        
        # 요약 통계
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎵 오디오 분석", len(results['audio_results']))
        with col2:
            st.metric("🖼️ 이미지 분석", len(results['image_results']))
        with col3:
            st.metric("🎬 비디오 수집", len(results['video_results']))
        with col4:
            processed_count = sum(1 for item in results['timeline'] if item['processed'])
            st.metric("✅ 처리 성공", processed_count)
        
        # 종합 스토리
        st.subheader("📖 상황 재구성 스토리")
        
        story_parts = generate_comprehensive_story(results)
        
        if story_parts:
            for i, part in enumerate(story_parts):
                with st.expander(f"{i+1}. {part['file']} ({part['type'].upper()})"):
                    st.write(f"**시간:** {part['timestamp']}")
                    st.write(f"**내용:** {part['content']}")
        else:
            st.info("분석된 내용이 없습니다.")
        
        # 상세 결과
        if results['audio_results']:
            st.subheader("🎵 오디오 분석 상세")
            for audio in results['audio_results']:
                with st.expander(f"🎵 {audio['file']}"):
                    st.write(f"**크기:** {audio['size_mb']:.1f}MB")
                    st.write(f"**시간:** {audio['timestamp']}")
                    st.write(f"**내용:** {audio['transcript']}")
        
        if results['image_results']:
            st.subheader("🖼️ 이미지 분석 상세")
            for image in results['image_results']:
                with st.expander(f"🖼️ {image['file']}"):
                    st.write(f"**텍스트 블록:** {image['text_blocks']}개")
                    st.write(f"**시간:** {image['timestamp']}")
                    st.write(f"**추출된 텍스트:** {image['extracted_text']}")
        
        # 결과 저장
        if st.button("💾 결과 저장"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_analysis_{timestamp}.json"
            
            save_data = {
                'analysis_time': datetime.now().isoformat(),
                'results': results,
                'story': story_parts
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            st.success(f"✅ 결과 저장 완료: {filename}")
    
    # 도움말
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 💡 사용 안내")
        st.markdown("""
        1. **파일 탐색**: user_files 폴더의 모든 파일 발견
        2. **종합 분석**: 오디오, 이미지, 비디오 통합 분석
        3. **상황 재구성**: 시간순으로 스토리 생성
        4. **결과 저장**: JSON 형태로 결과 저장
        """)

if __name__ == "__main__":
    main()