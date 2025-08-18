#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 Streamlit UI에 종합 분석 기능 추가
"""

def add_comprehensive_analysis_to_main_ui():
    """메인 UI에 종합 분석 기능 추가"""
    
    # 종합 분석 함수 코드
    comprehensive_analysis_code = '''
    def render_comprehensive_analysis(self):
        """종합 상황 분석 페이지"""
        st.header("🎯 종합 상황 분석")
        st.markdown("**user_files 폴더의 모든 파일을 하나의 상황으로 통합 분석**")
        
        # 초기화
        if 'comprehensive_analysis_complete' not in st.session_state:
            st.session_state.comprehensive_analysis_complete = False
        if 'comprehensive_results' not in st.session_state:
            st.session_state.comprehensive_results = None
        
        # 설정
        with st.sidebar:
            st.subheader("🔧 종합 분석 설정")
            max_audio_size = st.slider("최대 오디오 크기 (MB)", 1, 50, 20)
            max_image_size = st.slider("최대 이미지 크기 (MB)", 1, 20, 10)
            include_videos = st.checkbox("비디오 메타데이터 포함", value=True)
        
        # 1. 파일 발견
        st.subheader("📁 상황 파일 발견")
        
        if st.button("🔍 user_files 폴더 탐색", type="primary"):
            with st.spinner("파일 탐색 중..."):
                files = self._discover_user_files()
            
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
                
                # 파일 미리보기
                with st.expander("📋 발견된 파일 목록"):
                    for file_info in files:
                        st.write(f"- **{file_info['name']}** ({file_info['size_mb']:.1f}MB) - {file_info['timestamp']}")
                
                st.session_state.discovered_files = files
            else:
                st.warning("⚠️ user_files 폴더에서 파일을 찾을 수 없습니다.")
        
        # 2. 종합 분석 실행
        if 'discovered_files' in st.session_state:
            st.subheader("🎯 종합 분석 실행")
            
            if st.button("🚀 모든 파일 통합 분석 시작", type="primary"):
                files = st.session_state.discovered_files
                
                st.info(f"📊 {len(files)}개 파일의 종합 분석을 시작합니다...")
                
                # 진행률 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 결과 저장용
                comprehensive_results = {
                    'audio_results': [],
                    'image_results': [],
                    'video_results': [],
                    'timeline': [],
                    'comprehensive_story': ''
                }
                
                total_files = len(files)
                
                # 파일별 순차 분석
                for i, file_info in enumerate(files):
                    progress = (i + 1) / total_files
                    progress_bar.progress(progress)
                    status_text.text(f"분석 중: {file_info['name']} ({i+1}/{total_files})")
                    
                    try:
                        result = self._analyze_single_file_comprehensive(file_info, max_audio_size, max_image_size)
                        
                        if result:
                            if result['type'] == 'audio':
                                comprehensive_results['audio_results'].append(result)
                            elif result['type'] == 'image':
                                comprehensive_results['image_results'].append(result)
                            elif result['type'] == 'video':
                                comprehensive_results['video_results'].append(result)
                            
                            comprehensive_results['timeline'].append({
                                'timestamp': file_info['timestamp'],
                                'file': file_info['name'],
                                'type': result['type'],
                                'content': result.get('content', '')[:200]
                            })
                    
                    except Exception as e:
                        st.error(f"❌ {file_info['name']} 분석 실패: {str(e)[:100]}")
                        continue
                
                # 종합 스토리 생성
                comprehensive_results['comprehensive_story'] = self._generate_comprehensive_story(comprehensive_results)
                
                st.session_state.comprehensive_results = comprehensive_results
                st.session_state.comprehensive_analysis_complete = True
                
                progress_bar.progress(1.0)
                status_text.text("✅ 종합 분석 완료!")
                
                st.success("🎉 모든 파일의 종합 분석이 완료되었습니다!")
                st.rerun()
        
        # 3. 결과 표시
        if st.session_state.comprehensive_analysis_complete and st.session_state.comprehensive_results:
            self._display_comprehensive_results(st.session_state.comprehensive_results)
    
    def _discover_user_files(self):
        """user_files 폴더에서 파일 발견"""
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
                            'timestamp': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                            'ext': file_path.suffix.lower()
                        }
                        all_files.append(file_info)
                    except Exception:
                        continue
        
        # 시간순 정렬
        all_files.sort(key=lambda x: x['timestamp'])
        return all_files
    
    def _analyze_single_file_comprehensive(self, file_info, max_audio_size, max_image_size):
        """단일 파일 종합 분석"""
        ext = file_info['ext']
        
        try:
            if ext in ['.m4a', '.wav', '.mp3']:
                # 오디오 분석
                if file_info['size_mb'] <= max_audio_size:
                    # 기존 분석 엔진 활용
                    if REAL_ANALYSIS_AVAILABLE:
                        result = analyze_file_real(
                            file_info['path'],
                            "audio",
                            language="ko"
                        )
                        
                        if result and result.get('success'):
                            return {
                                'type': 'audio',
                                'file': file_info['name'],
                                'content': result.get('stt_result', {}).get('text', ''),
                                'processing_time': result.get('processing_time', 0),
                                'timestamp': file_info['timestamp']
                            }
                
            elif ext in ['.jpg', '.jpeg', '.png']:
                # 이미지 분석
                if file_info['size_mb'] <= max_image_size:
                    if REAL_ANALYSIS_AVAILABLE:
                        result = analyze_file_real(
                            file_info['path'],
                            "image",
                            language="ko"
                        )
                        
                        if result and result.get('success'):
                            ocr_result = result.get('ocr_result', {})
                            return {
                                'type': 'image',
                                'file': file_info['name'],
                                'content': ' '.join([block.get('text', '') for block in ocr_result.get('text_blocks', [])]),
                                'text_blocks': len(ocr_result.get('text_blocks', [])),
                                'timestamp': file_info['timestamp']
                            }
            
            elif ext in ['.mov', '.mp4']:
                # 비디오 메타데이터
                return {
                    'type': 'video',
                    'file': file_info['name'],
                    'content': f"비디오 파일 ({file_info['size_mb']:.1f}MB)",
                    'size_mb': file_info['size_mb'],
                    'timestamp': file_info['timestamp']
                }
        
        except Exception as e:
            raise Exception(f"파일 분석 중 오류: {str(e)}")
        
        return None
    
    def _generate_comprehensive_story(self, results):
        """종합 스토리 생성"""
        story_parts = []
        
        # 시간순 정렬
        timeline = sorted(results['timeline'], key=lambda x: x['timestamp'])
        
        story_parts.append("=== 종합 상황 분석 스토리 ===\\n")
        
        for i, event in enumerate(timeline):
            if event['content'].strip():
                story_parts.append(f"{i+1}. [{event['type'].upper()}] {event['file']}")
                story_parts.append(f"   시간: {event['timestamp']}")
                story_parts.append(f"   내용: {event['content'][:300]}...")
                story_parts.append("")
        
        return "\\n".join(story_parts)
    
    def _display_comprehensive_results(self, results):
        """종합 결과 표시"""
        st.subheader("📊 종합 분석 결과")
        
        # 요약 통계
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎵 오디오 분석", len(results['audio_results']))
        with col2:
            st.metric("🖼️ 이미지 분석", len(results['image_results']))
        with col3:
            st.metric("🎬 비디오 수집", len(results['video_results']))
        with col4:
            st.metric("📅 총 이벤트", len(results['timeline']))
        
        # 종합 스토리
        st.subheader("📖 종합 상황 스토리")
        with st.expander("📜 전체 스토리 보기", expanded=True):
            st.text_area("종합 스토리", results['comprehensive_story'], height=400)
        
        # 타임라인 시각화
        if results['timeline']:
            st.subheader("📅 시간순 타임라인")
            for i, event in enumerate(results['timeline']):
                with st.expander(f"{i+1}. {event['file']} ({event['type']}) - {event['timestamp']}"):
                    st.write(f"**내용:** {event['content']}")
        
        # 상세 결과
        if results['audio_results']:
            st.subheader("🎵 오디오 분석 상세")
            for audio in results['audio_results']:
                with st.expander(f"🎵 {audio['file']}"):
                    st.write(f"**처리 시간:** {audio.get('processing_time', 0):.1f}초")
                    st.write(f"**내용:** {audio['content']}")
        
        if results['image_results']:
            st.subheader("🖼️ 이미지 분석 상세")
            for image in results['image_results']:
                with st.expander(f"🖼️ {image['file']}"):
                    st.write(f"**텍스트 블록:** {image.get('text_blocks', 0)}개")
                    st.write(f"**추출된 텍스트:** {image['content']}")
        
        # 결과 저장
        if st.button("💾 종합 분석 결과 저장"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_situation_analysis_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            st.success(f"✅ 종합 분석 결과 저장: {filename}")
    '''
    
    # 기존 파일 읽기
    with open('jewelry_stt_ui_v23_real.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 메인 UI 클래스에 메서드 추가할 위치 찾기
    insert_position = content.find('    def run(self):')
    
    if insert_position != -1:
        # 종합 분석 메서드들 추가
        new_content = (
            content[:insert_position] + 
            comprehensive_analysis_code + 
            '\n\n    ' + 
            content[insert_position:]
        )
        
        # run 메서드에 종합 분석 페이지 추가
        run_method_start = new_content.find('        # 사이드바 - 페이지 선택')
        if run_method_start != -1:
            # 페이지 선택에 종합 분석 추가
            page_selection_end = new_content.find('        # 선택된 페이지 렌더링', run_method_start)
            if page_selection_end != -1:
                page_addition = '''
            "🎯 종합 상황 분석": "comprehensive_analysis",'''
                
                # 페이지 딕셔너리에 추가
                pages_start = new_content.find('        pages = {', run_method_start)
                if pages_start != -1:
                    first_page_end = new_content.find('\n', pages_start + 20)
                    new_content = (
                        new_content[:first_page_end] + 
                        page_addition + 
                        new_content[first_page_end:]
                    )
                
                # 렌더링 로직에 추가
                elif_chain_pos = new_content.find('        elif selected_page == "step4_report":', page_selection_end)
                if elif_chain_pos != -1:
                    elif_addition = '''
        elif selected_page == "comprehensive_analysis":
            self.render_comprehensive_analysis()
        '''
                    
                    new_content = (
                        new_content[:elif_chain_pos] + 
                        elif_addition + 
                        new_content[elif_chain_pos:]
                    )
        
        # 수정된 내용 저장
        with open('jewelry_stt_ui_v23_real_enhanced.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ 종합 분석 기능이 추가된 새로운 UI 파일 생성: jewelry_stt_ui_v23_real_enhanced.py")
        
        return True
    else:
        print("❌ UI 클래스 구조를 찾을 수 없습니다.")
        return False

if __name__ == "__main__":
    add_comprehensive_analysis_to_main_ui()