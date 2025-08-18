#!/usr/bin/env python3
"""
사용자 설정 UI 컴포넌트 v2.6
Streamlit 기반 사용자 설정 인터페이스
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import logging
from datetime import datetime

try:
    from .user_settings_manager import (
        get_global_settings_manager, 
        SettingType, 
        SettingScope,
        UserProfile,
        UserSetting
    )
    USER_SETTINGS_AVAILABLE = True
except ImportError:
    USER_SETTINGS_AVAILABLE = False

def render_user_settings_panel() -> None:
    """사용자 설정 패널 렌더링"""
    if not USER_SETTINGS_AVAILABLE:
        st.error("❌ 사용자 설정 시스템을 사용할 수 없습니다.")
        return
    
    st.header("⚙️ 사용자 설정")
    
    try:
        settings_manager = get_global_settings_manager()
        
        # 탭 구성
        tabs = st.tabs([
            "🤖 AI 모델", 
            "📊 분석 설정", 
            "⚡ 성능", 
            "🎨 UI 테마", 
            "👤 프로필",
            "💾 백업/복원"
        ])
        
        # AI 모델 설정
        with tabs[0]:
            _render_ai_model_settings(settings_manager)
        
        # 분석 설정
        with tabs[1]:
            _render_analysis_settings(settings_manager)
        
        # 성능 설정
        with tabs[2]:
            _render_performance_settings(settings_manager)
        
        # UI 테마 설정
        with tabs[3]:
            _render_ui_theme_settings(settings_manager)
        
        # 프로필 설정
        with tabs[4]:
            _render_profile_settings(settings_manager)
        
        # 백업/복원
        with tabs[5]:
            _render_backup_restore(settings_manager)
    
    except Exception as e:
        st.error(f"❌ 설정 패널 렌더링 오류: {e}")

def _render_ai_model_settings(settings_manager) -> None:
    """AI 모델 설정 렌더링"""
    st.subheader("🤖 AI 모델 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Whisper STT 모델")
        
        current_whisper = settings_manager.get_setting("ai.whisper_model_size", "base")
        whisper_options = ["tiny", "base", "small", "medium", "large"]
        whisper_index = whisper_options.index(current_whisper) if current_whisper in whisper_options else 1
        
        new_whisper = st.selectbox(
            "모델 크기",
            options=whisper_options,
            index=whisper_index,
            help="크기가 클수록 정확도가 높지만 메모리와 처리 시간이 증가합니다.",
            key="whisper_model_size"
        )
        
        if new_whisper != current_whisper:
            settings_manager.set_setting(
                "ai.whisper_model_size", 
                new_whisper,
                SettingType.AI_MODEL,
                SettingScope.GLOBAL,
                "Whisper 모델 크기"
            )
            st.success(f"✅ Whisper 모델이 {new_whisper}로 변경되었습니다.")
            st.rerun()
        
        # 모델 정보 표시
        model_info = {
            "tiny": "39MB, 32x 속도, 낮은 정확도",
            "base": "142MB, 16x 속도, 보통 정확도",
            "small": "244MB, 6x 속도, 좋은 정확도", 
            "medium": "769MB, 2x 속도, 높은 정확도",
            "large": "1550MB, 1x 속도, 최고 정확도"
        }
        st.info(f"📊 {new_whisper} 모델: {model_info.get(new_whisper, '정보 없음')}")
    
    with col2:
        st.markdown("### EasyOCR 언어")
        
        current_languages = settings_manager.get_setting("ai.easyocr_languages", ["ko", "en"])
        
        language_options = {
            "ko": "한국어",
            "en": "영어",
            "ja": "일본어",
            "zh": "중국어 (간체)",
            "zh_tra": "중국어 (번체)",
            "th": "태국어",
            "vi": "베트남어",
            "ar": "아랍어",
            "hi": "힌디어",
            "fr": "프랑스어",
            "de": "독일어",
            "es": "스페인어",
            "ru": "러시아어"
        }
        
        selected_languages = st.multiselect(
            "지원 언어",
            options=list(language_options.keys()),
            default=current_languages,
            format_func=lambda x: language_options.get(x, x),
            help="여러 언어를 선택할 수 있습니다. 언어가 많을수록 메모리 사용량이 증가합니다.",
            key="easyocr_languages"
        )
        
        if selected_languages != current_languages:
            settings_manager.set_setting(
                "ai.easyocr_languages",
                selected_languages,
                SettingType.AI_MODEL,
                SettingScope.GLOBAL,
                "EasyOCR 지원 언어"
            )
            st.success(f"✅ EasyOCR 언어가 업데이트되었습니다.")
            st.rerun()
        
        if selected_languages:
            lang_names = [language_options.get(lang, lang) for lang in selected_languages]
            st.info(f"📝 선택된 언어: {', '.join(lang_names)}")
    
    st.markdown("### Transformers NLP 모델")
    
    current_transformers = settings_manager.get_setting("ai.transformers_model", "facebook/bart-large-cnn")
    
    transformers_options = {
        "facebook/bart-base": "BART Base (140MB) - 빠른 속도, 기본 성능",
        "facebook/bart-large": "BART Large (400MB) - 보통 속도, 좋은 성능",
        "facebook/bart-large-cnn": "BART Large CNN (400MB) - 요약 특화",
        "bert-base-multilingual-cased": "BERT Multilingual (680MB) - 다국어 지원",
        "gogamza/kobart-base-v2": "KoBART (140MB) - 한국어 특화",
        "ainize/kobart-news": "KoBART News (140MB) - 한국어 뉴스 요약"
    }
    
    new_transformers = st.selectbox(
        "NLP 모델",
        options=list(transformers_options.keys()),
        index=list(transformers_options.keys()).index(current_transformers) if current_transformers in transformers_options else 0,
        format_func=lambda x: transformers_options.get(x, x),
        help="텍스트 요약 및 자연어 처리에 사용되는 모델입니다.",
        key="transformers_model"
    )
    
    if new_transformers != current_transformers:
        settings_manager.set_setting(
            "ai.transformers_model",
            new_transformers,
            SettingType.AI_MODEL,
            SettingScope.GLOBAL,
            "Transformers NLP 모델"
        )
        st.success(f"✅ NLP 모델이 업데이트되었습니다.")
        st.rerun()
    
    st.info(f"🤖 현재 모델: {transformers_options.get(new_transformers, new_transformers)}")

def _render_analysis_settings(settings_manager) -> None:
    """분석 설정 렌더링"""
    st.subheader("📊 분석 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 최대 파일 크기
        current_max_size = settings_manager.get_setting("analysis.max_file_size_mb", 500.0)
        
        new_max_size = st.slider(
            "최대 파일 크기 (MB)",
            min_value=50.0,
            max_value=2000.0,
            value=current_max_size,
            step=50.0,
            help="이 크기를 초과하는 파일은 처리되지 않습니다.",
            key="max_file_size"
        )
        
        if new_max_size != current_max_size:
            settings_manager.set_setting(
                "analysis.max_file_size_mb",
                new_max_size,
                SettingType.ANALYSIS,
                SettingScope.GLOBAL,
                "최대 파일 크기 (MB)"
            )
            st.success(f"✅ 최대 파일 크기가 {new_max_size}MB로 설정되었습니다.")
        
        # 배치 크기
        current_batch_size = settings_manager.get_setting("analysis.batch_size", 5)
        
        new_batch_size = st.slider(
            "배치 처리 크기",
            min_value=1,
            max_value=20,
            value=current_batch_size,
            help="한 번에 처리할 파일 수입니다. 클수록 빠르지만 메모리를 많이 사용합니다.",
            key="batch_size"
        )
        
        if new_batch_size != current_batch_size:
            settings_manager.set_setting(
                "analysis.batch_size",
                new_batch_size,
                SettingType.ANALYSIS,
                SettingScope.GLOBAL,
                "배치 처리 크기"
            )
            st.success(f"✅ 배치 크기가 {new_batch_size}개로 설정되었습니다.")
    
    with col2:
        # 스트리밍 활성화
        current_streaming = settings_manager.get_setting("analysis.enable_streaming", True)
        
        new_streaming = st.checkbox(
            "스트리밍 분석 활성화",
            value=current_streaming,
            help="대용량 파일의 메모리 효율적 처리를 위한 스트리밍 모드입니다.",
            key="enable_streaming"
        )
        
        if new_streaming != current_streaming:
            settings_manager.set_setting(
                "analysis.enable_streaming",
                new_streaming,
                SettingType.ANALYSIS,
                SettingScope.GLOBAL,
                "스트리밍 분석 활성화"
            )
            status = "활성화" if new_streaming else "비활성화"
            st.success(f"✅ 스트리밍 분석이 {status}되었습니다.")
        
        # 자동 저장
        current_auto_save = settings_manager.get_setting("system.auto_save", True)
        
        new_auto_save = st.checkbox(
            "자동 저장 활성화",
            value=current_auto_save,
            help="분석 결과를 자동으로 저장합니다.",
            key="auto_save"
        )
        
        if new_auto_save != current_auto_save:
            settings_manager.set_setting(
                "system.auto_save",
                new_auto_save,
                SettingType.SYSTEM,
                SettingScope.GLOBAL,
                "자동 저장 활성화"
            )
            status = "활성화" if new_auto_save else "비활성화"
            st.success(f"✅ 자동 저장이 {status}되었습니다.")

def _render_performance_settings(settings_manager) -> None:
    """성능 설정 렌더링"""
    st.subheader("⚡ 성능 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 최대 메모리
        current_max_memory = settings_manager.get_setting("performance.max_memory_mb", 2048.0)
        
        new_max_memory = st.slider(
            "최대 메모리 사용량 (MB)",
            min_value=512.0,
            max_value=8192.0,
            value=current_max_memory,
            step=256.0,
            help="AI 모델이 사용할 수 있는 최대 메모리입니다.",
            key="max_memory"
        )
        
        if new_max_memory != current_max_memory:
            settings_manager.set_setting(
                "performance.max_memory_mb",
                new_max_memory,
                SettingType.PERFORMANCE,
                SettingScope.GLOBAL,
                "최대 메모리 사용량 (MB)"
            )
            st.success(f"✅ 최대 메모리가 {new_max_memory}MB로 설정되었습니다.")
        
        # 메모리 사용량 표시
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        used_gb = memory.used / (1024**3)
        usage_percent = memory.percent
        
        st.metric(
            "현재 시스템 메모리",
            f"{used_gb:.1f} / {total_gb:.1f} GB",
            f"{usage_percent:.1f}%"
        )
    
    with col2:
        # GPU 사용
        current_gpu = settings_manager.get_setting("performance.enable_gpu", False)
        
        # GPU 가용성 확인
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except:
            pass
        
        if gpu_available:
            new_gpu = st.checkbox(
                "GPU 사용 활성화",
                value=current_gpu,
                help="GPU를 사용하여 AI 모델 처리 속도를 향상시킵니다.",
                key="enable_gpu"
            )
            
            if new_gpu != current_gpu:
                settings_manager.set_setting(
                    "performance.enable_gpu",
                    new_gpu,
                    SettingType.PERFORMANCE,
                    SettingScope.GLOBAL,
                    "GPU 사용 활성화"
                )
                status = "활성화" if new_gpu else "비활성화"
                st.success(f"✅ GPU 사용이 {status}되었습니다.")
                st.warning("⚠️ 변경사항은 다음 분석부터 적용됩니다.")
            
            # GPU 정보 표시
            if gpu_available:
                try:
                    import torch
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    st.info(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                except:
                    st.info("🎮 GPU 사용 가능")
        else:
            st.warning("🚫 GPU를 사용할 수 없습니다.")
            st.info("PyTorch CUDA가 설치되지 않았거나 호환되는 GPU가 없습니다.")

def _render_ui_theme_settings(settings_manager) -> None:
    """UI 테마 설정 렌더링"""
    st.subheader("🎨 UI 테마 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 테마 선택
        current_theme = settings_manager.get_setting("ui.theme", "auto")
        
        theme_options = {
            "auto": "자동 (시스템 설정 따름)",
            "light": "라이트 모드",
            "dark": "다크 모드"
        }
        
        new_theme = st.selectbox(
            "테마",
            options=list(theme_options.keys()),
            index=list(theme_options.keys()).index(current_theme) if current_theme in theme_options else 0,
            format_func=lambda x: theme_options.get(x, x),
            help="애플리케이션의 색상 테마를 설정합니다.",
            key="ui_theme"
        )
        
        if new_theme != current_theme:
            settings_manager.set_setting(
                "ui.theme",
                new_theme,
                SettingType.UI_THEME,
                SettingScope.GLOBAL,
                "UI 테마"
            )
            st.success(f"✅ 테마가 {theme_options[new_theme]}로 변경되었습니다.")
            st.info("🔄 브라우저를 새로고침하면 테마가 적용됩니다.")
    
    with col2:
        # 언어 선택
        current_language = settings_manager.get_setting("ui.language", "ko")
        
        language_options = {
            "ko": "한국어",
            "en": "English",
            "ja": "日本語",
            "zh": "中文"
        }
        
        new_language = st.selectbox(
            "인터페이스 언어",
            options=list(language_options.keys()),
            index=list(language_options.keys()).index(current_language) if current_language in language_options else 0,
            format_func=lambda x: language_options.get(x, x),
            help="사용자 인터페이스 언어를 설정합니다.",
            key="ui_language"
        )
        
        if new_language != current_language:
            settings_manager.set_setting(
                "ui.language",
                new_language,
                SettingType.UI_THEME,
                SettingScope.GLOBAL,
                "인터페이스 언어"
            )
            st.success(f"✅ 언어가 {language_options[new_language]}로 변경되었습니다.")
            st.info("🔄 페이지를 새로고침하면 언어가 적용됩니다.")

def _render_profile_settings(settings_manager) -> None:
    """프로필 설정 렌더링"""
    st.subheader("👤 사용자 프로필")
    
    profile = settings_manager.get_user_profile()
    
    if profile:
        col1, col2 = st.columns(2)
        
        with col1:
            # 사용자명
            new_username = st.text_input(
                "사용자명",
                value=profile.username,
                help="표시될 사용자명을 입력하세요.",
                key="username"
            )
            
            # 이메일
            new_email = st.text_input(
                "이메일",
                value=profile.email or "",
                help="알림을 받을 이메일 주소를 입력하세요.",
                key="email"
            )
        
        with col2:
            # 선호 언어
            new_preferred_language = st.selectbox(
                "선호 언어",
                options=["ko", "en", "ja", "zh"],
                index=["ko", "en", "ja", "zh"].index(profile.preferred_language) if profile.preferred_language in ["ko", "en", "ja", "zh"] else 0,
                format_func=lambda x: {"ko": "한국어", "en": "English", "ja": "日本語", "zh": "中文"}.get(x, x),
                help="분석 결과에 사용될 기본 언어입니다.",
                key="preferred_language"
            )
            
            # 시간대
            new_timezone = st.selectbox(
                "시간대",
                options=["Asia/Seoul", "UTC", "America/New_York", "Europe/London", "Asia/Tokyo"],
                index=["Asia/Seoul", "UTC", "America/New_York", "Europe/London", "Asia/Tokyo"].index(profile.timezone) if profile.timezone in ["Asia/Seoul", "UTC", "America/New_York", "Europe/London", "Asia/Tokyo"] else 0,
                help="시간 표시에 사용될 시간대입니다.",
                key="timezone"
            )
        
        # 변경사항 저장
        if st.button("💾 프로필 저장", key="save_profile"):
            update_data = {}
            
            if new_username != profile.username:
                update_data['username'] = new_username
            
            if new_email != (profile.email or ""):
                update_data['email'] = new_email if new_email else None
            
            if new_preferred_language != profile.preferred_language:
                update_data['preferred_language'] = new_preferred_language
            
            if new_timezone != profile.timezone:
                update_data['timezone'] = new_timezone
            
            if update_data:
                success = settings_manager.update_user_profile(**update_data)
                if success:
                    st.success("✅ 프로필이 저장되었습니다.")
                    st.rerun()
                else:
                    st.error("❌ 프로필 저장에 실패했습니다.")
            else:
                st.info("ℹ️ 변경된 내용이 없습니다.")
        
        # 프로필 정보 표시
        with st.expander("📊 프로필 정보"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("사용자 ID", profile.user_id)
                st.metric("생성일", profile.created_at.strftime("%Y-%m-%d %H:%M"))
            
            with col2:
                if profile.last_login:
                    st.metric("마지막 로그인", profile.last_login.strftime("%Y-%m-%d %H:%M"))
                st.metric("설정 버전", profile.settings_version)
    else:
        st.error("❌ 프로필 정보를 로드할 수 없습니다.")

def _render_backup_restore(settings_manager) -> None:
    """백업/복원 설정 렌더링"""
    st.subheader("💾 백업 및 복원")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📤 설정 내보내기")
        
        if st.button("📁 설정 백업", key="export_settings"):
            try:
                export_path = settings_manager.export_settings()
                st.success(f"✅ 설정이 백업되었습니다:")
                st.code(str(export_path))
                
                # 다운로드 링크 제공
                with open(export_path, 'r', encoding='utf-8') as f:
                    backup_data = f.read()
                
                st.download_button(
                    label="💾 백업 파일 다운로드",
                    data=backup_data,
                    file_name=f"solomond_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"❌ 백업 실패: {e}")
        
        # 설정 요약
        summary = settings_manager.get_settings_summary()
        
        with st.expander("📊 현재 설정 요약"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("총 설정 수", summary['total_settings'])
                st.metric("사용자", summary['user_profile']['username'] or "알 수 없음")
            
            with col_b:
                st.write("**타입별 분포:**")
                for setting_type, count in summary['settings_by_type'].items():
                    st.write(f"- {setting_type}: {count}개")
    
    with col2:
        st.markdown("### 📥 설정 가져오기")
        
        uploaded_file = st.file_uploader(
            "백업 파일 선택",
            type=['json'],
            help="이전에 백업한 설정 파일을 선택하세요.",
            key="import_file"
        )
        
        if uploaded_file is not None:
            merge_mode = st.radio(
                "가져오기 모드",
                options=["merge", "overwrite"],
                format_func=lambda x: "병합 (기존 설정 유지)" if x == "merge" else "덮어쓰기 (기존 설정 교체)",
                help="병합: 새 설정과 기존 설정을 합치고, 충돌 시 더 최신 설정 사용\n덮어쓰기: 기존 설정을 모두 교체",
                key="import_mode"
            )
            
            if st.button("📥 설정 가져오기", key="import_settings"):
                try:
                    # 임시 파일로 저장
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                        temp_file.write(uploaded_file.getvalue().decode('utf-8'))
                        temp_path = Path(temp_file.name)
                    
                    # 설정 가져오기
                    success = settings_manager.import_settings(temp_path, merge=(merge_mode == "merge"))
                    
                    # 임시 파일 삭제
                    temp_path.unlink()
                    
                    if success:
                        mode_text = "병합" if merge_mode == "merge" else "덮어쓰기"
                        st.success(f"✅ 설정을 성공적으로 가져왔습니다. (모드: {mode_text})")
                        st.info("🔄 페이지를 새로고침하면 새 설정이 적용됩니다.")
                        
                        if st.button("🔄 지금 새로고침", key="refresh_after_import"):
                            st.rerun()
                    else:
                        st.error("❌ 설정 가져오기에 실패했습니다.")
                        
                except Exception as e:
                    st.error(f"❌ 설정 가져오기 오류: {e}")
        
        # 기본값으로 리셋
        st.markdown("### 🔄 설정 초기화")
        
        reset_type = st.selectbox(
            "초기화 범위",
            options=["all", "ai_model", "analysis", "performance", "ui_theme"],
            format_func=lambda x: {
                "all": "모든 설정",
                "ai_model": "AI 모델 설정만",
                "analysis": "분석 설정만", 
                "performance": "성능 설정만",
                "ui_theme": "UI 테마만"
            }.get(x, x),
            key="reset_type"
        )
        
        if st.button("⚠️ 설정 초기화", key="reset_settings"):
            try:
                if reset_type == "all":
                    settings_manager.reset_settings_to_default()
                else:
                    setting_type_map = {
                        "ai_model": SettingType.AI_MODEL,
                        "analysis": SettingType.ANALYSIS,
                        "performance": SettingType.PERFORMANCE,
                        "ui_theme": SettingType.UI_THEME
                    }
                    settings_manager.reset_settings_to_default(setting_type_map[reset_type])
                
                st.success("✅ 설정이 기본값으로 초기화되었습니다.")
                st.info("🔄 페이지를 새로고침하면 초기화된 설정이 적용됩니다.")
                
                if st.button("🔄 지금 새로고침", key="refresh_after_reset"):
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ 설정 초기화 오류: {e}")

def render_settings_sidebar() -> None:
    """사이드바에 간단한 설정 표시"""
    if not USER_SETTINGS_AVAILABLE:
        return
    
    try:
        settings_manager = get_global_settings_manager()
        profile = settings_manager.get_user_profile()
        
        if profile:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 👤 현재 사용자")
            st.sidebar.write(f"**{profile.username}**")
            
            # 주요 설정 표시
            whisper_model = settings_manager.get_setting("ai.whisper_model_size", "base")
            max_memory = settings_manager.get_setting("performance.max_memory_mb", 2048.0)
            
            st.sidebar.write(f"🤖 Whisper: {whisper_model}")
            st.sidebar.write(f"💾 최대 메모리: {max_memory:.0f}MB")
            
            if st.sidebar.button("⚙️ 설정 열기", key="open_settings_sidebar"):
                st.session_state.show_settings = True
    
    except Exception as e:
        st.sidebar.error(f"설정 로드 오류: {e}")

# 사용 예시
if __name__ == "__main__":
    import streamlit as st
    
    st.set_page_config(
        page_title="사용자 설정",
        page_icon="⚙️",
        layout="wide"
    )
    
    render_user_settings_panel()