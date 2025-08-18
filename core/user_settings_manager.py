#!/usr/bin/env python3
"""
사용자 설정 관리 시스템 v2.6
개인화된 사용자 설정 저장 및 관리
"""

import os
import json
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
from enum import Enum
import hashlib
import uuid

class SettingType(Enum):
    """설정 타입"""
    SYSTEM = "system"
    USER_PREFERENCE = "user_preference"
    AI_MODEL = "ai_model"
    ANALYSIS = "analysis"
    UI_THEME = "ui_theme"
    PERFORMANCE = "performance"

class SettingScope(Enum):
    """설정 범위"""
    GLOBAL = "global"
    SESSION = "session"
    TEMPORARY = "temporary"

@dataclass
class UserSetting:
    """사용자 설정 항목"""
    key: str
    value: Any
    setting_type: SettingType
    scope: SettingScope
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_encrypted: bool = False
    version: str = "1.0"

@dataclass
class UserProfile:
    """사용자 프로필"""
    user_id: str
    username: str
    email: Optional[str] = None
    preferred_language: str = "ko"
    timezone: str = "Asia/Seoul"
    theme: str = "auto"
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    settings_version: str = "2.6"
    custom_preferences: Dict[str, Any] = field(default_factory=dict)

class UserSettingsManager:
    """사용자 설정 관리 시스템"""
    
    def __init__(self, settings_dir: Optional[Path] = None, enable_encryption: bool = True):
        self.settings_dir = settings_dir or Path.home() / ".solomond_ai" / "settings"
        self.enable_encryption = enable_encryption
        self.lock = threading.RLock()
        self.logger = self._setup_logging()
        
        # 현재 사용자 프로필
        self.current_profile: Optional[UserProfile] = None
        self.current_user_id: Optional[str] = None
        
        # 설정 저장소
        self.settings_cache: Dict[str, UserSetting] = {}
        self.global_settings: Dict[str, UserSetting] = {}
        self.session_settings: Dict[str, UserSetting] = {}
        
        # 변경 감지 콜백
        self.change_callbacks: List[Callable] = []
        
        # 기본 설정
        self.default_settings = {
            # 시스템 설정
            "system.auto_save": UserSetting(
                key="system.auto_save",
                value=True,
                setting_type=SettingType.SYSTEM,
                scope=SettingScope.GLOBAL,
                description="자동 저장 활성화"
            ),
            "system.auto_backup": UserSetting(
                key="system.auto_backup",
                value=True,
                setting_type=SettingType.SYSTEM,
                scope=SettingScope.GLOBAL,
                description="자동 백업 활성화"
            ),
            
            # AI 모델 설정
            "ai.whisper_model_size": UserSetting(
                key="ai.whisper_model_size",
                value="base",
                setting_type=SettingType.AI_MODEL,
                scope=SettingScope.GLOBAL,
                description="Whisper 모델 크기"
            ),
            "ai.easyocr_languages": UserSetting(
                key="ai.easyocr_languages",
                value=["ko", "en"],
                setting_type=SettingType.AI_MODEL,
                scope=SettingScope.GLOBAL,
                description="EasyOCR 지원 언어"
            ),
            "ai.transformers_model": UserSetting(
                key="ai.transformers_model",
                value="facebook/bart-large-cnn",
                setting_type=SettingType.AI_MODEL,
                scope=SettingScope.GLOBAL,
                description="Transformers 요약 모델"
            ),
            
            # 분석 설정
            "analysis.max_file_size_mb": UserSetting(
                key="analysis.max_file_size_mb",
                value=500.0,
                setting_type=SettingType.ANALYSIS,
                scope=SettingScope.GLOBAL,
                description="최대 파일 크기 (MB)"
            ),
            "analysis.batch_size": UserSetting(
                key="analysis.batch_size",
                value=5,
                setting_type=SettingType.ANALYSIS,
                scope=SettingScope.GLOBAL,
                description="배치 처리 크기"
            ),
            "analysis.enable_streaming": UserSetting(
                key="analysis.enable_streaming",
                value=True,
                setting_type=SettingType.ANALYSIS,
                scope=SettingScope.GLOBAL,
                description="스트리밍 분석 활성화"
            ),
            
            # UI 테마 설정
            "ui.theme": UserSetting(
                key="ui.theme",
                value="auto",
                setting_type=SettingType.UI_THEME,
                scope=SettingScope.GLOBAL,
                description="UI 테마 (light/dark/auto)"
            ),
            "ui.language": UserSetting(
                key="ui.language",
                value="ko",
                setting_type=SettingType.UI_THEME,
                scope=SettingScope.GLOBAL,
                description="인터페이스 언어"
            ),
            
            # 성능 설정
            "performance.max_memory_mb": UserSetting(
                key="performance.max_memory_mb",
                value=2048.0,
                setting_type=SettingType.PERFORMANCE,
                scope=SettingScope.GLOBAL,
                description="최대 메모리 사용량 (MB)"
            ),
            "performance.enable_gpu": UserSetting(
                key="performance.enable_gpu",
                value=False,
                setting_type=SettingType.PERFORMANCE,
                scope=SettingScope.GLOBAL,
                description="GPU 사용 활성화"
            )
        }
        
        # 디렉토리 생성
        self._ensure_directories()
        
        # 기본 사용자 생성 또는 로드
        self._initialize_default_user()
        
        self.logger.info("⚙️ 사용자 설정 관리 시스템 초기화 완료")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _ensure_directories(self) -> None:
        """필요한 디렉토리 생성"""
        directories = [
            self.settings_dir,
            self.settings_dir / "profiles",
            self.settings_dir / "global",
            self.settings_dir / "sessions",
            self.settings_dir / "backups"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _initialize_default_user(self) -> None:
        """기본 사용자 초기화"""
        default_user_id = "default_user"
        profile_path = self.settings_dir / "profiles" / f"{default_user_id}.json"
        
        if profile_path.exists():
            # 기존 프로필 로드
            self.load_user_profile(default_user_id)
        else:
            # 새 프로필 생성
            profile = UserProfile(
                user_id=default_user_id,
                username="기본 사용자",
                preferred_language="ko",
                theme="auto"
            )
            self.create_user_profile(profile)
            self.load_user_profile(default_user_id)
    
    def create_user_profile(self, profile: UserProfile) -> bool:
        """사용자 프로필 생성"""
        try:
            with self.lock:
                profile_path = self.settings_dir / "profiles" / f"{profile.user_id}.json"
                
                if profile_path.exists():
                    self.logger.warning(f"⚠️ 프로필이 이미 존재함: {profile.user_id}")
                    return False
                
                # 프로필 저장
                profile_data = asdict(profile)
                profile_data['created_at'] = profile.created_at.isoformat()
                if profile.last_login:
                    profile_data['last_login'] = profile.last_login.isoformat()
                
                with open(profile_path, 'w', encoding='utf-8') as f:
                    json.dump(profile_data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"👤 새 사용자 프로필 생성: {profile.username} ({profile.user_id})")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ 프로필 생성 실패 {profile.user_id}: {e}")
            return False
    
    def load_user_profile(self, user_id: str) -> bool:
        """사용자 프로필 로드"""
        try:
            with self.lock:
                profile_path = self.settings_dir / "profiles" / f"{user_id}.json"
                
                if not profile_path.exists():
                    self.logger.error(f"❌ 프로필을 찾을 수 없음: {user_id}")
                    return False
                
                # 프로필 로드
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                
                # datetime 필드 변환
                profile_data['created_at'] = datetime.fromisoformat(profile_data['created_at'])
                if profile_data.get('last_login'):
                    profile_data['last_login'] = datetime.fromisoformat(profile_data['last_login'])
                
                self.current_profile = UserProfile(**profile_data)
                self.current_user_id = user_id
                
                # 사용자 설정 로드
                self._load_user_settings(user_id)
                
                # 마지막 로그인 시간 업데이트
                self.current_profile.last_login = datetime.now()
                self._save_user_profile()
                
                self.logger.info(f"👤 사용자 프로필 로드: {self.current_profile.username}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ 프로필 로드 실패 {user_id}: {e}")
            return False
    
    def _save_user_profile(self) -> None:
        """현재 사용자 프로필 저장"""
        if not self.current_profile:
            return
        
        try:
            profile_path = self.settings_dir / "profiles" / f"{self.current_profile.user_id}.json"
            
            profile_data = asdict(self.current_profile)
            profile_data['created_at'] = self.current_profile.created_at.isoformat()
            if self.current_profile.last_login:
                profile_data['last_login'] = self.current_profile.last_login.isoformat()
            
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"❌ 프로필 저장 실패: {e}")
    
    def _load_user_settings(self, user_id: str) -> None:
        """사용자 설정 로드"""
        try:
            # 전역 설정 로드
            global_settings_path = self.settings_dir / "global" / f"{user_id}_global.json"
            if global_settings_path.exists():
                with open(global_settings_path, 'r', encoding='utf-8') as f:
                    global_data = json.load(f)
                
                for key, setting_data in global_data.items():
                    # datetime 필드 변환
                    setting_data['created_at'] = datetime.fromisoformat(setting_data['created_at'])
                    setting_data['updated_at'] = datetime.fromisoformat(setting_data['updated_at'])
                    setting_data['setting_type'] = SettingType(setting_data['setting_type'])
                    setting_data['scope'] = SettingScope(setting_data['scope'])
                    
                    setting = UserSetting(**setting_data)
                    self.global_settings[key] = setting
                    self.settings_cache[key] = setting
            
            # 기본 설정으로 누락된 항목 채우기
            for key, default_setting in self.default_settings.items():
                if key not in self.settings_cache:
                    self.settings_cache[key] = default_setting
                    if default_setting.scope == SettingScope.GLOBAL:
                        self.global_settings[key] = default_setting
            
            self.logger.debug(f"📥 사용자 설정 로드: {len(self.settings_cache)}개 설정")
            
        except Exception as e:
            self.logger.error(f"❌ 사용자 설정 로드 실패: {e}")
            # 기본 설정 사용
            self.settings_cache.update(self.default_settings)
            for key, setting in self.default_settings.items():
                if setting.scope == SettingScope.GLOBAL:
                    self.global_settings[key] = setting
    
    def set_setting(self, key: str, value: Any, setting_type: Optional[SettingType] = None,
                   scope: Optional[SettingScope] = None, description: str = "") -> bool:
        """설정 값 설정"""
        try:
            with self.lock:
                # 기존 설정이 있으면 타입과 범위 유지
                if key in self.settings_cache:
                    existing = self.settings_cache[key]
                    setting_type = setting_type or existing.setting_type
                    scope = scope or existing.scope
                    description = description or existing.description
                else:
                    setting_type = setting_type or SettingType.USER_PREFERENCE
                    scope = scope or SettingScope.GLOBAL
                
                # 새 설정 생성
                setting = UserSetting(
                    key=key,
                    value=value,
                    setting_type=setting_type,
                    scope=scope,
                    description=description,
                    updated_at=datetime.now()
                )
                
                # 캐시 업데이트
                self.settings_cache[key] = setting
                
                # 범위별 저장소 업데이트
                if scope == SettingScope.GLOBAL:
                    self.global_settings[key] = setting
                elif scope == SettingScope.SESSION:
                    self.session_settings[key] = setting
                
                # 영구 저장 (TEMPORARY 제외)
                if scope != SettingScope.TEMPORARY:
                    self._save_settings()
                
                # 변경 콜백 호출
                self._trigger_change_callbacks(key, value)
                
                self.logger.debug(f"⚙️ 설정 업데이트: {key} = {value}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ 설정 저장 실패 {key}: {e}")
            return False
    
    def get_setting(self, key: str, default_value: Any = None) -> Any:
        """설정 값 가져오기"""
        with self.lock:
            if key in self.settings_cache:
                setting = self.settings_cache[key]
                self.logger.debug(f"📖 설정 조회: {key} = {setting.value}")
                return setting.value
            
            # 기본값 반환
            if default_value is not None:
                return default_value
            
            # 기본 설정에서 찾기
            if key in self.default_settings:
                return self.default_settings[key].value
            
            return None
    
    def get_setting_info(self, key: str) -> Optional[UserSetting]:
        """설정 정보 가져오기"""
        with self.lock:
            return self.settings_cache.get(key)
    
    def delete_setting(self, key: str) -> bool:
        """설정 삭제"""
        try:
            with self.lock:
                if key in self.settings_cache:
                    setting = self.settings_cache[key]
                    
                    # 캐시에서 제거
                    del self.settings_cache[key]
                    
                    # 범위별 저장소에서 제거
                    if key in self.global_settings:
                        del self.global_settings[key]
                    if key in self.session_settings:
                        del self.session_settings[key]
                    
                    # 영구 저장
                    if setting.scope != SettingScope.TEMPORARY:
                        self._save_settings()
                    
                    self.logger.info(f"🗑️ 설정 삭제: {key}")
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 설정 삭제 실패 {key}: {e}")
            return False
    
    def get_settings_by_type(self, setting_type: SettingType) -> Dict[str, UserSetting]:
        """타입별 설정 조회"""
        with self.lock:
            return {
                key: setting for key, setting in self.settings_cache.items()
                if setting.setting_type == setting_type
            }
    
    def get_settings_by_scope(self, scope: SettingScope) -> Dict[str, UserSetting]:
        """범위별 설정 조회"""
        with self.lock:
            return {
                key: setting for key, setting in self.settings_cache.items()
                if setting.scope == scope
            }
    
    def reset_settings_to_default(self, setting_type: Optional[SettingType] = None) -> None:
        """설정을 기본값으로 리셋"""
        with self.lock:
            settings_to_reset = self.default_settings
            
            if setting_type:
                settings_to_reset = {
                    key: setting for key, setting in self.default_settings.items()
                    if setting.setting_type == setting_type
                }
            
            for key, default_setting in settings_to_reset.items():
                self.settings_cache[key] = default_setting
                if default_setting.scope == SettingScope.GLOBAL:
                    self.global_settings[key] = default_setting
            
            self._save_settings()
            
            reset_type = setting_type.value if setting_type else "모든"
            self.logger.info(f"🔄 설정 리셋: {reset_type} 설정을 기본값으로 복원")
    
    def export_settings(self, export_path: Optional[Path] = None) -> Path:
        """설정 내보내기"""
        try:
            if not export_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = self.settings_dir / "backups" / f"settings_export_{timestamp}.json"
            
            export_data = {
                'profile': asdict(self.current_profile) if self.current_profile else None,
                'settings': {},
                'export_timestamp': datetime.now().isoformat(),
                'version': "2.6"
            }
            
            # 설정 데이터 준비
            for key, setting in self.settings_cache.items():
                setting_data = asdict(setting)
                setting_data['created_at'] = setting.created_at.isoformat()
                setting_data['updated_at'] = setting.updated_at.isoformat()
                setting_data['setting_type'] = setting.setting_type.value
                setting_data['scope'] = setting.scope.value
                export_data['settings'][key] = setting_data
            
            # 프로필 datetime 필드 처리
            if export_data['profile']:
                export_data['profile']['created_at'] = self.current_profile.created_at.isoformat()
                if self.current_profile.last_login:
                    export_data['profile']['last_login'] = self.current_profile.last_login.isoformat()
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📤 설정 내보내기 완료: {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"❌ 설정 내보내기 실패: {e}")
            raise
    
    def import_settings(self, import_path: Path, merge: bool = True) -> bool:
        """설정 가져오기"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            with self.lock:
                # 프로필 가져오기 (선택적)
                if import_data.get('profile') and not merge:
                    profile_data = import_data['profile']
                    profile_data['created_at'] = datetime.fromisoformat(profile_data['created_at'])
                    if profile_data.get('last_login'):
                        profile_data['last_login'] = datetime.fromisoformat(profile_data['last_login'])
                    
                    self.current_profile = UserProfile(**profile_data)
                
                # 설정 가져오기
                imported_settings = import_data.get('settings', {})
                
                for key, setting_data in imported_settings.items():
                    # datetime 필드 변환
                    setting_data['created_at'] = datetime.fromisoformat(setting_data['created_at'])
                    setting_data['updated_at'] = datetime.fromisoformat(setting_data['updated_at'])
                    setting_data['setting_type'] = SettingType(setting_data['setting_type'])
                    setting_data['scope'] = SettingScope(setting_data['scope'])
                    
                    setting = UserSetting(**setting_data)
                    
                    # 병합 모드에 따라 처리
                    if merge and key in self.settings_cache:
                        # 기존 설정이 더 최신이면 스킵
                        existing = self.settings_cache[key]
                        if existing.updated_at > setting.updated_at:
                            continue
                    
                    self.settings_cache[key] = setting
                    if setting.scope == SettingScope.GLOBAL:
                        self.global_settings[key] = setting
                
                # 저장
                self._save_settings()
                self._save_user_profile()
                
                mode = "병합" if merge else "덮어쓰기"
                self.logger.info(f"📥 설정 가져오기 완료: {len(imported_settings)}개 설정 ({mode})")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ 설정 가져오기 실패: {e}")
            return False
    
    def _save_settings(self) -> None:
        """설정 저장"""
        if not self.current_user_id:
            return
        
        try:
            # 전역 설정 저장
            global_settings_path = self.settings_dir / "global" / f"{self.current_user_id}_global.json"
            global_data = {}
            
            for key, setting in self.global_settings.items():
                setting_data = asdict(setting)
                setting_data['created_at'] = setting.created_at.isoformat()
                setting_data['updated_at'] = setting.updated_at.isoformat()
                setting_data['setting_type'] = setting.setting_type.value
                setting_data['scope'] = setting.scope.value
                global_data[key] = setting_data
            
            with open(global_settings_path, 'w', encoding='utf-8') as f:
                json.dump(global_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"💾 전역 설정 저장: {len(global_data)}개 설정")
            
        except Exception as e:
            self.logger.error(f"❌ 설정 저장 실패: {e}")
    
    def add_change_callback(self, callback: Callable[[str, Any], None]) -> None:
        """설정 변경 콜백 추가"""
        with self.lock:
            if callback not in self.change_callbacks:
                self.change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[str, Any], None]) -> None:
        """설정 변경 콜백 제거"""
        with self.lock:
            if callback in self.change_callbacks:
                self.change_callbacks.remove(callback)
    
    def _trigger_change_callbacks(self, key: str, value: Any) -> None:
        """설정 변경 콜백 트리거"""
        for callback in self.change_callbacks:
            try:
                callback(key, value)
            except Exception as e:
                self.logger.error(f"❌ 설정 변경 콜백 오류: {e}")
    
    def get_user_profile(self) -> Optional[UserProfile]:
        """현재 사용자 프로필 반환"""
        return self.current_profile
    
    def update_user_profile(self, **kwargs) -> bool:
        """사용자 프로필 업데이트"""
        if not self.current_profile:
            return False
        
        try:
            with self.lock:
                for key, value in kwargs.items():
                    if hasattr(self.current_profile, key):
                        setattr(self.current_profile, key, value)
                
                self._save_user_profile()
                self.logger.info(f"👤 프로필 업데이트: {', '.join(kwargs.keys())}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ 프로필 업데이트 실패: {e}")
            return False
    
    def get_settings_summary(self) -> Dict[str, Any]:
        """설정 요약 정보"""
        with self.lock:
            type_counts = {}
            scope_counts = {}
            
            for setting in self.settings_cache.values():
                type_name = setting.setting_type.value
                scope_name = setting.scope.value
                
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
                scope_counts[scope_name] = scope_counts.get(scope_name, 0) + 1
            
            return {
                'total_settings': len(self.settings_cache),
                'user_profile': {
                    'user_id': self.current_profile.user_id if self.current_profile else None,
                    'username': self.current_profile.username if self.current_profile else None,
                    'last_login': self.current_profile.last_login.isoformat() if self.current_profile and self.current_profile.last_login else None
                },
                'settings_by_type': type_counts,
                'settings_by_scope': scope_counts,
                'settings_directory': str(self.settings_dir),
                'encryption_enabled': self.enable_encryption
            }
    
    def cleanup(self) -> None:
        """리소스 정리"""
        with self.lock:
            # 설정 저장
            self._save_settings()
            self._save_user_profile()
            
            # 콜백 정리
            self.change_callbacks.clear()
            
            self.logger.info("🧹 사용자 설정 관리 시스템 정리 완료")

# 전역 사용자 설정 관리자
_global_settings_manager = None
_global_settings_lock = threading.Lock()

def get_global_settings_manager() -> UserSettingsManager:
    """전역 사용자 설정 관리자 가져오기"""
    global _global_settings_manager
    
    with _global_settings_lock:
        if _global_settings_manager is None:
            _global_settings_manager = UserSettingsManager()
        return _global_settings_manager

# 편의 함수들
def get_user_setting(key: str, default_value: Any = None) -> Any:
    """사용자 설정 값 가져오기 (편의 함수)"""
    manager = get_global_settings_manager()
    return manager.get_setting(key, default_value)

def set_user_setting(key: str, value: Any, setting_type: Optional[SettingType] = None,
                    scope: Optional[SettingScope] = None, description: str = "") -> bool:
    """사용자 설정 값 설정 (편의 함수)"""
    manager = get_global_settings_manager()
    return manager.set_setting(key, value, setting_type, scope, description)

def get_user_profile() -> Optional[UserProfile]:
    """현재 사용자 프로필 가져오기 (편의 함수)"""
    manager = get_global_settings_manager()
    return manager.get_user_profile()

# 사용 예시
if __name__ == "__main__":
    # 사용자 설정 관리자 테스트
    manager = UserSettingsManager()
    
    # 설정 값 설정
    manager.set_setting("ai.whisper_model_size", "large", SettingType.AI_MODEL, SettingScope.GLOBAL, "Whisper 모델 크기")
    manager.set_setting("ui.theme", "dark", SettingType.UI_THEME, SettingScope.GLOBAL, "UI 테마")
    manager.set_setting("temp.current_session", "test_session", SettingType.SYSTEM, SettingScope.TEMPORARY, "임시 세션 ID")
    
    # 설정 값 조회
    whisper_model = manager.get_setting("ai.whisper_model_size")
    ui_theme = manager.get_setting("ui.theme")
    max_memory = manager.get_setting("performance.max_memory_mb")
    
    print(f"Whisper 모델: {whisper_model}")
    print(f"UI 테마: {ui_theme}")
    print(f"최대 메모리: {max_memory}MB")
    
    # 프로필 업데이트
    manager.update_user_profile(username="테스트 사용자", preferred_language="en")
    
    # 타입별 설정 조회
    ai_settings = manager.get_settings_by_type(SettingType.AI_MODEL)
    print(f"\nAI 모델 설정: {len(ai_settings)}개")
    for key, setting in ai_settings.items():
        print(f"  {key}: {setting.value}")
    
    # 설정 요약
    summary = manager.get_settings_summary()
    print(f"\n설정 요약:")
    print(f"  총 설정 수: {summary['total_settings']}")
    print(f"  사용자: {summary['user_profile']['username']}")
    print(f"  타입별 분포: {summary['settings_by_type']}")
    
    # 설정 내보내기
    export_path = manager.export_settings()
    print(f"\n설정 내보내기: {export_path}")
    
    # 정리
    manager.cleanup()
    print("\n✅ 사용자 설정 관리 시스템 테스트 완료!")