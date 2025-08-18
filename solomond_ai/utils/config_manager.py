"""
설정 관리자
프로젝트별 설정 파일 로딩 및 관리
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

class ConfigManager:
    """설정 관리 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config_data = {}
        self.default_config = self._get_default_config()
        
        if config_path:
            self.load_config(config_path)
        else:
            self.config_data = self.default_config.copy()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            "project": {
                "name": "SolomondAI Analysis",
                "domain": "general",
                "version": "1.0.0"
            },
            "engines": {
                "audio": {
                    "model": "whisper-base",
                    "language": "ko",
                    "enabled": True
                },
                "image": {
                    "ocr_engine": "easyocr",
                    "languages": ["ko", "en"],
                    "enabled": True
                },
                "video": {
                    "sample_frames": 5,
                    "enabled": True
                },
                "text": {
                    "language": "ko",
                    "use_transformers": False,
                    "enabled": True
                }
            },
            "ui": {
                "layout": "four_step",
                "theme": "default",
                "title": "솔로몬드 AI 분석 시스템"
            },
            "analysis": {
                "cross_validation": True,
                "confidence_threshold": 0.5,
                "report_format": "standard"
            },
            "processing": {
                "max_workers": 4,
                "timeout_seconds": 300,
                "memory_limit_mb": 2048
            },
            "output": {
                "save_intermediate": True,
                "output_format": "json",
                "include_raw_data": True
            }
        }
    
    def load_config(self, config_path: str) -> bool:
        """설정 파일 로딩"""
        try:
            config_file = Path(config_path)
            
            if not config_file.exists():
                logging.warning(f"Config file not found: {config_path}. Using default config.")
                self.config_data = self.default_config.copy()
                return False
            
            # 파일 확장자에 따른 로딩
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_file.suffix}")
            
            # 기본 설정과 병합
            self.config_data = self._merge_config(self.default_config, loaded_config)
            self.config_path = config_path
            
            logging.info(f"Config loaded successfully: {config_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            self.config_data = self.default_config.copy()
            return False
    
    def _merge_config(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """기본 설정과 로딩된 설정 병합"""
        merged = default.copy()
        
        for key, value in loaded.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_config(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """점 표기법으로 설정값 가져오기 (예: 'engines.audio.model')"""
        keys = key_path.split('.')
        current = self.config_data
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """점 표기법으로 설정값 설정"""
        keys = key_path.split('.')
        current = self.config_data
        
        # 중간 경로 생성
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # 마지막 키에 값 설정
        current[keys[-1]] = value
    
    def get_engine_config(self, engine_name: str) -> Dict[str, Any]:
        """특정 엔진의 설정 반환"""
        return self.get(f'engines.{engine_name}', {})
    
    def is_engine_enabled(self, engine_name: str) -> bool:
        """엔진 활성화 상태 확인"""
        return self.get(f'engines.{engine_name}.enabled', True)
    
    def get_enabled_engines(self):
        """활성화된 엔진 목록"""
        engines = self.get('engines', {})
        return [name for name, config in engines.items() if config.get('enabled', True)]
    
    def save_config(self, output_path: Optional[str] = None) -> bool:
        """설정 파일 저장"""
        try:
            save_path = output_path or self.config_path
            if not save_path:
                raise ValueError("No output path specified")
            
            save_file = Path(save_path)
            
            # 확장자에 따른 저장
            if save_file.suffix.lower() in ['.yaml', '.yml']:
                with open(save_file, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)
            elif save_file.suffix.lower() == '.json':
                with open(save_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported save format: {save_file.suffix}")
            
            logging.info(f"Config saved: {save_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
            return False
    
    def create_sample_config(self, output_path: str) -> bool:
        """샘플 설정 파일 생성"""
        try:
            sample_config = {
                "project": {
                    "name": "My Analysis Project",
                    "domain": "medical",  # jewelry, medical, education 등
                    "version": "1.0.0"
                },
                "engines": {
                    "audio": {
                        "model": "whisper-base",  # tiny, base, small, medium, large
                        "language": "ko",
                        "enabled": True
                    },
                    "image": {
                        "ocr_engine": "easyocr",
                        "languages": ["ko", "en"],
                        "enabled": True
                    },
                    "video": {
                        "sample_frames": 5,
                        "enabled": True
                    },
                    "text": {
                        "language": "ko",
                        "use_transformers": True,
                        "enabled": True
                    }
                },
                "ui": {
                    "layout": "four_step",  # four_step, dashboard, simple
                    "theme": "medical",     # default, medical, jewelry, business
                    "title": "의료 컨퍼런스 분석 시스템"
                },
                "analysis": {
                    "cross_validation": True,
                    "confidence_threshold": 0.75,
                    "report_format": "medical_standard"  # standard, medical_standard, business
                }
            }
            
            output_file = Path(output_path)
            
            if output_file.suffix.lower() in ['.yaml', '.yml']:
                with open(output_file, 'w', encoding='utf-8') as f:
                    yaml.dump(sample_config, f, default_flow_style=False, allow_unicode=True)
            else:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(sample_config, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Sample config created: {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to create sample config: {e}")
            return False
    
    def validate_config(self) -> Dict[str, Any]:
        """설정 유효성 검증"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # 필수 섹션 확인
        required_sections = ['project', 'engines', 'ui']
        for section in required_sections:
            if section not in self.config_data:
                validation_result["errors"].append(f"Missing required section: {section}")
                validation_result["valid"] = False
        
        # 엔진 설정 확인
        engines = self.config_data.get('engines', {})
        if not engines:
            validation_result["warnings"].append("No engines configured")
        
        # UI 설정 확인
        ui_layout = self.get('ui.layout')
        valid_layouts = ['four_step', 'dashboard', 'simple']
        if ui_layout and ui_layout not in valid_layouts:
            validation_result["warnings"].append(f"Unknown UI layout: {ui_layout}")
        
        return validation_result
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'ConfigManager':
        """파일에서 ConfigManager 인스턴스 생성"""
        manager = cls()
        manager.load_config(config_path)
        return manager
    
    def __repr__(self) -> str:
        return f"ConfigManager(config_path='{self.config_path}', sections={list(self.config_data.keys())})"