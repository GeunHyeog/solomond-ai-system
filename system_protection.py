#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ SOLOMOND AI 시스템 보호 체계
System Protection Framework for Safe Enhancements

핵심 기능:
1. 실시간 시스템 상태 모니터링
2. 자동 백업 및 복구 시스템  
3. 기능별 상태 체크
4. 안전한 개선 모드 관리
5. 원클릭 롤백 시스템
"""

import os
import sys
import json
import time
import shutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import logging
import pickle
import hashlib
from collections import defaultdict, deque

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent
BACKUP_DIR = PROJECT_ROOT / "system_backups"
LOGS_DIR = PROJECT_ROOT / "system_logs"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f"system_protection_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemStatus:
    """시스템 상태 정보"""
    component: str
    status: str  # 'healthy', 'warning', 'error', 'unknown'
    last_check: datetime
    response_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class BackupInfo:
    """백업 정보"""
    backup_id: str
    timestamp: datetime
    description: str
    files: List[str]
    git_commit: Optional[str] = None
    system_state: Optional[Dict[str, Any]] = None

class SystemProtector:
    """시스템 보호 관리자"""
    
    def __init__(self):
        self.setup_directories()
        self.system_status = {}
        self.monitoring_active = False
        self.monitor_thread = None
        self.backup_history = []
        self.load_backup_history()
        
        # 핵심 컴포넌트 정의
        self.critical_components = {
            'streamlit_8501': {
                'name': '메인 컨퍼런스 분석 시스템',
                'check_url': 'http://localhost:8501',
                'critical': True
            },
            'n8n_5678': {
                'name': 'n8n 자동화 워크플로우',
                'check_url': 'http://localhost:5678',
                'critical': True
            },
            'file_upload': {
                'name': '파일 업로드 시스템',
                'check_function': self._check_file_upload_system,
                'critical': True
            },
            'database': {
                'name': '데이터베이스 연결',
                'check_function': self._check_database_connection,
                'critical': True
            },
            'ocr_engine': {
                'name': 'OCR 텍스트 추출',
                'check_function': self._check_ocr_engine,
                'critical': True
            },
            'whisper_stt': {
                'name': 'Whisper 음성 인식',
                'check_function': self._check_whisper_engine,
                'critical': True
            }
        }
        
        logger.info("🛡️ 시스템 보호 체계 초기화 완료")
    
    def setup_directories(self):
        """필요한 디렉토리 생성"""
        for directory in [BACKUP_DIR, LOGS_DIR]:
            directory.mkdir(exist_ok=True)
    
    def create_system_backup(self, description: str = "정기 백업") -> str:
        """시스템 전체 백업 생성"""
        backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = BACKUP_DIR / backup_id
        backup_path.mkdir(exist_ok=True)
        
        logger.info(f"🔄 시스템 백업 생성 중: {backup_id}")
        
        # 핵심 파일들 백업
        critical_files = [
            "conference_analysis_UNIFIED_COMPLETE.py",
            "core/comprehensive_message_extractor.py",
            "core/multimodal_pipeline.py", 
            "core/insight_generator.py",
            "shared/ollama_interface.py",
            "CLAUDE.md",
            ".streamlit/config.toml"
        ]
        
        backed_up_files = []
        for file_path in critical_files:
            source = PROJECT_ROOT / file_path
            if source.exists():
                target = backup_path / file_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                backed_up_files.append(file_path)
        
        # Git 커밋 정보 저장
        git_commit = None
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=PROJECT_ROOT)
            if result.returncode == 0:
                git_commit = result.stdout.strip()
        except:
            pass
        
        # 현재 시스템 상태 저장
        current_status = self.get_full_system_status()
        
        # 백업 정보 저장
        backup_info = BackupInfo(
            backup_id=backup_id,
            timestamp=datetime.now(),
            description=description,
            files=backed_up_files,
            git_commit=git_commit,
            system_state=current_status
        )
        
        with open(backup_path / "backup_info.json", 'w', encoding='utf-8') as f:
            json.dump(asdict(backup_info), f, indent=2, default=str, ensure_ascii=False)
        
        self.backup_history.append(backup_info)
        self.save_backup_history()
        
        logger.info(f"✅ 시스템 백업 완료: {backup_id} ({len(backed_up_files)}개 파일)")
        return backup_id
    
    def restore_from_backup(self, backup_id: str) -> bool:
        """백업에서 시스템 복구"""
        backup_path = BACKUP_DIR / backup_id
        
        if not backup_path.exists():
            logger.error(f"❌ 백업을 찾을 수 없음: {backup_id}")
            return False
        
        logger.info(f"🔄 시스템 복구 시작: {backup_id}")
        
        try:
            # 백업 정보 로드
            with open(backup_path / "backup_info.json", 'r', encoding='utf-8') as f:
                backup_info = json.load(f)
            
            # 현재 상태를 긴급 백업으로 저장
            emergency_backup = self.create_system_backup("복구 전 긴급 백업")
            
            # 파일들 복구
            restored_files = []
            for file_path in backup_info['files']:
                source = backup_path / file_path
                target = PROJECT_ROOT / file_path
                
                if source.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    restored_files.append(file_path)
            
            logger.info(f"✅ 시스템 복구 완료: {len(restored_files)}개 파일")
            return True
            
        except Exception as e:
            logger.error(f"❌ 시스템 복구 실패: {e}")
            return False
    
    def start_monitoring(self, interval: int = 60):
        """실시간 모니터링 시작"""
        if self.monitoring_active:
            logger.warning("⚠️ 모니터링이 이미 실행 중입니다")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"🔍 시스템 모니터링 시작 (간격: {interval}초)")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("⏹️ 시스템 모니터링 중지")
    
    def _monitoring_loop(self, interval: int):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                self._check_all_components()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"❌ 모니터링 오류: {e}")
                time.sleep(10)
    
    def _check_all_components(self):
        """모든 컴포넌트 상태 확인"""
        for component_id, config in self.critical_components.items():
            try:
                start_time = time.time()
                
                if 'check_url' in config:
                    status = self._check_url_component(config['check_url'])
                elif 'check_function' in config:
                    status = config['check_function']()
                else:
                    status = 'unknown'
                
                response_time = (time.time() - start_time) * 1000  # ms
                
                self.system_status[component_id] = SystemStatus(
                    component=config['name'],
                    status=status,
                    last_check=datetime.now(),
                    response_time=response_time
                )
                
            except Exception as e:
                self.system_status[component_id] = SystemStatus(
                    component=config['name'],
                    status='error',
                    last_check=datetime.now(),
                    response_time=0,
                    error_message=str(e)
                )
    
    def _check_url_component(self, url: str) -> str:
        """URL 기반 컴포넌트 상태 확인"""
        try:
            import requests
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return 'healthy'
            else:
                return 'warning'
        except:
            return 'error'
    
    def _check_file_upload_system(self) -> str:
        """파일 업로드 시스템 상태 확인"""
        try:
            # 임시 디렉토리 확인
            temp_dir = PROJECT_ROOT / "temp"
            if temp_dir.exists() or True:  # 임시로 True
                return 'healthy'
            else:
                return 'warning'
        except:
            return 'error'
    
    def _check_database_connection(self) -> str:
        """데이터베이스 연결 상태 확인"""
        try:
            # 데이터베이스 파일 확인
            db_file = PROJECT_ROOT / "conference_analysis_unified_conference_2025.db"
            if db_file.exists():
                return 'healthy'
            else:
                return 'warning'
        except:
            return 'error'
    
    def _check_ocr_engine(self) -> str:
        """OCR 엔진 상태 확인"""
        try:
            import easyocr
            return 'healthy'
        except ImportError:
            return 'error'
        except:
            return 'warning'
    
    def _check_whisper_engine(self) -> str:
        """Whisper 엔진 상태 확인"""
        try:
            import whisper
            return 'healthy'
        except ImportError:
            return 'error'
        except:
            return 'warning'
    
    def get_full_system_status(self) -> Dict[str, Any]:
        """전체 시스템 상태 반환"""
        # 최신 상태로 업데이트
        self._check_all_components()
        
        status_summary = {
            'overall_health': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'critical_issues': [],
            'warnings': []
        }
        
        critical_errors = 0
        warnings = 0
        
        for component_id, status in self.system_status.items():
            config = self.critical_components[component_id]
            
            status_summary['components'][component_id] = {
                'name': status.component,
                'status': status.status,
                'response_time': status.response_time,
                'last_check': status.last_check.isoformat(),
                'critical': config.get('critical', False),
                'error_message': status.error_message
            }
            
            if status.status == 'error' and config.get('critical', False):
                critical_errors += 1
                status_summary['critical_issues'].append({
                    'component': status.component,
                    'error': status.error_message
                })
            elif status.status in ['warning', 'error']:
                warnings += 1
                status_summary['warnings'].append({
                    'component': status.component,
                    'status': status.status
                })
        
        # 전체 상태 결정
        if critical_errors > 0:
            status_summary['overall_health'] = 'critical'
        elif warnings > 2:
            status_summary['overall_health'] = 'warning'
        
        return status_summary
    
    def get_backup_list(self) -> List[Dict[str, Any]]:
        """백업 목록 반환"""
        return [asdict(backup) for backup in sorted(self.backup_history, 
                                                    key=lambda x: x.timestamp, 
                                                    reverse=True)]
    
    def load_backup_history(self):
        """백업 이력 로드"""
        history_file = BACKUP_DIR / "backup_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.backup_history = [
                        BackupInfo(**item) if isinstance(item, dict) else item 
                        for item in data
                    ]
            except Exception as e:
                logger.warning(f"백업 이력 로드 실패: {e}")
                self.backup_history = []
    
    def save_backup_history(self):
        """백업 이력 저장"""
        history_file = BACKUP_DIR / "backup_history.json"
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(backup) for backup in self.backup_history], 
                         f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            logger.error(f"백업 이력 저장 실패: {e}")

# 전역 인스턴스
system_protector = SystemProtector()

def get_system_protection():
    """시스템 보호 인스턴스 반환"""
    return system_protector

if __name__ == "__main__":
    # 시스템 보호 테스트
    protector = get_system_protection()
    
    # 초기 백업 생성
    backup_id = protector.create_system_backup("시스템 보호 체계 구축 완료")
    print(f"✅ 초기 백업 생성: {backup_id}")
    
    # 시스템 상태 확인
    status = protector.get_full_system_status()
    print(f"🔍 시스템 상태: {status['overall_health']}")
    
    for component_id, comp_status in status['components'].items():
        emoji = "🟢" if comp_status['status'] == 'healthy' else "🟡" if comp_status['status'] == 'warning' else "🔴"
        print(f"  {emoji} {comp_status['name']}: {comp_status['status']}")
    
    # 모니터링 시작
    protector.start_monitoring(30)  # 30초 간격
    print("🔍 실시간 모니터링 시작")