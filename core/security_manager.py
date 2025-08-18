#!/usr/bin/env python3
"""
보안 강화 시스템
입력 검증, 파일 안전성 검사, 경로 검증, 리소스 제한
"""

import os
import hashlib
import mimetypes
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
import time
import psutil

class SecurityLevel(Enum):
    """보안 수준"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    """위협 유형"""
    MALICIOUS_FILE = "malicious_file"
    PATH_TRAVERSAL = "path_traversal"
    OVERSIZED_FILE = "oversized_file"
    SUSPICIOUS_EXTENSION = "suspicious_extension"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    ENCODING_ATTACK = "encoding_attack"

@dataclass
class SecurityResult:
    """보안 검사 결과"""
    is_safe: bool
    security_level: SecurityLevel
    threats: List[ThreatType]
    details: Dict[str, Any]
    recommendations: List[str]

class SecurityManager:
    """보안 관리자"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH):
        self.logger = logging.getLogger(__name__)
        self.security_level = security_level
        
        # 허용된 파일 확장자
        self.allowed_extensions = {
            'audio': {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'},
            'image': {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'},
            'video': {'.mp4', '.mov', '.avi', '.mkv', '.webm'},
            'document': {'.pdf', '.txt', '.docx', '.doc'}
        }
        
        # 위험한 확장자
        self.dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.scr', '.pif', '.com', '.dll',
            '.vbs', '.js', '.jar', '.ps1', '.sh', '.py', '.pl'
        }
        
        # MIME 타입 매핑
        self.safe_mime_types = {
            'audio/wav', 'audio/mpeg', 'audio/flac', 'audio/x-m4a', 'audio/mp4', 'audio/ogg',
            'image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp',
            'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska',
            'application/pdf', 'text/plain'
        }
        
        # 파일 크기 제한 (바이트)
        self.max_file_sizes = {
            'audio': 5 * 1024 * 1024 * 1024,  # 5GB
            'image': 500 * 1024 * 1024,       # 500MB
            'video': 5 * 1024 * 1024 * 1024,  # 5GB
            'document': 200 * 1024 * 1024      # 200MB
        }
        
        # 허용된 디렉토리 패턴
        self.allowed_path_patterns = [
            r'^[A-Za-z]:[\\\/][^<>:"|?*]*$',  # Windows 경로
            r'^\/[^<>:"|?*]*$',               # Unix 경로
            r'^\.\.?[\\\/][^<>:"|?*]*$'       # 상대 경로 (제한적)
        ]
        
        # 금지된 경로 패턴
        self.forbidden_path_patterns = [
            r'\.\.[\\/]',                     # 경로 순회
            r'[<>:"|?*]',                     # 특수 문자
            r'(?i)(system32|windows|program files|etc|usr|bin)', # 시스템 디렉토리
        ]
        
        # 리소스 제한
        self.resource_limits = {
            'max_memory_mb': 2048,            # 최대 메모리 2GB
            'max_cpu_percent': 80,            # 최대 CPU 80%
            'max_processing_time': 1800,     # 최대 처리 시간 30분
            'max_concurrent_files': 10       # 최대 동시 처리 파일 수
        }
        
        self.logger.info(f"보안 관리자 초기화: {security_level.value} 수준")
    
    def validate_file_path(self, file_path: str) -> SecurityResult:
        """파일 경로 검증"""
        threats = []
        details = {}
        recommendations = []
        
        try:
            # 경로 정규화
            normalized_path = os.path.normpath(file_path)
            abs_path = os.path.abspath(normalized_path)
            
            details['original_path'] = file_path
            details['normalized_path'] = normalized_path
            details['absolute_path'] = abs_path
            
            # 경로 순회 공격 검사
            if '..' in normalized_path or any(re.search(pattern, normalized_path) for pattern in self.forbidden_path_patterns):
                threats.append(ThreatType.PATH_TRAVERSAL)
                recommendations.append("안전한 경로를 사용하세요")
            
            # 파일 존재 확인
            if not os.path.exists(abs_path):
                details['file_exists'] = False
                recommendations.append("파일이 존재하는지 확인하세요")
            else:
                details['file_exists'] = True
                
                # 파일인지 확인
                if not os.path.isfile(abs_path):
                    details['is_file'] = False
                    recommendations.append("디렉토리가 아닌 파일을 선택하세요")
                else:
                    details['is_file'] = True
            
            # 경로 길이 검사
            if len(abs_path) > 260:  # Windows 경로 제한
                recommendations.append("경로가 너무 깁니다")
            
            is_safe = len(threats) == 0 and details.get('file_exists', False) and details.get('is_file', False)
            
            return SecurityResult(
                is_safe=is_safe,
                security_level=SecurityLevel.MEDIUM if is_safe else SecurityLevel.HIGH,
                threats=threats,
                details=details,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"경로 검증 오류: {e}")
            return SecurityResult(
                is_safe=False,
                security_level=SecurityLevel.CRITICAL,
                threats=[ThreatType.PATH_TRAVERSAL],
                details={'error': str(e)},
                recommendations=["경로 검증에 실패했습니다"]
            )
    
    def validate_file_properties(self, file_path: str) -> SecurityResult:
        """파일 속성 검증"""
        threats = []
        details = {}
        recommendations = []
        
        try:
            file_path = os.path.abspath(file_path)
            
            if not os.path.exists(file_path):
                return SecurityResult(
                    is_safe=False,
                    security_level=SecurityLevel.HIGH,
                    threats=[],
                    details={'error': 'File not found'},
                    recommendations=['파일이 존재하지 않습니다']
                )
            
            # 기본 정보
            stat = os.stat(file_path)
            file_size = stat.st_size
            file_ext = Path(file_path).suffix.lower()
            
            details['file_size_bytes'] = file_size
            details['file_size_mb'] = file_size / (1024 * 1024)
            details['file_extension'] = file_ext
            details['modification_time'] = stat.st_mtime
            
            # 확장자 검사
            if file_ext in self.dangerous_extensions:
                threats.append(ThreatType.SUSPICIOUS_EXTENSION)
                recommendations.append(f"위험한 파일 확장자입니다: {file_ext}")
            
            # 파일 타입 결정 및 크기 검사
            file_category = self._determine_file_category(file_ext)
            details['file_category'] = file_category
            
            if file_category and file_size > self.max_file_sizes.get(file_category, float('inf')):
                threats.append(ThreatType.OVERSIZED_FILE)
                max_size_mb = self.max_file_sizes[file_category] / (1024 * 1024)
                recommendations.append(f"파일이 너무 큽니다. 최대 {max_size_mb:.0f}MB")
            
            # MIME 타입 검사
            try:
                mime_type, _ = mimetypes.guess_type(file_path)
                details['mime_type'] = mime_type
                
                if mime_type and mime_type not in self.safe_mime_types:
                    if self.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                        threats.append(ThreatType.MALICIOUS_FILE)
                        recommendations.append(f"지원하지 않는 MIME 타입: {mime_type}")
            except Exception as e:
                details['mime_type_error'] = str(e)
            
            # 파일 매직 바이트 검사 (가능한 경우)
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(512)  # 처음 512바이트
                details['file_header'] = header[:16].hex()  # 처음 16바이트만 로깅
                
                # 기본적인 파일 시그니처 검사
                if self._check_file_signature(header, file_ext):
                    details['signature_valid'] = True
                else:
                    details['signature_valid'] = False
                    if self.security_level == SecurityLevel.CRITICAL:
                        threats.append(ThreatType.MALICIOUS_FILE)
                        recommendations.append("파일 시그니처가 확장자와 일치하지 않습니다")
                        
            except Exception as e:
                details['signature_check_error'] = str(e)
            
            is_safe = len(threats) == 0
            security_level = SecurityLevel.LOW if is_safe else SecurityLevel.MEDIUM
            
            return SecurityResult(
                is_safe=is_safe,
                security_level=security_level,
                threats=threats,
                details=details,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"파일 속성 검증 오류: {e}")
            return SecurityResult(
                is_safe=False,
                security_level=SecurityLevel.CRITICAL,
                threats=[ThreatType.MALICIOUS_FILE],
                details={'error': str(e)},
                recommendations=["파일 속성 검증에 실패했습니다"]
            )
    
    def _determine_file_category(self, file_ext: str) -> Optional[str]:
        """파일 카테고리 결정"""
        for category, extensions in self.allowed_extensions.items():
            if file_ext in extensions:
                return category
        return None
    
    def _check_file_signature(self, header: bytes, file_ext: str) -> bool:
        """파일 시그니처 검사"""
        # 기본적인 파일 시그니처들
        signatures = {
            '.jpg': [b'\xff\xd8\xff'],
            '.jpeg': [b'\xff\xd8\xff'],
            '.png': [b'\x89PNG\r\n\x1a\n'],
            '.pdf': [b'%PDF'],
            '.wav': [b'RIFF', b'WAVE'],
            '.mp3': [b'ID3', b'\xff\xfb'],
            '.mp4': [b'ftyp'],
            '.avi': [b'RIFF', b'AVI ']
        }
        
        if file_ext not in signatures:
            return True  # 알 수 없는 확장자는 통과
        
        expected_sigs = signatures[file_ext]
        return any(header.startswith(sig) for sig in expected_sigs)
    
    def check_resource_usage(self) -> SecurityResult:
        """리소스 사용량 검사"""
        threats = []
        details = {}
        recommendations = []
        
        try:
            # 메모리 사용량 체크
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            details['memory_usage_mb'] = memory_mb
            details['memory_limit_mb'] = self.resource_limits['max_memory_mb']
            
            if memory_mb > self.resource_limits['max_memory_mb']:
                threats.append(ThreatType.RESOURCE_EXHAUSTION)
                recommendations.append(f"메모리 사용량이 제한을 초과했습니다: {memory_mb:.1f}MB")
            
            # CPU 사용률 체크
            cpu_percent = process.cpu_percent(interval=1)
            details['cpu_usage_percent'] = cpu_percent
            details['cpu_limit_percent'] = self.resource_limits['max_cpu_percent']
            
            if cpu_percent > self.resource_limits['max_cpu_percent']:
                threats.append(ThreatType.RESOURCE_EXHAUSTION)
                recommendations.append(f"CPU 사용률이 제한을 초과했습니다: {cpu_percent:.1f}%")
            
            # 시스템 전체 리소스
            system_memory = psutil.virtual_memory()
            details['system_memory_percent'] = system_memory.percent
            
            if system_memory.percent > 90:
                threats.append(ThreatType.RESOURCE_EXHAUSTION)
                recommendations.append("시스템 메모리 부족")
            
            is_safe = len(threats) == 0
            
            return SecurityResult(
                is_safe=is_safe,
                security_level=SecurityLevel.LOW if is_safe else SecurityLevel.MEDIUM,
                threats=threats,
                details=details,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"리소스 사용량 검사 오류: {e}")
            return SecurityResult(
                is_safe=False,
                security_level=SecurityLevel.MEDIUM,
                threats=[ThreatType.RESOURCE_EXHAUSTION],
                details={'error': str(e)},
                recommendations=["리소스 검사에 실패했습니다"]
            )
    
    def sanitize_input(self, input_text: str) -> str:
        """입력 문자열 정화"""
        if not input_text:
            return ""
        
        # 기본적인 HTML/스크립트 태그 제거
        input_text = re.sub(r'<[^>]*>', '', input_text)
        
        # 특수 문자 제한
        input_text = re.sub(r'[<>:"|?*]', '', input_text)
        
        # 길이 제한
        if len(input_text) > 1000:
            input_text = input_text[:1000]
        
        # 인코딩 공격 방지
        try:
            input_text = input_text.encode('utf-8', errors='ignore').decode('utf-8')
        except:
            input_text = ""
        
        return input_text.strip()
    
    def validate_file_comprehensive(self, file_path: str) -> SecurityResult:
        """종합적인 파일 검증"""
        self.logger.info(f"파일 보안 검증 시작: {file_path}")
        
        # 경로 검증
        path_result = self.validate_file_path(file_path)
        if not path_result.is_safe:
            return path_result
        
        # 파일 속성 검증
        props_result = self.validate_file_properties(file_path)
        if not props_result.is_safe and self.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            return props_result
        
        # 리소스 검사
        resource_result = self.check_resource_usage()
        
        # 결과 통합
        all_threats = path_result.threats + props_result.threats + resource_result.threats
        all_details = {**path_result.details, **props_result.details, **resource_result.details}
        all_recommendations = path_result.recommendations + props_result.recommendations + resource_result.recommendations
        
        is_safe = len(all_threats) == 0
        
        # 보안 수준 결정
        if ThreatType.MALICIOUS_FILE in all_threats or ThreatType.PATH_TRAVERSAL in all_threats:
            security_level = SecurityLevel.CRITICAL
        elif len(all_threats) > 2:
            security_level = SecurityLevel.HIGH
        elif len(all_threats) > 0:
            security_level = SecurityLevel.MEDIUM
        else:
            security_level = SecurityLevel.LOW
        
        result = SecurityResult(
            is_safe=is_safe,
            security_level=security_level,
            threats=all_threats,
            details=all_details,
            recommendations=all_recommendations
        )
        
        self.logger.info(
            f"파일 보안 검증 완료: {file_path} "
            f"(안전: {is_safe}, 수준: {security_level.value}, 위협: {len(all_threats)}개)"
        )
        
        return result
    
    def get_security_config(self) -> Dict[str, Any]:
        """보안 설정 조회"""
        return {
            'security_level': self.security_level.value,
            'allowed_extensions': self.allowed_extensions,
            'max_file_sizes': {k: v/(1024*1024) for k, v in self.max_file_sizes.items()},  # MB 단위
            'resource_limits': self.resource_limits
        }
    
    def update_security_level(self, new_level: SecurityLevel) -> None:
        """보안 수준 업데이트"""
        old_level = self.security_level
        self.security_level = new_level
        self.logger.info(f"보안 수준 변경: {old_level.value} -> {new_level.value}")

# 전역 인스턴스
global_security_manager = SecurityManager(SecurityLevel.HIGH)

def get_global_security_manager() -> SecurityManager:
    """전역 보안 관리자 반환"""
    return global_security_manager

# 편의 함수들
def validate_file_security(file_path: str) -> SecurityResult:
    """파일 보안 검증 (편의 함수)"""
    return global_security_manager.validate_file_comprehensive(file_path)

def sanitize_user_input(input_text: str) -> str:
    """사용자 입력 정화 (편의 함수)"""
    return global_security_manager.sanitize_input(input_text)

def check_system_resources() -> SecurityResult:
    """시스템 리소스 검사 (편의 함수)"""
    return global_security_manager.check_resource_usage()

def is_file_safe(file_path: str) -> bool:
    """파일 안전성 간단 체크 (편의 함수)"""
    result = validate_file_security(file_path)
    return result.is_safe