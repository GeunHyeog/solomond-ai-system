"""
🛡️ SOLOMOND AI 진실 강제 시스템 (Truth Enforcement System)
허위 정보 완전 차단을 위한 영구적 검증 프레임워크

사용자 지시: "시스템 전체적으로 허위정보는 없도록하자. 이건 가장 기본적이고 가장 중요한거야."
원칙: 모든 상태 표시는 실제 검증 후에만, 확실하지 않으면 솔직하게 표시
"""

import logging
import inspect
from typing import Callable, Dict, Any, Optional, Union
from functools import wraps
from datetime import datetime
from enum import Enum

# 진실성 로거
truth_logger = logging.getLogger('TRUTH_SYSTEM')
truth_logger.setLevel(logging.INFO)

class TruthLevel(Enum):
    """진실성 수준"""
    VERIFIED = "🟢 검증됨"        # 실제 확인 완료
    PARTIAL = "🟡 부분 확인"      # 일부만 확인됨
    UNVERIFIED = "🔴 미검증"      # 확인되지 않음
    UNKNOWN = "⚪ 상태 불명"       # 확인 불가능

class StatusType(Enum):
    """상태 유형"""
    AVAILABLE = "사용가능"
    ACTIVATED = "활성화"
    COMPLETED = "완료" 
    READY = "준비"
    SUCCESS = "성공"
    INITIALIZED = "초기화"

class TruthEnforcer:
    """허위 정보 방지 강제자"""
    
    def __init__(self):
        self.violation_log = []
        self.verification_stats = {
            'total_checks': 0,
            'verified': 0,
            'unverified': 0,
            'violations_prevented': 0
        }
    
    def verify_status(self, 
                     status_type: StatusType,
                     system_name: str,
                     verification_func: Callable[[], bool],
                     context: str = "") -> str:
        """
        상태 검증 강제 함수
        
        Args:
            status_type: 상태 유형 (사용가능, 활성화 등)
            system_name: 시스템명
            verification_func: 실제 검증 함수 (반드시 제공)
            context: 추가 컨텍스트
            
        Returns:
            진실한 상태 메시지
        """
        self.verification_stats['total_checks'] += 1
        
        try:
            # 검증 함수가 제공되지 않으면 강제로 미검증 처리
            if verification_func is None:
                self._log_violation(f"검증 함수 없음: {system_name} {status_type.value}")
                return f"🚨 {system_name}: {status_type.value} 확인 불가 (검증 함수 누락)"
            
            # 실제 검증 실행
            result = verification_func()
            
            if result is True:
                self.verification_stats['verified'] += 1
                truth_logger.info(f"✅ 진실성 확인: {system_name} {status_type.value}")
                return f"{TruthLevel.VERIFIED.value} {system_name}: {status_type.value}"
                
            elif result is False:
                self.verification_stats['unverified'] += 1
                truth_logger.warning(f"❌ 상태 거짓: {system_name} {status_type.value}")
                return f"{TruthLevel.UNVERIFIED.value} {system_name}: {status_type.value} 불가"
                
            else:
                # 부분적 결과 (예: 3/4 완료)
                return f"{TruthLevel.PARTIAL.value} {system_name}: {result}"
                
        except Exception as e:
            truth_logger.error(f"검증 오류: {system_name} - {e}")
            return f"{TruthLevel.UNKNOWN.value} {system_name}: 검증 실패 ({e})"
    
    def prevent_false_claim(self, claim: str, evidence: Any = None) -> str:
        """
        허위 주장 방지 데코레이터
        
        Args:
            claim: 주장하려는 내용
            evidence: 증거/검증 데이터
            
        Returns:
            진실성 검증된 메시지
        """
        if evidence is None or not evidence:
            self._log_violation(f"증거 없는 주장: {claim}")
            self.verification_stats['violations_prevented'] += 1
            return f"⚠️ 미확인: {claim} (증거 부족)"
        
        # 증거 기반 진실한 메시지 반환
        if isinstance(evidence, (list, dict)) and len(evidence) == 0:
            return f"❌ {claim} (결과 없음)"
        elif isinstance(evidence, bool) and evidence:
            return f"✅ {claim} (검증됨)"
        elif isinstance(evidence, int) and evidence > 0:
            return f"✅ {claim} ({evidence}개 확인됨)"
        else:
            return f"✅ {claim} (증거 확인됨)"
    
    def _log_violation(self, violation: str):
        """허위 정보 시도 로깅"""
        timestamp = datetime.now().isoformat()
        violation_record = {
            'timestamp': timestamp,
            'violation': violation,
            'caller': inspect.stack()[2].function if len(inspect.stack()) > 2 else 'unknown'
        }
        self.violation_log.append(violation_record)
        truth_logger.warning(f"🚨 허위정보 시도 차단: {violation}")
    
    def get_truth_report(self) -> Dict[str, Any]:
        """진실성 보고서 생성"""
        total = self.verification_stats['total_checks']
        verified_rate = (self.verification_stats['verified'] / total * 100) if total > 0 else 0
        
        return {
            'verification_stats': self.verification_stats,
            'truth_rate': f"{verified_rate:.1f}%",
            'violations_prevented': self.verification_stats['violations_prevented'],
            'recent_violations': self.violation_log[-10:],  # 최근 10개만
            'system_integrity': 'HIGH' if verified_rate >= 90 else 'MEDIUM' if verified_rate >= 70 else 'LOW'
        }

# 전역 진실 강제자
global_truth_enforcer = TruthEnforcer()

# 편의 함수들
def verify_available(system_name: str, check_func: Callable[[], bool]) -> str:
    """'사용가능' 상태 검증"""
    return global_truth_enforcer.verify_status(
        StatusType.AVAILABLE, system_name, check_func
    )

def verify_activated(system_name: str, check_func: Callable[[], bool]) -> str:
    """'활성화' 상태 검증"""
    return global_truth_enforcer.verify_status(
        StatusType.ACTIVATED, system_name, check_func
    )

def verify_completed(task_name: str, check_func: Callable[[], bool]) -> str:
    """'완료' 상태 검증"""
    return global_truth_enforcer.verify_status(
        StatusType.COMPLETED, task_name, check_func
    )

def prevent_false_completion(task_name: str, results: Any) -> str:
    """허위 완료 표시 방지"""
    return global_truth_enforcer.prevent_false_claim(f"{task_name} 완료", results)

def prevent_false_availability(system_name: str, evidence: Any) -> str:
    """허위 사용가능 표시 방지"""
    return global_truth_enforcer.prevent_false_claim(f"{system_name} 사용가능", evidence)

def no_lies_decorator(func: Callable):
    """
    허위 정보 방지 데코레이터
    함수가 허위 상태를 반환하려 하면 강제로 차단
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            # 결과가 허위 패턴인지 검사
            if isinstance(result, str):
                false_patterns = [
                    "✅.*완료" and "완료" in result and not any(["검증", "확인", "개"] in result),
                    "✅.*활성화" and "활성화" in result and not any(["검증", "확인"] in result),
                    "✅.*사용가능" and "사용가능" in result and not any(["검증", "확인"] in result)
                ]
                
                for pattern in false_patterns:
                    if pattern:
                        truth_logger.warning(f"🚨 허위 패턴 감지: {result}")
                        return f"⚠️ 검증 필요: {result.replace('✅', '❓')}"
            
            return result
            
        except Exception as e:
            truth_logger.error(f"함수 실행 오류: {func.__name__} - {e}")
            return f"❌ {func.__name__} 실행 실패: {e}"
    
    return wrapper

# 사용 예시
def example_usage():
    """사용 예시"""
    
    # 올바른 검증 방식
    def check_ollama_available():
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    # 진실한 상태 표시
    status = verify_activated("Ollama AI", check_ollama_available)
    print(status)  # 🟢 검증됨 Ollama AI: 활성화 또는 🔴 미검증 Ollama AI: 활성화 불가
    
    # 허위 완료 방지
    results = []  # 빈 결과
    completion_msg = prevent_false_completion("데이터 분석", results)
    print(completion_msg)  # ❌ 데이터 분석 완료 (결과 없음)
    
    # 진실성 보고서
    report = global_truth_enforcer.get_truth_report()
    print(f"진실성 비율: {report['truth_rate']}")

if __name__ == "__main__":
    print("🛡️ SOLOMOND AI 진실 강제 시스템 초기화됨")
    print("🎯 목표: 허위 정보 완전 차단")
    print("📋 원칙: 검증된 진실만 표시")
    example_usage()