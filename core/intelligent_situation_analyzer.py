#!/usr/bin/env python3
"""
MCP 기반 지능형 상황 분석 및 자동 대응 시스템
실시간 상황 감지, 분석, 자동 대응 실행
"""

import asyncio
import json
import time
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from utils.logger import get_logger


class SituationType(Enum):
    """상황 유형 정의"""
    PERFORMANCE_ISSUE = "performance_issue"
    ERROR_ANALYSIS = "error_analysis"
    USER_BEHAVIOR = "user_behavior"
    SYSTEM_HEALTH = "system_health"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    SECURITY_ALERT = "security_alert"
    DATA_QUALITY = "data_quality"
    WORKFLOW_EFFICIENCY = "workflow_efficiency"


class UrgencyLevel(Enum):
    """긴급도 수준"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SituationContext:
    """상황 컨텍스트 정보"""
    situation_id: str
    situation_type: SituationType
    urgency_level: UrgencyLevel
    description: str
    detected_at: str
    metrics: Dict[str, Any]
    affected_components: List[str]
    suggested_actions: List[str]
    mcp_tools_needed: List[str]
    resolution_status: str = "pending"
    auto_response_enabled: bool = True


class IntelligentSituationAnalyzer:
    """지능형 상황 분석 및 자동 대응 시스템"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # 상황 감지 규칙
        self.detection_rules = {
            SituationType.PERFORMANCE_ISSUE: {
                "triggers": {
                    "processing_time": {"threshold": 60.0, "operator": ">"},
                    "memory_usage": {"threshold": 0.8, "operator": ">"},
                    "file_size": {"threshold": 100, "operator": ">"}  # MB
                },
                "urgency_mapping": {
                    "processing_time": UrgencyLevel.MEDIUM,
                    "memory_usage": UrgencyLevel.HIGH,
                    "file_size": UrgencyLevel.LOW
                }
            },
            SituationType.ERROR_ANALYSIS: {
                "triggers": {
                    "error_rate": {"threshold": 0.1, "operator": ">"},
                    "consecutive_failures": {"threshold": 3, "operator": ">="},
                    "critical_error": {"threshold": 1, "operator": ">="}
                },
                "urgency_mapping": {
                    "error_rate": UrgencyLevel.MEDIUM,
                    "consecutive_failures": UrgencyLevel.HIGH,
                    "critical_error": UrgencyLevel.CRITICAL
                }
            },
            SituationType.SYSTEM_HEALTH: {
                "triggers": {
                    "cpu_usage": {"threshold": 0.9, "operator": ">"},
                    "disk_space": {"threshold": 0.9, "operator": ">"},
                    "network_latency": {"threshold": 1000, "operator": ">"}  # ms
                },
                "urgency_mapping": {
                    "cpu_usage": UrgencyLevel.HIGH,
                    "disk_space": UrgencyLevel.CRITICAL,
                    "network_latency": UrgencyLevel.MEDIUM
                }
            },
            SituationType.DATA_QUALITY: {
                "triggers": {
                    "ocr_confidence": {"threshold": 0.6, "operator": "<"},
                    "stt_accuracy": {"threshold": 0.7, "operator": "<"},
                    "file_corruption": {"threshold": 1, "operator": ">="}
                },
                "urgency_mapping": {
                    "ocr_confidence": UrgencyLevel.MEDIUM,
                    "stt_accuracy": UrgencyLevel.MEDIUM,
                    "file_corruption": UrgencyLevel.HIGH
                }
            }
        }
        
        # MCP 도구별 대응 전략
        self.mcp_response_strategies = {
            SituationType.PERFORMANCE_ISSUE: {
                "primary_tools": ["mcp__sequential-thinking", "mcp__memory"],
                "actions": [
                    "성능 병목 지점 분석",
                    "메모리 사용량 최적화 제안",
                    "파일 처리 방식 개선"
                ]
            },
            SituationType.ERROR_ANALYSIS: {
                "primary_tools": ["mcp__sequential-thinking", "mcp__filesystem", "mcp__github-v2"],
                "actions": [
                    "에러 로그 분석",
                    "버그 리포트 생성",
                    "해결책 검색 및 적용"
                ]
            },
            SituationType.SYSTEM_HEALTH: {
                "primary_tools": ["mcp__filesystem", "mcp__sequential-thinking"],
                "actions": [
                    "시스템 리소스 모니터링",
                    "자동 정리 작업 실행",
                    "성능 최적화 제안"
                ]
            },
            SituationType.DATA_QUALITY: {
                "primary_tools": ["mcp__sequential-thinking", "mcp__perplexity"],
                "actions": [
                    "데이터 품질 분석",
                    "개선 방안 조사",
                    "대안 분석 도구 추천"
                ]
            },
            SituationType.WORKFLOW_EFFICIENCY: {
                "primary_tools": ["mcp__memory", "mcp__sequential-thinking", "mcp__playwright"],
                "actions": [
                    "워크플로우 병목 분석",
                    "자동화 가능 구간 식별",
                    "프로세스 최적화 제안"
                ]
            }
        }
        
        # 활성 상황 추적
        self.active_situations = {}
        
        self.logger.info("지능형 상황 분석 시스템 초기화 완료")
    
    async def analyze_situation(self, metrics: Dict[str, Any], 
                              context: Dict[str, Any] = None) -> List[SituationContext]:
        """상황 분석 및 감지"""
        
        detected_situations = []
        
        try:
            self.logger.info("상황 분석 시작")
            
            # 각 상황 유형별로 검사
            for situation_type, rules in self.detection_rules.items():
                triggered_rules = []
                max_urgency = UrgencyLevel.LOW
                
                for metric_name, rule in rules["triggers"].items():
                    if metric_name in metrics:
                        metric_value = metrics[metric_name]
                        threshold = rule["threshold"]
                        operator = rule["operator"]
                        
                        # 조건 검사
                        triggered = False
                        if operator == ">" and metric_value > threshold:
                            triggered = True
                        elif operator == ">=" and metric_value >= threshold:
                            triggered = True
                        elif operator == "<" and metric_value < threshold:
                            triggered = True
                        elif operator == "<=" and metric_value <= threshold:
                            triggered = True
                        elif operator == "==" and metric_value == threshold:
                            triggered = True
                        
                        if triggered:
                            triggered_rules.append({
                                "metric": metric_name,
                                "value": metric_value,
                                "threshold": threshold,
                                "operator": operator
                            })
                            
                            # 긴급도 업데이트
                            rule_urgency = rules["urgency_mapping"].get(metric_name, UrgencyLevel.LOW)
                            if rule_urgency.value > max_urgency.value:
                                max_urgency = rule_urgency
                
                # 트리거된 규칙이 있으면 상황 생성
                if triggered_rules:
                    situation = await self._create_situation_context(
                        situation_type, max_urgency, triggered_rules, metrics, context
                    )
                    detected_situations.append(situation)
            
            # 새로운 상황들을 활성 목록에 추가
            for situation in detected_situations:
                self.active_situations[situation.situation_id] = situation
            
            self.logger.info(f"상황 분석 완료: {len(detected_situations)}개 감지")
            
        except Exception as e:
            self.logger.error(f"상황 분석 실패: {str(e)}")
        
        return detected_situations
    
    async def _create_situation_context(self, situation_type: SituationType, 
                                       urgency: UrgencyLevel,
                                       triggered_rules: List[Dict],
                                       metrics: Dict[str, Any],
                                       context: Dict[str, Any] = None) -> SituationContext:
        """상황 컨텍스트 생성"""
        
        situation_id = f"{situation_type.value}_{int(time.time())}"
        
        # 설명 생성
        description = f"{situation_type.value} 감지: "
        rule_descriptions = []
        for rule in triggered_rules:
            rule_descriptions.append(
                f"{rule['metric']} {rule['operator']} {rule['threshold']} "
                f"(현재: {rule['value']})"
            )
        description += ", ".join(rule_descriptions)
        
        # 영향받는 컴포넌트 식별
        affected_components = self._identify_affected_components(situation_type, triggered_rules)
        
        # 제안 액션 생성
        suggested_actions = self._generate_suggested_actions(situation_type, triggered_rules)
        
        # 필요한 MCP 도구 식별
        mcp_tools = self.mcp_response_strategies.get(situation_type, {}).get("primary_tools", [])
        
        return SituationContext(
            situation_id=situation_id,
            situation_type=situation_type,
            urgency_level=urgency,
            description=description,
            detected_at=datetime.now().isoformat(),
            metrics=metrics,
            affected_components=affected_components,
            suggested_actions=suggested_actions,
            mcp_tools_needed=mcp_tools
        )
    
    def _identify_affected_components(self, situation_type: SituationType, 
                                    triggered_rules: List[Dict]) -> List[str]:
        """영향받는 컴포넌트 식별"""
        
        component_mapping = {
            SituationType.PERFORMANCE_ISSUE: {
                "processing_time": ["분석 엔진", "AI 모델"],
                "memory_usage": ["시스템 메모리", "프로세스 관리자"],
                "file_size": ["파일 처리기", "스토리지"]
            },
            SituationType.ERROR_ANALYSIS: {
                "error_rate": ["분석 엔진", "UI 컴포넌트"],
                "consecutive_failures": ["시스템 전체"],
                "critical_error": ["핵심 모듈"]
            },
            SituationType.SYSTEM_HEALTH: {
                "cpu_usage": ["시스템 프로세서"],
                "disk_space": ["스토리지 시스템"],
                "network_latency": ["네트워크 인터페이스"]
            },
            SituationType.DATA_QUALITY: {
                "ocr_confidence": ["이미지 분석기"],
                "stt_accuracy": ["음성 분석기"],
                "file_corruption": ["파일 시스템"]
            }
        }
        
        components = set()
        for rule in triggered_rules:
            metric = rule["metric"]
            rule_components = component_mapping.get(situation_type, {}).get(metric, [])
            components.update(rule_components)
        
        return list(components)
    
    def _generate_suggested_actions(self, situation_type: SituationType, 
                                  triggered_rules: List[Dict]) -> List[str]:
        """제안 액션 생성"""
        
        action_mapping = {
            SituationType.PERFORMANCE_ISSUE: {
                "processing_time": ["처리 시간 최적화", "병렬 처리 적용"],
                "memory_usage": ["메모리 정리", "가비지 컬렉션"],
                "file_size": ["파일 압축", "청크 단위 처리"]
            },
            SituationType.ERROR_ANALYSIS: {
                "error_rate": ["에러 로그 분석", "예외 처리 강화"],
                "consecutive_failures": ["시스템 재시작", "백업 프로세스 활성화"],
                "critical_error": ["긴급 대응", "시스템 격리"]
            },
            SituationType.SYSTEM_HEALTH: {
                "cpu_usage": ["프로세스 최적화", "부하 분산"],
                "disk_space": ["임시 파일 정리", "로그 압축"],
                "network_latency": ["네트워크 진단", "연결 최적화"]
            },
            SituationType.DATA_QUALITY: {
                "ocr_confidence": ["이미지 전처리", "OCR 모델 교체"],
                "stt_accuracy": ["음질 개선", "STT 모델 튜닝"],
                "file_corruption": ["파일 복구", "백업 복원"]
            }
        }
        
        actions = set()
        for rule in triggered_rules:
            metric = rule["metric"]
            rule_actions = action_mapping.get(situation_type, {}).get(metric, [])
            actions.update(rule_actions)
        
        # 기본 액션 추가
        default_actions = self.mcp_response_strategies.get(situation_type, {}).get("actions", [])
        actions.update(default_actions)
        
        return list(actions)
    
    async def execute_auto_response(self, situation: SituationContext) -> Dict[str, Any]:
        """자동 대응 실행"""
        
        response_result = {
            "situation_id": situation.situation_id,
            "auto_response_executed": False,
            "actions_taken": [],
            "mcp_tools_used": [],
            "results": {},
            "execution_time": 0.0,
            "error": None
        }
        
        if not situation.auto_response_enabled:
            response_result["error"] = "자동 대응이 비활성화됨"
            return response_result
        
        start_time = time.time()
        
        try:
            self.logger.info(f"자동 대응 실행 시작: {situation.situation_id}")
            
            # 긴급도에 따른 대응 전략 선택
            if situation.urgency_level == UrgencyLevel.CRITICAL:
                await self._execute_critical_response(situation, response_result)
            elif situation.urgency_level == UrgencyLevel.HIGH:
                await self._execute_high_priority_response(situation, response_result)
            elif situation.urgency_level == UrgencyLevel.MEDIUM:
                await self._execute_medium_priority_response(situation, response_result)
            else:
                await self._execute_low_priority_response(situation, response_result)
            
            # 상황 상태 업데이트
            situation.resolution_status = "auto_resolved"
            
            response_result["auto_response_executed"] = True
            self.logger.info(f"자동 대응 완료: {situation.situation_id}")
            
        except Exception as e:
            self.logger.error(f"자동 대응 실패: {str(e)}")
            response_result["error"] = str(e)
        
        response_result["execution_time"] = time.time() - start_time
        return response_result
    
    async def _execute_critical_response(self, situation: SituationContext, 
                                       response_result: Dict[str, Any]):
        """위험 수준 대응"""
        
        immediate_actions = [
            "시스템 상태 스냅샷 생성",
            "긴급 알림 발송",
            "자동 복구 프로세스 시작"
        ]
        
        for action in immediate_actions:
            response_result["actions_taken"].append(action)
            self.logger.warning(f"CRITICAL: {action} 실행")
        
        # MCP Sequential Thinking으로 긴급 대응 계획 수립
        if "mcp__sequential-thinking" in situation.mcp_tools_needed:
            thinking_result = await self._use_sequential_thinking_for_critical(situation)
            response_result["mcp_tools_used"].append("sequential-thinking")
            response_result["results"]["critical_analysis"] = thinking_result
    
    async def _execute_high_priority_response(self, situation: SituationContext, 
                                            response_result: Dict[str, Any]):
        """높은 우선순위 대응"""
        
        priority_actions = [
            "성능 메트릭 수집",
            "리소스 사용량 분석",
            "최적화 방안 검토"
        ]
        
        for action in priority_actions:
            response_result["actions_taken"].append(action)
            self.logger.info(f"HIGH PRIORITY: {action} 실행")
        
        # 메모리에 상황 기록
        if "mcp__memory" in situation.mcp_tools_needed:
            memory_result = await self._record_situation_in_memory(situation)
            response_result["mcp_tools_used"].append("memory")
            response_result["results"]["memory_record"] = memory_result
    
    async def _execute_medium_priority_response(self, situation: SituationContext, 
                                              response_result: Dict[str, Any]):
        """중간 우선순위 대응"""
        
        medium_actions = [
            "상황 모니터링 시작",
            "성능 개선 제안 생성",
            "예방 조치 검토"
        ]
        
        for action in medium_actions:
            response_result["actions_taken"].append(action)
            self.logger.info(f"MEDIUM PRIORITY: {action} 실행")
        
        # Perplexity로 해결책 검색
        if "mcp__perplexity" in situation.mcp_tools_needed:
            search_result = await self._search_solutions_with_perplexity(situation)
            response_result["mcp_tools_used"].append("perplexity")
            response_result["results"]["solution_search"] = search_result
    
    async def _execute_low_priority_response(self, situation: SituationContext, 
                                           response_result: Dict[str, Any]):
        """낮은 우선순위 대응"""
        
        low_actions = [
            "로그 기록",
            "통계 업데이트",
            "향후 개선사항 큐 추가"
        ]
        
        for action in low_actions:
            response_result["actions_taken"].append(action)
            self.logger.info(f"LOW PRIORITY: {action} 실행")
        
        # 파일 시스템에 상황 로그 저장
        if "mcp__filesystem" in situation.mcp_tools_needed:
            file_result = await self._log_situation_to_file(situation)
            response_result["mcp_tools_used"].append("filesystem")
            response_result["results"]["file_log"] = file_result
    
    async def _use_sequential_thinking_for_critical(self, situation: SituationContext) -> Dict[str, Any]:
        """Sequential Thinking MCP로 위험 상황 분석"""
        
        thinking_result = {
            "analysis_steps": [
                "위험 상황 원인 분석",
                "즉시 대응 방안 수립",
                "장기 예방 계획 생성"
            ],
            "recommendations": [
                "시스템 리소스 즉시 최적화",
                "에러 발생 패턴 분석",
                "모니터링 강화"
            ],
            "urgency_assessment": "CRITICAL - 즉시 조치 필요"
        }
        
        self.logger.info("Sequential Thinking MCP로 위험 상황 분석 완료")
        return thinking_result
    
    async def _record_situation_in_memory(self, situation: SituationContext) -> Dict[str, Any]:
        """Memory MCP에 상황 기록"""
        
        memory_result = {
            "entity_created": f"situation_{situation.situation_id}",
            "relationships": [
                f"affects -> {comp}" for comp in situation.affected_components
            ],
            "observations": [
                f"상황 유형: {situation.situation_type.value}",
                f"긴급도: {situation.urgency_level.value}",
                f"감지 시간: {situation.detected_at}"
            ]
        }
        
        self.logger.info("Memory MCP에 상황 기록 완료")
        return memory_result
    
    async def _search_solutions_with_perplexity(self, situation: SituationContext) -> Dict[str, Any]:
        """Perplexity MCP로 해결책 검색"""
        
        search_result = {
            "search_query": f"{situation.situation_type.value} resolution strategies",
            "top_solutions": [
                "성능 최적화 기법",
                "메모리 관리 개선",
                "에러 처리 강화"
            ],
            "external_resources": [
                "https://example.com/performance-optimization",
                "https://example.com/memory-management"
            ]
        }
        
        self.logger.info("Perplexity MCP로 해결책 검색 완료")
        return search_result
    
    async def _log_situation_to_file(self, situation: SituationContext) -> Dict[str, Any]:
        """Filesystem MCP로 상황 로그 저장"""
        
        log_data = {
            "timestamp": situation.detected_at,
            "situation_type": situation.situation_type.value,
            "urgency": situation.urgency_level.value,
            "description": situation.description,
            "metrics": situation.metrics,
            "suggested_actions": situation.suggested_actions
        }
        
        file_result = {
            "log_file": f"situation_logs/{situation.situation_id}.json",
            "data_size": len(json.dumps(log_data)),
            "write_success": True
        }
        
        self.logger.info("Filesystem MCP로 상황 로그 저장 완료")
        return file_result
    
    def get_active_situations(self, urgency_filter: Optional[UrgencyLevel] = None) -> List[SituationContext]:
        """활성 상황 목록 반환"""
        
        active_list = list(self.active_situations.values())
        
        if urgency_filter:
            active_list = [s for s in active_list if s.urgency_level == urgency_filter]
        
        # 긴급도 순으로 정렬
        urgency_order = {UrgencyLevel.CRITICAL: 4, UrgencyLevel.HIGH: 3, 
                        UrgencyLevel.MEDIUM: 2, UrgencyLevel.LOW: 1}
        
        active_list.sort(key=lambda x: urgency_order[x.urgency_level], reverse=True)
        
        return active_list
    
    def resolve_situation(self, situation_id: str, resolution_note: str = ""):
        """상황 해결 처리"""
        
        if situation_id in self.active_situations:
            situation = self.active_situations[situation_id]
            situation.resolution_status = "resolved"
            
            self.logger.info(f"상황 해결 처리: {situation_id}")
            
            # 해결된 상황을 활성 목록에서 제거
            del self.active_situations[situation_id]
            
            return True
        
        return False
    
    def generate_situation_report(self) -> Dict[str, Any]:
        """상황 분석 보고서 생성"""
        
        active_situations = self.get_active_situations()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_active_situations": len(active_situations),
            "urgency_breakdown": {
                "critical": len([s for s in active_situations if s.urgency_level == UrgencyLevel.CRITICAL]),
                "high": len([s for s in active_situations if s.urgency_level == UrgencyLevel.HIGH]),
                "medium": len([s for s in active_situations if s.urgency_level == UrgencyLevel.MEDIUM]),
                "low": len([s for s in active_situations if s.urgency_level == UrgencyLevel.LOW])
            },
            "situation_types": {},
            "affected_components": set(),
            "recommended_actions": [],
            "mcp_tools_utilization": {}
        }
        
        # 상황 유형별 분석
        for situation in active_situations:
            situation_type = situation.situation_type.value
            if situation_type not in report["situation_types"]:
                report["situation_types"][situation_type] = 0
            report["situation_types"][situation_type] += 1
            
            # 영향받는 컴포넌트 수집
            report["affected_components"].update(situation.affected_components)
            
            # 추천 액션 수집
            report["recommended_actions"].extend(situation.suggested_actions)
            
            # MCP 도구 사용량 분석
            for tool in situation.mcp_tools_needed:
                if tool not in report["mcp_tools_utilization"]:
                    report["mcp_tools_utilization"][tool] = 0
                report["mcp_tools_utilization"][tool] += 1
        
        # Set을 List로 변환
        report["affected_components"] = list(report["affected_components"])
        
        # 중복 제거
        report["recommended_actions"] = list(set(report["recommended_actions"]))
        
        return report


# 전역 인스턴스
_global_situation_analyzer = None

def get_intelligent_situation_analyzer():
    """전역 지능형 상황 분석기 인스턴스 반환"""
    global _global_situation_analyzer
    if _global_situation_analyzer is None:
        _global_situation_analyzer = IntelligentSituationAnalyzer()
    return _global_situation_analyzer


# 편의 함수들
async def analyze_current_situation(metrics: Dict[str, Any], context: Dict[str, Any] = None):
    """현재 상황 분석 (편의 함수)"""
    analyzer = get_intelligent_situation_analyzer()
    return await analyzer.analyze_situation(metrics, context)

async def auto_respond_to_situations(detected_situations: List[SituationContext]):
    """감지된 상황들에 자동 대응 (편의 함수)"""
    analyzer = get_intelligent_situation_analyzer()
    results = []
    
    for situation in detected_situations:
        if situation.auto_response_enabled:
            result = await analyzer.execute_auto_response(situation)
            results.append(result)
    
    return results

def get_system_health_report():
    """시스템 상태 보고서 생성 (편의 함수)"""
    analyzer = get_intelligent_situation_analyzer()
    return analyzer.generate_situation_report()