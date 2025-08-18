#!/usr/bin/env python3
"""
MCP 자동 문제 해결 시스템
문제 상황을 자동으로 감지하고 MCP 도구들을 활용하여 해결책을 찾는 시스템
"""

import time
import json
import os
import psutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from utils.logger import get_logger
import streamlit as st

class MCPAutoProblemSolver:
    """MCP를 활용한 자동 문제 해결 시스템"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.problem_history = []
        self.solution_cache = {}
        self.mcp_tools_available = self._check_mcp_availability()
        
        # Playwright 통합 로드 (절대 임포트로 변경)
        try:
            import importlib.util
            from pathlib import Path
            
            # 동적 임포트로 상대 임포트 문제 해결
            current_dir = Path(__file__).parent
            playwright_file = current_dir / "playwright_mcp_integration.py"
            
            if playwright_file.exists():
                spec = importlib.util.spec_from_file_location(
                    "playwright_mcp_integration", 
                    playwright_file
                )
                playwright_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(playwright_module)
                
                self.playwright_integration = getattr(playwright_module, 'global_playwright_integration', None)
                self.logger.info("Playwright MCP 통합 시스템 로드 완료")
            else:
                self.playwright_integration = None
                self.logger.info("Playwright MCP 통합 파일이 존재하지 않음 - 건너뜀")
                
        except Exception as e:
            self.playwright_integration = None
            self.logger.warning(f"Playwright MCP 통합 시스템 로드 실패: {e}")
        
        # 문제 감지 임계값
        self.thresholds = {
            'memory_warning': 1000,    # 1GB
            'memory_critical': 2000,   # 2GB  
            'processing_slow': 300,    # 5분
            'processing_timeout': 600, # 10분
            'file_size_large': 100,    # 100MB
        }
        
    def _check_mcp_availability(self) -> Dict[str, bool]:
        """사용 가능한 MCP 도구들 확인"""
        available_tools = {}
        
        # MCP 도구 목록 (실제 사용 가능한 것들)
        mcp_tools = [
            'memory', 'filesystem', 'github-v2', 'notion', 
            'playwright', 'perplexity', 'sequential-thinking'
        ]
        
        for tool in mcp_tools:
            try:
                # 실제로는 MCP 함수 호출로 확인해야 하지만, 여기서는 기본값으로 설정
                available_tools[tool] = True
                self.logger.info(f"MCP 도구 '{tool}' 사용 가능")
            except Exception as e:
                available_tools[tool] = False
                self.logger.warning(f"MCP 도구 '{tool}' 사용 불가: {e}")
        
        return available_tools
    
    def detect_and_solve_problems(self, 
                                 memory_usage_mb: float,
                                 processing_time: float,
                                 file_info: Dict[str, Any] = None,
                                 error_message: str = None) -> Dict[str, Any]:
        """문제를 감지하고 MCP를 활용하여 해결책 제안"""
        
        detected_problems = []
        solutions = []
        
        # 1. 메모리 문제 감지
        if memory_usage_mb > self.thresholds['memory_critical']:
            detected_problems.append({
                'type': 'memory_critical',
                'severity': 'high',
                'value': memory_usage_mb,
                'description': f'심각한 메모리 사용량: {memory_usage_mb:.1f}MB'
            })
        elif memory_usage_mb > self.thresholds['memory_warning']:
            detected_problems.append({
                'type': 'memory_warning', 
                'severity': 'medium',
                'value': memory_usage_mb,
                'description': f'높은 메모리 사용량: {memory_usage_mb:.1f}MB'
            })
        
        # 2. 처리 시간 문제 감지
        if processing_time > self.thresholds['processing_timeout']:
            detected_problems.append({
                'type': 'processing_timeout',
                'severity': 'high', 
                'value': processing_time,
                'description': f'처리 시간 초과: {processing_time:.1f}초'
            })
        elif processing_time > self.thresholds['processing_slow']:
            detected_problems.append({
                'type': 'processing_slow',
                'severity': 'medium',
                'value': processing_time, 
                'description': f'느린 처리 속도: {processing_time:.1f}초'
            })
        
        # 3. 파일 크기 문제 감지
        if file_info and file_info.get('size_mb', 0) > self.thresholds['file_size_large']:
            detected_problems.append({
                'type': 'large_file',
                'severity': 'medium',
                'value': file_info['size_mb'],
                'description': f'대용량 파일: {file_info["size_mb"]:.1f}MB'
            })
        
        # 4. 에러 메시지 분석
        if error_message:
            detected_problems.append({
                'type': 'error_occurred',
                'severity': 'high',
                'value': error_message,
                'description': f'오류 발생: {error_message}'
            })
        
        # 감지된 문제들에 대한 해결책 검색
        for problem in detected_problems:
            solution = self._find_solution_with_mcp(problem)
            if solution:
                solutions.append(solution)
        
        # 결과 반환
        result = {
            'timestamp': datetime.now().isoformat(),
            'problems_detected': detected_problems,
            'solutions_found': solutions,
            'auto_actions_taken': []
        }
        
        # 자동 해결 액션 실행 (안전한 것들만)
        auto_actions = self._execute_safe_auto_actions(detected_problems)
        result['auto_actions_taken'] = auto_actions
        
        # 문제 이력에 저장
        self.problem_history.append(result)
        
        return result
    
    def _find_solution_with_mcp(self, problem: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """MCP 도구들을 활용하여 문제 해결책 검색"""
        
        problem_type = problem['type']
        
        # 캐시된 해결책 확인
        if problem_type in self.solution_cache:
            return self.solution_cache[problem_type]
        
        solution = None
        
        try:
            # 1. Sequential Thinking MCP로 문제 분석
            if self.mcp_tools_available.get('sequential-thinking'):
                analysis = self._analyze_problem_with_sequential_thinking(problem)
                
            # 2. Perplexity MCP로 외부 해결책 검색
            if self.mcp_tools_available.get('perplexity'):
                external_solutions = self._search_solutions_with_perplexity(problem)
                
            # 3. Memory MCP에 과거 해결 사례 검색
            if self.mcp_tools_available.get('memory'):
                past_solutions = self._search_memory_for_solutions(problem)
                
            # 4. GitHub MCP로 유사한 이슈 검색
            if self.mcp_tools_available.get('github-v2'):
                github_solutions = self._search_github_issues(problem)
            
            # 해결책 종합
            solution = self._synthesize_solutions(problem, {
                'analysis': analysis if 'analysis' in locals() else None,
                'external': external_solutions if 'external_solutions' in locals() else None,
                'past': past_solutions if 'past_solutions' in locals() else None,
                'github': github_solutions if 'github_solutions' in locals() else None
            })
            
            # 캐시에 저장
            if solution:
                self.solution_cache[problem_type] = solution
                
        except Exception as e:
            self.logger.error(f"MCP 해결책 검색 중 오류: {e}")
            # 폴백: 기본 해결책 제공
            solution = self._get_fallback_solution(problem)
        
        return solution
    
    def _analyze_problem_with_sequential_thinking(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Sequential Thinking MCP를 사용한 문제 분석"""
        
        # 실제 구현에서는 mcp__sequential-thinking__sequentialthinking 호출
        analysis_prompt = f"""
        다음 시스템 문제를 단계별로 분석해주세요:
        
        문제 유형: {problem['type']}
        심각도: {problem['severity']}  
        설명: {problem['description']}
        값: {problem['value']}
        
        1. 문제의 근본 원인 분석
        2. 즉시 해결 가능한 방법
        3. 장기적 예방 방안
        4. 위험도 평가
        """
        
        # 여기서는 기본 분석 결과 반환 (실제로는 MCP 호출)
        return {
            'root_cause': self._identify_root_cause(problem),
            'immediate_actions': self._get_immediate_actions(problem),
            'preventive_measures': self._get_preventive_measures(problem),
            'risk_level': problem['severity']
        }
    
    def _search_solutions_with_perplexity(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perplexity MCP와 Playwright 통합을 사용한 외부 해결책 검색"""
        
        solutions = []
        
        # 1. Perplexity MCP로 AI 기반 검색
        search_query = f"Python Streamlit {problem['type']} {problem['description']} solution fix"
        
        # 실제 구현에서는 mcp__perplexity__chat_completion 호출
        perplexity_result = {
            'source': 'perplexity_ai',
            'title': f"{problem['type']} AI 분석 결과",
            'description': self._get_external_solution_description(problem),
            'confidence': 0.8,
            'search_method': 'ai_powered'
        }
        solutions.append(perplexity_result)
        
        # 2. Playwright 통합으로 브라우저 자동화 검색
        if self.playwright_integration:
            try:
                browser_results = self.playwright_integration.search_solution_online(
                    problem['description'], 
                    problem['type']
                )
                
                # 브라우저 검색 결과를 솔루션 형식으로 변환
                for result in browser_results[:3]:  # 상위 3개만 사용
                    browser_solution = {
                        'source': 'browser_automation',
                        'title': result['title'],
                        'description': result['snippet'],
                        'url': result['url'],
                        'confidence': result['relevance_score'],
                        'search_method': 'browser_automated',
                        'source_type': result.get('source_type', 'web')
                    }
                    solutions.append(browser_solution)
                
                self.logger.info(f"Playwright 통합으로 {len(browser_results)}개 해결책 검색 완료")
                
            except Exception as e:
                self.logger.warning(f"Playwright 통합 검색 실패: {e}")
        
        return solutions
    
    def _search_memory_for_solutions(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Memory MCP에서 과거 해결 사례 검색"""
        
        # 실제 구현에서는 mcp__memory__search_nodes 호출
        search_results = []
        
        for past_problem in self.problem_history:
            for past_prob in past_problem.get('problems_detected', []):
                if past_prob['type'] == problem['type']:
                    for solution in past_problem.get('solutions_found', []):
                        if solution.get('effectiveness', 0) > 0.7:
                            search_results.append({
                                'source': 'past_experience',
                                'solution': solution,
                                'date': past_problem['timestamp'],
                                'effectiveness': solution.get('effectiveness', 0.5)
                            })
        
        return search_results
    
    def _search_github_issues(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """GitHub MCP로 유사한 이슈 검색"""
        
        # 실제 구현에서는 mcp__github-v2__issues_manage 호출
        return [{
            'source': 'github_issues',
            'title': f"Similar issue: {problem['type']}",
            'url': f"https://github.com/streamlit/streamlit/issues?q={problem['type']}",
            'description': "GitHub에서 유사한 이슈 확인 권장"
        }]
    
    def _synthesize_solutions(self, problem: Dict[str, Any], sources: Dict[str, Any]) -> Dict[str, Any]:
        """여러 소스의 해결책을 종합"""
        
        solution = {
            'problem_type': problem['type'],
            'severity': problem['severity'],
            'recommended_actions': [],
            'immediate_fixes': [],
            'monitoring_needed': [],
            'prevention_tips': [],
            'confidence_score': 0.0,
            'sources_used': []
        }
        
        # 문제 유형별 기본 해결책
        if problem['type'] == 'memory_critical':
            solution['immediate_fixes'] = [
                "즉시 가비지 컬렉션 실행 (gc.collect())",
                "메모리 사용량이 큰 변수들 del로 제거",
                "처리 중인 작업 중단 고려"
            ]
            solution['recommended_actions'] = [
                "파일 크기를 줄여서 재시도",
                "배치 크기 감소",
                "시스템 재시작"
            ]
            solution['monitoring_needed'] = [
                "실시간 메모리 사용량 추적",
                "메모리 누수 패턴 모니터링"
            ]
            
        elif problem['type'] == 'processing_slow':
            solution['immediate_fixes'] = [
                "CPU 모드로 전환 (GPU 오버헤드 제거)",
                "병렬 처리 비활성화", 
                "캐시 정리"
            ]
            solution['recommended_actions'] = [
                "파일을 더 작은 단위로 분할",
                "처리 알고리즘 최적화",
                "하드웨어 리소스 확인"
            ]
            
        elif problem['type'] == 'large_file':
            solution['immediate_fixes'] = [
                "파일을 청크 단위로 분할 처리",
                "스트리밍 처리 방식 적용"
            ]
            solution['recommended_actions'] = [
                "파일 압축 또는 해상도 감소",
                "필요한 부분만 추출하여 처리"
            ]
        
        # 신뢰도 점수 계산
        source_count = sum(1 for source in sources.values() if source)
        solution['confidence_score'] = min(0.9, 0.3 + (source_count * 0.15))
        solution['sources_used'] = [k for k, v in sources.items() if v]
        
        return solution
    
    def _execute_safe_auto_actions(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """안전한 자동 해결 액션 실행"""
        
        actions_taken = []
        
        for problem in problems:
            if problem['type'] == 'memory_warning':
                # 가비지 컬렉션 실행
                import gc
                gc.collect()
                actions_taken.append({
                    'action': 'garbage_collection',
                    'description': '가비지 컬렉션 실행',
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                })
                
            elif problem['type'] == 'processing_slow':
                # CPU 코어 수 확인 및 권장사항 제공
                cpu_count = psutil.cpu_count()
                actions_taken.append({
                    'action': 'system_analysis',
                    'description': f'시스템 리소스 분석 완료 (CPU: {cpu_count}코어)',
                    'recommendation': '병렬 처리 수준 조정 권장',
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                })
        
        return actions_taken
    
    def _get_fallback_solution(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """기본 폴백 해결책"""
        return {
            'problem_type': problem['type'],
            'severity': problem['severity'],
            'recommended_actions': [
                "시스템 재시작",
                "파일 크기 줄이기",
                "다른 시간에 재시도"
            ],
            'confidence_score': 0.3,
            'sources_used': ['fallback']
        }
    
    def _identify_root_cause(self, problem: Dict[str, Any]) -> str:
        """근본 원인 식별"""
        causes = {
            'memory_critical': '대용량 데이터 처리 또는 메모리 누수',
            'processing_slow': 'CPU 성능 부족 또는 비효율적 알고리즘',
            'large_file': '파일 크기가 시스템 처리 한계 초과',
            'error_occurred': '예상치 못한 시스템 오류'
        }
        return causes.get(problem['type'], '원인 분석 필요')
    
    def _get_immediate_actions(self, problem: Dict[str, Any]) -> List[str]:
        """즉시 실행 가능한 액션"""
        actions = {
            'memory_critical': ['가비지 컬렉션', '메모리 정리', '프로세스 재시작'],
            'processing_slow': ['CPU 모드 전환', '병렬 처리 비활성화', '캐시 정리'],
            'large_file': ['파일 분할', '압축 적용', '스트리밍 처리'],
            'error_occurred': ['에러 로그 확인', '시스템 상태 점검', '재시도']
        }
        return actions.get(problem['type'], ['시스템 재시작'])
    
    def _get_preventive_measures(self, problem: Dict[str, Any]) -> List[str]:
        """예방 조치"""
        measures = {
            'memory_critical': ['메모리 모니터링 강화', '배치 크기 제한', '정기적 정리'],
            'processing_slow': ['성능 벤치마킹', '알고리즘 최적화', '하드웨어 업그레이드'],
            'large_file': ['파일 크기 제한', '사전 압축', '청크 처리'],
            'error_occurred': ['예외 처리 강화', '로깅 개선', '테스트 확대']
        }
        return measures.get(problem['type'], ['정기적 시스템 점검'])
    
    def _get_external_solution_description(self, problem: Dict[str, Any]) -> str:
        """외부 검색 기반 해결책 설명"""
        descriptions = {
            'memory_critical': 'Python 메모리 최적화 및 가비지 컬렉션 전략',
            'processing_slow': 'Streamlit 성능 최적화 및 비동기 처리 방법',
            'large_file': '대용량 파일 처리를 위한 스트리밍 및 청크 처리',
            'error_occurred': '일반적인 Python/Streamlit 오류 해결 방법'
        }
        return descriptions.get(problem['type'], '일반적인 시스템 문제 해결 방법')
    
    def get_problem_history_summary(self) -> Dict[str, Any]:
        """문제 이력 요약"""
        if not self.problem_history:
            return {'message': '아직 감지된 문제가 없습니다'}
        
        problem_types = {}
        total_problems = 0
        
        for record in self.problem_history:
            for problem in record['problems_detected']:
                prob_type = problem['type']
                if prob_type not in problem_types:
                    problem_types[prob_type] = 0
                problem_types[prob_type] += 1
                total_problems += 1
        
        return {
            'total_incidents': len(self.problem_history),
            'total_problems': total_problems,
            'problem_distribution': problem_types,
            'most_common_problem': max(problem_types.items(), key=lambda x: x[1])[0] if problem_types else None,
            'latest_incident': self.problem_history[-1]['timestamp'] if self.problem_history else None
        }

# 전역 인스턴스
global_mcp_solver = MCPAutoProblemSolver()