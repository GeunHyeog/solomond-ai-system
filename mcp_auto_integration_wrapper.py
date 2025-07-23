#!/usr/bin/env python3
"""
MCP 자동 통합 래퍼 - 모든 작업에 MCP 자동 적용
사용자 요청 → 자동 MCP 분석 → 실행 → 결과 향상
"""

import asyncio
import logging
import inspect
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
import functools
from mcp_comprehensive_orchestrator import (
    global_mcp_orchestrator, 
    analyze_request_and_recommend_mcps,
    execute_with_optimal_mcps
)

class MCPAutoIntegrationWrapper:
    """모든 작업에 MCP를 자동으로 통합하는 래퍼 시스템"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.mcp_usage_stats = {
            "total_requests": 0,
            "mcp_enhanced_requests": 0,
            "improvement_rate": 0.0,
            "popular_mcps": {},
            "success_rate": 0.0
        }
        self.logger.info("🔄 MCP 자동 통합 래퍼 시스템 초기화 완료")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.MCPAutoIntegrationWrapper')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def auto_mcp_enhance(self, require_mcp: bool = True, mcp_priority: str = "auto"):
        """MCP 자동 향상 데코레이터"""
        
        def decorator(func: Callable) -> Callable:
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._execute_with_mcp_enhancement(
                    func, args, kwargs, require_mcp, mcp_priority, is_async=True
                )
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if asyncio.iscoroutinefunction(func):
                    # 비동기 함수는 비동기 래퍼 사용
                    return async_wrapper(*args, **kwargs)
                else:
                    # 동기 함수는 동기 실행
                    return asyncio.run(self._execute_with_mcp_enhancement(
                        func, args, kwargs, require_mcp, mcp_priority, is_async=False
                    ))
            
            # 함수가 비동기인지 확인하여 적절한 래퍼 반환
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
                
        return decorator
    
    async def _execute_with_mcp_enhancement(self, func: Callable, args: tuple, kwargs: dict,
                                          require_mcp: bool, mcp_priority: str, is_async: bool):
        """MCP 향상 로직 실행"""
        
        self.mcp_usage_stats["total_requests"] += 1
        
        try:
            # 1. 함수 정보 및 인자 분석
            func_info = self._analyze_function_call(func, args, kwargs)
            
            # 2. 사용자 요청 추출 (첫 번째 문자열 인자에서)
            user_request = self._extract_user_request(args, kwargs)
            
            # 3. 컨텍스트 정보 수집
            context = self._build_context_from_args(args, kwargs, func_info)
            
            # 4. MCP 전략 분석
            should_use_mcp = self._should_use_mcp(user_request, context, require_mcp, mcp_priority)
            
            if should_use_mcp:
                self.logger.info(f"🎯 MCP 향상 적용: {func.__name__}")
                return await self._execute_with_mcp_strategy(
                    func, args, kwargs, user_request, context, is_async
                )
            else:
                # MCP 없이 기본 실행
                self.logger.info(f"📋 기본 실행: {func.__name__}")
                if is_async:
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        except Exception as e:
            self.logger.error(f"❌ MCP 통합 중 오류: {e}")
            # 오류 시 기본 함수 실행으로 폴백
            if is_async:
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
    
    def _analyze_function_call(self, func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """함수 호출 정보 분석"""
        
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            return {
                "function_name": func.__name__,
                "module": func.__module__,
                "arguments": dict(bound_args.arguments),
                "parameter_names": list(sig.parameters.keys()),
                "has_context": "context" in sig.parameters,
                "has_user_input": any("user" in param or "request" in param or "query" in param 
                                    for param in sig.parameters.keys())
            }
        except Exception as e:
            self.logger.warning(f"함수 분석 실패: {e}")
            return {
                "function_name": func.__name__,
                "module": getattr(func, '__module__', 'unknown'),
                "arguments": {},
                "parameter_names": [],
                "has_context": False,
                "has_user_input": False
            }
    
    def _extract_user_request(self, args: tuple, kwargs: dict) -> str:
        """사용자 요청 문자열 추출"""
        
        # 1. kwargs에서 찾기
        for key in ['user_request', 'request', 'query', 'prompt', 'text', 'message']:
            if key in kwargs and isinstance(kwargs[key], str):
                return kwargs[key]
        
        # 2. args에서 첫 번째 문자열 찾기
        for arg in args:
            if isinstance(arg, str) and len(arg.strip()) > 5:
                return arg
        
        # 3. 기본값
        return "사용자 요청 분석"
    
    def _build_context_from_args(self, args: tuple, kwargs: dict, func_info: Dict[str, Any]) -> Dict[str, Any]:
        """인자들로부터 컨텍스트 구성"""
        
        context = {
            "function_name": func_info["function_name"],
            "module": func_info["module"],
            "timestamp": datetime.now().isoformat(),
            "arg_count": len(args),
            "kwarg_count": len(kwargs)
        }
        
        # 특수한 인자들 분석
        for key, value in kwargs.items():
            if key in ['context', 'file_path', 'files', 'data', 'config']:
                context[key] = value
            elif key.endswith('_count') or key.endswith('_size'):
                context[key] = value
            elif isinstance(value, (int, float, bool)):
                context[key] = value
        
        # 파일 관련 정보 추출
        for arg in args:
            if hasattr(arg, 'name') and hasattr(arg, 'size'):  # 업로드된 파일
                context.setdefault('files', []).append({
                    'name': arg.name,
                    'size': getattr(arg, 'size', 0)
                })
        
        return context
    
    def _should_use_mcp(self, user_request: str, context: Dict[str, Any], 
                       require_mcp: bool, mcp_priority: str) -> bool:
        """MCP 사용 여부 결정"""
        
        # 1. 강제 사용 설정
        if require_mcp:
            return True
        
        # 2. 우선순위 설정에 따른 판단
        if mcp_priority == "never":
            return False
        elif mcp_priority == "always":
            return True
        
        # 3. 자동 판단 (기본값)
        auto_triggers = [
            len(user_request) > 20,  # 충분한 길이의 요청
            any(keyword in user_request.lower() for keyword in [
                "분석", "조사", "연구", "최적화", "개선", "해결", "찾아", "만들어",
                "analyze", "research", "optimize", "improve", "solve", "create"
            ]),
            context.get('files') and len(context['files']) > 0,  # 파일이 있는 경우
            context.get('complexity_level') == 'high',  # 높은 복잡도
            context.get('comprehensive_mode'),  # 종합 모드
            'customer_id' in context  # 고객 분석인 경우
        ]
        
        return sum(auto_triggers) >= 2  # 2개 이상 조건 만족시 MCP 사용
    
    async def _execute_with_mcp_strategy(self, func: Callable, args: tuple, kwargs: dict,
                                       user_request: str, context: Dict[str, Any], is_async: bool):
        """MCP 전략을 적용한 실행"""
        
        # 1. 기본 함수 실행 (원본 결과 획득)
        if is_async:
            base_result = await func(*args, **kwargs)
        else:
            base_result = func(*args, **kwargs)
        
        # 2. MCP 전략 분석
        mcp_strategy = analyze_request_and_recommend_mcps(user_request, context)
        
        # 3. MCP 적용 가치 판단
        if not mcp_strategy["recommended_mcps"]:
            self.logger.info("📋 MCP 전략 없음, 기본 결과 반환")
            return base_result
        
        # 4. MCP로 결과 향상
        try:
            enhanced_result = await execute_with_optimal_mcps(user_request, base_result, context)
            
            self.mcp_usage_stats["mcp_enhanced_requests"] += 1
            self._update_mcp_stats(mcp_strategy["recommended_mcps"])
            
            self.logger.info(f"✅ MCP 향상 완료: {len(mcp_strategy['recommended_mcps'])}개 서버 활용")
            return enhanced_result
            
        except Exception as e:
            self.logger.warning(f"⚠️ MCP 향상 실패, 기본 결과 반환: {e}")
            return base_result
    
    def _update_mcp_stats(self, used_mcps: List[str]):
        """MCP 사용 통계 업데이트"""
        
        for mcp in used_mcps:
            self.mcp_usage_stats["popular_mcps"][mcp] = \
                self.mcp_usage_stats["popular_mcps"].get(mcp, 0) + 1
        
        # 성공률 계산
        if self.mcp_usage_stats["total_requests"] > 0:
            self.mcp_usage_stats["improvement_rate"] = \
                self.mcp_usage_stats["mcp_enhanced_requests"] / self.mcp_usage_stats["total_requests"]
    
    def get_mcp_usage_report(self) -> Dict[str, Any]:
        """MCP 사용 현황 보고서"""
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "statistics": self.mcp_usage_stats.copy(),
            "most_used_mcp": max(self.mcp_usage_stats["popular_mcps"], 
                               key=self.mcp_usage_stats["popular_mcps"].get) \
                             if self.mcp_usage_stats["popular_mcps"] else "none",
            "recommendations": self._generate_usage_recommendations()
        }
    
    def _generate_usage_recommendations(self) -> List[str]:
        """사용 개선 권장사항"""
        
        recommendations = []
        
        if self.mcp_usage_stats["improvement_rate"] < 0.3:
            recommendations.append("MCP 활용도가 낮습니다. require_mcp=True 옵션 고려")
        
        if not self.mcp_usage_stats["popular_mcps"]:
            recommendations.append("MCP 서버 연결 상태를 확인하세요")
        
        if self.mcp_usage_stats["total_requests"] > 10:
            recommendations.append(f"총 {self.mcp_usage_stats['total_requests']}건 중 "
                                 f"{self.mcp_usage_stats['mcp_enhanced_requests']}건 MCP 향상 적용")
        
        return recommendations


# 전역 인스턴스 생성
global_mcp_wrapper = MCPAutoIntegrationWrapper()

# 편의 함수들
def mcp_enhance(require_mcp: bool = True, priority: str = "auto"):
    """MCP 자동 향상 데코레이터 - 편의 함수"""
    return global_mcp_wrapper.auto_mcp_enhance(require_mcp=require_mcp, mcp_priority=priority)

def smart_mcp_enhance(func: Callable = None, *, 
                     require_mcp: bool = False, 
                     priority: str = "auto"):
    """스마트 MCP 향상 - 상황에 따라 자동 판단"""
    
    if func is None:
        # 데코레이터로 사용된 경우
        return global_mcp_wrapper.auto_mcp_enhance(require_mcp=require_mcp, mcp_priority=priority)
    else:
        # 직접 함수에 적용된 경우
        return global_mcp_wrapper.auto_mcp_enhance(require_mcp=require_mcp, mcp_priority=priority)(func)

async def enhance_result_with_mcp(user_request: str, base_result: Any, context: Dict[str, Any] = None) -> Any:
    """기존 결과를 MCP로 향상시키는 함수"""
    
    if context is None:
        context = {}
    
    try:
        enhanced_result = await execute_with_optimal_mcps(user_request, base_result, context)
        global_mcp_wrapper.mcp_usage_stats["mcp_enhanced_requests"] += 1
        return enhanced_result
    except Exception as e:
        global_mcp_wrapper.logger.warning(f"MCP 향상 실패: {e}")
        return base_result

def get_mcp_usage_stats() -> Dict[str, Any]:
    """MCP 사용 통계 반환"""
    return global_mcp_wrapper.get_mcp_usage_report()


# 사용 예시
if __name__ == "__main__":
    
    # 예시 1: 자동 MCP 향상 함수
    @smart_mcp_enhance
    async def analyze_customer_request(user_input: str, context: Dict[str, Any] = None):
        """고객 요청 분석 함수 (MCP 자동 향상 적용)"""
        basic_result = {
            "status": "analyzed",
            "user_input": user_input,
            "basic_analysis": "기본 분석 완료",
            "timestamp": datetime.now().isoformat()
        }
        return basic_result
    
    # 예시 2: 필수 MCP 적용 함수
    @mcp_enhance(require_mcp=True, priority="always")
    def optimize_system_performance(system_data: Dict[str, Any]):
        """시스템 성능 최적화 (MCP 필수 적용)"""
        return {
            "status": "optimized",
            "improvements": ["기본 최적화 적용"],
            "performance_gain": "10%"
        }
    
    # 테스트 실행
    async def test_mcp_integration():
        print("🧪 MCP 자동 통합 시스템 테스트")
        
        # 테스트 1: 고객 분석
        result1 = await analyze_customer_request(
            "3GB 동영상에서 고객 상담 내용을 분석하고 다이아몬드 반지 가격 조사해줘",
            {"customer_id": "CUST_001", "file_size": 3000}
        )
        print(f"✅ 고객 분석 완료: MCP 향상 여부 = {bool(result1.get('mcp_integration_summary'))}")
        
        # 테스트 2: 시스템 최적화
        result2 = optimize_system_performance({"cpu_usage": 80, "memory_usage": 70})
        print(f"✅ 성능 최적화 완료: MCP 향상 여부 = {bool(result2.get('mcp_integration_summary'))}")
        
        # 사용 통계 출력
        stats = get_mcp_usage_stats()
        print(f"📊 MCP 활용 통계: {stats['statistics']['improvement_rate']:.1%} 향상률")
    
    # 비동기 테스트 실행
    asyncio.run(test_mcp_integration())