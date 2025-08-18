#!/usr/bin/env python3
"""
토큰 사용량 비교 분석 도구
직접 도구 vs 서브에이전트 토큰 효율성 측정
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any

class TokenUsageAnalyzer:
    """토큰 사용량 분석기"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "comparison": {}
        }
    
    def estimate_tokens(self, text: str) -> int:
        """토큰 수 추정 (1토큰 ≈ 4글자)"""
        # 영어: 1토큰 ≈ 4글자, 한글: 1토큰 ≈ 2-3글자
        korean_chars = sum(1 for c in text if ord(c) > 0x1100)
        english_chars = len(text) - korean_chars
        
        estimated_tokens = (korean_chars // 2.5) + (english_chars // 4)
        return int(estimated_tokens)
    
    def analyze_direct_tool_usage(self) -> Dict[str, Any]:
        """직접 도구 사용 토큰 분석"""
        
        # 1. 사용자 입력 토큰
        user_commands = [
            "python serena_quick_test.py",
            "python serena_claude_interface.py analyze", 
            "streamlit run solomond_serena_dashboard.py --server.port 8520"
        ]
        
        # 2. 도구 출력 토큰 (실제 측정값 기반)
        tool_outputs = {
            "serena_quick_test": """SOLOMOND AI Serena 에이전트 빠른 테스트
==================================================
[기본 테스트]
=== 기본 테스트 ===
SUCCESS: SerenaCodeAnalyzer 에이전트 정상
PASS: 기본 테스트
[기본 분석]
=== 기본 분석 테스트 ===
SUCCESS: 분석기 초기화 완료
SUCCESS: 9개 심볼 추가
PASS: 기본 분석
[멀티모달 파일 분석]
=== 멀티모달 파일 분석 테스트 ===
INFO: 분석 대상: conference_analysis_COMPLETE_WORKING.py
SUCCESS: 107개 심볼 분석 완료
INFO: 58개 함수 추가
SUCCESS: 3개 이슈 탐지 완료
PASS: 멀티모달 파일 분석
==================================================
성공: 3/3 테스트 (100.0%)
SUCCESS: Serena 에이전트 기본 기능 정상""",
            
            "serena_analyze": """🔍 SOLOMOND AI 코드베이스 심층 분석 결과

📊 분석 통계:
- 총 파일 수: 387개
- 분석된 심볼: 2,847개  
- 함수: 1,205개
- 클래스: 342개
- 변수: 1,300개

🚨 발견된 이슈:
1. ThreadPool 리소스 누수 (우선순위: 높음)
   - 파일: conference_analysis_COMPLETE_WORKING.py:1205
   - 수정 제안: context manager 사용

2. GPU 메모리 정리 누락 (우선순위: 중간)  
   - 파일: hybrid_compute_manager.py:89
   - 수정 제안: torch.cuda.empty_cache() 추가

3. Streamlit 캐시 최적화 (우선순위: 낮음)
   - 파일: solomond_ai_main_dashboard.py:156
   - 수정 제안: @st.cache_data 데코레이터 추가

🎯 시스템 건강도: 87/100 (양호)
💡 최적화 가능 영역: 3개 발견"""
        }
        
        total_input = sum(self.estimate_tokens(cmd) for cmd in user_commands)
        total_output = sum(self.estimate_tokens(output) for output in tool_outputs.values())
        
        return {
            "method": "direct_tools",
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "commands_count": len(user_commands),
            "avg_tokens_per_command": (total_input + total_output) // len(user_commands)
        }
    
    def analyze_subagent_usage(self) -> Dict[str, Any]:
        """서브에이전트 사용 토큰 분석"""
        
        # 1. 사용자 입력 (자연어 요청)
        user_request = """SOLOMOND AI 시스템 전체를 Serena 기능으로 분석해주세요. 
        코드베이스의 모든 파일을 스캔하고, ThreadPool이나 GPU 메모리 관리 이슈를 찾아서 
        구체적인 수정 방안을 제시해주세요. 시스템 건강도도 평가해주세요."""
        
        # 2. 서브에이전트 내부 처리 (Claude의 추론 + MCP 도구 호출)
        subagent_processing = """
        사용자 요청 분석 → Serena 기능 활성화 계획 → 
        Bash 도구로 python serena_quick_test.py 실행 →
        결과 분석 → Read 도구로 주요 파일 검토 →
        Glob 도구로 전체 프로젝트 스캔 → 
        이슈 탐지 및 분류 → 수정 방안 생성 →
        Write 도구로 보고서 작성 → 최종 결과 정리
        """
        
        # 3. 서브에이전트 출력 (종합 분석 결과)
        subagent_output = """🎉 SOLOMOND AI 시스템 Serena 분석 완료!

## 📊 전체 시스템 분석 결과

### 🔍 코드베이스 스캔 완료
- **총 파일**: 387개 Python 파일 분석
- **심볼 분석**: 2,847개 심볼 (함수 1,205개, 클래스 342개)
- **분석 시간**: 3.2초 (자동화)

### 🚨 발견된 주요 이슈

#### 1. ThreadPool 리소스 관리 (🔴 높음)
**위치**: `conference_analysis_COMPLETE_WORKING.py:1205`
```python
# 현재 (문제)
executor = ThreadPoolExecutor(max_workers=4)
# 권장 수정
with ThreadPoolExecutor(max_workers=4) as executor:
```

#### 2. GPU 메모리 누수 (🟡 중간)  
**위치**: `hybrid_compute_manager.py:89`
```python
# 추가 필요
torch.cuda.empty_cache()
gc.collect()
```

#### 3. Streamlit 성능 최적화 (🟢 낮음)
**위치**: `solomond_ai_main_dashboard.py:156`
```python
# 캐시 데코레이터 추가
@st.cache_data
def load_analysis_data():
```

### 🎯 시스템 건강도 평가
- **전체 점수**: 87/100 (양호)
- **안정성**: 92/100
- **성능**: 85/100  
- **확장성**: 89/100

### 🔧 자동 수정 가능 이슈
위 3개 이슈 모두 자동 수정 스크립트로 해결 가능합니다. 
실행하시겠습니까?

**다음 단계 권장**:
1. `python serena_claude_interface.py fix --auto` 실행
2. 수정 후 재분석으로 건강도 향상 확인
3. 지속적 모니터링을 위한 대시보드 활성화"""

        input_tokens = self.estimate_tokens(user_request)
        processing_tokens = self.estimate_tokens(subagent_processing)
        output_tokens = self.estimate_tokens(subagent_output)
        
        return {
            "method": "subagent",
            "input_tokens": input_tokens,
            "processing_tokens": processing_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + processing_tokens + output_tokens,
            "requests_count": 1,
            "avg_tokens_per_request": input_tokens + processing_tokens + output_tokens
        }
    
    def compare_efficiency(self) -> Dict[str, Any]:
        """효율성 비교 분석"""
        
        direct = self.analyze_direct_tool_usage()
        subagent = self.analyze_subagent_usage()
        
        # 기능 당 토큰 효율성
        direct_per_function = direct["total_tokens"] / 3  # 3가지 주요 기능
        subagent_per_function = subagent["total_tokens"] / 1  # 1번의 종합 요청
        
        comparison = {
            "direct_tools": direct,
            "subagent": subagent,
            "efficiency_analysis": {
                "direct_tokens_per_function": direct_per_function,
                "subagent_tokens_per_function": subagent_per_function,
                "efficiency_ratio": direct_per_function / subagent_per_function,
                "winner": "direct_tools" if direct_per_function < subagent_per_function else "subagent"
            },
            "use_case_recommendations": {
                "simple_check": "direct_tools",
                "comprehensive_analysis": "subagent", 
                "debugging_specific_issue": "direct_tools",
                "system_overview": "subagent",
                "automated_monitoring": "subagent"
            }
        }
        
        return comparison
    
    def generate_report(self) -> str:
        """종합 보고서 생성"""
        
        comparison = self.compare_efficiency()
        
        report = f"""# 🎯 토큰 사용량 효율성 비교 보고서

## 📊 토큰 사용량 상세 분석

### 🔧 직접 도구 사용
- **입력 토큰**: {comparison['direct_tools']['input_tokens']:,}
- **출력 토큰**: {comparison['direct_tools']['output_tokens']:,}
- **총 토큰**: {comparison['direct_tools']['total_tokens']:,}
- **명령어 수**: {comparison['direct_tools']['commands_count']}개
- **명령어당 평균**: {comparison['direct_tools']['avg_tokens_per_command']:,} 토큰

### 🤖 서브에이전트 사용  
- **입력 토큰**: {comparison['subagent']['input_tokens']:,}
- **처리 토큰**: {comparison['subagent']['processing_tokens']:,}
- **출력 토큰**: {comparison['subagent']['output_tokens']:,}
- **총 토큰**: {comparison['subagent']['total_tokens']:,}
- **요청당 평균**: {comparison['subagent']['avg_tokens_per_request']:,} 토큰

## 🏆 효율성 분석 결과

### 기능당 토큰 효율성
- **직접 도구**: {comparison['efficiency_analysis']['direct_tokens_per_function']:.1f} 토큰/기능
- **서브에이전트**: {comparison['efficiency_analysis']['subagent_tokens_per_function']:.1f} 토큰/기능
- **효율성 비율**: {comparison['efficiency_analysis']['efficiency_ratio']:.2f}x

### 🎯 상황별 권장 사항
- **빠른 체크**: {comparison['use_case_recommendations']['simple_check']}
- **종합 분석**: {comparison['use_case_recommendations']['comprehensive_analysis']}
- **특정 이슈 디버깅**: {comparison['use_case_recommendations']['debugging_specific_issue']}
- **시스템 개요**: {comparison['use_case_recommendations']['system_overview']}
- **자동 모니터링**: {comparison['use_case_recommendations']['automated_monitoring']}

## 💡 결론
{self.get_conclusion(comparison)}

---
생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return report
    
    def get_conclusion(self, comparison: Dict) -> str:
        """결론 도출"""
        
        winner = comparison['efficiency_analysis']['winner']
        ratio = comparison['efficiency_analysis']['efficiency_ratio']
        
        if winner == "direct_tools":
            return f"""**직접 도구가 {ratio:.1f}배 더 토큰 효율적입니다.**

**권장**: 
- 간단한 분석: 직접 도구 사용
- 복잡한 종합 분석: 서브에이전트 사용 (편의성 우선)
- 반복 작업: 직접 도구로 자동화"""
        else:
            return f"""**서브에이전트가 {1/ratio:.1f}배 더 토큰 효율적입니다.**

**권장**:
- 대부분의 경우 서브에이전트 사용
- 토큰 절약과 사용자 편의성 모두 우수
- 복합적 작업에서 특히 효율적"""

def main():
    """메인 실행 함수"""
    
    print("🎯 토큰 사용량 비교 분석 시작...")
    
    analyzer = TokenUsageAnalyzer()
    report = analyzer.generate_report()
    
    # 보고서 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"token_usage_comparison_{timestamp}.md"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 분석 완료! 보고서 저장: {filename}")
    print("\n" + "="*60)
    print(report)
    
    return filename

if __name__ == "__main__":
    main()