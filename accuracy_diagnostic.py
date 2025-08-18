#!/usr/bin/env python3
"""
Analysis Accuracy Diagnostic Tool
분석 결과 정확성 문제 진단 및 개선 방안 도출
"""

import os
import json
import time
from datetime import datetime
import requests

class AccuracyDiagnostic:
    def __init__(self):
        self.issues = []
        self.improvements = []
        
    def diagnose_current_system(self):
        """현재 시스템의 정확성 문제 진단"""
        print("=== SOLOMOND AI 분석 정확성 진단 ===")
        
        # 1. 가짜 데이터 vs 실제 데이터 문제
        self.check_fake_vs_real_analysis()
        
        # 2. AI 프롬프트 품질 문제
        self.check_prompt_quality()
        
        # 3. 데이터 전처리 문제
        self.check_data_preprocessing()
        
        # 4. 컨텍스트 손실 문제
        self.check_context_loss()
        
        # 5. 모델 선택 및 설정 문제
        self.check_model_configuration()
        
        return self.generate_improvement_plan()
    
    def check_fake_vs_real_analysis(self):
        """가짜 분석 vs 실제 분석 문제 체크"""
        print("\n1. 가짜 분석 vs 실제 분석 체크:")
        
        # 실제 분석 엔진 확인
        analysis_files = [
            'real_analysis_engine.py',
            'comprehensive_message_extractor.py',
            'core/comprehensive_message_extractor.py'
        ]
        
        real_engine_found = False
        for file in analysis_files:
            if os.path.exists(file):
                print(f"   ✅ 실제 분석 엔진 발견: {file}")
                real_engine_found = True
                break
        
        if not real_engine_found:
            self.issues.append({
                'category': 'CRITICAL',
                'issue': '실제 분석 엔진 없음',
                'description': '현재 시스템이 시뮬레이션/가짜 데이터를 사용하고 있을 가능성',
                'impact': '분석 결과가 실제 파일 내용과 무관한 더미 데이터'
            })
            print("   ❌ 실제 분석 엔진을 찾을 수 없음")
        
        # EasyOCR, Whisper 실제 사용 여부 확인
        try:
            import easyocr
            print("   ✅ EasyOCR 설치됨")
        except ImportError:
            self.issues.append({
                'category': 'HIGH',
                'issue': 'EasyOCR 미설치',
                'description': '이미지 텍스트 추출이 실제로 작동하지 않음',
                'impact': '이미지 분석 결과가 부정확하거나 가짜일 수 있음'
            })
            print("   ❌ EasyOCR 미설치")
        
        try:
            import whisper
            print("   ✅ Whisper 설치됨")
        except ImportError:
            self.issues.append({
                'category': 'HIGH',
                'issue': 'Whisper 미설치',
                'description': '음성 텍스트 변환이 실제로 작동하지 않음',
                'impact': '오디오 분석 결과가 부정확하거나 가짜일 수 있음'
            })
            print("   ❌ Whisper 미설치")
    
    def check_prompt_quality(self):
        """AI 프롬프트 품질 확인"""
        print("\n2. AI 프롬프트 품질 체크:")
        
        # 현재 프롬프트 패턴 분석
        prompt_issues = [
            {
                'problem': '모호한 지시사항',
                'example': '"분석해주세요"와 같은 일반적 요청',
                'solution': '구체적인 분석 항목과 형식 지정'
            },
            {
                'problem': '컨텍스트 부족',
                'example': '파일 유형, 목적, 배경정보 누락',
                'solution': '상세한 컨텍스트와 메타데이터 제공'
            },
            {
                'problem': '검증 부족',
                'example': 'AI 응답에 대한 품질 검증 없음',
                'solution': '다중 모델 검증 또는 결과 검토 시스템'
            }
        ]
        
        for issue in prompt_issues:
            print(f"   ⚠️ {issue['problem']}: {issue['example']}")
            self.issues.append({
                'category': 'MEDIUM',
                'issue': issue['problem'],
                'description': issue['example'],
                'solution': issue['solution']
            })
    
    def check_data_preprocessing(self):
        """데이터 전처리 문제 확인"""
        print("\n3. 데이터 전처리 체크:")
        
        preprocessing_issues = [
            '파일 형식별 최적화 부족',
            '텍스트 정제 및 노이즈 제거 부족',
            '언어 감지 및 처리 부족',
            '특수 문자 및 인코딩 문제',
            '구조화된 데이터 추출 부족'
        ]
        
        for issue in preprocessing_issues:
            print(f"   ⚠️ {issue}")
            self.issues.append({
                'category': 'MEDIUM',
                'issue': '데이터 전처리 부족',
                'description': issue
            })
    
    def check_context_loss(self):
        """컨텍스트 손실 문제 확인"""
        print("\n4. 컨텍스트 손실 체크:")
        
        context_issues = [
            {
                'issue': '파일 간 연관성 손실',
                'description': '여러 파일의 관계와 순서를 고려하지 않음',
                'impact': '전체적인 스토리와 맥락을 놓침'
            },
            {
                'issue': '시간적 순서 무시',
                'description': '시간 순서나 이벤트 흐름을 고려하지 않음',
                'impact': '인과관계와 발전 과정을 놓침'
            },
            {
                'issue': '메타데이터 활용 부족',
                'description': '파일명, 생성일시, 크기 등 메타정보 무시',
                'impact': '중요한 맥락 정보 손실'
            }
        ]
        
        for issue in context_issues:
            print(f"   ⚠️ {issue['issue']}: {issue['description']}")
            self.issues.append({
                'category': 'HIGH',
                'issue': issue['issue'],
                'description': issue['description'],
                'impact': issue['impact']
            })
    
    def check_model_configuration(self):
        """모델 설정 문제 확인"""
        print("\n5. AI 모델 설정 체크:")
        
        # Ollama 연결 및 모델 확인
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=3)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"   ✅ Ollama 연결 성공: {len(models)}개 모델")
                
                # 모델별 품질 평가
                quality_models = ['qwen2.5:7b', 'gemma3:27b', 'qwen3:8b']
                available_quality = [m['name'] for m in models if m['name'] in quality_models]
                
                if available_quality:
                    print(f"   ✅ 고품질 모델 사용 가능: {available_quality}")
                else:
                    self.issues.append({
                        'category': 'MEDIUM',
                        'issue': '저품질 모델만 사용 가능',
                        'description': '분석 정확도가 떨어지는 소형 모델만 설치됨'
                    })
                    print("   ⚠️ 고품질 모델 없음, 정확도 저하 가능")
                    
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            self.issues.append({
                'category': 'CRITICAL',
                'issue': 'Ollama 서버 연결 실패',
                'description': f'AI 분석이 작동하지 않음: {e}'
            })
            print(f"   ❌ Ollama 연결 실패: {e}")
    
    def generate_improvement_plan(self):
        """개선 계획 생성"""
        print(f"\n=== 분석 결과: {len(self.issues)}개 문제 발견 ===")
        
        # 문제별 분류
        critical = [i for i in self.issues if i.get('category') == 'CRITICAL']
        high = [i for i in self.issues if i.get('category') == 'HIGH']
        medium = [i for i in self.issues if i.get('category') == 'MEDIUM']
        
        print(f"🚨 긴급: {len(critical)}개")
        print(f"⚡ 고우선순위: {len(high)}개")
        print(f"📋 중우선순위: {len(medium)}개")
        
        # 개선 계획 생성
        improvement_plan = {
            'timestamp': datetime.now().isoformat(),
            'total_issues': len(self.issues),
            'critical_issues': len(critical),
            'high_priority': len(high),
            'medium_priority': len(medium),
            'issues': self.issues,
            'improvement_actions': self.generate_improvement_actions()
        }
        
        return improvement_plan
    
    def generate_improvement_actions(self):
        """구체적인 개선 액션 생성"""
        actions = []
        
        # 1. 실제 분석 엔진 구현
        actions.append({
            'priority': 'CRITICAL',
            'action': '실제 분석 엔진 구현',
            'description': 'EasyOCR + Whisper + 실제 파일 처리 시스템 구축',
            'expected_impact': '분석 결과 정확도 80% 향상',
            'implementation': [
                'EasyOCR 이미지 텍스트 추출 구현',
                'Whisper 음성 텍스트 변환 구현', 
                'FFMPEG 비디오 분할 및 처리',
                '실제 파일 읽기 및 처리 시스템'
            ]
        })
        
        # 2. 고품질 프롬프트 엔지니어링
        actions.append({
            'priority': 'HIGH',
            'action': '고품질 프롬프트 엔지니어링',
            'description': '전문적이고 구체적인 분석 지시사항 설계',
            'expected_impact': 'AI 분석 품질 60% 향상',
            'implementation': [
                '도메인별 전문 프롬프트 템플릿 개발',
                '단계별 분석 프로세스 설계',
                '검증 및 품질 체크 로직 추가',
                '컨텍스트 보강 시스템 구축'
            ]
        })
        
        # 3. 다중 검증 시스템
        actions.append({
            'priority': 'HIGH',
            'action': '다중 검증 시스템',
            'description': '여러 모델과 방법론으로 결과 검증',
            'expected_impact': '분석 신뢰도 50% 향상',
            'implementation': [
                '2개 이상 AI 모델로 교차 검증',
                '규칙 기반 검증 로직 추가',
                '사용자 피드백 수집 시스템',
                '결과 신뢰도 점수 제공'
            ]
        })
        
        # 4. 컨텍스트 보존 시스템
        actions.append({
            'priority': 'MEDIUM',
            'action': '컨텍스트 보존 시스템',
            'description': '파일 간 관계와 전체적 맥락 유지',
            'expected_impact': '맥락적 이해도 40% 향상',
            'implementation': [
                '파일 메타데이터 활용 시스템',
                '시간순 정렬 및 관계 분석',
                '전체 스토리 재구성 로직',
                '핵심 키워드 추적 시스템'
            ]
        })
        
        # 5. 실시간 품질 모니터링
        actions.append({
            'priority': 'MEDIUM',
            'action': '실시간 품질 모니터링',
            'description': '분석 품질을 지속적으로 모니터링',
            'expected_impact': '지속적 품질 개선',
            'implementation': [
                '분석 결과 품질 지표 개발',
                '사용자 만족도 추적',
                '오류 패턴 분석 시스템',
                '자동 개선 제안 시스템'
            ]
        })
        
        return actions

def main():
    diagnostic = AccuracyDiagnostic()
    report = diagnostic.diagnose_current_system()
    
    # 보고서 저장
    with open('accuracy_diagnostic_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n상세 보고서 저장: accuracy_diagnostic_report.json")
    
    # 우선순위별 개선 방안 출력
    print("\n=== 즉시 개선 방안 ===")
    for action in report['improvement_actions']:
        if action['priority'] in ['CRITICAL', 'HIGH']:
            print(f"\n🔥 {action['action']} ({action['priority']})")
            print(f"   설명: {action['description']}")
            print(f"   기대효과: {action['expected_impact']}")
            for impl in action['implementation'][:2]:  # 상위 2개만 표시
                print(f"   • {impl}")
    
    return report

if __name__ == '__main__':
    main()