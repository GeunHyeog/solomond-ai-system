"""
🚀 Solomond AI v2.1 - 품질 강화 데모 스크립트
새로운 품질 검증, 다국어 처리, 다중파일 통합 기능 종합 테스트

Author: 전근혁 (Solomond)
Created: 2025.07.11
Version: 2.1.0
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'demo_v21_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 프로젝트 루트 디렉토리 추가
sys.path.append('core')

try:
    from quality_analyzer_v21 import QualityAnalyzerV21
    from multilingual_processor_v21 import MultilingualProcessorV21
    from multi_file_integrator_v21 import MultiFileIntegratorV21
    from korean_summary_engine_v21 import KoreanSummaryEngineV21
    
    logger.info("✅ v2.1 모든 모듈 import 성공")
except ImportError as e:
    logger.error(f"❌ v2.1 모듈 import 실패: {e}")
    sys.exit(1)

class SolomondAIv21Demo:
    """솔로몬드 AI v2.1 데모 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.demo_start_time = time.time()
        
        # v2.1 모듈 초기화
        self.logger.info("🔧 v2.1 모듈 초기화 중...")
        
        self.quality_analyzer = QualityAnalyzerV21()
        self.multilingual_processor = MultilingualProcessorV21()
        self.file_integrator = MultiFileIntegratorV21()
        self.korean_engine = KoreanSummaryEngineV21()
        
        self.logger.info("✅ v2.1 모든 모듈 초기화 완료")
        
        # 데모 결과 저장
        self.demo_results = {
            'version': '2.1.0',
            'demo_start_time': self.demo_start_time,
            'test_results': {}
        }
    
    def create_sample_files(self):
        """데모용 샘플 파일 생성"""
        self.logger.info("📁 데모용 샘플 파일 생성 중...")
        
        # 샘플 디렉토리 생성
        sample_dir = Path("demo_samples_v21")
        sample_dir.mkdir(exist_ok=True)
        
        # 1. 샘플 텍스트 파일 (한국어)
        korean_sample = """
        주얼리 업계 회의 내용
        
        오늘 회의에서는 다이아몬드 시장의 최신 트렌드에 대해 논의했습니다.
        4C (캐럿, 컬러, 클래리티, 커팅) 기준으로 품질 평가가 이루어지고 있으며,
        GIA 인증서의 중요성이 더욱 강조되고 있습니다.
        
        주요 결정사항:
        1. 새로운 다이아몬드 공급업체와 계약 체결
        2. 품질 관리 시스템 도입 결정
        3. 직원 교육 프로그램 시작 예정
        
        다음 회의는 내주 화요일 오후 2시에 예정되어 있습니다.
        """
        
        with open(sample_dir / "korean_meeting.txt", "w", encoding="utf-8") as f:
            f.write(korean_sample)
        
        # 2. 샘플 영어 텍스트
        english_sample = """
        Jewelry Industry Conference Notes
        
        Today's discussion focused on emerging trends in the diamond market.
        We covered the importance of 4C grading (Carat, Color, Clarity, Cut)
        and the critical role of GIA certification in ensuring quality.
        
        Market insights:
        - Growing demand for sustainable diamonds
        - Technology integration in manufacturing
        - Customer preference shifting towards customization
        
        Action items:
        - Follow up with new suppliers by Friday
        - Review quality control procedures
        - Schedule staff training sessions
        """
        
        with open(sample_dir / "english_conference.txt", "w", encoding="utf-8") as f:
            f.write(english_sample)
        
        # 3. 샘플 중국어 텍스트
        chinese_sample = """
        珠宝行业分析报告
        
        钻石市场概况：
        - 4C标准：克拉、颜色、净度、切工
        - GIA认证的重要性日益增长
        - 市场需求持续增长
        
        技术发展：
        - 人工智能在钻石分级中的应用
        - 3D打印技术在珠宝制造中的使用
        - 区块链技术确保钻石溯源
        
        市场机会：
        - 中国市场潜力巨大
        - 年轻消费者群体增长
        - 在线销售渠道扩展
        """
        
        with open(sample_dir / "chinese_analysis.txt", "w", encoding="utf-8") as f:
            f.write(chinese_sample)
        
        # 4. 종합 보고서 (혼합 언어)
        mixed_sample = """
        International Jewelry Trade Show Summary
        국제 주얼리 전시회 요약
        
        Key Findings / 주요 발견사항:
        - Diamond quality standards are becoming more stringent globally
        - 전 세계적으로 다이아몬드 품질 기준이 더욱 엄격해지고 있음
        - GIA certification remains the gold standard
        - Technology integration is accelerating industry transformation
        
        Market Trends / 시장 트렌드:
        1. Sustainable sourcing - 지속가능한 원료 조달
        2. Customization demand - 맞춤화 수요 증가
        3. Digital transformation - 디지털 변환
        
        Next Steps / 다음 단계:
        - Establish partnerships with certified suppliers
        - 인증된 공급업체와 파트너십 구축
        - Implement quality management system
        - 품질 관리 시스템 도입
        """
        
        with open(sample_dir / "mixed_language_report.txt", "w", encoding="utf-8") as f:
            f.write(mixed_sample)
        
        sample_files = [
            sample_dir / "korean_meeting.txt",
            sample_dir / "english_conference.txt", 
            sample_dir / "chinese_analysis.txt",
            sample_dir / "mixed_language_report.txt"
        ]
        
        self.logger.info(f"✅ {len(sample_files)}개 샘플 파일 생성 완료")
        return sample_files
    
    def test_quality_analyzer(self, sample_files: List[Path]):
        """품질 분석기 테스트"""
        self.logger.info("🔍 품질 분석기 테스트 시작...")
        
        start_time = time.time()
        
        try:
            # 파일 경로를 문자열로 변환
            file_paths = [str(f) for f in sample_files]
            
            # 품질 분석 실행
            quality_results = self.quality_analyzer.analyze_batch_quality(file_paths)
            
            processing_time = time.time() - start_time
            
            # 결과 검증
            if quality_results.get('processing_complete'):
                stats = quality_results['batch_statistics']
                
                self.logger.info(f"✅ 품질 분석 완료 - {processing_time:.2f}초")
                self.logger.info(f"   📊 분석 파일: {stats['total_files']}개")
                self.logger.info(f"   📈 평균 품질: {stats['average_quality']:.1f}점")
                self.logger.info(f"   🏆 고품질 파일: {stats['high_quality_count']}개")
                
                # 권장사항 출력
                recommendations = stats.get('recommendations', [])
                if recommendations:
                    self.logger.info("💡 품질 개선 권장사항:")
                    for rec in recommendations[:3]:
                        self.logger.info(f"   • {rec}")
                
                self.demo_results['test_results']['quality_analyzer'] = {
                    'status': 'success',
                    'processing_time': processing_time,
                    'average_quality': stats['average_quality'],
                    'recommendations_count': len(recommendations)
                }
                
                return quality_results
            else:
                raise Exception("품질 분석 실패")
                
        except Exception as e:
            self.logger.error(f"❌ 품질 분석기 테스트 실패: {e}")
            self.demo_results['test_results']['quality_analyzer'] = {
                'status': 'failed',
                'error': str(e)
            }
            return None
    
    def test_multilingual_processor(self, sample_files: List[Path]):
        """다국어 처리기 테스트"""
        self.logger.info("🌍 다국어 처리기 테스트 시작...")
        
        start_time = time.time()
        
        try:
            # 파일 내용 읽기
            file_contents = []
            for file_path in sample_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_contents.append(f.read())
            
            # 다국어 처리 실행
            multilingual_results = self.multilingual_processor.process_multilingual_content(
                file_contents, "text"
            )
            
            processing_time = time.time() - start_time
            
            # 결과 검증
            if multilingual_results.get('processing_statistics'):
                stats = multilingual_results['processing_statistics']
                lang_dist = multilingual_results.get('language_distribution', {})
                
                self.logger.info(f"✅ 다국어 처리 완료 - {processing_time:.2f}초")
                self.logger.info(f"   📂 처리 파일: {stats['successful_files']}개")
                self.logger.info(f"   🎯 평균 신뢰도: {stats['average_confidence']:.1%}")
                
                # 언어 분포 출력
                self.logger.info("🗣️ 감지된 언어 분포:")
                lang_names = {'ko': '한국어', 'en': '영어', 'zh': '중국어', 'ja': '일본어'}
                for lang, ratio in lang_dist.items():
                    lang_name = lang_names.get(lang, lang)
                    self.logger.info(f"   • {lang_name}: {ratio:.1%}")
                
                # 통합 결과 확인
                integrated = multilingual_results.get('integrated_result', {})
                if integrated and not integrated.get('error'):
                    korean_text_length = len(integrated.get('final_korean_text', ''))
                    jewelry_terms = integrated.get('jewelry_terms_count', 0)
                    
                    self.logger.info(f"   📝 한국어 통합 텍스트: {korean_text_length:,}자")
                    self.logger.info(f"   💎 주얼리 전문용어: {jewelry_terms}개")
                
                self.demo_results['test_results']['multilingual_processor'] = {
                    'status': 'success',
                    'processing_time': processing_time,
                    'average_confidence': stats['average_confidence'],
                    'language_distribution': lang_dist,
                    'korean_text_length': korean_text_length
                }
                
                return multilingual_results
            else:
                raise Exception("다국어 처리 실패")
                
        except Exception as e:
            self.logger.error(f"❌ 다국어 처리기 테스트 실패: {e}")
            self.demo_results['test_results']['multilingual_processor'] = {
                'status': 'failed',
                'error': str(e)
            }
            return None
    
    def test_file_integrator(self, sample_files: List[Path]):
        """파일 통합 분석기 테스트"""
        self.logger.info("📊 다중 파일 통합 분석기 테스트 시작...")
        
        start_time = time.time()
        
        try:
            file_paths = [str(f) for f in sample_files]
            
            # 파일 통합 분석 실행
            integration_results = self.file_integrator.integrate_multiple_files(file_paths)
            
            processing_time = time.time() - start_time
            
            # 결과 검증
            if integration_results.get('processing_statistics'):
                stats = integration_results['processing_statistics']
                timeline = integration_results.get('timeline_analysis', {})
                
                self.logger.info(f"✅ 파일 통합 분석 완료 - {processing_time:.2f}초")
                self.logger.info(f"   📁 총 파일: {stats['total_files']}개")
                self.logger.info(f"   🎯 감지된 세션: {stats['total_sessions']}개")
                self.logger.info(f"   ⏱️ 분석 기간: {timeline.get('total_duration_hours', 0):.1f}시간")
                
                # 세션별 정보
                sessions = integration_results.get('individual_sessions', [])
                if sessions:
                    self.logger.info("📋 세션별 분석 결과:")
                    for i, session in enumerate(sessions[:3]):  # 상위 3개만 표시
                        self.logger.info(f"   • 세션 #{i+1}: {session.session_type} - {len(session.files)}개 파일")
                        self.logger.info(f"     제목: {session.title[:50]}...")
                        self.logger.info(f"     신뢰도: {session.confidence_score:.1%}")
                
                # 전체 통합 결과
                overall = integration_results.get('overall_integration', {})
                if overall and not overall.get('error'):
                    insights_count = len(overall.get('overall_insights', []))
                    self.logger.info(f"   💡 추출된 인사이트: {insights_count}개")
                
                self.demo_results['test_results']['file_integrator'] = {
                    'status': 'success',
                    'processing_time': processing_time,
                    'total_sessions': stats['total_sessions'],
                    'insights_count': insights_count
                }
                
                return integration_results
            else:
                raise Exception("파일 통합 분석 실패")
                
        except Exception as e:
            self.logger.error(f"❌ 파일 통합 분석기 테스트 실패: {e}")
            self.demo_results['test_results']['file_integrator'] = {
                'status': 'failed',
                'error': str(e)
            }
            return None
    
    def test_korean_engine(self, integrated_content: str):
        """한국어 분석 엔진 테스트"""
        self.logger.info("🎯 한국어 통합 분석 엔진 테스트 시작...")
        
        start_time = time.time()
        
        try:
            if not integrated_content:
                integrated_content = "주얼리 업계 분석 샘플 텍스트입니다. 다이아몬드 품질과 GIA 인증에 대해 논의했습니다."
            
            # 한국어 분석 실행
            korean_results = self.korean_engine.analyze_korean_content(
                integrated_content, "comprehensive"
            )
            
            processing_time = time.time() - start_time
            
            # 결과 검증
            if korean_results:
                self.logger.info(f"✅ 한국어 분석 완료 - {processing_time:.2f}초")
                self.logger.info(f"   🎯 분석 신뢰도: {korean_results.confidence_score:.1%}")
                
                # 인사이트 개수 확인
                business_insights = len(korean_results.business_insights)
                technical_insights = len(korean_results.technical_insights)
                market_insights = len(korean_results.market_insights)
                action_items = len(korean_results.action_items)
                
                self.logger.info(f"   💼 비즈니스 인사이트: {business_insights}개")
                self.logger.info(f"   🔧 기술적 인사이트: {technical_insights}개")
                self.logger.info(f"   🌍 시장 인사이트: {market_insights}개")
                self.logger.info(f"   📋 액션 아이템: {action_items}개")
                
                # 주얼리 전문용어 분석
                jewelry_terms = len(korean_results.jewelry_terminology)
                if jewelry_terms > 0:
                    self.logger.info(f"   💎 주얼리 전문용어: {jewelry_terms}개")
                    top_terms = list(korean_results.jewelry_terminology.items())[:3]
                    for term, count in top_terms:
                        self.logger.info(f"     • {term}: {count}회")
                
                # 종합 리포트 생성 테스트
                report = self.korean_engine.generate_comprehensive_report(korean_results)
                report_length = len(report)
                self.logger.info(f"   📄 종합 리포트: {report_length:,}자 생성됨")
                
                self.demo_results['test_results']['korean_engine'] = {
                    'status': 'success',
                    'processing_time': processing_time,
                    'confidence_score': korean_results.confidence_score,
                    'total_insights': business_insights + technical_insights + market_insights,
                    'action_items': action_items,
                    'jewelry_terms': jewelry_terms,
                    'report_length': report_length
                }
                
                return korean_results, report
            else:
                raise Exception("한국어 분석 결과 없음")
                
        except Exception as e:
            self.logger.error(f"❌ 한국어 분석 엔진 테스트 실패: {e}")
            self.demo_results['test_results']['korean_engine'] = {
                'status': 'failed',
                'error': str(e)
            }
            return None, None
    
    def run_comprehensive_demo(self):
        """종합 데모 실행"""
        self.logger.info("🚀 솔로몬드 AI v2.1 종합 데모 시작!")
        self.logger.info("=" * 60)
        
        try:
            # 1. 샘플 파일 생성
            sample_files = self.create_sample_files()
            
            # 2. 품질 분석 테스트
            quality_results = self.test_quality_analyzer(sample_files)
            
            # 3. 다국어 처리 테스트
            multilingual_results = self.test_multilingual_processor(sample_files)
            
            # 4. 파일 통합 분석 테스트
            integration_results = self.test_file_integrator(sample_files)
            
            # 5. 한국어 분석 테스트
            integrated_content = ""
            if integration_results and integration_results.get('overall_integration'):
                integrated_content = integration_results['overall_integration'].get('integrated_content', '')
            
            if not integrated_content and multilingual_results and multilingual_results.get('integrated_result'):
                integrated_content = multilingual_results['integrated_result'].get('final_korean_text', '')
            
            korean_results, final_report = self.test_korean_engine(integrated_content)
            
            # 6. 데모 결과 요약
            self.generate_demo_summary(final_report)
            
        except Exception as e:
            self.logger.error(f"❌ 종합 데모 실행 중 오류: {e}")
            self.demo_results['demo_status'] = 'failed'
            self.demo_results['demo_error'] = str(e)
    
    def generate_demo_summary(self, final_report: str = None):
        """데모 결과 요약 생성"""
        total_time = time.time() - self.demo_start_time
        
        self.logger.info("=" * 60)
        self.logger.info("🏆 솔로몬드 AI v2.1 데모 완료!")
        self.logger.info(f"⏱️ 총 실행 시간: {total_time:.2f}초")
        
        # 각 모듈별 테스트 결과 요약
        test_results = self.demo_results['test_results']
        
        self.logger.info("\n📊 모듈별 테스트 결과:")
        for module, result in test_results.items():
            status = "✅ 성공" if result['status'] == 'success' else "❌ 실패"
            time_taken = result.get('processing_time', 0)
            self.logger.info(f"   • {module}: {status} ({time_taken:.2f}초)")
        
        # 성공률 계산
        success_count = sum(1 for r in test_results.values() if r['status'] == 'success')
        total_tests = len(test_results)
        success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
        
        self.logger.info(f"\n🎯 전체 성공률: {success_rate:.1f}% ({success_count}/{total_tests})")
        
        # 성능 지표 요약
        if success_count > 0:
            self.logger.info("\n📈 성능 지표:")
            
            # 품질 분석 결과
            if 'quality_analyzer' in test_results and test_results['quality_analyzer']['status'] == 'success':
                qa_result = test_results['quality_analyzer']
                self.logger.info(f"   • 평균 품질 점수: {qa_result['average_quality']:.1f}/100")
            
            # 다국어 처리 결과
            if 'multilingual_processor' in test_results and test_results['multilingual_processor']['status'] == 'success':
                ml_result = test_results['multilingual_processor']
                self.logger.info(f"   • 다국어 처리 신뢰도: {ml_result['average_confidence']:.1%}")
                self.logger.info(f"   • 한국어 통합 텍스트: {ml_result['korean_text_length']:,}자")
            
            # 한국어 분석 결과
            if 'korean_engine' in test_results and test_results['korean_engine']['status'] == 'success':
                ke_result = test_results['korean_engine']
                self.logger.info(f"   • 한국어 분석 신뢰도: {ke_result['confidence_score']:.1%}")
                self.logger.info(f"   • 추출된 인사이트: {ke_result['total_insights']}개")
                self.logger.info(f"   • 액션 아이템: {ke_result['action_items']}개")
                self.logger.info(f"   • 주얼리 전문용어: {ke_result['jewelry_terms']}개")
        
        # 최종 리포트 저장
        if final_report:
            report_file = f"demo_v21_final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(final_report)
            self.logger.info(f"\n📄 최종 리포트 저장: {report_file}")
        
        # 데모 결과 JSON 저장
        self.demo_results['demo_end_time'] = time.time()
        self.demo_results['demo_total_time'] = total_time
        self.demo_results['demo_success_rate'] = success_rate
        
        results_file = f"demo_v21_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.demo_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📊 데모 결과 저장: {results_file}")
        
        # 성공 메시지
        if success_rate >= 80:
            self.logger.info("\n🎉 솔로몬드 AI v2.1 데모가 성공적으로 완료되었습니다!")
            self.logger.info("💎 모든 주요 기능이 정상 작동하고 있습니다.")
        else:
            self.logger.warning(f"\n⚠️ 일부 모듈에서 오류가 발생했습니다. (성공률: {success_rate:.1f}%)")
            self.logger.info("🔧 오류 로그를 확인하여 문제를 해결해주세요.")

def main():
    """메인 함수"""
    print("🏆 솔로몬드 AI v2.1 - 품질 강화 데모")
    print("💎 주얼리 업계 특화 AI 분석 플랫폼")
    print("=" * 60)
    
    try:
        # 데모 인스턴스 생성
        demo = SolomondAIv21Demo()
        
        # 종합 데모 실행
        demo.run_comprehensive_demo()
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 데모가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 데모 실행 중 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
