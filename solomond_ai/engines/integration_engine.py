"""
통합 분석 엔진
다중 엔진 결과의 교차 검증 및 통합 분석
"""

import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from .base_engine import BaseEngine

class IntegrationEngine(BaseEngine):
    """통합 분석 엔진 - 교차 검증 및 일관성 분석"""
    
    def __init__(self, engines: Dict[str, BaseEngine]):
        super().__init__("integration")
        self.engines = engines
        self.is_initialized = True
    
    def initialize(self) -> bool:
        """통합 엔진은 별도 초기화 불필요"""
        return True
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """단일 파일에 대한 통합 분석 (사용되지 않음)"""
        return {"error": "IntegrationEngine requires multiple engine results"}
    
    def cross_validate(self, engine_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """다중 엔진 결과의 교차 검증"""
        try:
            # 각 엔진별 성공/실패 통계
            engine_stats = self._calculate_engine_stats(engine_results)
            
            # 공통 키워드 추출
            common_keywords = self._extract_common_keywords(engine_results)
            
            # 일관성 점수 계산
            consistency_score = self._calculate_consistency_score(engine_results)
            
            # 종합 인사이트 생성
            insights = self._generate_insights(engine_results, common_keywords)
            
            # 권장 조치사항
            recommendations = self._generate_recommendations(engine_stats, consistency_score)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "engine_stats": engine_stats,
                "common_keywords": common_keywords,
                "consistency_score": consistency_score,
                "insights": insights,
                "recommendations": recommendations,
                "cross_validation_status": "completed"
            }
            
        except Exception as e:
            logging.error(f"Cross validation failed: {e}")
            return {
                "error": str(e),
                "cross_validation_status": "failed"
            }
    
    def _calculate_engine_stats(self, engine_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """각 엔진별 성능 통계"""
        stats = {}
        
        for engine_name, results in engine_results.items():
            successful = sum(1 for r in results if r.get('success', False))
            total = len(results)
            
            stats[engine_name] = {
                "total_files": total,
                "successful": successful,
                "failed": total - successful,
                "success_rate": successful / total if total > 0 else 0,
                "average_processing_time": self._calculate_avg_processing_time(results)
            }
        
        return stats
    
    def _calculate_avg_processing_time(self, results: List[Dict[str, Any]]) -> float:
        """평균 처리 시간 계산"""
        processing_times = []
        for result in results:
            if 'processing_time' in result:
                processing_times.append(result['processing_time'])
        
        return sum(processing_times) / len(processing_times) if processing_times else 0
    
    def _extract_common_keywords(self, engine_results: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """공통 키워드 추출"""
        all_keywords = {}
        
        # 각 엔진에서 키워드 수집
        for engine_name, results in engine_results.items():
            for result in results:
                if result.get('success', False):
                    # 오디오 엔진 키워드
                    if engine_name == 'audio' and 'keywords' in result:
                        for keyword in result['keywords']:
                            all_keywords[keyword] = all_keywords.get(keyword, 0) + 1
                    
                    # 이미지 엔진에서 텍스트 추출
                    elif engine_name == 'image' and 'full_text' in result:
                        text_keywords = self._extract_keywords_from_text(result['full_text'])
                        for keyword in text_keywords:
                            all_keywords[keyword] = all_keywords.get(keyword, 0) + 1
                    
                    # 텍스트 엔진 키워드
                    elif engine_name == 'text' and 'keywords' in result:
                        for keyword in result['keywords']:
                            all_keywords[keyword] = all_keywords.get(keyword, 0) + 1
        
        # 빈도순 정렬하여 상위 키워드 반환
        sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, count in sorted_keywords[:15] if count >= 2]  # 2회 이상 등장
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출 (간단 버전)"""
        import re
        
        # 한국어 단어 추출
        words = re.findall(r'[가-힣]{2,}', text)
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # 상위 키워드 반환
        top_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:10]
        return [word for word, count in top_words]
    
    def _calculate_consistency_score(self, engine_results: Dict[str, List[Dict[str, Any]]]) -> float:
        """일관성 점수 계산 (0-100)"""
        scores = []
        
        # 각 엔진별 성공률 기반 점수
        for engine_name, results in engine_results.items():
            if results:
                success_rate = sum(1 for r in results if r.get('success', False)) / len(results)
                scores.append(success_rate * 100)
        
        # 전체 평균
        return sum(scores) / len(scores) if scores else 0
    
    def _generate_insights(self, engine_results: Dict[str, List[Dict[str, Any]]], 
                          common_keywords: List[str]) -> Dict[str, Any]:
        """종합 인사이트 생성"""
        insights = {
            "total_files_processed": sum(len(results) for results in engine_results.values()),
            "engines_used": list(engine_results.keys()),
            "most_common_topics": common_keywords[:5],
            "processing_summary": {}
        }
        
        # 각 엔진별 주요 발견사항
        for engine_name, results in engine_results.items():
            successful_results = [r for r in results if r.get('success', False)]
            
            if engine_name == 'audio' and successful_results:
                total_duration = sum(r.get('duration', 0) for r in successful_results)
                insights["processing_summary"]["audio"] = {
                    "total_duration_minutes": round(total_duration / 60, 1),
                    "files_processed": len(successful_results),
                    "average_confidence": self._calculate_average_confidence(successful_results)
                }
            
            elif engine_name == 'image' and successful_results:
                total_text_blocks = sum(r.get('total_blocks', 0) for r in successful_results)
                insights["processing_summary"]["image"] = {
                    "total_text_blocks": total_text_blocks,
                    "files_processed": len(successful_results),
                    "average_confidence": self._calculate_average_confidence(successful_results, 'average_confidence')
                }
            
            elif engine_name == 'video' and successful_results:
                insights["processing_summary"]["video"] = {
                    "files_processed": len(successful_results),
                    "total_sample_frames": sum(r.get('frame_count', 0) for r in successful_results)
                }
            
            elif engine_name == 'text' and successful_results:
                total_words = sum(r.get('basic_analysis', {}).get('word_count', 0) for r in successful_results)
                insights["processing_summary"]["text"] = {
                    "total_words": total_words,
                    "files_processed": len(successful_results)
                }
        
        return insights
    
    def _calculate_average_confidence(self, results: List[Dict[str, Any]], 
                                    confidence_key: str = 'confidence') -> float:
        """평균 신뢰도 계산"""
        confidences = []
        for result in results:
            if confidence_key in result:
                confidences.append(result[confidence_key])
            elif 'segments' in result:  # 오디오 결과의 경우
                segment_confidences = [seg.get('confidence', 0) for seg in result['segments']]
                if segment_confidences:
                    confidences.append(sum(segment_confidences) / len(segment_confidences))
        
        return sum(confidences) / len(confidences) if confidences else 0
    
    def _generate_recommendations(self, engine_stats: Dict[str, Any], 
                                consistency_score: float) -> List[str]:
        """권장 조치사항 생성"""
        recommendations = []
        
        # 일관성 점수 기반 권장사항
        if consistency_score < 50:
            recommendations.append("⚠️ 분석 일관성이 낮습니다. 입력 데이터 품질을 확인해보세요.")
        elif consistency_score < 75:
            recommendations.append("📊 분석 결과가 양호합니다. 추가 검증을 고려해보세요.")
        else:
            recommendations.append("✅ 높은 일관성의 분석 결과입니다.")
        
        # 엔진별 성능 기반 권장사항
        for engine_name, stats in engine_stats.items():
            if stats['success_rate'] < 0.5:
                recommendations.append(f"🔧 {engine_name} 엔진의 성공률이 낮습니다. 설정을 확인해보세요.")
            elif stats['success_rate'] == 1.0:
                recommendations.append(f"✅ {engine_name} 엔진이 완벽하게 작동했습니다.")
        
        # 일반적인 권장사항
        if len(recommendations) == 0:
            recommendations.append("📈 모든 엔진이 정상적으로 작동했습니다.")
        
        return recommendations
    
    def get_supported_formats(self) -> List[str]:
        """통합 엔진은 파일 형식을 직접 지원하지 않음"""
        return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "engines": list(self.engines.keys()),
            "initialized": self.is_initialized,
            "capabilities": ["cross_validation", "consistency_analysis", "insight_generation"]
        }