#!/usr/bin/env python3
"""
웹 데이터 통합 엔진
MCP 브라우저에서 수집한 웹 데이터를 분석 워크플로우에 통합
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from utils.logger import get_logger

class WebDataIntegration:
    """웹 데이터 통합 시스템"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # 통합 규칙 정의
        self.integration_rules = {
            "jewelry_search": {
                "priority": "high",
                "merge_strategy": "contextual",
                "data_types": ["product_info", "price_comparison", "market_analysis"]
            },
            "competitive_analysis": {
                "priority": "medium", 
                "merge_strategy": "comparative",
                "data_types": ["brand_analysis", "feature_comparison", "pricing_strategy"]
            },
            "market_research": {
                "priority": "medium",
                "merge_strategy": "aggregative", 
                "data_types": ["trend_analysis", "consumer_behavior", "market_forecast"]
            }
        }
        
        # 데이터 품질 기준
        self.quality_thresholds = {
            "min_confidence": 0.7,
            "min_data_completeness": 0.6,
            "max_processing_time": 30.0,
            "required_fields": ["query", "results", "timestamp"]
        }
        
        self.logger.info("웹 데이터 통합 엔진 초기화 완료")
    
    def integrate_web_data_to_workflow(self, web_search_result: Dict[str, Any], 
                                     workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """웹 검색 결과를 워크플로우에 통합"""
        
        integration_result = {
            "timestamp": datetime.now().isoformat(),
            "integration_success": False,
            "web_data_summary": {},
            "enhanced_context": {},
            "recommendations": [],
            "quality_assessment": {},
            "integration_metadata": {}
        }
        
        try:
            self.logger.info("웹 데이터 워크플로우 통합 시작")
            
            # 1. 웹 데이터 품질 검증
            quality_check = self._validate_web_data_quality(web_search_result)
            integration_result["quality_assessment"] = quality_check
            
            if not quality_check["is_valid"]:
                integration_result["error"] = "웹 데이터 품질 검증 실패"
                return integration_result
            
            # 2. 웹 데이터 요약 생성
            web_summary = self._generate_web_data_summary(web_search_result)
            integration_result["web_data_summary"] = web_summary
            
            # 3. 워크플로우 컨텍스트와 결합
            enhanced_context = self._merge_contexts(web_summary, workflow_context)
            integration_result["enhanced_context"] = enhanced_context
            
            # 4. 통합 추천사항 생성
            recommendations = self._generate_integrated_recommendations(
                web_summary, workflow_context, enhanced_context
            )
            integration_result["recommendations"] = recommendations
            
            # 5. 메타데이터 생성
            integration_result["integration_metadata"] = self._generate_integration_metadata(
                web_search_result, workflow_context
            )
            
            integration_result["integration_success"] = True
            self.logger.info("웹 데이터 워크플로우 통합 완료")
            
        except Exception as e:
            self.logger.error(f"웹 데이터 통합 실패: {str(e)}")
            integration_result["error"] = str(e)
        
        return integration_result
    
    def _validate_web_data_quality(self, web_data: Dict[str, Any]) -> Dict[str, Any]:
        """웹 데이터 품질 검증"""
        
        quality_result = {
            "is_valid": False,
            "quality_score": 0.0,
            "issues": [],
            "strengths": []
        }
        
        score = 0.0
        max_score = 100.0
        
        # 필수 필드 확인
        required_fields = self.quality_thresholds["required_fields"]
        missing_fields = [field for field in required_fields if field not in web_data]
        
        if missing_fields:
            quality_result["issues"].append(f"필수 필드 누락: {missing_fields}")
        else:
            score += 25.0
            quality_result["strengths"].append("모든 필수 필드 존재")
        
        # 성공 여부 확인
        if web_data.get("success", False):
            score += 25.0
            quality_result["strengths"].append("검색 성공")
        else:
            quality_result["issues"].append("웹 검색 실패")
        
        # 데이터 완성도 확인
        search_results = web_data.get("search_results", {})
        if search_results:
            successful_searches = 0
            total_searches = 0
            
            for category, results in search_results.items():
                if isinstance(results, dict):
                    total_searches += 1
                    if results.get("success"):
                        successful_searches += 1
                elif isinstance(results, list):
                    for result in results:
                        total_searches += 1
                        if result.get("success"):
                            successful_searches += 1
            
            if total_searches > 0:
                completeness = successful_searches / total_searches
                if completeness >= self.quality_thresholds["min_data_completeness"]:
                    score += 25.0
                    quality_result["strengths"].append(f"데이터 완성도 양호 ({completeness:.1%})")
                else:
                    quality_result["issues"].append(f"데이터 완성도 부족 ({completeness:.1%})")
        
        # 추천사항 존재 확인
        recommendations = web_data.get("recommendations", [])
        if recommendations and len(recommendations) >= 3:
            score += 25.0
            quality_result["strengths"].append(f"풍부한 추천사항 ({len(recommendations)}개)")
        else:
            quality_result["issues"].append("추천사항 부족")
        
        quality_result["quality_score"] = score / max_score
        quality_result["is_valid"] = quality_result["quality_score"] >= self.quality_thresholds["min_confidence"]
        
        return quality_result
    
    def _generate_web_data_summary(self, web_data: Dict[str, Any]) -> Dict[str, Any]:
        """웹 데이터 요약 생성"""
        
        summary = {
            "search_query": web_data.get("query", ""),
            "search_context": web_data.get("context", {}),
            "total_sites_searched": 0,
            "successful_searches": 0,
            "key_findings": [],
            "price_information": {},
            "brand_insights": [],
            "market_trends": [],
            "search_recommendations": web_data.get("recommendations", [])
        }
        
        # 검색 결과 분석
        search_results = web_data.get("search_results", {})
        
        # 구글 검색 결과 처리
        google_result = search_results.get("google", {})
        if google_result.get("success"):
            summary["successful_searches"] += 1
            google_data = google_result.get("data", {})
            if "top_results" in google_data:
                summary["key_findings"].extend([
                    f"구글 검색 상위 결과: {len(google_data['top_results'])}개",
                    f"예상 검색 결과: {google_data.get('estimated_results', 'N/A')}"
                ])
        summary["total_sites_searched"] += 1
        
        # 쇼핑몰 검색 결과 처리
        shopping_results = search_results.get("shopping", [])
        price_data = []
        
        for shop_result in shopping_results:
            summary["total_sites_searched"] += 1
            if shop_result.get("success"):
                summary["successful_searches"] += 1
                shop_data = shop_result.get("data", {})
                
                # 가격 정보 수집
                price_range = shop_data.get("price_range", "")
                if price_range:
                    price_data.append({
                        "site": shop_result.get("site", "Unknown"),
                        "price_range": price_range,
                        "product_count": shop_data.get("products_found", "N/A")
                    })
                
                # 브랜드 정보 수집
                brands = shop_data.get("popular_brands", [])
                if brands:
                    summary["brand_insights"].extend(brands)
        
        if price_data:
            summary["price_information"] = {
                "price_sources": len(price_data),
                "price_data": price_data,
                "price_analysis": self._analyze_price_data(price_data)
            }
        
        # 전문점 검색 결과 처리
        jewelry_results = search_results.get("jewelry", [])
        
        for jewelry_result in jewelry_results:
            summary["total_sites_searched"] += 1
            if jewelry_result.get("success"):
                summary["successful_searches"] += 1
                jewelry_data = jewelry_result.get("data", {})
                
                # 전문점 특화 정보
                specialty = jewelry_data.get("specialty", "")
                if specialty:
                    summary["market_trends"].append(f"{jewelry_result.get('site', 'Unknown')}: {specialty}")
                
                # 서비스 혜택 정보
                benefits = jewelry_data.get("service_benefits", [])
                if benefits:
                    summary["key_findings"].append(f"{jewelry_result.get('site', 'Unknown')} 혜택: {', '.join(benefits[:3])}")
        
        # 브랜드 인사이트 중복 제거 및 정리
        summary["brand_insights"] = list(set(summary["brand_insights"]))
        
        return summary
    
    def _analyze_price_data(self, price_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """가격 데이터 분석"""
        
        analysis = {
            "price_range_analysis": "다양한 가격대 확인됨",
            "market_positioning": "중간 가격대",
            "price_competitiveness": "경쟁적",
            "recommendations": []
        }
        
        # 간단한 가격 분석 (실제로는 더 정교한 분석 필요)
        if len(price_data) >= 3:
            analysis["recommendations"].append("여러 매장 가격 비교 완료")
        
        if any("만원" in data.get("price_range", "") for data in price_data):
            analysis["market_positioning"] = "프리미엄 시장"
            analysis["recommendations"].append("고품질 제품군 확인")
        
        return analysis
    
    def _merge_contexts(self, web_summary: Dict[str, Any], workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """웹 데이터와 워크플로우 컨텍스트 결합"""
        
        enhanced_context = {
            "original_context": workflow_context.copy(),
            "web_insights": web_summary,
            "merged_insights": {},
            "enhanced_recommendations": []
        }
        
        # 기본 정보 결합
        search_context = web_summary.get("search_context", {})
        
        # 상황 정보 결합
        if "situation" in search_context and "customer_situation" in workflow_context:
            enhanced_context["merged_insights"]["situation_alignment"] = {
                "web_search_situation": search_context["situation"],
                "workflow_situation": workflow_context["customer_situation"],
                "consistency": search_context["situation"] == workflow_context.get("customer_situation")
            }
        
        # 예산 정보 결합
        if "budget" in search_context and "budget_info" in workflow_context:
            enhanced_context["merged_insights"]["budget_analysis"] = {
                "search_budget": search_context["budget"],
                "workflow_budget": workflow_context["budget_info"],
                "market_price_data": web_summary.get("price_information", {})
            }
        
        # 키워드 및 관심사 결합
        search_query = web_summary.get("search_query", "")
        workflow_keywords = workflow_context.get("key_topics", [])
        
        enhanced_context["merged_insights"]["topic_relevance"] = {
            "search_terms": search_query.split(),
            "workflow_keywords": workflow_keywords,
            "market_trends": web_summary.get("market_trends", [])
        }
        
        return enhanced_context
    
    def _generate_integrated_recommendations(self, web_summary: Dict[str, Any], 
                                           workflow_context: Dict[str, Any],
                                           enhanced_context: Dict[str, Any]) -> List[str]:
        """통합 추천사항 생성"""
        
        recommendations = []
        
        # 웹 검색 기반 추천사항
        web_recommendations = web_summary.get("search_recommendations", [])
        if web_recommendations:
            recommendations.append("🌐 온라인 시장 조사 결과:")
            recommendations.extend([f"  • {rec}" for rec in web_recommendations[:3]])
        
        # 가격 분석 기반 추천사항
        price_info = web_summary.get("price_information", {})
        if price_info:
            price_analysis = price_info.get("price_analysis", {})
            price_recs = price_analysis.get("recommendations", [])
            if price_recs:
                recommendations.append("💰 가격 분석 결과:")
                recommendations.extend([f"  • {rec}" for rec in price_recs])
        
        # 브랜드 분석 기반 추천사항
        brands = web_summary.get("brand_insights", [])
        if brands:
            recommendations.append(f"🏷️ 주요 브랜드 확인: {', '.join(brands[:3])}")
        
        # 통합 분석 기반 추천사항
        merged_insights = enhanced_context.get("merged_insights", {})
        
        # 상황별 맞춤 추천
        situation_data = merged_insights.get("situation_alignment", {})
        if situation_data.get("consistency"):
            recommendations.append("✅ 검색 목적과 상담 내용이 일치하여 신뢰성 높은 정보 제공 가능")
        
        # 예산 분석 기반 추천
        budget_data = merged_insights.get("budget_analysis", {})
        if budget_data:
            recommendations.append("💡 예산 대비 시장 가격 정보를 종합하여 최적 구매 시점 제안")
        
        # 종합 추천사항
        recommendations.extend([
            "🔍 온라인 정보와 실제 상담 내용을 결합한 맞춤형 솔루션 제공",
            "📊 시장 동향과 개인 요구사항을 균형있게 고려한 의사결정 지원",
            "🎯 웹 검색으로 확인된 최신 정보를 바탕으로 한 실시간 상담 업데이트"
        ])
        
        return recommendations
    
    def _generate_integration_metadata(self, web_data: Dict[str, Any], 
                                     workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """통합 메타데이터 생성"""
        
        metadata = {
            "integration_timestamp": datetime.now().isoformat(),
            "web_data_source": "MCP Browser Integration",
            "integration_version": "1.0.0",
            "data_sources": {
                "web_search_count": 0,
                "workflow_files": 0,
                "integration_points": []
            },
            "quality_metrics": {
                "web_data_completeness": 0.0,
                "context_alignment": 0.0,
                "recommendation_relevance": 0.0
            },
            "processing_stats": {
                "total_processing_time": 0.0,
                "web_search_time": web_data.get("timestamp", ""),
                "integration_complexity": "medium"
            }
        }
        
        # 데이터 소스 카운트
        search_results = web_data.get("search_results", {})
        web_search_count = 0
        
        for category, results in search_results.items():
            if isinstance(results, list):
                web_search_count += len(results)
            else:
                web_search_count += 1
        
        metadata["data_sources"]["web_search_count"] = web_search_count
        metadata["data_sources"]["workflow_files"] = len(workflow_context.get("analyzed_files", []))
        
        # 통합 포인트 식별
        integration_points = []
        if "customer_situation" in workflow_context:
            integration_points.append("situation_context")
        if "budget_info" in workflow_context:
            integration_points.append("budget_analysis")
        if "key_topics" in workflow_context:
            integration_points.append("topic_relevance")
        
        metadata["data_sources"]["integration_points"] = integration_points
        
        return metadata
    
    def create_comprehensive_report(self, integration_result: Dict[str, Any], 
                                  original_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """웹 데이터가 통합된 종합 보고서 생성"""
        
        comprehensive_report = {
            "report_id": f"comprehensive_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "report_type": "web_integrated_analysis",
            "executive_summary": {},
            "detailed_analysis": {},
            "web_market_insights": {},
            "integrated_recommendations": {},
            "appendix": {}
        }
        
        try:
            # 요약 섹션
            comprehensive_report["executive_summary"] = self._create_executive_summary(
                integration_result, original_analysis
            )
            
            # 상세 분석 섹션
            comprehensive_report["detailed_analysis"] = self._create_detailed_analysis(
                integration_result, original_analysis
            )
            
            # 웹 시장 인사이트 섹션
            comprehensive_report["web_market_insights"] = self._create_market_insights(
                integration_result
            )
            
            # 통합 추천사항 섹션
            comprehensive_report["integrated_recommendations"] = self._create_integrated_recommendations_section(
                integration_result
            )
            
            # 부록 섹션
            comprehensive_report["appendix"] = self._create_appendix(
                integration_result, original_analysis
            )
            
            self.logger.info("종합 보고서 생성 완료")
            
        except Exception as e:
            self.logger.error(f"종합 보고서 생성 실패: {str(e)}")
            comprehensive_report["error"] = str(e)
        
        return comprehensive_report
    
    def _create_executive_summary(self, integration_result: Dict[str, Any], 
                                original_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """경영진 요약 생성"""
        
        summary = {
            "overview": "",
            "key_highlights": [],
            "critical_insights": [],
            "action_items": []
        }
        
        # 개요 생성
        web_summary = integration_result.get("web_data_summary", {})
        search_query = web_summary.get("search_query", "주얼리 상품")
        
        summary["overview"] = (
            f"'{search_query}' 관련 종합 분석 결과입니다. "
            f"온라인 시장 조사와 실제 상담 내용을 통합하여 "
            f"고객 맞춤형 솔루션을 제안합니다."
        )
        
        # 주요 하이라이트
        successful_searches = web_summary.get("successful_searches", 0)
        total_searches = web_summary.get("total_sites_searched", 0)
        
        summary["key_highlights"] = [
            f"온라인 조사: {successful_searches}/{total_searches} 사이트 성공적 분석",
            f"시장 트렌드: {len(web_summary.get('market_trends', []))}개 핵심 동향 파악",
            f"브랜드 분석: {len(web_summary.get('brand_insights', []))}개 주요 브랜드 확인",
            f"통합 추천사항: {len(integration_result.get('recommendations', []))}개 제안"
        ]
        
        # 핵심 인사이트
        enhanced_context = integration_result.get("enhanced_context", {})
        merged_insights = enhanced_context.get("merged_insights", {})
        
        if merged_insights.get("situation_alignment", {}).get("consistency"):
            summary["critical_insights"].append("✅ 온라인 검색과 실제 상담 목적이 일치하여 신뢰성 높음")
        
        price_info = web_summary.get("price_information", {})
        if price_info:
            summary["critical_insights"].append(f"💰 {price_info.get('price_sources', 0)}개 매장 가격 정보 확보")
        
        # 액션 아이템
        summary["action_items"] = [
            "시장 조사 결과를 바탕으로 한 고객 맞춤 상품 제안",
            "경쟁력 있는 가격대 확인 및 최적 구매 시점 안내", 
            "브랜드별 특화 서비스 및 혜택 정보 제공"
        ]
        
        return summary
    
    def _create_detailed_analysis(self, integration_result: Dict[str, Any], 
                                original_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """상세 분석 섹션 생성"""
        
        detailed = {
            "original_analysis_summary": {},
            "web_research_findings": {},
            "integration_analysis": {},
            "comparative_insights": {}
        }
        
        # 원본 분석 요약
        detailed["original_analysis_summary"] = {
            "analysis_type": original_analysis.get("analysis_type", "파일 분석"),
            "file_count": len(original_analysis.get("files_analyzed", [])),
            "key_findings": original_analysis.get("key_insights", [])[:5],
            "processing_time": original_analysis.get("processing_time", "N/A")
        }
        
        # 웹 조사 결과
        web_summary = integration_result.get("web_data_summary", {})
        detailed["web_research_findings"] = {
            "search_scope": f"{web_summary.get('total_sites_searched', 0)}개 사이트 조사",
            "success_rate": f"{web_summary.get('successful_searches', 0)}/{web_summary.get('total_sites_searched', 0)}",
            "price_analysis": web_summary.get("price_information", {}),
            "market_trends": web_summary.get("market_trends", []),
            "brand_landscape": web_summary.get("brand_insights", [])
        }
        
        # 통합 분석
        enhanced_context = integration_result.get("enhanced_context", {})
        detailed["integration_analysis"] = {
            "context_alignment": enhanced_context.get("merged_insights", {}),
            "data_quality": integration_result.get("quality_assessment", {}),
            "integration_points": len(enhanced_context.get("merged_insights", {}))
        }
        
        return detailed
    
    def _create_market_insights(self, integration_result: Dict[str, Any]) -> Dict[str, Any]:
        """시장 인사이트 섹션 생성"""
        
        insights = {
            "market_overview": {},
            "competitive_landscape": {},
            "pricing_analysis": {},
            "trend_analysis": {}
        }
        
        web_summary = integration_result.get("web_data_summary", {})
        
        # 시장 개요
        insights["market_overview"] = {
            "search_query": web_summary.get("search_query", ""),
            "market_coverage": f"{web_summary.get('total_sites_searched', 0)}개 플랫폼 조사",
            "data_reliability": "높음" if web_summary.get("successful_searches", 0) > 3 else "보통"
        }
        
        # 경쟁 환경
        brands = web_summary.get("brand_insights", [])
        insights["competitive_landscape"] = {
            "major_brands": brands[:5] if brands else [],
            "market_segments": ["프리미엄", "중간가", "저가"] if brands else [],
            "brand_count": len(brands)
        }
        
        # 가격 분석
        price_info = web_summary.get("price_information", {})
        insights["pricing_analysis"] = price_info if price_info else {"status": "데이터 부족"}
        
        # 트렌드 분석
        trends = web_summary.get("market_trends", [])
        insights["trend_analysis"] = {
            "identified_trends": trends,
            "trend_count": len(trends),
            "trend_reliability": "높음" if len(trends) > 2 else "보통"
        }
        
        return insights
    
    def _create_integrated_recommendations_section(self, integration_result: Dict[str, Any]) -> Dict[str, Any]:
        """통합 추천사항 섹션 생성"""
        
        recommendations_section = {
            "immediate_actions": [],
            "short_term_strategy": [],
            "long_term_considerations": [],
            "risk_assessments": []
        }
        
        recommendations = integration_result.get("recommendations", [])
        
        # 추천사항을 카테고리별로 분류
        for rec in recommendations:
            if "온라인" in rec or "실시간" in rec:
                recommendations_section["immediate_actions"].append(rec)
            elif "예산" in rec or "가격" in rec:
                recommendations_section["short_term_strategy"].append(rec)
            elif "시장" in rec or "브랜드" in rec:
                recommendations_section["long_term_considerations"].append(rec)
        
        # 기본 추천사항이 없는 경우 기본값 제공
        if not recommendations_section["immediate_actions"]:
            recommendations_section["immediate_actions"] = [
                "웹 조사 결과를 바탕으로 한 즉시 실행 가능한 액션 플랜 수립"
            ]
        
        if not recommendations_section["short_term_strategy"]:
            recommendations_section["short_term_strategy"] = [
                "가격 경쟁력 확보를 위한 단기 전략 수립"
            ]
        
        # 리스크 평가
        quality_assessment = integration_result.get("quality_assessment", {})
        if quality_assessment.get("quality_score", 0) < 0.8:
            recommendations_section["risk_assessments"].append(
                "웹 데이터 품질이 제한적이므로 추가 검증 필요"
            )
        
        return recommendations_section
    
    def _create_appendix(self, integration_result: Dict[str, Any], 
                       original_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """부록 섹션 생성"""
        
        appendix = {
            "technical_details": integration_result.get("integration_metadata", {}),
            "data_sources": {},
            "quality_metrics": integration_result.get("quality_assessment", {}),
            "processing_logs": [],
            "raw_data_summary": {}
        }
        
        # 데이터 소스 정보
        web_summary = integration_result.get("web_data_summary", {})
        appendix["data_sources"] = {
            "web_searches": web_summary.get("total_sites_searched", 0),
            "successful_web_searches": web_summary.get("successful_searches", 0),
            "original_files": len(original_analysis.get("files_analyzed", [])),
            "search_query": web_summary.get("search_query", "")
        }
        
        # 처리 로그
        appendix["processing_logs"] = [
            f"웹 데이터 통합 시작: {integration_result.get('timestamp', '')}",
            f"품질 검증 완료: {integration_result.get('quality_assessment', {}).get('is_valid', False)}",
            f"통합 추천사항 생성: {len(integration_result.get('recommendations', []))}개"
        ]
        
        return appendix

# 전역 인스턴스
_global_web_data_integration = None

def get_web_data_integration():
    """전역 웹 데이터 통합 인스턴스 반환"""
    global _global_web_data_integration
    if _global_web_data_integration is None:
        _global_web_data_integration = WebDataIntegration()
    return _global_web_data_integration