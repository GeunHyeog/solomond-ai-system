#!/usr/bin/env python3
"""
브라우저 자동화 엔진 - Playwright MCP 통합
실시간 웹 검색, 스크린샷, 자동 크롤링
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from utils.logger import get_logger

class BrowserAutomationEngine:
    """브라우저 자동화 엔진"""
    
    def __init__(self):
        self.logger = get_logger(f'{__name__}.BrowserAutomationEngine')
        
        # 주얼리 관련 검색 사이트들
        self.jewelry_sites = {
            "search_engines": {
                "google": "https://www.google.com/search?q=",
                "naver": "https://search.naver.com/search.naver?query=",
                "daum": "https://search.daum.net/search?q="
            },
            "jewelry_sites": {
                "golden_dew": "https://www.goldendew.co.kr",
                "lotte_jewelry": "https://www.lottejewelry.co.kr", 
                "hyundai_jewelry": "https://www.hyundaijewelry.co.kr",
                "shinsegae_jewelry": "https://www.ssg.com/search.ssg?target=all&query="
            },
            "price_comparison": {
                "danawa": "https://www.danawa.com/search/?query=",
                "enuri": "https://www.enuri.com/search.jsp?keyword="
            }
        }
        
        # 검색 시나리오별 키워드
        self.search_scenarios = {
            "product_search": {
                "keywords": ["결혼반지", "목걸이", "다이아몬드", "골드", "플래티넘"],
                "modifiers": ["가격", "할인", "이벤트", "신상품", "추천"]
            },
            "market_research": {
                "keywords": ["주얼리 시장", "보석 트렌드", "브라이달 주얼리"],
                "modifiers": ["2025", "최신", "동향", "전망"]
            },
            "competitor_analysis": {
                "keywords": ["주얼리 브랜드", "보석 업체", "온라인 쇼핑몰"],
                "modifiers": ["비교", "순위", "리뷰", "평점"]
            }
        }
        
        self.logger.info("브라우저 자동화 엔진 초기화 완료")
    
    async def search_jewelry_information(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """주얼리 정보 종합 검색"""
        
        search_result = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "context": context or {},
            "search_results": {},
            "screenshots": [],
            "extracted_data": {},
            "recommendations": []
        }
        
        try:
            self.logger.info(f"주얼리 정보 검색 시작: {query}")
            
            # 1. 네이버 검색 (한국 시장 정보)
            naver_result = await self._search_naver(query)
            search_result["search_results"]["naver"] = naver_result
            
            # 2. 주요 주얼리 사이트 검색
            jewelry_results = await self._search_jewelry_sites(query)
            search_result["search_results"]["jewelry_sites"] = jewelry_results
            
            # 3. 가격 비교 사이트 확인
            price_results = await self._search_price_comparison(query)
            search_result["search_results"]["price_comparison"] = price_results
            
            # 4. 데이터 통합 및 분석
            search_result["extracted_data"] = self._extract_and_analyze_data(search_result["search_results"])
            
            # 5. 추천사항 생성
            search_result["recommendations"] = self._generate_search_recommendations(
                search_result["extracted_data"], context
            )
            
            self.logger.info("주얼리 정보 검색 완료")
            return search_result
            
        except Exception as e:
            self.logger.error(f"주얼리 정보 검색 실패: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "error": str(e),
                "status": "failed"
            }
    
    async def _search_naver(self, query: str) -> Dict[str, Any]:
        """네이버 검색 실행"""
        
        try:
            # MCP Playwright를 사용한 네이버 검색
            from utils.mcp_functions import mcp__playwright__browser_navigate, mcp__playwright__browser_snapshot
            
            search_url = f"https://search.naver.com/search.naver?query={query} 주얼리"
            
            # 네이버 검색 페이지로 이동
            await mcp__playwright__browser_navigate(url=search_url)
            
            # 페이지 로딩 대기
            await asyncio.sleep(3)
            
            # 페이지 스냅샷 캡처
            snapshot = await mcp__playwright__browser_snapshot()
            
            return {
                "site": "naver",
                "url": search_url,
                "snapshot_captured": True,
                "page_content": self._parse_naver_results(snapshot),
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"네이버 검색 실패: {str(e)}")
            return {
                "site": "naver",
                "error": str(e),
                "status": "failed"
            }
    
    async def _search_jewelry_sites(self, query: str) -> Dict[str, Any]:
        """주얼리 전문 사이트 검색"""
        
        results = {}
        
        for site_name, base_url in self.jewelry_sites["jewelry_sites"].items():
            try:
                # 각 사이트별 검색 실행
                site_result = await self._search_single_site(site_name, base_url, query)
                results[site_name] = site_result
                
                # 사이트간 간격 두기
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"{site_name} 검색 실패: {str(e)}")
                results[site_name] = {"error": str(e), "status": "failed"}
        
        return results
    
    async def _search_single_site(self, site_name: str, base_url: str, query: str) -> Dict[str, Any]:
        """개별 사이트 검색"""
        
        try:
            from utils.mcp_functions import (
                mcp__playwright__browser_navigate, 
                mcp__playwright__browser_snapshot,
                mcp__playwright__browser_type,
                mcp__playwright__browser_click
            )
            
            # 사이트 메인 페이지로 이동
            await mcp__playwright__browser_navigate(url=base_url)
            await asyncio.sleep(2)
            
            # 검색창 찾기 및 검색어 입력 시도
            try:
                # 일반적인 검색창 선택자들
                search_selectors = [
                    "input[type='search']",
                    "input[name='query']", 
                    "input[name='keyword']",
                    "#search",
                    ".search-input"
                ]
                
                search_executed = False
                for selector in search_selectors:
                    try:
                        # 검색창에 검색어 입력
                        await mcp__playwright__browser_type(
                            element=f"검색창 ({selector})",
                            ref=selector,
                            text=query
                        )
                        
                        # Enter 키로 검색 실행
                        await mcp__playwright__browser_type(
                            element=f"검색창 ({selector})",
                            ref=selector,
                            text="",
                            submit=True
                        )
                        
                        search_executed = True
                        break
                        
                    except:
                        continue
                
                if search_executed:
                    await asyncio.sleep(3)  # 검색 결과 로딩 대기
                
            except Exception as search_error:
                self.logger.warning(f"{site_name} 검색창 입력 실패: {str(search_error)}")
            
            # 페이지 스냅샷 캡처
            snapshot = await mcp__playwright__browser_snapshot()
            
            # 결과 파싱
            parsed_data = self._parse_site_content(site_name, snapshot)
            
            return {
                "site": site_name,
                "url": base_url,
                "search_executed": search_executed,
                "snapshot_captured": True,
                "extracted_data": parsed_data,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "site": site_name,
                "error": str(e),
                "status": "failed"
            }
    
    async def _search_price_comparison(self, query: str) -> Dict[str, Any]:
        """가격 비교 사이트 검색"""
        
        results = {}
        
        for site_name, search_url in self.jewelry_sites["price_comparison"].items():
            try:
                full_url = f"{search_url}{query}"
                
                from utils.mcp_functions import mcp__playwright__browser_navigate, mcp__playwright__browser_snapshot
                
                await mcp__playwright__browser_navigate(url=full_url)
                await asyncio.sleep(3)
                
                snapshot = await mcp__playwright__browser_snapshot()
                
                results[site_name] = {
                    "url": full_url,
                    "snapshot_captured": True,
                    "price_data": self._extract_price_info(snapshot),
                    "status": "success"
                }
                
            except Exception as e:
                results[site_name] = {"error": str(e), "status": "failed"}
        
        return results
    
    def _parse_naver_results(self, snapshot: str) -> Dict[str, Any]:
        """네이버 검색 결과 파싱"""
        
        # 스냅샷에서 주요 정보 추출
        parsed_data = {
            "organic_results": [],
            "shopping_results": [],
            "news_results": [],
            "blog_results": []
        }
        
        try:
            # 네이버 검색 결과 구조 분석
            if "쇼핑" in snapshot:
                parsed_data["has_shopping"] = True
            if "뉴스" in snapshot:
                parsed_data["has_news"] = True
            if "블로그" in snapshot:
                parsed_data["has_blog"] = True
                
            # 가격 정보 추출
            price_patterns = ["만원", "원", "가격", "할인"]
            for pattern in price_patterns:
                if pattern in snapshot:
                    parsed_data["price_mentions"] = True
                    break
                    
        except Exception as e:
            self.logger.error(f"네이버 결과 파싱 실패: {str(e)}")
        
        return parsed_data
    
    def _parse_site_content(self, site_name: str, snapshot: str) -> Dict[str, Any]:
        """사이트 콘텐츠 파싱"""
        
        parsed_data = {
            "site": site_name,
            "content_type": "unknown",
            "products_found": False,
            "price_info": [],
            "brand_info": {},
            "key_features": []
        }
        
        try:
            # 제품 관련 키워드 확인
            product_keywords = ["반지", "목걸이", "귀걸이", "팔찌", "다이아몬드", "골드"]
            found_products = [keyword for keyword in product_keywords if keyword in snapshot]
            
            if found_products:
                parsed_data["products_found"] = True
                parsed_data["found_keywords"] = found_products
            
            # 가격 정보 추출
            if "원" in snapshot or "만원" in snapshot:
                parsed_data["has_price_info"] = True
            
            # 브랜드 정보
            brand_keywords = ["브랜드", "컬렉션", "시리즈"]
            for keyword in brand_keywords:
                if keyword in snapshot:
                    parsed_data["brand_info"]["has_brand_content"] = True
                    break
                    
        except Exception as e:
            self.logger.error(f"{site_name} 콘텐츠 파싱 실패: {str(e)}")
        
        return parsed_data
    
    def _extract_price_info(self, snapshot: str) -> Dict[str, Any]:
        """가격 정보 추출"""
        
        price_data = {
            "price_found": False,
            "price_ranges": [],
            "discount_info": [],
            "comparison_available": False
        }
        
        try:
            # 가격 패턴 확인
            import re
            
            # 가격 패턴 (예: 1,000,000원, 100만원)
            price_patterns = [
                r'(\d{1,3}(?:,\d{3})*)\s*원',
                r'(\d+)\s*만원',
                r'(\d{1,3}(?:,\d{3})*)\s*~\s*(\d{1,3}(?:,\d{3})*)\s*원'
            ]
            
            for pattern in price_patterns:
                matches = re.findall(pattern, snapshot)
                if matches:
                    price_data["price_found"] = True
                    price_data["price_ranges"].extend(matches)
            
            # 할인 정보 확인
            discount_keywords = ["할인", "세일", "이벤트", "%"]
            for keyword in discount_keywords:
                if keyword in snapshot:
                    price_data["discount_info"].append(keyword)
                    
        except Exception as e:
            self.logger.error(f"가격 정보 추출 실패: {str(e)}")
        
        return price_data
    
    def _extract_and_analyze_data(self, search_results: Dict) -> Dict[str, Any]:
        """검색 결과 데이터 통합 분석"""
        
        analysis = {
            "market_overview": {},
            "product_availability": {},
            "price_analysis": {},
            "trend_insights": {},
            "competition_analysis": {}
        }
        
        try:
            # 1. 시장 개요 분석
            total_sites_searched = 0
            successful_searches = 0
            
            for category, results in search_results.items():
                if isinstance(results, dict):
                    for site, result in results.items():
                        total_sites_searched += 1
                        # result가 문자열인 경우 처리
                        if isinstance(result, str):
                            # 문자열 결과는 성공으로 간주
                            successful_searches += 1
                        elif isinstance(result, dict) and result.get("status") == "success":
                            successful_searches += 1
            
            analysis["market_overview"] = {
                "sites_searched": total_sites_searched,
                "successful_searches": successful_searches,
                "search_success_rate": successful_searches / max(1, total_sites_searched),
                "data_completeness": "high" if successful_searches >= 3 else "medium" if successful_searches >= 1 else "low"
            }
            
            # 2. 제품 가용성 분석
            products_found = 0
            price_info_available = 0
            
            for category, results in search_results.items():
                if isinstance(results, dict):
                    for site, result in results.items():
                        # result 타입에 따른 처리
                        if isinstance(result, str):
                            # 문자열 결과에서 제품 정보 확인
                            if any(keyword in result for keyword in ["반지", "목걸이", "귀걸이"]):
                                products_found += 1
                            if "원" in result or "만원" in result:
                                price_info_available += 1
                        elif isinstance(result, dict) and result.get("status") == "success":
                            if result.get("extracted_data", {}).get("products_found"):
                                products_found += 1
                            if result.get("extracted_data", {}).get("has_price_info"):
                                price_info_available += 1
            
            analysis["product_availability"] = {
                "sites_with_products": products_found,
                "sites_with_prices": price_info_available,
                "availability_score": products_found / max(1, successful_searches)
            }
            
            # 3. 가격 분석
            price_data_sources = []
            for category, results in search_results.items():
                if "price_comparison" in category:
                    for site, result in results.items():
                        # result 타입에 따른 가격 데이터 확인
                        if isinstance(result, str):
                            if "원" in result or "가격" in result:
                                price_data_sources.append(site)
                        elif isinstance(result, dict) and result.get("status") == "success" and result.get("price_data", {}).get("price_found"):
                            price_data_sources.append(site)
            
            analysis["price_analysis"] = {
                "price_sources": price_data_sources,
                "price_comparison_available": len(price_data_sources) > 0,
                "market_price_transparency": "high" if len(price_data_sources) >= 2 else "medium" if len(price_data_sources) == 1 else "low"
            }
            
        except Exception as e:
            self.logger.error(f"데이터 분석 실패: {str(e)}")
            analysis["error"] = str(e)
        
        return analysis
    
    def _generate_search_recommendations(self, extracted_data: Dict, context: Dict) -> List[str]:
        """검색 기반 추천사항 생성"""
        
        recommendations = []
        
        try:
            market_overview = extracted_data.get("market_overview", {})
            product_availability = extracted_data.get("product_availability", {})
            price_analysis = extracted_data.get("price_analysis", {})
            
            # 검색 성공률 기반 추천
            success_rate = market_overview.get("search_success_rate", 0)
            if success_rate < 0.5:
                recommendations.append("추가 검색 채널 활용 권장 (소셜미디어, 전문 커뮤니티)")
            
            # 제품 가용성 기반 추천
            availability_score = product_availability.get("availability_score", 0)
            if availability_score > 0.7:
                recommendations.append("다양한 온라인 옵션 확인 가능, 비교 구매 권장")
            elif availability_score > 0.3:
                recommendations.append("제한적 온라인 옵션, 오프라인 매장 방문 고려")
            else:
                recommendations.append("온라인 정보 부족, 전문 상담사와 직접 상담 권장")
            
            # 가격 투명성 기반 추천
            price_transparency = price_analysis.get("market_price_transparency", "low")
            if price_transparency == "high":
                recommendations.append("가격 비교가 용이, 최적 가격대 선택 가능")
            elif price_transparency == "medium":
                recommendations.append("일부 가격 정보 확인 가능, 추가 견적 요청 권장")
            else:
                recommendations.append("가격 정보 부족, 직접 문의를 통한 견적 필요")
            
            # 컨텍스트 기반 추천
            if context and context.get("situation"):
                situation = context["situation"].lower()
                if "결혼" in situation:
                    recommendations.append("브라이달 전문 매장 및 커스텀 서비스 고려")
                if "예산" in situation:
                    recommendations.append("예산별 옵션 비교 및 할부 서비스 확인")
                    
        except Exception as e:
            self.logger.error(f"추천사항 생성 실패: {str(e)}")
            recommendations.append("검색 결과 분석 중 오류 발생, 전문가 상담 권장")
        
        return recommendations if recommendations else ["추가 정보 수집을 위한 심화 검색 권장"]
    
    async def capture_competitive_analysis(self, competitor_urls: List[str]) -> Dict[str, Any]:
        """경쟁사 분석을 위한 스크린샷 및 데이터 수집"""
        
        analysis_result = {
            "timestamp": datetime.now().isoformat(),
            "competitor_data": {},
            "comparison_matrix": {},
            "insights": []
        }
        
        try:
            from utils.mcp_functions import (
                mcp__playwright__browser_navigate,
                mcp__playwright__browser_snapshot, 
                mcp__playwright__browser_take_screenshot
            )
            
            for i, url in enumerate(competitor_urls):
                competitor_name = f"competitor_{i+1}"
                
                try:
                    # 경쟁사 사이트 방문
                    await mcp__playwright__browser_navigate(url=url)
                    await asyncio.sleep(3)
                    
                    # 스냅샷 캡처
                    snapshot = await mcp__playwright__browser_snapshot()
                    
                    # 스크린샷 저장
                    screenshot_filename = f"competitor_{i+1}_{int(time.time())}.png"
                    await mcp__playwright__browser_take_screenshot(filename=screenshot_filename)
                    
                    # 데이터 분석
                    competitor_analysis = self._analyze_competitor_page(snapshot)
                    
                    analysis_result["competitor_data"][competitor_name] = {
                        "url": url,
                        "screenshot": screenshot_filename,
                        "analysis": competitor_analysis,
                        "status": "success"
                    }
                    
                except Exception as e:
                    analysis_result["competitor_data"][competitor_name] = {
                        "url": url,
                        "error": str(e),
                        "status": "failed"
                    }
                
                # 사이트간 간격
                await asyncio.sleep(2)
            
            # 비교 매트릭스 생성
            analysis_result["comparison_matrix"] = self._create_comparison_matrix(
                analysis_result["competitor_data"]
            )
            
            # 인사이트 생성
            analysis_result["insights"] = self._generate_competitive_insights(
                analysis_result["comparison_matrix"]
            )
            
        except Exception as e:
            analysis_result["error"] = str(e)
        
        return analysis_result
    
    def _analyze_competitor_page(self, snapshot: str) -> Dict[str, Any]:
        """경쟁사 페이지 분석"""
        
        analysis = {
            "ui_elements": {},
            "content_strategy": {},
            "pricing_visibility": {},
            "user_experience": {}
        }
        
        try:
            # UI 요소 분석
            ui_keywords = ["button", "link", "search", "menu", "navigation"]
            analysis["ui_elements"] = {
                keyword: keyword in snapshot.lower() for keyword in ui_keywords
            }
            
            # 콘텐츠 전략 분석
            content_keywords = ["product", "collection", "brand", "story", "about"]
            analysis["content_strategy"] = {
                keyword: keyword in snapshot.lower() for keyword in content_keywords
            }
            
            # 가격 가시성 분석
            price_keywords = ["price", "cost", "원", "만원", "할인", "sale"]
            analysis["pricing_visibility"] = {
                "price_displayed": any(keyword in snapshot.lower() for keyword in price_keywords),
                "discount_offered": "할인" in snapshot or "sale" in snapshot.lower()
            }
            
            # 사용자 경험 분석
            ux_keywords = ["search", "filter", "sort", "compare", "wishlist"]
            analysis["user_experience"] = {
                keyword: keyword in snapshot.lower() for keyword in ux_keywords
            }
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def _create_comparison_matrix(self, competitor_data: Dict) -> Dict[str, Any]:
        """경쟁사 비교 매트릭스 생성"""
        
        matrix = {
            "feature_comparison": {},
            "strengths_weaknesses": {},
            "market_positioning": {}
        }
        
        try:
            # 기능 비교
            features = ["search", "pricing_visibility", "user_experience", "content_strategy"]
            
            for feature in features:
                matrix["feature_comparison"][feature] = {}
                for comp_name, comp_data in competitor_data.items():
                    if comp_data.get("status") == "success":
                        analysis = comp_data.get("analysis", {})
                        if feature in analysis:
                            matrix["feature_comparison"][feature][comp_name] = analysis[feature]
            
            # 강점/약점 분석
            for comp_name, comp_data in competitor_data.items():
                if comp_data.get("status") == "success":
                    analysis = comp_data.get("analysis", {})
                    
                    strengths = []
                    weaknesses = []
                    
                    # 가격 표시가 잘 되어 있으면 강점
                    if analysis.get("pricing_visibility", {}).get("price_displayed"):
                        strengths.append("가격 투명성")
                    else:
                        weaknesses.append("가격 정보 부족")
                    
                    # 사용자 경험 요소가 많으면 강점
                    ux_features = analysis.get("user_experience", {})
                    ux_count = sum(1 for v in ux_features.values() if v)
                    if ux_count >= 3:
                        strengths.append("우수한 사용자 경험")
                    elif ux_count <= 1:
                        weaknesses.append("제한적 사용자 기능")
                    
                    matrix["strengths_weaknesses"][comp_name] = {
                        "strengths": strengths,
                        "weaknesses": weaknesses
                    }
                    
        except Exception as e:
            matrix["error"] = str(e)
        
        return matrix
    
    def _generate_competitive_insights(self, comparison_matrix: Dict) -> List[str]:
        """경쟁 분석 인사이트 생성"""
        
        insights = []
        
        try:
            strengths_weaknesses = comparison_matrix.get("strengths_weaknesses", {})
            
            # 공통 강점 파악
            all_strengths = []
            all_weaknesses = []
            
            for comp_data in strengths_weaknesses.values():
                all_strengths.extend(comp_data.get("strengths", []))
                all_weaknesses.extend(comp_data.get("weaknesses", []))
            
            # 가장 많이 언급된 강점
            if all_strengths:
                strength_counts = {}
                for strength in all_strengths:
                    strength_counts[strength] = strength_counts.get(strength, 0) + 1
                
                most_common_strength = max(strength_counts.items(), key=lambda x: x[1])
                insights.append(f"시장 공통 강점: {most_common_strength[0]} ({most_common_strength[1]}개 업체)")
            
            # 가장 많이 언급된 약점
            if all_weaknesses:
                weakness_counts = {}
                for weakness in all_weaknesses:
                    weakness_counts[weakness] = weakness_counts.get(weakness, 0) + 1
                
                most_common_weakness = max(weakness_counts.items(), key=lambda x: x[1])
                insights.append(f"시장 공통 개선점: {most_common_weakness[0]} ({most_common_weakness[1]}개 업체)")
            
            # 차별화 기회
            insights.append("차별화 기회: 가격 투명성과 사용자 경험 개선에 집중")
            insights.append("권장 전략: 모바일 최적화 및 실시간 상담 기능 강화")
            
        except Exception as e:
            insights.append(f"인사이트 생성 중 오류: {str(e)}")
        
        return insights if insights else ["경쟁 분석 데이터 부족으로 추가 조사 필요"]

# 사용 예시
async def test_browser_automation():
    """브라우저 자동화 테스트"""
    
    engine = BrowserAutomationEngine()
    
    # 주얼리 검색 테스트
    search_result = await engine.search_jewelry_information(
        "결혼반지 200만원",
        context={
            "situation": "결혼 준비",
            "budget": "200만원",
            "preferences": "심플한 디자인"
        }
    )
    
    print("검색 결과:", json.dumps(search_result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(test_browser_automation())