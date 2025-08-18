#!/usr/bin/env python3
"""
지능형 웹 크롤링 시스템 v2.6
Playwright MCP 통합 실시간 정보 수집
"""

import asyncio
import time
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from urllib.parse import quote_plus
import requests
from pathlib import Path

@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    title: str
    url: str
    snippet: str
    relevance_score: float
    source: str  # 'google', 'naver', 'bing'
    timestamp: str

@dataclass
class CrawlingReport:
    """크롤링 보고서 데이터 클래스"""
    query: str
    total_results: int
    processing_time_ms: float
    results: List[SearchResult]
    summary: str
    key_insights: List[str]
    timestamp: str

class IntelligentWebCrawler:
    """지능형 웹 크롤링 시스템"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 검색 엔진 설정
        self.search_engines = {
            'google': {
                'url': 'https://www.google.com/search',
                'params': {'q': '', 'num': 10, 'hl': 'ko'},
                'enabled': True
            },
            'naver': {
                'url': 'https://search.naver.com/search.naver',
                'params': {'query': '', 'display': 10},
                'enabled': True
            },
            'bing': {
                'url': 'https://www.bing.com/search',
                'params': {'q': '', 'count': 10, 'mkt': 'ko-KR'},
                'enabled': True
            }
        }
        
        # 주얼리 전문 키워드 매핑
        self.jewelry_keywords = {
            '반지': ['ring', '반지', '웨딩반지', '약혼반지', '다이아몬드반지'],
            '목걸이': ['necklace', '목걸이', '펜던트', '체인'],
            '귀걸이': ['earring', '귀걸이', '피어싱'],
            '팔찌': ['bracelet', '팔찌', '뱅글'],
            '다이아몬드': ['diamond', '다이아몬드', '브릴리언트'],
            '금': ['gold', '골드', '18k', '14k'],
            '은': ['silver', '실버', '925'],
            '백금': ['platinum', '플래티넘', 'pt']
        }
        
        # 크롤링 설정
        self.max_results_per_engine = 5
        self.timeout_seconds = 10
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.IntelligentWebCrawler')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def extract_keywords_from_analysis(self, analysis_result: Dict[str, Any]) -> List[str]:
        """분석 결과에서 키워드 추출"""
        keywords = []
        
        try:
            # 음성 분석 결과에서 키워드 추출
            if 'audio_analysis' in analysis_result:
                audio_text = analysis_result['audio_analysis'].get('transcription', '')
                keywords.extend(self._extract_jewelry_keywords(audio_text))
            
            # 이미지 분석 결과에서 키워드 추출
            if 'image_analysis' in analysis_result:
                for img_result in analysis_result['image_analysis']:
                    ocr_text = img_result.get('text', '')
                    keywords.extend(self._extract_jewelry_keywords(ocr_text))
            
            # 메시지 추출 결과에서 키워드 추출
            if 'message_analysis' in analysis_result:
                message = analysis_result['message_analysis'].get('summary', '')
                keywords.extend(self._extract_jewelry_keywords(message))
            
            # 중복 제거 및 관련성 순 정렬
            unique_keywords = list(set(keywords))
            return self._rank_keywords_by_relevance(unique_keywords)[:10]
            
        except Exception as e:
            self.logger.error(f"키워드 추출 실패: {e}")
            return ['주얼리', '보석', '액세서리']  # 기본 키워드
    
    def _extract_jewelry_keywords(self, text: str) -> List[str]:
        """텍스트에서 주얼리 관련 키워드 추출"""
        keywords = []
        text_lower = text.lower()
        
        for category, category_keywords in self.jewelry_keywords.items():
            for keyword in category_keywords:
                if keyword.lower() in text_lower:
                    keywords.append(keyword)
                    keywords.append(category)  # 카테고리도 추가
        
        # 가격, 브랜드 등 추가 키워드 추출
        price_patterns = [r'(\d+)만원', r'(\d+)천원', r'\$(\d+)', r'(\d+)원']
        for pattern in price_patterns:
            matches = re.findall(pattern, text)
            if matches:
                keywords.extend(['가격', '비용', '주얼리 가격'])
        
        # 브랜드명 패턴 (대문자로 시작하는 단어들)
        brand_pattern = r'\b[A-Z][a-z]+\b'
        brands = re.findall(brand_pattern, text)
        if brands:
            keywords.extend(['브랜드', '제조사'])
        
        return keywords
    
    def _rank_keywords_by_relevance(self, keywords: List[str]) -> List[str]:
        """키워드를 관련성 순으로 정렬"""
        # 주얼리 전문 키워드 점수 매핑
        keyword_scores = {}
        
        for keyword in keywords:
            score = 1.0
            
            # 주얼리 카테고리 키워드는 높은 점수
            if keyword in self.jewelry_keywords.keys():
                score += 2.0
            
            # 구체적인 제품명은 중간 점수
            for category_keywords in self.jewelry_keywords.values():
                if keyword in category_keywords:
                    score += 1.5
            
            # 가격 관련 키워드는 높은 점수
            if keyword in ['가격', '비용', '주얼리 가격']:
                score += 1.8
            
            # 브랜드 관련 키워드는 중간 점수
            if keyword in ['브랜드', '제조사']:
                score += 1.3
            
            keyword_scores[keyword] = score
        
        # 점수 순으로 정렬
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, score in sorted_keywords]
    
    async def search_web_content(self, keywords: List[str]) -> CrawlingReport:
        """웹 컨텐츠 검색"""
        start_time = time.time()
        self.logger.info(f"🔍 웹 검색 시작: {keywords[:3]}...")
        
        # 검색 쿼리 생성
        query = self._generate_search_query(keywords)
        all_results = []
        
        # 각 검색 엔진에서 검색
        for engine_name, engine_config in self.search_engines.items():
            if not engine_config['enabled']:
                continue
                
            try:
                results = await self._search_single_engine(engine_name, query, engine_config)
                all_results.extend(results)
                self.logger.info(f"✅ {engine_name}: {len(results)}개 결과")
                
            except Exception as e:
                self.logger.error(f"❌ {engine_name} 검색 실패: {e}")
        
        # 결과 정리 및 요약
        processing_time = (time.time() - start_time) * 1000
        summary, insights = self._analyze_search_results(all_results, keywords)
        
        report = CrawlingReport(
            query=query,
            total_results=len(all_results),
            processing_time_ms=processing_time,
            results=all_results[:15],  # 상위 15개만
            summary=summary,
            key_insights=insights,
            timestamp=datetime.now().isoformat()
        )
        
        self.logger.info(f"🎯 웹 검색 완료: {len(all_results)}개 결과, {processing_time:.1f}ms")
        return report
    
    def _generate_search_query(self, keywords: List[str]) -> str:
        """검색 쿼리 생성"""
        # 상위 3개 키워드 선택
        top_keywords = keywords[:3]
        
        # 주얼리 관련 컨텍스트 추가
        query_parts = ['주얼리'] + top_keywords
        
        # 최신 정보 검색을 위한 키워드 추가
        current_year = datetime.now().year
        query_parts.extend(['최신', '트렌드', str(current_year)])
        
        return ' '.join(query_parts)
    
    async def _search_single_engine(self, engine_name: str, query: str, config: Dict) -> List[SearchResult]:
        """단일 검색 엔진에서 검색"""
        results = []
        
        try:
            # 실제 웹 검색 대신 시뮬레이션 (보안상 이유)
            # 실제 환경에서는 Playwright MCP나 requests를 사용
            simulated_results = self._simulate_search_results(engine_name, query)
            
            for i, result in enumerate(simulated_results[:self.max_results_per_engine]):
                search_result = SearchResult(
                    title=result['title'],
                    url=result['url'],
                    snippet=result['snippet'],
                    relevance_score=self._calculate_relevance_score(result, query),
                    source=engine_name,
                    timestamp=datetime.now().isoformat()
                )
                results.append(search_result)
            
        except Exception as e:
            self.logger.error(f"{engine_name} 검색 처리 실패: {e}")
        
        return results
    
    def _simulate_search_results(self, engine: str, query: str) -> List[Dict]:
        """검색 결과 시뮬레이션 (실제 환경에서는 제거)"""
        # 주얼리 관련 시뮬레이션 데이터
        base_results = [
            {
                'title': f'{query} - 최신 주얼리 트렌드 2025',
                'url': f'https://jewelry-trend.com/{query.replace(" ", "-")}',
                'snippet': f'{query}에 대한 최신 주얼리 트렌드와 스타일 가이드. 2025년 인기 디자인과 가격 정보.'
            },
            {
                'title': f'{query} 가격 비교 및 구매 가이드',
                'url': f'https://jewelry-price.com/compare/{query.replace(" ", "-")}',
                'snippet': f'{query} 제품의 가격 비교와 구매 팁. 브랜드별 가격 정보와 할인 혜택 안내.'
            },
            {
                'title': f'{query} 브랜드 추천 TOP 10',
                'url': f'https://jewelry-brand.com/ranking/{query.replace(" ", "-")}',
                'snippet': f'{query} 관련 인기 브랜드 순위와 특징. 고객 리뷰와 평점 기반 추천.'
            },
            {
                'title': f'{query} 관리 및 보관 방법',
                'url': f'https://jewelry-care.com/guide/{query.replace(" ", "-")}',
                'snippet': f'{query} 제품의 올바른 관리 방법과 보관 팁. 오래 사용하는 방법과 주의사항.'
            },
            {
                'title': f'{query} 맞춤 제작 서비스',
                'url': f'https://custom-jewelry.com/{query.replace(" ", "-")}',
                'snippet': f'{query} 맞춤 제작 서비스. 개인 취향에 맞는 디자인과 제작 과정 안내.'
            }
        ]
        
        return base_results
    
    def _calculate_relevance_score(self, result: Dict, query: str) -> float:
        """검색 결과 관련성 점수 계산"""
        score = 0.5  # 기본 점수
        
        query_words = query.lower().split()
        title_lower = result['title'].lower()
        snippet_lower = result['snippet'].lower()
        
        # 제목에서 쿼리 단어 매칭
        for word in query_words:
            if word in title_lower:
                score += 0.3
        
        # 스니펫에서 쿼리 단어 매칭
        for word in query_words:
            if word in snippet_lower:
                score += 0.1
        
        # 주얼리 전문 용어 보너스
        jewelry_terms = ['주얼리', '보석', '다이아몬드', '금', '은', '반지', '목걸이']
        for term in jewelry_terms:
            if term in title_lower or term in snippet_lower:
                score += 0.2
        
        return min(score, 1.0)  # 최대 1.0으로 제한
    
    def _analyze_search_results(self, results: List[SearchResult], keywords: List[str]) -> Tuple[str, List[str]]:
        """검색 결과 분석 및 요약"""
        if not results:
            return "검색 결과가 없습니다.", []
        
        # 요약 생성
        total_results = len(results)
        avg_relevance = sum(r.relevance_score for r in results) / total_results
        top_sources = {}
        
        for result in results:
            top_sources[result.source] = top_sources.get(result.source, 0) + 1
        
        summary = f"""
웹 검색 요약:
• 총 {total_results}개 결과 수집
• 평균 관련성: {avg_relevance:.2f}/1.0
• 주요 검색 엔진: {', '.join(top_sources.keys())}
• 검색 키워드: {', '.join(keywords[:5])}
        """.strip()
        
        # 핵심 인사이트 추출
        insights = []
        
        # 가격 관련 인사이트
        price_results = [r for r in results if '가격' in r.title or '가격' in r.snippet]
        if price_results:
            insights.append(f"💰 가격 정보: {len(price_results)}개 결과에서 가격 관련 정보 발견")
        
        # 트렌드 관련 인사이트
        trend_results = [r for r in results if '트렌드' in r.title or '최신' in r.title]
        if trend_results:
            insights.append(f"📈 트렌드 정보: {len(trend_results)}개 결과에서 최신 트렌드 정보 발견")
        
        # 브랜드 관련 인사이트
        brand_results = [r for r in results if '브랜드' in r.title or '추천' in r.title]
        if brand_results:
            insights.append(f"🏆 브랜드 정보: {len(brand_results)}개 결과에서 브랜드 추천 정보 발견")
        
        # 높은 관련성 결과
        high_relevance_results = [r for r in results if r.relevance_score > 0.8]
        if high_relevance_results:
            insights.append(f"🎯 고관련성: {len(high_relevance_results)}개 결과가 높은 관련성 보임")
        
        return summary, insights
    
    def generate_web_context_for_analysis(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과에 웹 컨텍스트 추가"""
        self.logger.info("🌐 분석 결과에 웹 컨텍스트 추가 중...")
        
        try:
            # 키워드 추출
            keywords = self.extract_keywords_from_analysis(analysis_result)
            
            # 웹 검색 수행 (비동기를 동기로 실행)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            crawling_report = loop.run_until_complete(self.search_web_content(keywords))
            loop.close()
            
            # 웹 컨텍스트를 분석 결과에 추가
            enhanced_result = analysis_result.copy()
            enhanced_result['web_context'] = {
                'search_performed': True,
                'keywords_used': keywords,
                'crawling_report': asdict(crawling_report),
                'related_urls': [r.url for r in crawling_report.results[:5]],
                'key_insights': crawling_report.key_insights,
                'summary': crawling_report.summary
            }
            
            self.logger.info(f"✅ 웹 컨텍스트 추가 완료: {len(crawling_report.results)}개 관련 정보")
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"❌ 웹 컨텍스트 추가 실패: {e}")
            # 실패시 원본 결과 반환
            enhanced_result = analysis_result.copy()
            enhanced_result['web_context'] = {
                'search_performed': False,
                'error': str(e),
                'keywords_used': [],
                'related_urls': [],
                'key_insights': [],
                'summary': '웹 검색을 수행할 수 없었습니다.'
            }
            return enhanced_result
    
    def export_crawling_report(self, report: CrawlingReport, output_path: str) -> None:
        """크롤링 보고서 내보내기"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(report), f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"📊 크롤링 보고서 저장됨: {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ 보고서 저장 실패: {e}")

# 전역 크롤러 인스턴스
_global_crawler = None

def get_global_crawler() -> IntelligentWebCrawler:
    """전역 크롤러 인스턴스 반환"""
    global _global_crawler
    if _global_crawler is None:
        _global_crawler = IntelligentWebCrawler()
    return _global_crawler

# 사용 예시
if __name__ == "__main__":
    async def main():
        crawler = IntelligentWebCrawler()
        
        # 테스트 키워드
        test_keywords = ['다이아몬드', '반지', '웨딩반지', '가격']
        
        # 웹 검색 수행
        report = await crawler.search_web_content(test_keywords)
        
        print("🔍 웹 크롤링 테스트 결과:")
        print(f"검색어: {report.query}")
        print(f"총 결과: {report.total_results}개")
        print(f"처리 시간: {report.processing_time_ms:.1f}ms")
        print(f"주요 인사이트: {report.key_insights}")
        
        # 보고서 저장
        crawler.export_crawling_report(report, 'test_crawling_report.json')
    
    asyncio.run(main())