#!/usr/bin/env python3
"""
🕷️ Module 2: 웹 크롤러 마이크로서비스
- Streamlit UI의 웹 크롤링 기능을 FastAPI로 변환
- 스마트 메모리 매니저와 완전 통합
- 포트 8002에서 실행
"""

import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import uvicorn
import httpx
from bs4 import BeautifulSoup
import requests

# 최적화된 컴포넌트들 import
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from core.smart_memory_manager import get_memory_stats
from config.security_config import get_system_config

logger = logging.getLogger(__name__)

# Pydantic 모델들
class CrawlRequest(BaseModel):
    """크롤링 요청 모델"""
    urls: List[str]
    max_depth: int = 1
    max_pages: int = 10
    include_images: bool = True
    include_links: bool = True
    custom_headers: Optional[Dict[str, str]] = None

class CrawlResult(BaseModel):
    """크롤링 결과 모델"""
    session_id: str
    status: str
    results: Dict[str, Any]
    processing_time: float
    timestamp: datetime

class WebCrawlerService:
    """웹 크롤러 서비스"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}
        self.crawl_cache: Dict[str, Any] = {}
        self.user_agent = "SolomondAI-WebCrawler/4.0"
        
    async def crawl_websites(self, urls: List[str], max_depth: int = 1, 
                           max_pages: int = 10, include_images: bool = True,
                           include_links: bool = True, 
                           custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """웹사이트 크롤링 실행"""
        session_id = str(uuid.uuid4())
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 세션 정보 저장
            self.active_sessions[session_id] = {
                'start_time': start_time,
                'status': 'processing',
                'urls_count': len(urls),
                'max_depth': max_depth,
                'max_pages': max_pages
            }
            
            results = {
                'session_id': session_id,
                'crawled_pages': [],
                'total_pages': 0,
                'total_images': 0,
                'total_links': 0,
                'errors': [],
                'summary': {}
            }
            
            # 각 URL 크롤링
            for url in urls[:max_pages]:  # max_pages 제한 적용
                try:
                    page_result = await self._crawl_single_page(
                        url, include_images, include_links, custom_headers
                    )
                    results['crawled_pages'].append(page_result)
                    
                    # 통계 업데이트
                    results['total_pages'] += 1
                    if page_result.get('images'):
                        results['total_images'] += len(page_result['images'])
                    if page_result.get('links'):
                        results['total_links'] += len(page_result['links'])
                        
                except Exception as e:
                    error_info = {'url': url, 'error': str(e)}
                    results['errors'].append(error_info)
                    logger.error(f"URL 크롤링 실패 {url}: {e}")
            
            # 깊이별 크롤링 (depth > 1인 경우)
            if max_depth > 1 and results['crawled_pages']:
                await self._crawl_deeper(
                    results, max_depth, max_pages, 
                    include_images, include_links, custom_headers
                )
            
            # 요약 생성
            results['summary'] = await self._generate_crawl_summary(results)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # 세션 완료
            self.active_sessions[session_id]['status'] = 'completed'
            self.active_sessions[session_id]['processing_time'] = processing_time
            
            return {
                'session_id': session_id,
                'status': 'success',
                'results': results,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"크롤링 실패 {session_id}: {e}")
            
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = 'failed'
                self.active_sessions[session_id]['error'] = str(e)
            
            raise HTTPException(status_code=500, detail=f"크롤링 실패: {str(e)}")
    
    async def _crawl_single_page(self, url: str, include_images: bool, 
                               include_links: bool, 
                               custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """단일 페이지 크롤링"""
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        if custom_headers:
            headers.update(custom_headers)
        
        try:
            # 캐시 확인
            cache_key = f"{url}_{hash(str(headers))}"
            if cache_key in self.crawl_cache:
                logger.info(f"캐시에서 가져옴: {url}")
                return self.crawl_cache[cache_key]
            
            # HTTP 요청
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, follow_redirects=True)
                response.raise_for_status()
            
            # HTML 파싱
            soup = BeautifulSoup(response.text, 'html.parser')
            
            page_result = {
                'url': url,
                'title': self._extract_title(soup),
                'status_code': response.status_code,
                'content_length': len(response.text),
                'text_content': self._extract_text_content(soup),
                'meta_description': self._extract_meta_description(soup),
                'crawled_at': datetime.now().isoformat()
            }
            
            # 이미지 추출
            if include_images:
                page_result['images'] = self._extract_images(soup, url)
            
            # 링크 추출
            if include_links:
                page_result['links'] = self._extract_links(soup, url)
            
            # 캐시 저장
            self.crawl_cache[cache_key] = page_result
            
            return page_result
            
        except httpx.TimeoutException:
            raise Exception(f"요청 시간 초과: {url}")
        except httpx.HTTPStatusError as e:
            raise Exception(f"HTTP 오류 {e.response.status_code}: {url}")
        except Exception as e:
            raise Exception(f"페이지 크롤링 실패: {str(e)}")
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """페이지 제목 추출"""
        title_tag = soup.find('title')
        return title_tag.get_text(strip=True) if title_tag else "제목 없음"
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """텍스트 콘텐츠 추출"""
        # 스크립트, 스타일 태그 제거
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        # 여러 공백을 하나로 치환, 줄바꿈 정리
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text[:1000]  # 1000자로 제한
    
    def _extract_meta_description(self, soup: BeautifulSoup) -> Optional[str]:
        """메타 설명 추출"""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            return meta_desc.get('content', '').strip()
        return None
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """이미지 URL 추출"""
        images = []
        img_tags = soup.find_all('img', src=True)
        
        for img in img_tags[:20]:  # 최대 20개로 제한
            src = img.get('src')
            if src:
                # 상대 URL을 절대 URL로 변환
                from urllib.parse import urljoin
                full_url = urljoin(base_url, src)
                
                images.append({
                    'src': full_url,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', '')
                })
        
        return images
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """링크 URL 추출"""
        links = []
        a_tags = soup.find_all('a', href=True)
        
        for a in a_tags[:50]:  # 최대 50개로 제한
            href = a.get('href')
            if href:
                from urllib.parse import urljoin
                full_url = urljoin(base_url, href)
                
                # 외부 링크만 수집 (http/https로 시작하는 것)
                if full_url.startswith(('http://', 'https://')):
                    links.append({
                        'url': full_url,
                        'text': a.get_text(strip=True)[:100],  # 링크 텍스트 100자 제한
                        'title': a.get('title', '')
                    })
        
        return links
    
    async def _crawl_deeper(self, results: Dict[str, Any], max_depth: int, 
                          max_pages: int, include_images: bool, include_links: bool,
                          custom_headers: Optional[Dict[str, str]] = None):
        """깊이별 크롤링 (현재는 기본 구현)"""
        # 추후 구현: 첫 번째 레벨에서 찾은 링크들을 이용해 더 깊이 크롤링
        logger.info(f"깊이 {max_depth} 크롤링 기능 개발 예정")
    
    async def _generate_crawl_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """크롤링 결과 요약 생성"""
        return {
            'total_pages_crawled': results['total_pages'],
            'total_images_found': results['total_images'],
            'total_links_found': results['total_links'],
            'total_errors': len(results['errors']),
            'success_rate': f"{((results['total_pages'] / (results['total_pages'] + len(results['errors']))) * 100) if (results['total_pages'] + len(results['errors'])) > 0 else 0:.1f}%",
            'recommendations': self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        if results['total_errors'] > 0:
            recommendations.append("일부 페이지에서 오류가 발생했습니다. 네트워크 상태를 확인해보세요.")
        
        if results['total_images'] > 100:
            recommendations.append("이미지가 많이 발견되었습니다. 이미지 다운로드 기능을 고려해보세요.")
        
        if results['total_links'] > 200:
            recommendations.append("많은 링크가 발견되었습니다. 더 깊은 크롤링을 고려해보세요.")
        
        recommendations.append("크롤링 결과를 CSV/JSON으로 내보낼 수 있습니다.")
        
        return recommendations

# FastAPI 앱 생성
app = FastAPI(
    title="Module 2: 웹 크롤러 서비스",
    description="지능형 웹 크롤링 및 데이터 수집",
    version="4.0.0"
)

# 서비스 인스턴스
service = WebCrawlerService()

@app.get("/health")
async def health_check():
    """헬스체크"""
    memory_stats = get_memory_stats()
    
    return {
        "status": "healthy",
        "service": "module2_crawler",
        "version": "4.0.0",
        "memory": {
            "memory_percent": memory_stats.get('memory_info', {}).get('percent', 0)
        },
        "cache": {
            "cached_pages": len(service.crawl_cache),
            "active_sessions": len(service.active_sessions)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/crawl", response_model=CrawlResult)
async def crawl_websites(
    background_tasks: BackgroundTasks,
    urls: List[str] = Query(..., description="크롤링할 URL 목록"),
    max_depth: int = Query(1, description="최대 크롤링 깊이"),
    max_pages: int = Query(10, description="최대 페이지 수"),
    include_images: bool = Query(True, description="이미지 포함 여부"),
    include_links: bool = Query(True, description="링크 포함 여부")
):
    """웹사이트 크롤링 API"""
    if not urls:
        raise HTTPException(status_code=400, detail="크롤링할 URL이 없습니다")
    
    # URL 유효성 검사
    for url in urls:
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise HTTPException(status_code=400, detail=f"유효하지 않은 URL: {url}")
        except Exception:
            raise HTTPException(status_code=400, detail=f"URL 파싱 오류: {url}")
    
    # 크롤링 실행
    result = await service.crawl_websites(
        urls, max_depth, max_pages, include_images, include_links
    )
    
    return result

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """세션 정보 조회"""
    if session_id not in service.active_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    return service.active_sessions[session_id]

@app.get("/sessions")
async def list_sessions():
    """전체 세션 목록"""
    return {
        "total_sessions": len(service.active_sessions),
        "sessions": service.active_sessions
    }

@app.delete("/cache")
async def clear_cache():
    """크롤링 캐시 지우기"""
    cache_size = len(service.crawl_cache)
    service.crawl_cache.clear()
    
    return {
        "message": f"{cache_size}개의 캐시 항목을 삭제했습니다",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats")
async def get_service_stats():
    """서비스 통계"""
    memory_stats = get_memory_stats()
    
    return {
        "service_info": {
            "name": "module2_crawler",
            "version": "4.0.0",
            "uptime": "실행 중"
        },
        "memory": memory_stats,
        "sessions": {
            "total": len(service.active_sessions),
            "active": len([s for s in service.active_sessions.values() 
                          if s['status'] == 'processing']),
            "completed": len([s for s in service.active_sessions.values() 
                             if s['status'] == 'completed'])
        },
        "cache": {
            "cached_pages": len(service.crawl_cache),
            "cache_hit_rate": "추후 구현"
        }
    }

if __name__ == "__main__":
    logger.info("🕷️ Module 2 웹 크롤러 서비스 시작: http://localhost:8002")
    
    uvicorn.run(
        "module2_service:app",
        host="0.0.0.0", 
        port=8002,
        reload=True,
        log_level="info"
    )