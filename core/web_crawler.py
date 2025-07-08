"""
솔로몬드 AI 시스템 - 웹 크롤링 엔진
유튜브, 웹사이트, 뉴스 등에서 주얼리 관련 정보 자동 수집 모듈
"""

import os
import re
import asyncio
import aiohttp
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from urllib.parse import urljoin, urlparse, parse_qs
import logging
import json
from datetime import datetime, timedelta
import hashlib

# 웹 크롤링 라이브러리
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# 유튜브 다운로더
try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False

# RSS 피드 처리
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

class WebCrawler:
    """웹 크롤링 및 콘텐츠 수집 클래스"""
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 주얼리 관련 키워드
        self.jewelry_keywords = [
            'jewelry', 'diamond', 'gold', 'silver', 'platinum', 'gem', 'jewel',
            '주얼리', '다이아몬드', '보석', '금', '은', '반지', '목걸이', '귀걸이',
            'GIA', 'AGS', '4C', 'carat', 'clarity', 'color', 'cut',
            '캐럿', '컷', '컬러', '클래리티', '감정서'
        ]
        
        # 주요 주얼리 사이트들
        self.jewelry_news_sources = [
            {
                "name": "JCK Magazine",
                "url": "https://www.jckonline.com",
                "rss": "https://www.jckonline.com/rss.xml"
            },
            {
                "name": "National Jeweler",
                "url": "https://www.nationaljeweler.com",
                "rss": "https://www.nationaljeweler.com/rss.xml"
            },
            {
                "name": "Rapaport",
                "url": "https://www.rapaport.com",
                "rss": "https://www.rapaport.com/rss.xml"
            },
            {
                "name": "Korean Jewelry News",
                "url": "https://www.jewelry.co.kr",
                "rss": None
            }
        ]
        
        # 캐시 디렉토리
        self.cache_dir = Path("temp_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        logging.info("웹 크롤링 엔진 초기화 완료")
        logging.info(f"지원 라이브러리: BS4={BS4_AVAILABLE}, Requests={REQUESTS_AVAILABLE}, yt-dlp={YTDLP_AVAILABLE}")
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 시작"""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    def get_supported_sources(self) -> Dict[str, List[str]]:
        """지원하는 소스 타입 반환"""
        return {
            "video": ["youtube", "vimeo"] if YTDLP_AVAILABLE else [],
            "websites": ["general", "jewelry_specific", "news"],
            "feeds": ["rss", "atom"] if FEEDPARSER_AVAILABLE else [],
            "social": ["twitter_public", "instagram_public"]
        }
    
    async def process_url(self, 
                         url: str, 
                         content_type: str = "auto",
                         extract_video: bool = False) -> Dict:
        """
        URL 처리 메인 함수
        
        Args:
            url: 처리할 URL
            content_type: 콘텐츠 타입 ("auto", "video", "article", "feed")
            extract_video: 비디오 다운로드 여부
            
        Returns:
            처리 결과 딕셔너리
        """
        try:
            print(f"🌐 URL 처리 시작: {url}")
            
            # URL 유효성 검사
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {
                    "success": False,
                    "error": "유효하지 않은 URL 형식",
                    "url": url
                }
            
            # 콘텐츠 타입 자동 감지
            if content_type == "auto":
                content_type = self._detect_content_type(url)
            
            # 타입별 처리
            if content_type == "video":
                return await self.process_video_url(url, extract_video)
            elif content_type == "feed":
                return await self.process_rss_feed(url)
            else:
                return await self.process_web_page(url)
                
        except Exception as e:
            logging.error(f"URL 처리 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    def _detect_content_type(self, url: str) -> str:
        """URL에서 콘텐츠 타입 자동 감지"""
        url_lower = url.lower()
        
        # 비디오 플랫폼
        if 'youtube.com' in url_lower or 'youtu.be' in url_lower:
            return "video"
        elif 'vimeo.com' in url_lower:
            return "video"
        
        # RSS/Atom 피드
        if any(feed_word in url_lower for feed_word in ['rss', 'feed', 'atom', '.xml']):
            return "feed"
        
        # 기본적으로 웹페이지
        return "webpage"
    
    async def process_video_url(self, 
                               url: str, 
                               extract_video: bool = False) -> Dict:
        """
        비디오 URL 처리 (유튜브 등)
        
        Args:
            url: 비디오 URL
            extract_video: 비디오 파일 다운로드 여부
            
        Returns:
            처리 결과 딕셔너리
        """
        if not YTDLP_AVAILABLE:
            return {
                "success": False,
                "error": "비디오 처리를 위해 yt-dlp가 필요합니다. pip install yt-dlp로 설치하세요.",
                "url": url
            }
        
        try:
            print(f"🎥 비디오 처리 시작: {url}")
            
            # yt-dlp 옵션 설정
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': not extract_video,
            }
            
            if extract_video:
                # 음성만 추출하는 경우
                ydl_opts.update({
                    'format': 'bestaudio/best',
                    'outtmpl': str(self.cache_dir / '%(title)s.%(ext)s'),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                    }]
                })
            
            # 비디오 정보 추출
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=extract_video)
            
            # 결과 구성
            result = {
                "success": True,
                "url": url,
                "content_type": "video",
                "video_info": {
                    "title": info.get('title', ''),
                    "description": info.get('description', ''),
                    "duration": info.get('duration', 0),
                    "view_count": info.get('view_count', 0),
                    "upload_date": info.get('upload_date', ''),
                    "uploader": info.get('uploader', ''),
                    "channel": info.get('channel', ''),
                },
                "extracted_video": extract_video
            }
            
            # 다운로드된 파일 정보 추가
            if extract_video and 'requested_downloads' in info:
                result["downloaded_files"] = info['requested_downloads']
            
            # 주얼리 관련성 분석
            text_content = f"{info.get('title', '')} {info.get('description', '')}"
            result["jewelry_analysis"] = self._analyze_jewelry_relevance(text_content)
            
            print(f"✅ 비디오 처리 완료: {info.get('title', 'Unknown')}")
            return result
            
        except Exception as e:
            logging.error(f"비디오 처리 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "content_type": "video"
            }
    
    async def process_web_page(self, url: str) -> Dict:
        """
        웹페이지 크롤링 및 콘텐츠 추출
        
        Args:
            url: 웹페이지 URL
            
        Returns:
            처리 결과 딕셔너리
        """
        if not BS4_AVAILABLE:
            return {
                "success": False,
                "error": "웹페이지 처리를 위해 BeautifulSoup4가 필요합니다. pip install beautifulsoup4로 설치하세요.",
                "url": url
            }
        
        try:
            print(f"📄 웹페이지 처리 시작: {url}")
            
            # 세션이 없으면 생성
            if not self.session:
                self.session = aiohttp.ClientSession(headers=self.headers)
            
            # 웹페이지 다운로드
            async with self.session.get(url, timeout=30) as response:
                if response.status != 200:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {response.reason}",
                        "url": url
                    }
                
                html_content = await response.text()
                charset = response.charset or 'utf-8'
            
            # BeautifulSoup으로 파싱
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 메타데이터 추출
            metadata = self._extract_metadata(soup, url)
            
            # 본문 콘텐츠 추출
            content = self._extract_main_content(soup)
            
            # 링크 추출
            links = self._extract_links(soup, url)
            
            # 이미지 추출
            images = self._extract_images(soup, url)
            
            # 주얼리 관련성 분석
            full_text = f"{metadata.get('title', '')} {metadata.get('description', '')} {content}"
            jewelry_analysis = self._analyze_jewelry_relevance(full_text)
            
            result = {
                "success": True,
                "url": url,
                "content_type": "webpage",
                "metadata": metadata,
                "content": content,
                "links": links[:20],  # 최대 20개 링크
                "images": images[:10],  # 최대 10개 이미지
                "content_length": len(content),
                "jewelry_analysis": jewelry_analysis,
                "charset": charset,
                "processing_time": datetime.now().isoformat()
            }
            
            print(f"✅ 웹페이지 처리 완료: {len(content)}자 추출")
            return result
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "웹페이지 로딩 시간 초과 (30초)",
                "url": url
            }
        except Exception as e:
            logging.error(f"웹페이지 처리 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "content_type": "webpage"
            }
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """웹페이지 메타데이터 추출"""
        metadata = {"url": url}
        
        # 타이틀
        title_tag = soup.find('title')
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()
        
        # 메타 태그들
        meta_tags = {
            "description": ["description", "og:description"],
            "keywords": ["keywords"],
            "author": ["author", "article:author"],
            "published_time": ["article:published_time", "published_time"],
            "image": ["og:image", "twitter:image"],
            "site_name": ["og:site_name"],
            "article_section": ["article:section"]
        }
        
        for key, meta_names in meta_tags.items():
            for meta_name in meta_names:
                meta_tag = soup.find('meta', attrs={
                    lambda attr: attr and attr.lower() in ['name', 'property', 'itemprop'] and 
                    soup.find('meta', {attr: meta_name})
                })
                if not meta_tag:
                    meta_tag = soup.find('meta', {'name': meta_name}) or \
                               soup.find('meta', {'property': meta_name})
                
                if meta_tag and meta_tag.get('content'):
                    metadata[key] = meta_tag['content'].strip()
                    break
        
        # h1 태그 (제목 보완)
        if "title" not in metadata:
            h1_tag = soup.find('h1')
            if h1_tag:
                metadata["title"] = h1_tag.get_text().strip()
        
        return metadata
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """웹페이지 본문 콘텐츠 추출"""
        # 불필요한 태그 제거
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
            tag.decompose()
        
        # 본문 후보 태그들
        main_candidates = [
            soup.find('main'),
            soup.find('article'),
            soup.find('div', class_=re.compile(r'content|main|post|article', re.I)),
            soup.find('div', id=re.compile(r'content|main|post|article', re.I)),
        ]
        
        # 가장 적합한 본문 선택
        main_content = None
        for candidate in main_candidates:
            if candidate:
                main_content = candidate
                break
        
        # 후보가 없으면 body 전체 사용
        if not main_content:
            main_content = soup.find('body') or soup
        
        # 텍스트 추출 및 정리
        text = main_content.get_text(separator=' ', strip=True)
        
        # 공백 정리
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """링크 추출"""
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            if not href or href.startswith('#'):
                continue
            
            # 절대 URL로 변환
            full_url = urljoin(base_url, href)
            
            link_text = a_tag.get_text(strip=True)
            if link_text:
                links.append({
                    "url": full_url,
                    "text": link_text,
                    "title": a_tag.get('title', '')
                })
        
        return links
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """이미지 링크 추출"""
        images = []
        
        for img_tag in soup.find_all('img', src=True):
            src = img_tag['src'].strip()
            if not src:
                continue
            
            # 절대 URL로 변환
            full_url = urljoin(base_url, src)
            
            images.append({
                "url": full_url,
                "alt": img_tag.get('alt', ''),
                "title": img_tag.get('title', ''),
                "width": img_tag.get('width'),
                "height": img_tag.get('height')
            })
        
        return images
    
    async def process_rss_feed(self, feed_url: str) -> Dict:
        """
        RSS/Atom 피드 처리
        
        Args:
            feed_url: RSS 피드 URL
            
        Returns:
            처리 결과 딕셔너리
        """
        if not FEEDPARSER_AVAILABLE:
            return {
                "success": False,
                "error": "RSS 피드 처리를 위해 feedparser가 필요합니다. pip install feedparser로 설치하세요.",
                "url": feed_url
            }
        
        try:
            print(f"📡 RSS 피드 처리 시작: {feed_url}")
            
            # 피드 파싱
            feed = feedparser.parse(feed_url)
            
            if feed.bozo:
                return {
                    "success": False,
                    "error": f"잘못된 피드 형식: {feed.bozo_exception}",
                    "url": feed_url
                }
            
            # 피드 정보
            feed_info = {
                "title": feed.feed.get('title', ''),
                "description": feed.feed.get('description', ''),
                "link": feed.feed.get('link', ''),
                "language": feed.feed.get('language', ''),
                "updated": feed.feed.get('updated', ''),
                "total_entries": len(feed.entries)
            }
            
            # 엔트리 처리
            entries = []
            jewelry_entries = []
            
            for entry in feed.entries[:20]:  # 최대 20개 항목
                entry_data = {
                    "title": entry.get('title', ''),
                    "link": entry.get('link', ''),
                    "description": entry.get('description', ''),
                    "published": entry.get('published', ''),
                    "author": entry.get('author', ''),
                    "tags": [tag.term for tag in entry.get('tags', [])]
                }
                
                entries.append(entry_data)
                
                # 주얼리 관련 엔트리 필터링
                content = f"{entry_data['title']} {entry_data['description']}"
                if self._is_jewelry_related(content):
                    jewelry_entries.append(entry_data)
            
            result = {
                "success": True,
                "url": feed_url,
                "content_type": "rss_feed",
                "feed_info": feed_info,
                "entries": entries,
                "jewelry_entries": jewelry_entries,
                "jewelry_count": len(jewelry_entries),
                "total_count": len(entries)
            }
            
            print(f"✅ RSS 피드 처리 완료: {len(entries)}개 항목, {len(jewelry_entries)}개 주얼리 관련")
            return result
            
        except Exception as e:
            logging.error(f"RSS 피드 처리 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": feed_url,
                "content_type": "rss_feed"
            }
    
    async def crawl_jewelry_news(self, 
                                max_sources: int = 5,
                                max_articles_per_source: int = 10) -> Dict:
        """
        주얼리 관련 뉴스 자동 크롤링
        
        Args:
            max_sources: 최대 소스 수
            max_articles_per_source: 소스당 최대 기사 수
            
        Returns:
            크롤링 결과 딕셔너리
        """
        try:
            print("📰 주얼리 뉴스 크롤링 시작")
            
            all_articles = []
            source_results = []
            
            for source in self.jewelry_news_sources[:max_sources]:
                print(f"🔍 소스 처리 중: {source['name']}")
                
                try:
                    # RSS 피드가 있으면 우선 사용
                    if source.get('rss'):
                        result = await self.process_rss_feed(source['rss'])
                        if result['success']:
                            articles = result.get('jewelry_entries', [])[:max_articles_per_source]
                            all_articles.extend(articles)
                            source_results.append({
                                "source": source['name'],
                                "method": "rss",
                                "articles_found": len(articles),
                                "success": True
                            })
                        else:
                            source_results.append({
                                "source": source['name'],
                                "method": "rss",
                                "success": False,
                                "error": result.get('error', 'Unknown error')
                            })
                    else:
                        # 일반 웹페이지 크롤링
                        result = await self.process_web_page(source['url'])
                        if result['success']:
                            # 간단한 기사 추출 (실제로는 더 정교한 파싱 필요)
                            source_results.append({
                                "source": source['name'],
                                "method": "webpage",
                                "content_length": result.get('content_length', 0),
                                "success": True
                            })
                        else:
                            source_results.append({
                                "source": source['name'],
                                "method": "webpage",
                                "success": False,
                                "error": result.get('error', 'Unknown error')
                            })
                
                except Exception as e:
                    source_results.append({
                        "source": source['name'],
                        "success": False,
                        "error": str(e)
                    })
                
                # 소스 간 딜레이
                await asyncio.sleep(1)
            
            # 중복 제거 및 정렬
            unique_articles = self._deduplicate_articles(all_articles)
            
            result = {
                "success": True,
                "content_type": "jewelry_news_crawl",
                "articles": unique_articles,
                "total_articles": len(unique_articles),
                "sources_processed": len(source_results),
                "source_results": source_results,
                "crawl_time": datetime.now().isoformat()
            }
            
            print(f"✅ 뉴스 크롤링 완료: {len(unique_articles)}개 기사 수집")
            return result
            
        except Exception as e:
            logging.error(f"뉴스 크롤링 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "content_type": "jewelry_news_crawl"
            }
    
    def _analyze_jewelry_relevance(self, text: str) -> Dict:
        """텍스트의 주얼리 관련성 분석"""
        if not text:
            return {"relevance_score": 0.0, "found_keywords": [], "category": "irrelevant"}
        
        text_lower = text.lower()
        found_keywords = []
        
        # 키워드 검색
        for keyword in self.jewelry_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        # 관련성 점수 계산
        relevance_score = len(found_keywords) / len(self.jewelry_keywords) * 100
        
        # 카테고리 분류
        if relevance_score >= 20:
            category = "highly_relevant"
        elif relevance_score >= 10:
            category = "moderately_relevant"
        elif relevance_score >= 5:
            category = "slightly_relevant"
        else:
            category = "irrelevant"
        
        return {
            "relevance_score": round(relevance_score, 2),
            "found_keywords": found_keywords,
            "keyword_count": len(found_keywords),
            "category": category,
            "is_jewelry_related": relevance_score >= 5
        }
    
    def _is_jewelry_related(self, text: str, threshold: float = 5.0) -> bool:
        """텍스트가 주얼리 관련인지 판단"""
        analysis = self._analyze_jewelry_relevance(text)
        return analysis["relevance_score"] >= threshold
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """기사 중복 제거"""
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            title = article.get('title', '').strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        # 발행일 기준 정렬 (최신순)
        try:
            unique_articles.sort(
                key=lambda x: x.get('published', ''), 
                reverse=True
            )
        except:
            pass  # 정렬 실패 시 원본 순서 유지
        
        return unique_articles
    
    async def search_youtube_jewelry(self, 
                                    query: str = "jewelry making tutorial",
                                    max_results: int = 10) -> Dict:
        """
        유튜브에서 주얼리 관련 영상 검색
        
        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            
        Returns:
            검색 결과 딕셔너리
        """
        if not YTDLP_AVAILABLE:
            return {
                "success": False,
                "error": "유튜브 검색을 위해 yt-dlp가 필요합니다.",
                "query": query
            }
        
        try:
            print(f"🔍 유튜브 검색: {query}")
            
            # 검색 URL 구성
            search_url = f"ytsearch{max_results}:{query}"
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_results = ydl.extract_info(search_url, download=False)
            
            videos = []
            for entry in search_results.get('entries', []):
                if entry:
                    video_info = {
                        "title": entry.get('title', ''),
                        "url": entry.get('url', ''),
                        "id": entry.get('id', ''),
                        "duration": entry.get('duration', 0),
                        "view_count": entry.get('view_count', 0),
                        "uploader": entry.get('uploader', ''),
                        "description": entry.get('description', '')[:500]  # 처음 500자만
                    }
                    videos.append(video_info)
            
            result = {
                "success": True,
                "query": query,
                "content_type": "youtube_search",
                "videos": videos,
                "total_found": len(videos),
                "search_time": datetime.now().isoformat()
            }
            
            print(f"✅ 유튜브 검색 완료: {len(videos)}개 영상 발견")
            return result
            
        except Exception as e:
            logging.error(f"유튜브 검색 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "content_type": "youtube_search"
            }

# 전역 인스턴스
_web_crawler_instance = None

def get_web_crawler() -> WebCrawler:
    """전역 웹 크롤러 인스턴스 반환"""
    global _web_crawler_instance
    if _web_crawler_instance is None:
        _web_crawler_instance = WebCrawler()
    return _web_crawler_instance

# 편의 함수들
async def crawl_url(url: str, **kwargs) -> Dict:
    """URL 크롤링 편의 함수"""
    crawler = get_web_crawler()
    async with crawler:
        return await crawler.process_url(url, **kwargs)

async def crawl_jewelry_news(**kwargs) -> Dict:
    """주얼리 뉴스 크롤링 편의 함수"""
    crawler = get_web_crawler()
    async with crawler:
        return await crawler.crawl_jewelry_news(**kwargs)

async def search_youtube_jewelry(**kwargs) -> Dict:
    """유튜브 주얼리 검색 편의 함수"""
    crawler = get_web_crawler()
    async with crawler:
        return await crawler.search_youtube_jewelry(**kwargs)

def check_crawler_support() -> Dict:
    """웹 크롤러 지원 상태 확인"""
    crawler = get_web_crawler()
    return {
        "supported_sources": crawler.get_supported_sources(),
        "libraries": {
            "beautifulsoup4": BS4_AVAILABLE,
            "requests": REQUESTS_AVAILABLE,
            "yt-dlp": YTDLP_AVAILABLE,
            "feedparser": FEEDPARSER_AVAILABLE,
            "aiohttp": True  # 항상 사용 가능 (aiohttp는 기본 설치됨)
        },
        "jewelry_keywords": len(crawler.jewelry_keywords),
        "news_sources": len(crawler.jewelry_news_sources)
    }

if __name__ == "__main__":
    # 테스트 코드
    async def test_crawler():
        print("웹 크롤링 엔진 테스트")
        support_info = check_crawler_support()
        print(f"지원 상태: {support_info}")
        
        # 간단한 테스트
        result = await crawl_url("https://www.jckonline.com")
        print(f"테스트 결과: {result.get('success', False)}")
    
    asyncio.run(test_crawler())
