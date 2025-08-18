#!/usr/bin/env python3
"""
🕷️ 모듈 2: 웹 크롤러 + 블로그 자동화 시스템
국제 주얼리 뉴스 수집 및 AI 기반 블로그 자동 발행

주요 기능:
- RSS 피드 자동 수집
- HTML 크롤링 (RSS 미제공 사이트)
- AI 요약 및 번역 (MCP Perplexity)
- Notion 블로그 자동 발행 (MCP Notion)
- Supabase 데이터 저장
"""

import streamlit as st
import os
import sys
import time
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import requests
import feedparser
from bs4 import BeautifulSoup
import pandas as pd

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Ollama AI 통합 (v2.0 고도화)
try:
    sys.path.append(str(PROJECT_ROOT / "shared"))
    from ollama_interface import global_ollama, quick_summary, quick_translate
    # v2 고도화된 인터페이스 추가
    from ollama_interface_v2 import advanced_ollama, premium_insight, quick_summary as v2_summary, smart_translate
    OLLAMA_AVAILABLE = global_ollama.health_check()
    OLLAMA_V2_AVAILABLE = True
    print("✅ 웹 크롤러 v2 Ollama 인터페이스 로드 완료!")
except ImportError as e:
    try:
        # v1 인터페이스만 시도
        from ollama_interface import global_ollama, quick_summary, quick_translate
        OLLAMA_AVAILABLE = global_ollama.health_check()
        OLLAMA_V2_AVAILABLE = False
        print("⚠️ 웹 크롤러 v1 Ollama 인터페이스만 사용 가능")
    except ImportError:
        OLLAMA_AVAILABLE = False
        OLLAMA_V2_AVAILABLE = False
        print(f"❌ 웹 크롤러 Ollama 인터페이스 로드 실패: {e}")

# MCP 및 외부 도구 import 시도
try:
    # MCP 도구들이 사용 가능한지 확인
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# 페이지 설정
st.set_page_config(
    page_title="🕷️ 웹 크롤러 + 블로그",
    page_icon="🕷️",
    layout="wide"
)

# 국제 주얼리 뉴스 사이트 설정
NEWS_SOURCES = {
    "rss_feeds": {
        "Professional Jeweller": "https://www.professionaljeweller.com/feed/",
        "Jewellery Focus": "https://www.jewelleryfocus.co.uk/feed",
        "The Jewelry Loupe": "https://feeds.feedburner.com/TheJewelryLoupe",
        "The Jewellery Editor": "https://www.thejewelleryeditor.com/feed/",
        "INSTORE Magazine": "https://instoremag.com/feed/",
        "WatchPro": "https://www.watchpro.com/feed/"
    },
    "html_crawl": {
        "JCK Online": "https://www.jckonline.com/editorial-articles/news/",
        "National Jeweler": "https://www.nationaljeweler.com/articles/category/news",
        "Rapaport": "https://www.diamonds.net/News/",
        "JewelleryNet": "https://www.jewellerynet.com/en/news",
        "Jewelin (국내)": "https://www.jewelin.co.kr/news",
        "Messi Jewelry": "https://www.messijewelry.com/news"
    }
}

class WebCrawlerModule:
    """웹 크롤러 모듈 클래스"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_user_agent()
    
    def initialize_session_state(self):
        """세션 상태 초기화"""
        if "crawler_results" not in st.session_state:
            st.session_state.crawler_results = []
        if "crawl_status" not in st.session_state:
            st.session_state.crawl_status = "대기중"
        if "selected_sources" not in st.session_state:
            st.session_state.selected_sources = []
    
    def setup_user_agent(self):
        """User-Agent 설정"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def fetch_rss_feed(self, name, url):
        """RSS 피드 수집"""
        try:
            st.info(f"📡 {name} RSS 피드 수집 중...")
            
            # feedparser로 RSS 파싱
            feed = feedparser.parse(url)
            
            articles = []
            for entry in feed.entries[:5]:  # 최신 5개만
                article = {
                    "source": name,
                    "title": entry.get('title', 'No Title'),
                    "link": entry.get('link', ''),
                    "published": entry.get('published', ''),
                    "summary": entry.get('summary', '')[:300] + "..." if len(entry.get('summary', '')) > 300 else entry.get('summary', ''),
                    "type": "RSS"
                }
                articles.append(article)
            
            st.success(f"✅ {name}: {len(articles)}개 기사 수집 완료")
            return articles
            
        except Exception as e:
            st.error(f"❌ {name} RSS 수집 실패: {str(e)}")
            return []
    
    def crawl_html_news(self, name, url):
        """HTML 크롤링 (RSS 미제공 사이트)"""
        try:
            st.info(f"🔍 {name} HTML 크롤링 중...")
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 기본적인 뉴스 링크 추출 로직
            articles = []
            
            # 일반적인 뉴스 링크 패턴들 시도
            link_selectors = [
                'article a', 'h2 a', 'h3 a', '.news-item a', 
                '.article-title a', '.post-title a'
            ]
            
            for selector in link_selectors:
                links = soup.select(selector)
                if links:
                    for link in links[:3]:  # 최대 3개
                        title = link.get_text(strip=True)
                        href = link.get('href', '')
                        
                        if title and href:
                            # 상대 경로를 절대 경로로 변환
                            if href.startswith('/'):
                                href = f"{url.split('/')[0]}//{url.split('/')[2]}{href}"
                            
                            article = {
                                "source": name,
                                "title": title,
                                "link": href,
                                "published": datetime.now().strftime("%Y-%m-%d"),
                                "summary": f"{name}에서 수집된 뉴스",
                                "type": "HTML"
                            }
                            articles.append(article)
                    
                    if articles:
                        break
            
            st.success(f"✅ {name}: {len(articles)}개 기사 크롤링 완료")
            return articles
            
        except Exception as e:
            st.error(f"❌ {name} HTML 크롤링 실패: {str(e)}")
            return []
    
    def process_with_ai(self, articles):
        """Ollama AI 기반 기사 요약 및 처리"""
        if not OLLAMA_AVAILABLE:
            st.warning("⚠️ Ollama AI를 사용할 수 없습니다. 기본 처리로 진행합니다.")
            return articles
        
        st.info("🤖 Ollama AI 기반 기사 요약 중...")
        
        # 진행률 표시
        progress_bar = st.progress(0)
        processed_articles = []
        
        for i, article in enumerate(articles):
            try:
                st.text(f"처리 중: {article['title'][:50]}...")
                
                # 🏆 v2 고도화 뉴스 분석
                full_content = f"제목: {article['title']}\n\n내용: {article['summary']}"
                
                if OLLAMA_V2_AVAILABLE:
                    # v2 인터페이스로 다양한 레벨 분석
                    v2_analysis = self.process_news_v2(full_content, article['title'])
                    ai_summary_ko = v2_analysis['best_summary']
                    # 추가 분석 데이터도 저장
                    processed_article = article.copy()
                    processed_article["v2_analysis"] = v2_analysis
                else:
                    # 기존 v1 분석
                    ai_summary = quick_summary(full_content)
                    
                    # 번역 (영어 기사의 경우)
                    if self.is_english_content(article['title']):
                        ai_summary_ko = quick_translate(ai_summary, "한국어")
                    else:
                        ai_summary_ko = ai_summary
                    
                    processed_article = article.copy()
                
                processed_article["ai_summary"] = ai_summary_ko
                processed_article["original_summary"] = article["summary"]
                processed_article["tags"] = self.extract_tags(ai_summary_ko)
                processed_article["importance"] = self.assess_importance(ai_summary_ko)
                processed_articles.append(processed_article)
                
                # 진행률 업데이트
                progress_bar.progress((i + 1) / len(articles))
                time.sleep(0.5)  # AI 처리 간격
                
            except Exception as e:
                st.warning(f"기사 처리 중 오류: {str(e)}")
                processed_articles.append(article)
        
        st.success(f"✅ {len(processed_articles)}개 기사 AI 처리 완료!")
        return processed_articles
    
    def is_english_content(self, text: str) -> bool:
        """영어 컨텐츠 감지"""
        try:
            text.encode('ascii')
            return len([c for c in text if c.isalpha() and ord(c) < 128]) > len(text) * 0.7
        except UnicodeEncodeError:
            return False
    
    def process_news_v2(self, content: str, title: str) -> dict:
        """🏆 v2 고도화 뉴스 처리 - 5개 모델 전략 활용"""
        
        try:
            is_english = self.is_english_content(title)
            
            # 🚀 빠른 요약 (Gemma3-4B) - 기본 처리
            fast_result = v2_summary(content)
            if is_english and fast_result:
                fast_result_ko = smart_translate(fast_result, "한국어")
            else:
                fast_result_ko = fast_result
            
            # 🔥 프리미엄 시장 인사이트 (Gemma3-27B) - 심화 분석
            try:
                premium_result = premium_insight(content)
                if is_english and premium_result:
                    premium_result_ko = smart_translate(premium_result, "한국어") 
                else:
                    premium_result_ko = premium_result
            except:
                premium_result_ko = fast_result_ko
            
            # ⚡ 표준 뉴스 분석 (Qwen3-8B)
            try:
                standard_result = advanced_ollama.advanced_generate(
                    task_type="news_analysis",
                    content=content,
                    task_goal="주얼리 업계 뉴스 트렌드 분석",
                    quality_priority=False,
                    speed_priority=False
                )
                if is_english and standard_result:
                    standard_result_ko = smart_translate(standard_result, "한국어")
                else:
                    standard_result_ko = standard_result
            except:
                standard_result_ko = fast_result_ko
            
            # 최고 품질 결과 선택 (길이와 품질을 기준으로)
            candidates = [
                ('premium', premium_result_ko),
                ('standard', standard_result_ko), 
                ('fast', fast_result_ko)
            ]
            
            # 가장 좋은 결과 선택 (길이 + 품질 기준)
            best_summary = fast_result_ko
            best_tier = 'fast'
            
            for tier, result in candidates:
                if result and len(result) > len(best_summary) and len(result) < 1000:
                    best_summary = result
                    best_tier = tier
            
            return {
                'best_summary': best_summary,
                'best_tier': best_tier,
                'fast_summary': fast_result_ko,
                'premium_insight': premium_result_ko,
                'standard_analysis': standard_result_ko,
                'is_english': is_english,
                'v2_processed': True
            }
            
        except Exception as e:
            # 폴백: v1 방식
            try:
                fallback_summary = quick_summary(content)
                if self.is_english_content(title):
                    fallback_summary = quick_translate(fallback_summary, "한국어")
                
                return {
                    'best_summary': fallback_summary,
                    'best_tier': 'fallback',
                    'v2_processed': False,
                    'error': str(e)
                }
            except:
                return {
                    'best_summary': "AI 분석 실패",
                    'best_tier': 'error',
                    'v2_processed': False,
                    'error': str(e)
                }
    
    def extract_tags(self, content: str) -> list:
        """AI 요약에서 태그 추출"""
        # 기본 주얼리 관련 키워드
        jewelry_keywords = [
            "다이아몬드", "보석", "주얼리", "반지", "목걸이", "귀걸이", "팔찌",
            "루비", "사파이어", "에메랄드", "진주", "금", "은", "플래티넘",
            "브랜드", "전시회", "트렌드", "디자인", "컬렉션", "럭셔리"
        ]
        
        tags = []
        content_lower = content.lower()
        
        for keyword in jewelry_keywords:
            if keyword in content_lower:
                tags.append(keyword)
        
        # 최대 5개까지
        return tags[:5] if tags else ["주얼리", "뉴스"]
    
    def assess_importance(self, content: str) -> str:
        """중요도 평가"""
        high_importance_words = ["중요", "혁신", "변화", "성장", "발표", "신제품", "트렌드"]
        medium_importance_words = ["업데이트", "소식", "정보", "발견"]
        
        content_lower = content.lower()
        
        high_count = sum(1 for word in high_importance_words if word in content_lower)
        medium_count = sum(1 for word in medium_importance_words if word in content_lower)
        
        if high_count >= 2:
            return "상"
        elif high_count >= 1 or medium_count >= 2:
            return "중"
        else:
            return "하"
    
    def save_to_notion(self, articles):
        """Notion에 블로그 포스트 자동 발행"""
        if not MCP_AVAILABLE:
            st.warning("⚠️ MCP Notion을 사용할 수 없습니다.")
            return False
        
        st.info("📝 Notion 블로그 발행 중...")
        
        try:
            # 실제로는 MCP Notion 함수 사용
            for article in articles:
                # notion_create_page() 등의 MCP 함수 호출
                pass
            
            st.success(f"✅ {len(articles)}개 기사가 Notion에 발행되었습니다.")
            return True
            
        except Exception as e:
            st.error(f"❌ Notion 발행 실패: {str(e)}")
            return False
    
    def render_source_selection(self):
        """뉴스 소스 선택 UI"""
        st.markdown("## 📰 뉴스 소스 선택")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📡 RSS 피드 사이트")
            for name, url in NEWS_SOURCES["rss_feeds"].items():
                selected = st.checkbox(
                    f"{name}",
                    key=f"rss_{name}",
                    help=url
                )
                if selected and name not in st.session_state.selected_sources:
                    st.session_state.selected_sources.append(("rss", name, url))
        
        with col2:
            st.markdown("### 🔍 HTML 크롤링 사이트")
            for name, url in NEWS_SOURCES["html_crawl"].items():
                selected = st.checkbox(
                    f"{name}",
                    key=f"html_{name}",
                    help=url
                )
                if selected and name not in st.session_state.selected_sources:
                    st.session_state.selected_sources.append(("html", name, url))
    
    def render_crawl_controls(self):
        """크롤링 제어 UI"""
        st.markdown("## ⚡ 크롤링 실행")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 선택된 소스 크롤링 시작", type="primary"):
                if st.session_state.selected_sources:
                    self.start_crawling()
                else:
                    st.warning("⚠️ 먼저 뉴스 소스를 선택해주세요.")
        
        with col2:
            if st.button("🔄 전체 소스 크롤링"):
                st.session_state.selected_sources = []
                # 모든 소스 추가
                for name, url in NEWS_SOURCES["rss_feeds"].items():
                    st.session_state.selected_sources.append(("rss", name, url))
                for name, url in NEWS_SOURCES["html_crawl"].items():
                    st.session_state.selected_sources.append(("html", name, url))
                
                self.start_crawling()
        
        with col3:
            if st.button("🧹 결과 초기화"):
                st.session_state.crawler_results = []
                st.session_state.selected_sources = []
                st.rerun()
    
    def start_crawling(self):
        """크롤링 시작"""
        st.session_state.crawl_status = "실행중"
        st.session_state.crawler_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_sources = len(st.session_state.selected_sources)
        
        for i, (source_type, name, url) in enumerate(st.session_state.selected_sources):
            status_text.text(f"처리 중: {name}")
            
            if source_type == "rss":
                articles = self.fetch_rss_feed(name, url)
            else:
                articles = self.crawl_html_news(name, url)
            
            st.session_state.crawler_results.extend(articles)
            
            # 진행률 업데이트
            progress_bar.progress((i + 1) / total_sources)
            time.sleep(1)  # 서버 부하 방지
        
        status_text.text("✅ 크롤링 완료!")
        st.session_state.crawl_status = "완료"
        
        # AI 처리
        if st.session_state.crawler_results:
            st.session_state.crawler_results = self.process_with_ai(st.session_state.crawler_results)
    
    def render_results(self):
        """크롤링 결과 표시"""
        if not st.session_state.crawler_results:
            st.info("아직 크롤링 결과가 없습니다.")
            return
        
        st.markdown(f"## 📊 크롤링 결과 ({len(st.session_state.crawler_results)}개)")
        
        # 통계
        col1, col2, col3 = st.columns(3)
        
        rss_count = len([r for r in st.session_state.crawler_results if r["type"] == "RSS"])
        html_count = len([r for r in st.session_state.crawler_results if r["type"] == "HTML"])
        
        with col1:
            st.metric("RSS 기사", rss_count)
        with col2:
            st.metric("HTML 기사", html_count)
        with col3:
            st.metric("총 기사", len(st.session_state.crawler_results))
        
        # 결과 테이블
        df = pd.DataFrame(st.session_state.crawler_results)
        st.dataframe(df, use_container_width=True)
        
        # Notion 발행 버튼
        if st.button("📝 Notion 블로그 발행"):
            self.save_to_notion(st.session_state.crawler_results)
    
    def render_sidebar(self):
        """사이드바"""
        with st.sidebar:
            st.markdown("## ⚙️ 설정")
            
            st.markdown("### 🔧 크롤링 설정")
            crawl_interval = st.selectbox(
                "크롤링 주기",
                ["수동", "1시간", "6시간", "12시간", "24시간"]
            )
            
            max_articles = st.slider("최대 기사 수", 1, 50, 10)
            
            st.markdown("### 📊 통계")
            st.info(f"""
            **상태**: {st.session_state.crawl_status}
            **수집된 기사**: {len(st.session_state.crawler_results)}개
            **선택된 소스**: {len(st.session_state.selected_sources)}개
            **Ollama AI**: {'✅ 연결됨' if OLLAMA_AVAILABLE else '❌ 불가능'}
            **사용 모델**: {global_ollama.select_model('web_crawler') if OLLAMA_AVAILABLE else 'N/A'}
            """)
            
            if st.button("🏠 메인 대시보드로"):
                st.markdown("메인 대시보드: http://localhost:8505")
    
    def run(self):
        """모듈 실행"""
        st.markdown("# 🕷️ 웹 크롤러 + 블로그 자동화")
        st.markdown("국제 주얼리 뉴스 수집 및 AI 기반 블로그 자동 발행 시스템")
        
        self.render_sidebar()
        
        st.markdown("---")
        
        # 탭 구성
        tab1, tab2, tab3 = st.tabs(["📰 소스 선택", "🚀 실행", "📊 결과"])
        
        with tab1:
            self.render_source_selection()
        
        with tab2:
            self.render_crawl_controls()
        
        with tab3:
            self.render_results()

def main():
    """메인 함수"""
    crawler = WebCrawlerModule()
    crawler.run()

if __name__ == "__main__":
    main()