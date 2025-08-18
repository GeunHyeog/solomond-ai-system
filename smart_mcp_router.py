#!/usr/bin/env python3
"""
스마트 MCP 라우터
사용자 요청을 분석해서 자동으로 적절한 도구(MCP vs 통합툴킷)를 선택하여 실행
"""

import re
import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from integrated_development_toolkit import IntegratedDevelopmentToolkit

class SmartMCPRouter:
    """스마트 MCP 라우터 - 요청에 따라 자동으로 적절한 도구 선택"""
    
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.toolkit = IntegratedDevelopmentToolkit()
        
        # 키워드 매핑 규칙
        self.routing_rules = {
            # GitHub 관련
            'github': {
                'keywords': ['github', 'repository', 'repo', '저장소', 'issue', '이슈', 'pull request', 'pr', 'commit', '커밋'],
                'tool': 'toolkit',
                'methods': ['get_repo_info', 'list_issues', 'create_issue', 'github_api_request']
            },
            
            # 브라우저 자동화
            'browser': {
                'keywords': ['browser', '브라우저', 'screenshot', '스크린샷', 'webpage', '웹페이지', 'navigate', '탐색', 'playwright'],
                'tool': 'toolkit',
                'methods': ['launch_browser_session', 'capture_page_content']
            },
            
            # 웹 검색
            'web_search': {
                'keywords': ['search', '검색', 'web search', '웹검색', 'duckduckgo', 'google', 'find online', '온라인 찾기'],
                'tool': 'auto',  # Perplexity MCP vs DuckDuckGo 툴킷 자동 선택
                'methods': ['web_search', 'fetch_webpage_content']
            },
            
            # 파일 시스템
            'filesystem': {
                'keywords': ['file', '파일', 'directory', '디렉토리', 'folder', '폴더', 'read', '읽기', 'write', '쓰기', 'create', '생성'],
                'tool': 'mcp',  # MCP filesystem 우선
                'methods': ['read_file', 'write_file', 'list_directory', 'search_files']
            },
            
            # 메모리/지식
            'memory': {
                'keywords': ['remember', '기억', 'memory', '메모리', 'knowledge', '지식', 'store', '저장', 'recall', '회상'],
                'tool': 'mcp',  # MCP memory 사용
                'methods': ['create_entities', 'search_nodes', 'add_observations']
            },
            
            # 데이터베이스
            'database': {
                'keywords': ['database', '데이터베이스', 'supabase', 'db', 'table', '테이블', 'query', '쿼리', 'data', '데이터'],
                'tool': 'toolkit',
                'methods': ['supabase_query', 'save_development_log']
            },
            
            # 복잡한 사고
            'thinking': {
                'keywords': ['analyze', '분석', 'think', '생각', 'complex', '복잡', 'step by step', '단계별', 'problem solving', '문제해결'],
                'tool': 'mcp',  # Sequential thinking MCP
                'methods': ['sequential_thinking']
            }
        }
        
        print(f"[ROUTER] 스마트 MCP 라우터 초기화 완료 - Session: {self.session_id}")
    
    def analyze_request(self, user_request: str) -> Dict[str, Any]:
        """사용자 요청 분석하여 적절한 도구와 메서드 결정"""
        
        user_request_lower = user_request.lower()
        analysis = {
            'original_request': user_request,
            'detected_categories': [],
            'recommended_tool': 'mcp',  # 기본값
            'recommended_methods': [],
            'confidence': 0.0
        }
        
        # 각 카테고리별 키워드 매칭
        category_scores = {}
        
        for category, config in self.routing_rules.items():
            score = 0
            matched_keywords = []
            
            for keyword in config['keywords']:
                if keyword in user_request_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                category_scores[category] = {
                    'score': score,
                    'matched_keywords': matched_keywords,
                    'tool': config['tool'],
                    'methods': config['methods']
                }
        
        # 가장 높은 점수의 카테고리 선택
        if category_scores:
            best_category = max(category_scores.keys(), key=lambda k: category_scores[k]['score'])
            best_config = category_scores[best_category]
            
            analysis['detected_categories'] = list(category_scores.keys())
            analysis['recommended_tool'] = best_config['tool']
            analysis['recommended_methods'] = best_config['methods']
            analysis['confidence'] = best_config['score'] / len(self.routing_rules[best_category]['keywords'])
            analysis['best_category'] = best_category
            analysis['matched_keywords'] = best_config['matched_keywords']
        
        return analysis
    
    async def execute_request(self, user_request: str) -> Dict[str, Any]:
        """사용자 요청을 분석하고 자동으로 적절한 도구로 실행"""
        
        print(f"[ROUTER] 요청 분석 시작: {user_request}")
        
        # 1. 요청 분석
        analysis = self.analyze_request(user_request)
        
        print(f"[ROUTER] 분석 결과: {analysis['best_category']} -> {analysis['recommended_tool']}")
        print(f"[ROUTER] 매칭된 키워드: {analysis['matched_keywords']}")
        
        result = {
            'request': user_request,
            'analysis': analysis,
            'execution_result': None,
            'tool_used': None,
            'success': False,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # 2. 도구 선택 및 실행
            if analysis['recommended_tool'] == 'toolkit':
                result['execution_result'] = await self._execute_toolkit(analysis, user_request)
                result['tool_used'] = 'integrated_toolkit'
                
            elif analysis['recommended_tool'] == 'mcp':
                result['execution_result'] = await self._execute_mcp(analysis, user_request)
                result['tool_used'] = 'mcp_server'
                
            elif analysis['recommended_tool'] == 'auto':
                # Perplexity vs DuckDuckGo 자동 선택
                result['execution_result'] = await self._execute_auto_search(user_request)
                result['tool_used'] = 'auto_selected'
            
            result['success'] = True
            
        except Exception as e:
            print(f"[ERROR] 실행 오류: {e}")
            result['execution_result'] = f"실행 오류: {str(e)}"
            result['success'] = False
        
        return result
    
    async def _execute_toolkit(self, analysis: Dict, request: str) -> Any:
        """통합 툴킷 실행"""
        
        category = analysis.get('best_category', '')
        
        if category == 'github':
            return await self._handle_github_request(request)
        elif category == 'browser':
            return await self._handle_browser_request(request)
        elif category == 'web_search':
            return self._handle_web_search_request(request)
        elif category == 'database':
            return self._handle_database_request(request)
        else:
            return "툴킷 실행: 구체적인 구현 필요"
    
    async def _execute_mcp(self, analysis: Dict, request: str) -> Any:
        """MCP 서버 실행"""
        
        category = analysis.get('best_category', '')
        
        if category == 'filesystem':
            return "MCP Filesystem 실행 (mcp__filesystem__ 함수 사용)"
        elif category == 'memory':
            return "MCP Memory 실행 (mcp__memory__ 함수 사용)"
        elif category == 'thinking':
            return "MCP Sequential Thinking 실행 (mcp__sequential-thinking__ 함수 사용)"
        else:
            return "MCP 실행: 구체적인 구현 필요"
    
    async def _execute_auto_search(self, request: str) -> Any:
        """자동 검색 (Perplexity vs DuckDuckGo)"""
        
        # 간단한 검색은 DuckDuckGo, 복잡한 분석은 Perplexity
        if any(word in request.lower() for word in ['analyze', '분석', 'explain', '설명', 'compare', '비교']):
            return "Perplexity MCP 사용 권장 (mcp__perplexity__chat_completion)"
        else:
            return self.toolkit.web_search(request)
    
    async def _handle_github_request(self, request: str) -> Any:
        """GitHub 요청 처리"""
        
        request_lower = request.lower()
        
        # 저장소 정보 요청
        if any(word in request_lower for word in ['repo info', '저장소 정보', 'repository']):
            return self.toolkit.get_repo_info('GeunHyeog', 'solomond-ai-system')
        
        # 이슈 목록 요청
        elif any(word in request_lower for word in ['issues', '이슈', 'list issues']):
            return self.toolkit.list_issues('GeunHyeog', 'solomond-ai-system')
        
        else:
            return "GitHub 관련 요청을 구체적으로 지정해주세요"
    
    async def _handle_browser_request(self, request: str) -> Any:
        """브라우저 요청 처리"""
        
        # URL 추출
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, request)
        
        if urls:
            url = urls[0]
        else:
            url = 'https://www.google.com'  # 기본값
        
        session = await self.toolkit.launch_browser_session(url, headless=True)
        if session:
            result = {
                'url': url,
                'title': session['info']['title'],
                'screenshot': session['info']['screenshot']
            }
            await session['browser'].close()
            return result
        else:
            return f"브라우저 세션 실패: {url}"
    
    def _handle_web_search_request(self, request: str) -> Any:
        """웹 검색 요청 처리"""
        
        # 검색어 추출 (간단한 방법)
        search_terms = request.replace('검색', '').replace('search', '').replace('찾아', '').strip()
        
        return self.toolkit.web_search(search_terms)
    
    def _handle_database_request(self, request: str) -> Any:
        """데이터베이스 요청 처리"""
        
        # 로그 저장 예시
        return self.toolkit.save_development_log(
            action="user_request",
            details={"request": request, "timestamp": datetime.now().isoformat()}
        )

# 전역 라우터 인스턴스
smart_router = SmartMCPRouter()

async def auto_execute(user_request: str) -> Dict[str, Any]:
    """사용자 요청을 자동으로 분석하고 실행하는 메인 함수"""
    return await smart_router.execute_request(user_request)

# 동기 버전 (Claude Code에서 쉽게 사용)
def execute(user_request: str) -> Dict[str, Any]:
    """동기 버전 - Claude Code에서 바로 사용 가능"""
    return asyncio.run(auto_execute(user_request))

# 사용 예시
if __name__ == "__main__":
    
    # 테스트 요청들
    test_requests = [
        "GitHub 저장소 정보를 알려줘",
        "Python 개발 팁을 검색해줘", 
        "https://www.google.com 브라우저로 열어줘",
        "이 내용을 기억해줘: Claude Code는 좋은 도구다",
        "현재 디렉토리의 파일 목록을 보여줘"
    ]
    
    async def test_router():
        for request in test_requests:
            print(f"\n{'='*60}")
            print(f"테스트 요청: {request}")
            print(f"{'='*60}")
            
            result = await auto_execute(request)
            
            print(f"사용된 도구: {result['tool_used']}")
            print(f"실행 결과: {result['execution_result']}")
            print(f"성공 여부: {result['success']}")
    
    asyncio.run(test_router())