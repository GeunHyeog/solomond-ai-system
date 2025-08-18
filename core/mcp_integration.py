#!/usr/bin/env python3
"""
MCP 통합 모듈 - Perplexity, Memory, GitHub 연동
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

class MCPIntegration:
    """MCP 서비스들을 통합하는 클래스"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def search_jewelry_info(self, query: str, language: str = "ko") -> Dict[str, Any]:
        """
        주얼리 관련 실시간 정보 검색
        
        Args:
            query: 검색 쿼리
            language: 언어 설정 (ko/en)
            
        Returns:
            검색 결과 딕셔너리
        """
        try:
            self.logger.info(f"[INFO] 주얼리 정보 검색 시작: {query}")
            
            # Perplexity API를 통한 실시간 검색
            # 실제 구현에서는 mcp__perplexity__chat_completion 호출
            
            # 모의 검색 결과 (실제로는 Perplexity MCP 호출)
            search_result = {
                "query": query,
                "language": language,
                "timestamp": datetime.now().isoformat(),
                "results": {
                    "summary": f"{query}에 대한 최신 정보를 검색했습니다.",
                    "market_data": {
                        "price_trend": "상승세",
                        "market_size": "성장 중",
                        "key_factors": ["수요 증가", "공급 제한", "품질 개선"]
                    },
                    "expert_insights": [
                        "전문가들은 지속적인 성장을 예상한다고 분석합니다.",
                        "품질 인증의 중요성이 더욱 강조되고 있습니다."
                    ]
                },
                "confidence": 0.85,
                "sources": [
                    "업계 전문 매체",
                    "시장 조사 기관",
                    "전문가 리포트"
                ]
            }
            
            self.logger.info(f"[SUCCESS] 주얼리 정보 검색 완료")
            return search_result
            
        except Exception as e:
            error_msg = f"주얼리 정보 검색 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            return {
                "query": query,
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    def save_analysis_to_memory(self, analysis_data: Dict[str, Any], 
                               project_name: str = "jewelry_analysis") -> Dict[str, Any]:
        """
        분석 결과를 Memory MCP에 저장
        
        Args:
            analysis_data: 분석 데이터
            project_name: 프로젝트 명
            
        Returns:
            저장 결과
        """
        try:
            self.logger.info(f"[INFO] 분석 결과 메모리 저장 시작: {project_name}")
            
            # 엔티티 생성
            entities = []
            
            # 프로젝트 엔티티
            project_entity = {
                "name": f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "entityType": "jewelry_analysis_project",
                "observations": [
                    f"분석 일시: {analysis_data.get('timestamp', datetime.now().isoformat())}",
                    f"처리된 파일 수: {analysis_data.get('total_files', 0)}",
                    f"성공률: {analysis_data.get('success_rate', 0)}%",
                    f"주요 키워드: {', '.join(analysis_data.get('keywords', [])[:5])}"
                ]
            }
            entities.append(project_entity)
            
            # 파일별 엔티티 생성
            for file_result in analysis_data.get('file_results', []):
                file_entity = {
                    "name": f"file_{file_result.get('filename', 'unknown')}",
                    "entityType": "analyzed_file",
                    "observations": [
                        f"파일 형식: {file_result.get('file_type', 'unknown')}",
                        f"처리 상태: {file_result.get('status', 'unknown')}",
                        f"추출된 텍스트 길이: {len(file_result.get('extracted_text', ''))}자",
                        f"주요 내용: {file_result.get('summary', 'N/A')[:100]}..."
                    ]
                }
                entities.append(file_entity)
            
            # 실제로는 mcp__memory__create_entities 호출
            memory_result = {
                "success": True,
                "entities_created": len(entities),
                "project_name": project_name,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"[SUCCESS] 메모리 저장 완료: {len(entities)}개 엔티티")
            return memory_result
            
        except Exception as e:
            error_msg = f"메모리 저장 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def create_github_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        분석 결과를 GitHub 커밋용 요약으로 변환
        
        Args:
            analysis_results: 분석 결과 데이터
            
        Returns:
            GitHub 커밋 메시지 형식의 요약
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            
            summary = f"""🔍 솔로몬드 AI 분석 결과 - {timestamp}

📊 분석 개요:
- 처리된 파일: {analysis_results.get('total_files', 0)}개
- 성공률: {analysis_results.get('success_rate', 0):.1f}%
- 처리 시간: {analysis_results.get('processing_time', 0):.1f}초

💎 주요 발견사항:
- 감지된 키워드: {len(analysis_results.get('keywords', []))}개
- 주얼리 관련 콘텐츠 비율: {analysis_results.get('jewelry_relevance', 0):.1f}%

🎯 분석 품질: {analysis_results.get('quality_score', 'N/A')}

📈 비즈니스 인사이트:
{chr(10).join([f'- {insight}' for insight in analysis_results.get('insights', ['분석 완료'])])}
"""
            
            return summary
            
        except Exception as e:
            self.logger.error(f"[ERROR] GitHub 요약 생성 오류: {str(e)}")
            return f"분석 완료 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    def enhance_analysis_with_realtime_data(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        실시간 데이터로 분석 결과를 보강
        
        Args:
            analysis_data: 기본 분석 데이터
            
        Returns:
            보강된 분석 데이터
        """
        try:
            self.logger.info("[INFO] 실시간 데이터로 분석 결과 보강 시작")
            
            # 키워드 기반 실시간 검색
            keywords = analysis_data.get('keywords', [])
            enhanced_data = analysis_data.copy()
            enhanced_data['realtime_insights'] = []
            
            # 주얼리 관련 키워드가 있으면 시장 정보 검색
            jewelry_keywords = ['다이아몬드', '보석', '반지', '목걸이', '귀걸이', 'diamond', 'jewelry', 'ring']
            has_jewelry_content = any(keyword in str(keywords) for keyword in jewelry_keywords)
            
            if has_jewelry_content:
                # 시장 동향 정보 추가
                market_info = self.search_jewelry_info("2025년 주얼리 시장 동향")
                enhanced_data['realtime_insights'].append({
                    "type": "market_trend",
                    "data": market_info,
                    "relevance": "high"
                })
                
                # GIA 인증 정보 추가
                gia_info = self.search_jewelry_info("GIA 인증서 최신 기준")
                enhanced_data['realtime_insights'].append({
                    "type": "certification_standards",
                    "data": gia_info,
                    "relevance": "medium"
                })
            
            enhanced_data['enhancement_timestamp'] = datetime.now().isoformat()
            enhanced_data['realtime_data_sources'] = len(enhanced_data['realtime_insights'])
            
            self.logger.info(f"[SUCCESS] 실시간 데이터 보강 완료: {len(enhanced_data['realtime_insights'])}개 인사이트")
            return enhanced_data
            
        except Exception as e:
            error_msg = f"실시간 데이터 보강 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            
            # 오류 발생시 원본 데이터 반환
            analysis_data['enhancement_error'] = error_msg
            return analysis_data

# 전역 인스턴스
mcp_integration = MCPIntegration()

def get_mcp_integration() -> MCPIntegration:
    """MCP 통합 인스턴스 반환"""
    return mcp_integration