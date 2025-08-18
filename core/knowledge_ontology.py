#!/usr/bin/env python3
"""
🧠 SOLOMOND AI 온톨로지 지식 베이스 시스템
Knowledge Ontology System for Domain-Specific Intelligence

🎯 주요 기능:
1. 계층적 지식 구조 - 도메인별 전문 지식 분류
2. 시맨틱 관계 매핑 - 개념 간 의미적 연관성
3. 인사이트 추론 엔진 - 지식 기반 자동 추론
4. 동적 학습 시스템 - 새로운 지식 자동 통합
5. 컨텍스트 인식 검색 - 상황 맞춤 지식 제공
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
import re

# 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeNode:
    """지식 노드 구조"""
    id: str
    concept: str
    domain: str
    category: str
    description: str
    confidence: float
    evidence: List[str]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str

@dataclass
class SemanticRelation:
    """시맨틱 관계 구조"""
    source_id: str
    target_id: str
    relation_type: str
    strength: float
    context: str
    evidence: List[str]
    created_at: str

class KnowledgeOntology:
    """온톨로지 지식 베이스 핵심 엔진"""
    
    def __init__(self, domain: str = "conference_analysis"):
        self.domain = domain
        self.knowledge_graph = nx.DiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.relations: List[SemanticRelation] = []
        self.domain_schemas = self._initialize_domain_schemas()
        
        # 추론 규칙
        self.inference_rules = self._initialize_inference_rules()
        
        # 성능 메트릭
        self.stats = {
            'total_nodes': 0,
            'total_relations': 0,
            'domains_covered': 0,
            'inference_accuracy': 0.0,
            'last_updated': None
        }
        
        logger.info(f"🧠 온톨로지 시스템 초기화 - 도메인: {domain}")
        
    def _initialize_domain_schemas(self) -> Dict[str, Dict]:
        """도메인별 스키마 초기화"""
        schemas = {
            "conference_analysis": {
                "categories": [
                    "presentation_content",
                    "speaker_analysis", 
                    "audience_reaction",
                    "technical_concepts",
                    "business_insights",
                    "industry_trends"
                ],
                "relation_types": [
                    "causes", "influences", "relates_to", "precedes", 
                    "contradicts", "supports", "exemplifies", "generalizes"
                ],
                "quality_factors": [
                    "audio_clarity", "visual_quality", "content_relevance",
                    "speaker_expertise", "audience_engagement"
                ]
            },
            "jewelry_analysis": {
                "categories": [
                    "gemstone_properties",
                    "design_elements",
                    "market_trends", 
                    "manufacturing_process",
                    "cultural_significance",
                    "pricing_factors"
                ],
                "relation_types": [
                    "material_of", "style_influences", "market_affects",
                    "culturally_related", "price_correlates"
                ]
            },
            "business_intelligence": {
                "categories": [
                    "market_analysis",
                    "customer_behavior",
                    "competitive_landscape",
                    "financial_metrics",
                    "strategic_initiatives"
                ],
                "relation_types": [
                    "drives", "impacts", "competes_with", "depends_on",
                    "enables", "disrupts"
                ]
            }
        }
        
        return schemas
    
    def _initialize_inference_rules(self) -> List[Dict]:
        """추론 규칙 초기화"""
        rules = [
            {
                "name": "transitive_relation",
                "pattern": "A -> B, B -> C => A -> C",
                "relation_types": ["causes", "influences", "precedes"],
                "confidence_decay": 0.8
            },
            {
                "name": "contradiction_detection",
                "pattern": "A contradicts B, B supports C => A likely contradicts C",
                "confidence_threshold": 0.7
            },
            {
                "name": "evidence_aggregation",
                "pattern": "Multiple evidence points => Higher confidence",
                "weight_function": "logarithmic"
            },
            {
                "name": "temporal_decay",
                "pattern": "Older evidence => Lower weight",
                "decay_rate": 0.95  # per day
            },
            {
                "name": "domain_expertise",
                "pattern": "Expert source => Higher confidence",
                "expertise_multiplier": 1.5
            }
        ]
        
        return rules
    
    def add_knowledge_from_analysis(self, analysis_results: Dict[str, Any]) -> None:
        """분석 결과에서 지식 추출 및 추가"""
        logger.info("🔍 분석 결과에서 지식 추출 중...")
        
        # 멀티모달 결과에서 지식 추출
        if "multimodal_results" in analysis_results:
            self._extract_from_multimodal_results(analysis_results["multimodal_results"])
        
        # 크로스 모달 인사이트에서 지식 추출
        if "cross_modal_insights" in analysis_results:
            self._extract_from_cross_modal_insights(analysis_results["cross_modal_insights"])
        
        # 기존 분석 결과에서 지식 추출 (필요시)
        if "traditional_results" in analysis_results:
            self._extract_from_traditional_analysis(analysis_results["traditional_results"])
        
        # 지식 그래프 업데이트
        self._update_knowledge_graph()
        
        self.stats['last_updated'] = datetime.now().isoformat()
        logger.info(f"✅ 지식 추출 완료 - {len(self.nodes)}개 노드, {len(self.relations)}개 관계")
    
    def _extract_from_multimodal_results(self, multimodal_results: List[Dict]) -> None:
        """멀티모달 결과에서 지식 추출"""
        for result in multimodal_results:
            # 파일 타입별 지식 추출
            if result.get("file_type") == "audio":
                self._extract_audio_knowledge(result)
            elif result.get("file_type") == "image": 
                self._extract_image_knowledge(result)
            elif result.get("file_type") == "text":
                self._extract_text_knowledge(result)
    
    def _extract_audio_knowledge(self, audio_result: Dict) -> None:
        """오디오 분석 결과에서 지식 추출"""
        content = audio_result.get("content", "")
        confidence = audio_result.get("confidence", 0.0)
        
        if not content or confidence < 0.5:
            return
        
        # 핵심 개념 추출
        concepts = self._extract_concepts_from_text(content)
        
        for concept in concepts:
            node_id = f"audio_concept_{concept}_{datetime.now().timestamp()}"
            
            knowledge_node = KnowledgeNode(
                id=node_id,
                concept=concept,
                domain=self.domain,
                category="presentation_content",
                description=f"오디오에서 추출된 개념: {concept}",
                confidence=confidence * 0.9,  # 오디오는 약간 낮은 신뢰도
                evidence=[content[:200]],
                metadata={
                    "source_type": "audio",
                    "file_path": audio_result.get("file_path"),
                    "extraction_method": "speech_to_text"
                },
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            
            self.nodes[node_id] = knowledge_node
    
    def _extract_image_knowledge(self, image_result: Dict) -> None:
        """이미지 분석 결과에서 지식 추출"""
        content = image_result.get("content", "")
        confidence = image_result.get("confidence", 0.0)
        
        if not content or confidence < 0.4:
            return
        
        # 시각적 요소 추출
        visual_elements = self._extract_visual_concepts(content)
        
        for element in visual_elements:
            node_id = f"visual_element_{element}_{datetime.now().timestamp()}"
            
            knowledge_node = KnowledgeNode(
                id=node_id,
                concept=element,
                domain=self.domain,
                category="technical_concepts",
                description=f"이미지에서 추출된 시각적 요소: {element}",
                confidence=confidence * 1.1,  # 시각적 정보는 높은 신뢰도
                evidence=[content],
                metadata={
                    "source_type": "image",
                    "file_path": image_result.get("file_path"),
                    "extraction_method": "optical_character_recognition"
                },
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            
            self.nodes[node_id] = knowledge_node
    
    def _extract_text_knowledge(self, text_result: Dict) -> None:
        """텍스트 분석 결과에서 지식 추출"""
        content = text_result.get("content", "")
        
        if not content:
            return
        
        # 구조화된 지식 추출
        structured_knowledge = self._extract_structured_knowledge(content)
        
        for knowledge in structured_knowledge:
            node_id = f"text_knowledge_{knowledge['concept']}_{datetime.now().timestamp()}"
            
            knowledge_node = KnowledgeNode(
                id=node_id,
                concept=knowledge['concept'],
                domain=self.domain,
                category=knowledge.get('category', 'business_insights'),
                description=knowledge['description'],
                confidence=1.0,  # 텍스트는 최고 신뢰도
                evidence=[content[:300]],
                metadata={
                    "source_type": "text",
                    "file_path": text_result.get("file_path"),
                    "extraction_method": "natural_language_processing"
                },
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            
            self.nodes[node_id] = knowledge_node
    
    def _extract_from_cross_modal_insights(self, insights: Dict) -> None:
        """크로스 모달 인사이트에서 관계성 지식 추출"""
        # 고상관 쌍에서 관계 추출
        for pair in insights.get("high_correlation_pairs", []):
            relation = SemanticRelation(
                source_id=f"file_{pair['primary_file']}",
                target_id=f"files_{len(pair['related_files'])}",
                relation_type="strongly_correlates",
                strength=pair['correlation_score'],
                context="cross_modal_analysis",
                evidence=[f"상관계수: {pair['correlation_score']:.3f}"],
                created_at=datetime.now().isoformat()
            )
            
            self.relations.append(relation)
        
        # 주요 테마에서 도메인 지식 추출
        for theme in insights.get("dominant_themes", []):
            node_id = f"theme_{theme}_{datetime.now().timestamp()}"
            
            knowledge_node = KnowledgeNode(
                id=node_id,
                concept=theme,
                domain=self.domain,
                category="industry_trends",
                description=f"크로스 모달 분석에서 식별된 주요 테마: {theme}",
                confidence=0.8,
                evidence=["멀티모달 분석 결과"],
                metadata={
                    "source_type": "cross_modal_analysis",
                    "extraction_method": "theme_analysis"
                },
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            
            self.nodes[node_id] = knowledge_node
    
    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """텍스트에서 핵심 개념 추출"""
        # 간단한 키워드 추출 (실제로는 더 정교한 NER/NLP 사용)
        import re
        
        # 한국어 명사구 추출
        korean_concepts = re.findall(r'[가-힣]{3,8}', text)
        
        # 영어 기술 용어 추출
        english_concepts = re.findall(r'\b[A-Z][a-zA-Z]{2,15}\b', text)
        
        # 숫자 + 단위 패턴
        numerical_concepts = re.findall(r'\d+\s*[가-힣]{1,3}|\d+%|\d+\$', text)
        
        all_concepts = korean_concepts + english_concepts + numerical_concepts
        
        # 빈도 기반 필터링
        from collections import Counter
        concept_freq = Counter(all_concepts)
        
        # 최소 2회 이상 언급된 개념만 선택
        filtered_concepts = [concept for concept, freq in concept_freq.items() if freq >= 2]
        
        return filtered_concepts[:20]  # 상위 20개만
    
    def _extract_visual_concepts(self, ocr_text: str) -> List[str]:
        """OCR 텍스트에서 시각적 개념 추출"""
        visual_indicators = [
            "차트", "그래프", "표", "도표", "슬라이드", "이미지", 
            "사진", "그림", "도식", "다이어그램", "Chart", "Graph", 
            "Table", "Figure", "Image", "Slide"
        ]
        
        found_elements = []
        for indicator in visual_indicators:
            if indicator in ocr_text:
                found_elements.append(indicator)
        
        # OCR 텍스트의 핵심 키워드도 추출
        concepts = self._extract_concepts_from_text(ocr_text)
        found_elements.extend(concepts[:10])  # 상위 10개 추가
        
        return list(set(found_elements))
    
    def _extract_structured_knowledge(self, text: str) -> List[Dict]:
        """구조화된 지식 추출"""
        knowledge_items = []
        
        # 정의 패턴 인식 ("X는 Y이다", "X란 Y를 말한다")
        definition_patterns = [
            r'([가-힣A-Za-z0-9\s]{2,20})는\s+([가-힣A-Za-z0-9\s]{5,50})이다',
            r'([가-힣A-Za-z0-9\s]{2,20})란\s+([가-힣A-Za-z0-9\s]{5,50})를?\s*말한다',
            r'([가-힣A-Za-z0-9\s]{2,20})\s*:\s*([가-힣A-Za-z0-9\s]{5,50})'
        ]
        
        for pattern in definition_patterns:
            matches = re.findall(pattern, text)
            for concept, description in matches:
                knowledge_items.append({
                    'concept': concept.strip(),
                    'description': description.strip(),
                    'category': 'technical_concepts'
                })
        
        # 인과관계 패턴 ("X 때문에 Y", "X로 인해 Y")
        causal_patterns = [
            r'([가-힣A-Za-z0-9\s]{2,30})\s*때문에\s*([가-힣A-Za-z0-9\s]{2,30})',
            r'([가-힣A-Za-z0-9\s]{2,30})\s*로\s*인해\s*([가-힣A-Za-z0-9\s]{2,30})'
        ]
        
        for pattern in causal_patterns:
            matches = re.findall(pattern, text)
            for cause, effect in matches:
                knowledge_items.append({
                    'concept': f"{cause.strip()} → {effect.strip()}",
                    'description': f"인과관계: {cause.strip()}가 {effect.strip()}를 야기함",
                    'category': 'business_insights'
                })
        
        return knowledge_items[:15]  # 최대 15개
    
    def _update_knowledge_graph(self) -> None:
        """지식 그래프 업데이트"""
        # 노드 추가
        for node_id, node in self.nodes.items():
            self.knowledge_graph.add_node(
                node_id,
                concept=node.concept,
                domain=node.domain,
                category=node.category,
                confidence=node.confidence
            )
        
        # 관계 추가
        for relation in self.relations:
            if (relation.source_id in self.knowledge_graph and 
                relation.target_id in self.knowledge_graph):
                self.knowledge_graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    relation_type=relation.relation_type,
                    strength=relation.strength
                )
        
        # 통계 업데이트
        self.stats.update({
            'total_nodes': len(self.nodes),
            'total_relations': len(self.relations),
            'domains_covered': len(set(node.domain for node in self.nodes.values()))
        })
    
    def infer_insights(self, query_context: str = "") -> List[Dict[str, Any]]:
        """지식 기반 인사이트 추론"""
        logger.info("🔮 지식 기반 인사이트 추론 시작...")
        
        insights = []
        
        # 1. 중심성 분석 - 가장 중요한 개념들
        centrality_insights = self._analyze_concept_centrality()
        insights.extend(centrality_insights)
        
        # 2. 클러스터 분석 - 관련 개념 그룹
        cluster_insights = self._analyze_concept_clusters()
        insights.extend(cluster_insights)
        
        # 3. 트렌드 분석 - 시간적 패턴
        trend_insights = self._analyze_temporal_trends()
        insights.extend(trend_insights)
        
        # 4. 추론 규칙 적용
        rule_based_insights = self._apply_inference_rules()
        insights.extend(rule_based_insights)
        
        # 신뢰도 기준 정렬
        insights.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        logger.info(f"✅ {len(insights)}개 인사이트 추론 완료")
        return insights[:20]  # 상위 20개만 반환
    
    def _analyze_concept_centrality(self) -> List[Dict[str, Any]]:
        """개념 중심성 분석"""
        if len(self.knowledge_graph.nodes) == 0:
            return []
        
        # 다양한 중심성 메트릭 계산
        degree_centrality = nx.degree_centrality(self.knowledge_graph)
        betweenness_centrality = nx.betweenness_centrality(self.knowledge_graph)
        
        insights = []
        
        # 상위 5개 중심 개념
        top_central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for node_id, centrality_score in top_central:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                insights.append({
                    'type': 'central_concept',
                    'concept': node.concept,
                    'description': f"핵심 개념: {node.concept} (중심성: {centrality_score:.3f})",
                    'confidence': min(0.9, centrality_score * 2),
                    'evidence': [f"네트워크 중심성 점수: {centrality_score:.3f}"],
                    'category': 'structural_analysis'
                })
        
        return insights
    
    def _analyze_concept_clusters(self) -> List[Dict[str, Any]]:
        """개념 클러스터 분석"""
        insights = []
        
        if len(self.knowledge_graph.nodes) < 3:
            return insights
        
        # 커뮤니티 탐지
        try:
            communities = nx.community.greedy_modularity_communities(
                self.knowledge_graph.to_undirected()
            )
            
            for i, community in enumerate(communities):
                if len(community) >= 2:  # 최소 2개 노드
                    concepts = []
                    for node_id in community:
                        if node_id in self.nodes:
                            concepts.append(self.nodes[node_id].concept)
                    
                    if concepts:
                        insights.append({
                            'type': 'concept_cluster',
                            'description': f"관련 개념 그룹 {i+1}: {', '.join(concepts[:5])}",
                            'concepts': concepts,
                            'confidence': 0.7,
                            'evidence': [f"네트워크 클러스터링으로 {len(concepts)}개 개념 그룹화"],
                            'category': 'relationship_analysis'
                        })
                        
        except Exception as e:
            logger.warning(f"클러스터 분석 실패: {e}")
        
        return insights
    
    def _analyze_temporal_trends(self) -> List[Dict[str, Any]]:
        """시간적 트렌드 분석"""
        insights = []
        
        # 시간별 지식 생성 패턴
        time_distribution = defaultdict(int)
        
        for node in self.nodes.values():
            try:
                created_date = datetime.fromisoformat(node.created_at).date()
                time_distribution[created_date] += 1
            except:
                continue
        
        if len(time_distribution) > 1:
            dates = sorted(time_distribution.keys())
            recent_growth = time_distribution[dates[-1]] if len(dates) > 1 else 0
            
            insights.append({
                'type': 'temporal_trend',
                'description': f"지식 생성 추세: 최근 {recent_growth}개 새 개념 발견",
                'confidence': 0.6,
                'evidence': [f"총 {len(dates)}일간 {sum(time_distribution.values())}개 지식 생성"],
                'category': 'trend_analysis'
            })
        
        return insights
    
    def _apply_inference_rules(self) -> List[Dict[str, Any]]:
        """추론 규칙 적용"""
        insights = []
        
        # 추이적 관계 추론
        transitive_relations = self._find_transitive_relations()
        for source, target, path_length in transitive_relations[:3]:  # 상위 3개
            if source in self.nodes and target in self.nodes:
                source_concept = self.nodes[source].concept
                target_concept = self.nodes[target].concept
                
                insights.append({
                    'type': 'inferred_relation',
                    'description': f"추론된 관계: {source_concept} → {target_concept}",
                    'confidence': max(0.4, 0.9 ** path_length),  # 경로가 길수록 낮은 신뢰도
                    'evidence': [f"{path_length}단계 추론 경로"],
                    'category': 'logical_inference'
                })
        
        return insights
    
    def _find_transitive_relations(self) -> List[Tuple[str, str, int]]:
        """추이적 관계 탐지"""
        transitive_relations = []
        
        # 모든 노드 쌍에 대해 경로 확인
        nodes = list(self.knowledge_graph.nodes())[:20]  # 성능을 위해 제한
        
        for source in nodes:
            for target in nodes:
                if source != target:
                    try:
                        if nx.has_path(self.knowledge_graph, source, target):
                            shortest_path = nx.shortest_path(self.knowledge_graph, source, target)
                            if len(shortest_path) > 2:  # 직접 연결이 아닌 경우
                                transitive_relations.append((source, target, len(shortest_path) - 1))
                    except:
                        continue
        
        return transitive_relations
    
    def save_knowledge_base(self, file_path: str) -> None:
        """지식 베이스 저장"""
        knowledge_data = {
            'domain': self.domain,
            'nodes': {node_id: asdict(node) for node_id, node in self.nodes.items()},
            'relations': [asdict(relation) for relation in self.relations],
            'stats': self.stats,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 지식 베이스 저장 완료: {file_path}")
    
    def load_knowledge_base(self, file_path: str) -> None:
        """지식 베이스 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
            
            self.domain = knowledge_data['domain']
            
            # 노드 복원
            for node_id, node_data in knowledge_data['nodes'].items():
                self.nodes[node_id] = KnowledgeNode(**node_data)
            
            # 관계 복원
            for relation_data in knowledge_data['relations']:
                self.relations.append(SemanticRelation(**relation_data))
            
            # 통계 복원
            self.stats = knowledge_data['stats']
            
            # 지식 그래프 재구축
            self._update_knowledge_graph()
            
            logger.info(f"📚 지식 베이스 로드 완료: {file_path}")
            
        except Exception as e:
            logger.error(f"❌ 지식 베이스 로드 실패: {e}")
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """지식 베이스 요약 정보"""
        category_distribution = defaultdict(int)
        confidence_distribution = []
        
        for node in self.nodes.values():
            category_distribution[node.category] += 1
            confidence_distribution.append(node.confidence)
        
        return {
            'total_knowledge_nodes': len(self.nodes),
            'total_relations': len(self.relations),
            'domain': self.domain,
            'category_distribution': dict(category_distribution),
            'average_confidence': np.mean(confidence_distribution) if confidence_distribution else 0,
            'knowledge_graph_density': nx.density(self.knowledge_graph) if len(self.knowledge_graph) > 1 else 0,
            'last_updated': self.stats.get('last_updated', 'Never')
        }

# 사용 예제
def main():
    """사용 예제"""
    # 온톨로지 시스템 초기화
    ontology = KnowledgeOntology("conference_analysis")
    
    # 샘플 분석 결과 (실제 사용시에는 분석 시스템에서 제공)
    sample_analysis = {
        "multimodal_results": [
            {
                "file_type": "audio",
                "content": "주얼리 시장의 새로운 트렌드는 지속가능한 재료 사용입니다",
                "confidence": 0.85,
                "file_path": "sample.wav"
            },
            {
                "file_type": "image", 
                "content": "Chart 주얼리 매출 증가 트렌드 2023-2024",
                "confidence": 0.92,
                "file_path": "chart.png"
            }
        ],
        "cross_modal_insights": {
            "high_correlation_pairs": [
                {
                    "primary_file": "audio_1.wav",
                    "correlation_score": 0.87,
                    "related_files": [{"file": "chart_1.png", "similarity": 0.87}]
                }
            ],
            "dominant_themes": ["지속가능성", "트렌드", "매출", "성장"]
        }
    }
    
    # 지식 추출
    ontology.add_knowledge_from_analysis(sample_analysis)
    
    # 인사이트 추론
    insights = ontology.infer_insights()
    
    print("🧠 온톨로지 지식 베이스 분석 결과:")
    print(f"📊 요약: {ontology.get_knowledge_summary()}")
    
    print("\n🔮 추론된 인사이트:")
    for i, insight in enumerate(insights[:5], 1):
        print(f"{i}. {insight['description']} (신뢰도: {insight['confidence']:.2f})")

if __name__ == "__main__":
    main()