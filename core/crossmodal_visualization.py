#!/usr/bin/env python3
"""
📊 크로스모달 시각화 엔진 - SOLOMOND AI 진정한 멀티모달리티 구현
Advanced Cross-Modal Visualization Engine

🎯 주요 기능:
1. 모달간 상관관계 매트릭스 - 히트맵 및 네트워크 그래프
2. 융합 과정 3D 시각화 - 인코딩→융합→디코딩 전 과정
3. Ollama 모델 체인 모니터링 - 실시간 처리 상태 추적
4. 지식 온톨로지 그래프 - 학습된 지식 구조 시각화
5. 인터랙티브 대시보드 - 사용자 탐색 가능한 분석 결과
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
from dataclasses import dataclass
from pathlib import Path
import json
import time
from datetime import datetime

# 로컬 임포트
from .multimodal_encoder import EncodedResult
from .crossmodal_fusion import FusionResult
from .ollama_decoder import DecodingResult

@dataclass
class VisualizationConfig:
    """시각화 설정"""
    color_scheme: str = "viridis"
    network_layout: str = "spring"
    node_size_range: Tuple[int, int] = (20, 100)
    edge_width_range: Tuple[float, float] = (1.0, 5.0)
    animation_duration: int = 1000
    show_labels: bool = True

class CrossModalVisualizer:
    """크로스모달 시각화 엔진"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        
        # 색상 팔레트
        self.modality_colors = {
            'image': '#FF6B6B',    # 빨간색
            'audio': '#4ECDC4',    # 청록색
            'text': '#45B7D1',     # 파란색
            'video': '#96CEB4',    # 초록색
            'unknown': '#CCCCCC'   # 회색
        }
        
        # 처리 단계 색상
        self.stage_colors = {
            'encoding': '#FFD93D',      # 노란색
            'fusion': '#6BCF7F',        # 초록색
            'decoding': '#4D96FF',      # 파란색
            'ontology': '#FF6B9D'       # 분홍색
        }
    
    def create_correlation_matrix(self, fusion_result: FusionResult, 
                                encoded_results: List[EncodedResult]) -> go.Figure:
        """모달간 상관관계 매트릭스 시각화"""
        
        # 파일명과 모달리티 매핑
        file_modal_map = {
            Path(result.file_path).name: result.modality 
            for result in encoded_results
        }
        
        # 상관관계 매트릭스 구성
        correlations = fusion_result.cross_modal_correlations
        files = list(set().union(*[corr['file_pair'] for corr in correlations]))
        
        # 매트릭스 초기화
        matrix = np.zeros((len(files), len(files)))
        file_index = {file: i for i, file in enumerate(files)}
        
        # 상관계수 입력
        for corr in correlations:
            file1, file2 = corr['file_pair']
            if file1 in file_index and file2 in file_index:
                i, j = file_index[file1], file_index[file2]
                matrix[i, j] = corr['correlation']
                matrix[j, i] = corr['correlation']  # 대칭행렬
        
        # 대각선을 1로 설정 (자기 자신과의 상관관계)
        np.fill_diagonal(matrix, 1.0)
        
        # 히트맵 생성
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=files,
            y=files,
            colorscale='RdYlBu_r',
            colorbar=dict(title="상관계수"),
            hoverongaps=False,
            text=matrix,
            texttemplate='%{text:.2f}',
            textfont={'size': 10}
        ))
        
        fig.update_layout(
            title="📊 크로스모달 상관관계 매트릭스",
            xaxis_title="파일",
            yaxis_title="파일",
            width=800,
            height=800
        )
        
        return fig
    
    def create_fusion_network(self, fusion_result: FusionResult,
                            encoded_results: List[EncodedResult]) -> go.Figure:
        """융합 네트워크 그래프"""
        
        # NetworkX 그래프 생성
        G = nx.Graph()
        
        # 노드 추가 (파일들)
        for result in encoded_results:
            filename = Path(result.file_path).stem
            G.add_node(filename, 
                      modality=result.modality,
                      confidence=result.confidence,
                      size=result.confidence * 100)
        
        # 엣지 추가 (상관관계)
        for corr in fusion_result.cross_modal_correlations:
            if corr['correlation'] > 0.3:  # 임계값 이상만
                files = corr['file_pair']
                if len(files) == 2:
                    file1 = Path(files[0]).stem
                    file2 = Path(files[1]).stem
                    if file1 in G.nodes and file2 in G.nodes:
                        G.add_edge(file1, file2, 
                                 weight=corr['correlation'],
                                 width=corr['correlation'] * 5)
        
        # 레이아웃 계산
        if self.config.network_layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif self.config.network_layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G)
        
        # 엣지 그리기
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{edge[0]} ↔ {edge[1]}: {edge[2]['weight']:.2f}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            mode='lines'
        )
        
        # 노드 그리기
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        node_hover = []
        
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node[0])
            
            modality = node[1]['modality']
            node_color.append(self.modality_colors.get(modality, '#CCCCCC'))
            node_size.append(max(30, node[1]['confidence'] * 80))
            
            node_hover.append(
                f"파일: {node[0]}<br>"
                f"모달리티: {modality}<br>"
                f"신뢰도: {node[1]['confidence']:.2f}"
            )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_hover,
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            )
        )
        
        # 피그 생성
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title="🕸️ 크로스모달 융합 네트워크",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="노드 크기 = 신뢰도, 엣지 두께 = 상관관계",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    font=dict(color="black", size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def create_3d_embedding_space(self, encoded_results: List[EncodedResult]) -> go.Figure:
        """3D 임베딩 공간 시각화"""
        
        # PCA로 3차원으로 축소
        try:
            from sklearn.decomposition import PCA
            
            embeddings = np.array([result.encoding for result in encoded_results])
            pca = PCA(n_components=3)
            embeddings_3d = pca.fit_transform(embeddings)
            
            # 데이터 준비
            modalities = [result.modality for result in encoded_results]
            filenames = [Path(result.file_path).name for result in encoded_results]
            confidences = [result.confidence for result in encoded_results]
            
            # 3D 산점도
            fig = go.Figure(data=go.Scatter3d(
                x=embeddings_3d[:, 0],
                y=embeddings_3d[:, 1], 
                z=embeddings_3d[:, 2],
                mode='markers+text',
                marker=dict(
                    size=[conf * 20 + 5 for conf in confidences],
                    color=[self.modality_colors.get(mod, '#CCCCCC') for mod in modalities],
                    opacity=0.8,
                    line=dict(color='white', width=2)
                ),
                text=filenames,
                textposition="top center",
                hovertemplate='<b>%{text}</b><br>' +
                            '모달리티: %{customdata[0]}<br>' +
                            '신뢰도: %{customdata[1]:.2f}<br>' +
                            '<extra></extra>',
                customdata=list(zip(modalities, confidences))
            ))
            
            fig.update_layout(
                title="🌌 3D 멀티모달 임베딩 공간",
                scene=dict(
                    xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                    yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                    zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%})',
                    bgcolor='rgba(240,240,240,0.1)'
                ),
                width=900,
                height=700
            )
            
            return fig
            
        except ImportError:
            # PCA 없는 경우 간단한 3D 플롯
            fig = go.Figure()
            fig.add_annotation(
                text="scikit-learn이 필요합니다",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_ollama_processing_chain(self, decoding_result: DecodingResult) -> go.Figure:
        """Ollama 모델 체인 처리 시각화"""
        
        # 처리 체인 정보
        chain = decoding_result.processing_chain
        confidence_scores = decoding_result.confidence_scores
        
        # 단계별 데이터
        stages = []
        confidences = []
        models = []
        
        for i, stage_model in enumerate(chain):
            stage_name = stage_model.split('(')[0]
            model_name = stage_model.split('(')[1].rstrip(')')
            
            stages.append(f"단계 {i+1}")
            confidences.append(confidence_scores.get(stage_name, 0.5))
            models.append(model_name)
        
        # 플로우 차트 스타일 시각화
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('모델 체인 처리 흐름', '단계별 신뢰도'),
            vertical_spacing=0.15
        )
        
        # 상단: 플로우 차트
        x_positions = list(range(len(stages)))
        
        # 연결선
        for i in range(len(stages) - 1):
            fig.add_trace(
                go.Scatter(
                    x=[x_positions[i] + 0.4, x_positions[i+1] - 0.4],
                    y=[1, 1],
                    mode='lines',
                    line=dict(color='gray', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            # 화살표
            fig.add_annotation(
                x=x_positions[i] + 0.5,
                y=1,
                text="▶",
                showarrow=False,
                font=dict(size=20, color='gray'),
                row=1, col=1
            )
        
        # 단계 박스
        for i, (stage, model, conf) in enumerate(zip(stages, models, confidences)):
            color = px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
            
            fig.add_trace(
                go.Scatter(
                    x=[x_positions[i]],
                    y=[1],
                    mode='markers+text',
                    marker=dict(
                        size=80,
                        color=color,
                        line=dict(color='white', width=3)
                    ),
                    text=f"{stage}<br>{model}",
                    textposition="middle center",
                    showlegend=False,
                    hovertemplate=f'<b>{stage}</b><br>' +
                                f'모델: {model}<br>' +
                                f'신뢰도: {conf:.2f}<br>' +
                                '<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 하단: 신뢰도 바 차트
        fig.add_trace(
            go.Bar(
                x=stages,
                y=confidences,
                name='신뢰도',
                marker_color=[px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)] 
                            for i in range(len(stages))],
                text=[f"{conf:.2f}" for conf in confidences],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>신뢰도: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="🧠 Ollama 5단계 지능형 디코딩 체인",
            height=600,
            showlegend=False
        )
        
        fig.update_yaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(title_text="신뢰도", row=2, col=1)
        
        return fig
    
    def create_knowledge_graph(self, ontology_summary: Dict[str, Any]) -> go.Figure:
        """지식 온톨로지 그래프 시각화"""
        
        # 샘플 지식 그래프 (실제로는 온톨로지에서 가져옴)
        categories = ontology_summary.get('category_distribution', {})
        
        if not categories:
            # 기본 그래프
            fig = go.Figure()
            fig.add_annotation(
                text="지식 데이터가 없습니다",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # 계층적 그래프 구성
        G = nx.Graph()
        
        # 중앙 노드 (도메인)
        G.add_node("컨퍼런스 분석", type="domain", size=100)
        
        # 카테고리 노드들
        for category, count in categories.items():
            G.add_node(category, type="category", size=count * 10 + 20)
            G.add_edge("컨퍼런스 분석", category)
        
        # 레이아웃 (중앙 집중형)
        pos = nx.spring_layout(G, k=3, center=(0, 0))
        
        # 엣지 그리기
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(150,150,150,0.5)'),
            hoverinfo='none',
            mode='lines'
        )
        
        # 노드 그리기
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node[0])
            node_size.append(node[1]['size'])
            
            if node[1]['type'] == 'domain':
                node_color.append('#FF6B6B')  # 도메인은 빨간색
            else:
                node_color.append('#4ECDC4')  # 카테고리는 청록색
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=[f"{node}<br>크기: {size}" for node, size in zip(node_text, node_size)],
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white'),
                opacity=0.8
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title="📚 지식 온톨로지 그래프",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=700,
            height=500
        )
        
        return fig
    
    def create_real_time_monitor(self) -> go.Figure:
        """실시간 처리 모니터링 대시보드"""
        
        # 샘플 실시간 데이터 (실제로는 라이브 데이터)
        time_points = pd.date_range('now', periods=20, freq='1T')
        
        processing_load = np.random.randint(20, 80, 20)
        memory_usage = np.random.randint(40, 90, 20)
        accuracy_scores = np.random.uniform(0.7, 0.95, 20)
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('처리 부하 (%)', '메모리 사용량 (%)', '분석 정확도'),
            vertical_spacing=0.1
        )
        
        # 처리 부하
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=processing_load,
                mode='lines+markers',
                name='처리 부하',
                line=dict(color='#FF6B6B', width=3),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # 메모리 사용량
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=memory_usage,
                mode='lines+markers',
                name='메모리 사용량',
                line=dict(color='#4ECDC4', width=3),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # 정확도
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=accuracy_scores,
                mode='lines+markers',
                name='분석 정확도',
                line=dict(color='#45B7D1', width=3),
                fill='tonexty'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title="⚡ 실시간 시스템 모니터링",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_insight_sunburst(self, decoding_result: DecodingResult) -> go.Figure:
        """인사이트 선버스트 차트"""
        
        insights = decoding_result.insights
        
        # 인사이트를 타입과 우선순위로 분류
        data = []
        
        for insight in insights:
            insight_type = insight.get('type', 'unknown')
            priority = insight.get('priority', 'medium')
            stage = insight.get('stage', 1)
            
            data.append({
                'ids': f"{insight_type}_{priority}_{stage}",
                'labels': f"{insight_type}",
                'parents': priority,
                'values': 1
            })
        
        # 우선순위 레벨 추가
        priorities = ['high', 'medium', 'low']
        for priority in priorities:
            data.append({
                'ids': priority,
                'labels': priority.upper(),
                'parents': "",
                'values': 0
            })
        
        # 데이터프레임 생성
        df = pd.DataFrame(data)
        
        if len(df) > len(priorities):  # 실제 인사이트가 있는 경우만
            fig = go.Figure(go.Sunburst(
                ids=df['ids'],
                labels=df['labels'],
                parents=df['parents'],
                values=df['values'],
                branchvalues="total",
                maxdepth=3
            ))
            
            fig.update_layout(
                title="🌟 인사이트 분류 선버스트",
                font_size=12,
                width=600,
                height=600
            )
        else:
            # 데이터 없는 경우
            fig = go.Figure()
            fig.add_annotation(
                text="분석 인사이트가 없습니다",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        return fig

# 사용 예제
def create_visualization_dashboard(encoded_results: List[EncodedResult],
                                 fusion_result: FusionResult,
                                 decoding_result: DecodingResult,
                                 ontology_summary: Dict[str, Any]) -> Dict[str, go.Figure]:
    """통합 시각화 대시보드 생성"""
    
    visualizer = CrossModalVisualizer()
    
    figures = {
        'correlation_matrix': visualizer.create_correlation_matrix(fusion_result, encoded_results),
        'fusion_network': visualizer.create_fusion_network(fusion_result, encoded_results),
        '3d_embedding': visualizer.create_3d_embedding_space(encoded_results),
        'ollama_chain': visualizer.create_ollama_processing_chain(decoding_result),
        'knowledge_graph': visualizer.create_knowledge_graph(ontology_summary),
        'real_time_monitor': visualizer.create_real_time_monitor(),
        'insight_sunburst': visualizer.create_insight_sunburst(decoding_result)
    }
    
    return figures

if __name__ == "__main__":
    # 테스트 코드
    visualizer = CrossModalVisualizer()
    print("크로스모달 시각화 엔진이 준비되었습니다!")