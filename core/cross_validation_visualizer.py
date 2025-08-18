# Phase 2: 크로스 검증 시각화 시스템
# 고급 차트 및 대시보드 - 주얼리 AI 플랫폼

import json
import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

@dataclass
class ValidationMetrics:
    """검증 지표 데이터 구조"""
    file_id: str
    filename: str
    file_type: str
    
    # 품질 지표
    content_completeness: float  # 내용 완전성 (0-100)
    keyword_accuracy: float      # 키워드 정확도 (0-100)
    audio_quality: float         # 음질/화질 (0-100)
    time_accuracy: float         # 시간 정확도 (0-100)
    
    # 신뢰도 지표
    confidence_score: float      # 개별 파일 신뢰도 (0-1)
    cross_match_score: float     # 다른 파일과의 일치도 (0-1)
    
    # 주얼리 특화 지표
    jewelry_terms_found: int     # 발견된 주얼리 용어 개수
    price_accuracy: float        # 가격 정보 정확도 (0-100)
    technical_terms: int         # 기술 용어 개수

@dataclass
class CrossValidationResult:
    """크로스 검증 종합 결과"""
    session_id: str
    session_name: str
    
    # 전체 지표
    overall_confidence: float    # 전체 신뢰도 (0-1)
    content_overlap: float       # 내용 중복도 (0-1)
    quality_improvement: float   # 품질 개선도 (0-1)
    
    # 파일별 지표
    file_metrics: List[ValidationMetrics]
    
    # 크로스 매칭 매트릭스
    cross_matrix: List[List[float]]  # N x N 매트릭스
    
    # 키워드 분석
    common_keywords: List[str]
    unique_keywords: Dict[str, List[str]]  # 파일별 고유 키워드
    
    # 주얼리 인사이트
    jewelry_insights: Dict[str, Any]
    
    # 시간 정보
    validation_timestamp: str
    processing_time: float

class CrossValidationVisualizer:
    """크로스 검증 결과 시각화 생성기"""
    
    def __init__(self):
        self.colors = {
            'primary': '#2C5282',
            'secondary': '#E53E3E', 
            'success': '#38A169',
            'warning': '#D69E2E',
            'gold': '#B7791F',
            'gradient_start': '#667eea',
            'gradient_end': '#764ba2'
        }
    
    def generate_visualization_html(self, validation_result: CrossValidationResult) -> str:
        """크로스 검증 시각화 HTML 생성"""
        
        # 각 차트 데이터 생성
        radar_chart_data = self._generate_radar_chart_data(validation_result)
        matrix_heatmap_data = self._generate_matrix_heatmap_data(validation_result)
        confidence_timeline_data = self._generate_confidence_timeline_data(validation_result)
        keyword_network_data = self._generate_keyword_network_data(validation_result)
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>크로스 검증 분석 결과 - {validation_result.session_name}</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
            <style>
                {self._get_visualization_css()}
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                {self._generate_header_section(validation_result)}
                {self._generate_metrics_overview(validation_result)}
                {self._generate_charts_grid(validation_result)}
                {self._generate_detailed_analysis(validation_result)}
                {self._generate_jewelry_insights_section(validation_result)}
            </div>
            
            <script>
                {self._generate_charts_javascript(radar_chart_data, matrix_heatmap_data, confidence_timeline_data, keyword_network_data)}
            </script>
        </body>
        </html>
        """
        
        return html_template
    
    def _get_visualization_css(self) -> str:
        """시각화 전용 CSS 스타일 - 대형 함수 (리팩토링 고려 대상 - 457줄)"""
        return """
        :root {
            --primary-color: #2C5282;
            --secondary-color: #E53E3E;
            --success-color: #38A169;
            --warning-color: #D69E2E;
            --gold-color: #B7791F;
            --gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --surface: #FFFFFF;
            --background: #F7FAFC;
            --text-primary: #2D3748;
            --text-secondary: #4A5568;
            --border: #E2E8F0;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--background);
            color: var(--text-primary);
            line-height: 1.6;
        }
        
        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header-section {
            background: var(--gradient);
            color: white;
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: var(--shadow);
        }
        
        .header-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header-subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 20px;
        }
        
        .session-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .session-stat {
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        
        .stat-value {
            font-size: 1.8rem;
            font-weight: 700;
            display: block;
        }
        
        .stat-label {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 5px;
        }
        
        .metrics-overview {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: var(--surface);
            padding: 30px;
            border-radius: 20px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--gradient);
        }
        
        .metric-icon {
            font-size: 3rem;
            margin-bottom: 15px;
            display: block;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 800;
            color: var(--primary-color);
            margin-bottom: 10px;
        }
        
        .metric-label {
            font-size: 1rem;
            color: var(--text-secondary);
            font-weight: 500;
        }
        
        .metric-trend {
            font-size: 0.85rem;
            margin-top: 10px;
            padding: 5px 10px;
            border-radius: 20px;
            display: inline-block;
        }
        
        .trend-positive {
            background: rgba(56, 161, 105, 0.1);
            color: var(--success-color);
        }
        
        .trend-neutral {
            background: rgba(214, 158, 46, 0.1);
            color: var(--warning-color);
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }
        
        .chart-container {
            background: var(--surface);
            padding: 30px;
            border-radius: 20px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
        }
        
        .chart-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .chart-canvas {
            width: 100%;
            height: 400px;
        }
        
        .chart-legend {
            margin-top: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            justify-content: center;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 0.9rem;
        }
        
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }
        
        .full-width-chart {
            grid-column: 1 / -1;
        }
        
        .matrix-heatmap {
            position: relative;
            background: var(--surface);
            border-radius: 15px;
            padding: 20px;
        }
        
        .heatmap-cell {
            stroke: #fff;
            stroke-width: 2;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .heatmap-cell:hover {
            stroke-width: 3;
            filter: brightness(1.1);
        }
        
        .heatmap-tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            pointer-events: none;
            z-index: 1000;
        }
        
        .detailed-analysis {
            background: var(--surface);
            padding: 40px;
            border-radius: 20px;
            box-shadow: var(--shadow);
            margin-bottom: 30px;
            border: 1px solid var(--border);
        }
        
        .analysis-title {
            font-size: 1.6rem;
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .file-analysis-grid {
            display: grid;
            gap: 20px;
        }
        
        .file-analysis-item {
            background: var(--background);
            padding: 25px;
            border-radius: 15px;
            border-left: 5px solid var(--primary-color);
        }
        
        .file-header {
            display: flex;
            justify-content: between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .file-name {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .file-type-badge {
            background: var(--primary-color);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .file-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .file-metric {
            text-align: center;
        }
        
        .file-metric-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--success-color);
        }
        
        .file-metric-label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 2px;
        }
        
        .jewelry-insights {
            background: linear-gradient(135deg, rgba(183, 121, 31, 0.05) 0%, rgba(183, 121, 31, 0.1) 100%);
            border: 2px solid var(--gold-color);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
        }
        
        .insights-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .insights-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--gold-color);
        }
        
        .insights-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
        }
        
        .insight-category {
            background: white;
            padding: 25px;
            border-radius: 15px;
            border-left: 4px solid var(--gold-color);
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .insight-category-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--gold-color);
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .insight-list {
            list-style: none;
        }
        
        .insight-item {
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
            font-size: 0.95rem;
        }
        
        .insight-item:last-child {
            border-bottom: none;
        }
        
        .confidence-indicator {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-left: 10px;
        }
        
        .confidence-high {
            background: rgba(56, 161, 105, 0.1);
            color: var(--success-color);
        }
        
        .confidence-medium {
            background: rgba(214, 158, 46, 0.1);
            color: var(--warning-color);
        }
        
        .confidence-low {
            background: rgba(229, 62, 62, 0.1);
            color: var(--secondary-color);
        }
        
        @media (max-width: 768px) {
            .dashboard-container {
                padding: 10px;
            }
            
            .header-title {
                font-size: 1.8rem;
            }
            
            .charts-grid {
                grid-template-columns: 1fr;
            }
            
            .session-info {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .metrics-overview {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        
        /* 애니메이션 */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .fade-in {
            animation: fadeInUp 0.6s ease-out;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        /* 인터랙티브 요소 */
        .interactive-element {
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .interactive-element:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px -8px rgba(0, 0, 0, 0.2);
        }
        """
    
    def _generate_header_section(self, validation_result: CrossValidationResult) -> str:
        """헤더 섹션 생성"""
        confidence_percentage = round(validation_result.overall_confidence * 100, 1)
        overlap_percentage = round(validation_result.content_overlap * 100, 1)
        improvement_percentage = round(validation_result.quality_improvement * 100, 1)
        
        return f"""
        <div class="header-section fade-in">
            <div class="header-title">📊 크로스 검증 분석 결과</div>
            <div class="header-subtitle">{validation_result.session_name}</div>
            
            <div class="session-info">
                <div class="session-stat">
                    <span class="stat-value">{confidence_percentage}%</span>
                    <div class="stat-label">전체 신뢰도</div>
                </div>
                <div class="session-stat">
                    <span class="stat-value">{overlap_percentage}%</span>
                    <div class="stat-label">내용 중복도</div>
                </div>
                <div class="session-stat">
                    <span class="stat-value">{improvement_percentage}%</span>
                    <div class="stat-label">품질 개선</div>
                </div>
                <div class="session-stat">
                    <span class="stat-value">{len(validation_result.file_metrics)}</span>
                    <div class="stat-label">분석 파일</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_metrics_overview(self, validation_result: CrossValidationResult) -> str:
        """지표 개요 섹션 생성"""
        
        # 평균 지표 계산
        avg_completeness = np.mean([m.content_completeness for m in validation_result.file_metrics])
        avg_keyword_accuracy = np.mean([m.keyword_accuracy for m in validation_result.file_metrics])
        avg_audio_quality = np.mean([m.audio_quality for m in validation_result.file_metrics])
        total_jewelry_terms = sum([m.jewelry_terms_found for m in validation_result.file_metrics])
        
        return f"""
        <div class="metrics-overview fade-in">
            <div class="metric-card interactive-element">
                <div class="metric-icon">📝</div>
                <div class="metric-value">{avg_completeness:.1f}%</div>
                <div class="metric-label">내용 완전성</div>
                <div class="metric-trend trend-positive">+{avg_completeness-85:.1f}% vs 평균</div>
            </div>
            
            <div class="metric-card interactive-element">
                <div class="metric-icon">🎯</div>
                <div class="metric-value">{avg_keyword_accuracy:.1f}%</div>
                <div class="metric-label">키워드 정확도</div>
                <div class="metric-trend trend-positive">+{avg_keyword_accuracy-80:.1f}% vs 평균</div>
            </div>
            
            <div class="metric-card interactive-element">
                <div class="metric-icon">🎵</div>
                <div class="metric-value">{avg_audio_quality:.1f}%</div>
                <div class="metric-label">음질/화질</div>
                <div class="metric-trend trend-neutral">표준 품질</div>
            </div>
            
            <div class="metric-card interactive-element">
                <div class="metric-icon">💎</div>
                <div class="metric-value">{total_jewelry_terms}</div>
                <div class="metric-label">주얼리 전문용어</div>
                <div class="metric-trend trend-positive">고품질 인식</div>
            </div>
        </div>
        """
    
    def _generate_charts_grid(self, validation_result: CrossValidationResult) -> str:
        """차트 그리드 섹션 생성"""
        return f"""
        <div class="charts-grid fade-in">
            <!-- 레이더 차트: 파일별 품질 비교 -->
            <div class="chart-container interactive-element">
                <div class="chart-title">
                    <span>🕸️</span>
                    <span>파일별 품질 비교</span>
                </div>
                <canvas id="radarChart" class="chart-canvas"></canvas>
                <div class="chart-legend" id="radarLegend"></div>
            </div>
            
            <!-- 도넛 차트: 신뢰도 분포 -->
            <div class="chart-container interactive-element">
                <div class="chart-title">
                    <span>🍩</span>
                    <span>신뢰도 분포</span>
                </div>
                <canvas id="confidenceChart" class="chart-canvas"></canvas>
                <div class="chart-legend" id="confidenceLegend"></div>
            </div>
            
            <!-- 히트맵: 파일 간 일치도 매트릭스 -->
            <div class="chart-container full-width-chart interactive-element">
                <div class="chart-title">
                    <span>🔥</span>
                    <span>파일 간 일치도 매트릭스</span>
                </div>
                <div id="heatmapContainer" class="matrix-heatmap"></div>
            </div>
            
            <!-- 타임라인: 처리 시간 분석 -->
            <div class="chart-container full-width-chart interactive-element">
                <div class="chart-title">
                    <span>⏱️</span>
                    <span>처리 시간 분석</span>
                </div>
                <canvas id="timelineChart" class="chart-canvas"></canvas>
            </div>
        </div>
        """
    
    def _generate_detailed_analysis(self, validation_result: CrossValidationResult) -> str:
        """상세 분석 섹션 생성"""
        files_html = ""
        
        for file_metric in validation_result.file_metrics:
            confidence_class = self._get_confidence_class(file_metric.confidence_score)
            files_html += f"""
            <div class="file-analysis-item">
                <div class="file-header">
                    <div class="file-name">📁 {file_metric.filename}</div>
                    <div class="file-type-badge">{file_metric.file_type.upper()}</div>
                </div>
                
                <div class="file-metrics">
                    <div class="file-metric">
                        <div class="file-metric-value">{file_metric.content_completeness:.1f}%</div>
                        <div class="file-metric-label">완전성</div>
                    </div>
                    <div class="file-metric">
                        <div class="file-metric-value">{file_metric.keyword_accuracy:.1f}%</div>
                        <div class="file-metric-label">키워드 정확도</div>
                    </div>
                    <div class="file-metric">
                        <div class="file-metric-value">{file_metric.audio_quality:.1f}%</div>
                        <div class="file-metric-label">품질</div>
                    </div>
                    <div class="file-metric">
                        <div class="file-metric-value">{file_metric.jewelry_terms_found}</div>
                        <div class="file-metric-label">전문용어</div>
                    </div>
                </div>
                
                <div class="confidence-indicator {confidence_class}">
                    신뢰도: {file_metric.confidence_score:.1%}
                </div>
            </div>
            """
        
        return f"""
        <div class="detailed-analysis fade-in">
            <div class="analysis-title">
                <span>🔍</span>
                <span>파일별 상세 분석</span>
            </div>
            
            <div class="file-analysis-grid">
                {files_html}
            </div>
        </div>
        """
    
    def _generate_jewelry_insights_section(self, validation_result: CrossValidationResult) -> str:
        """주얼리 인사이트 섹션 생성"""
        insights = validation_result.jewelry_insights
        
        price_items = ""
        if insights.get('price_mentions'):
            for price in insights['price_mentions']:
                price_items += f'<li class="insight-item">{price}<span class="confidence-indicator confidence-high">높음</span></li>'
        
        quality_items = ""
        if insights.get('quality_grades'):
            for grade in insights['quality_grades']:
                quality_items += f'<li class="insight-item">{grade}<span class="confidence-indicator confidence-high">높음</span></li>'
        
        tech_items = ""
        if insights.get('technical_terms'):
            for term in insights['technical_terms']:
                tech_items += f'<li class="insight-item">{term}<span class="confidence-indicator confidence-medium">중간</span></li>'
        
        keyword_items = ""
        for keyword in validation_result.common_keywords:
            keyword_items += f'<li class="insight-item">{keyword}<span class="confidence-indicator confidence-high">검증완료</span></li>'
        
        return f"""
        <div class="jewelry-insights fade-in">
            <div class="insights-header">
                <span>💎</span>
                <div class="insights-title">주얼리 특화 인사이트</div>
            </div>
            
            <div class="insights-grid">
                <div class="insight-category">
                    <div class="insight-category-title">
                        <span>💰</span>
                        <span>가격 정보</span>
                    </div>
                    <ul class="insight-list">
                        {price_items or '<li class="insight-item">가격 정보가 발견되지 않았습니다.</li>'}
                    </ul>
                </div>
                
                <div class="insight-category">
                    <div class="insight-category-title">
                        <span>⭐</span>
                        <span>품질 등급</span>
                    </div>
                    <ul class="insight-list">
                        {quality_items or '<li class="insight-item">품질 등급 정보가 발견되지 않았습니다.</li>'}
                    </ul>
                </div>
                
                <div class="insight-category">
                    <div class="insight-category-title">
                        <span>🔧</span>
                        <span>기술 용어</span>
                    </div>
                    <ul class="insight-list">
                        {tech_items or '<li class="insight-item">기술 용어가 발견되지 않았습니다.</li>'}
                    </ul>
                </div>
                
                <div class="insight-category">
                    <div class="insight-category-title">
                        <span>🏷️</span>
                        <span>검증된 키워드</span>
                    </div>
                    <ul class="insight-list">
                        {keyword_items or '<li class="insight-item">공통 키워드가 발견되지 않았습니다.</li>'}
                    </ul>
                </div>
            </div>
        </div>
        """
    
    def _get_confidence_class(self, confidence_score: float) -> str:
        """신뢰도 점수에 따른 CSS 클래스 반환"""
        if confidence_score >= 0.8:
            return "confidence-high"
        elif confidence_score >= 0.6:
            return "confidence-medium"
        else:
            return "confidence-low"
    
    def _generate_radar_chart_data(self, validation_result: CrossValidationResult) -> dict:
        """레이더 차트 데이터 생성"""
        return {
            'labels': ['내용 완전성', '키워드 정확도', '음질/화질', '시간 정확도', '크로스 매칭'],
            'datasets': [
                {
                    'label': file_metric.filename,
                    'data': [
                        file_metric.content_completeness,
                        file_metric.keyword_accuracy,
                        file_metric.audio_quality,
                        file_metric.time_accuracy,
                        file_metric.cross_match_score * 100
                    ],
                    'borderColor': self._get_chart_color(i),
                    'backgroundColor': self._get_chart_color(i, alpha=0.2),
                    'pointBackgroundColor': self._get_chart_color(i),
                } for i, file_metric in enumerate(validation_result.file_metrics)
            ]
        }
    
    def _generate_matrix_heatmap_data(self, validation_result: CrossValidationResult) -> dict:
        """매트릭스 히트맵 데이터 생성"""
        return {
            'matrix': validation_result.cross_matrix,
            'labels': [f.filename for f in validation_result.file_metrics],
            'colorScale': ['#E53E3E', '#D69E2E', '#38A169']  # 낮음-중간-높음
        }
    
    def _generate_confidence_timeline_data(self, validation_result: CrossValidationResult) -> dict:
        """신뢰도 타임라인 데이터 생성"""
        return {
            'labels': [f.filename for f in validation_result.file_metrics],
            'data': [f.confidence_score * 100 for f in validation_result.file_metrics],
            'processingTimes': [f.processing_time if hasattr(f, 'processing_time') else 0 for f in validation_result.file_metrics]
        }
    
    def _generate_keyword_network_data(self, validation_result: CrossValidationResult) -> dict:
        """키워드 네트워크 데이터 생성"""
        return {
            'common_keywords': validation_result.common_keywords,
            'unique_keywords': validation_result.unique_keywords,
            'files': [f.filename for f in validation_result.file_metrics]
        }
    
    def _get_chart_color(self, index: int, alpha: float = 1.0) -> str:
        """차트 색상 생성"""
        colors = [
            f'rgba(44, 82, 130, {alpha})',    # 파란색
            f'rgba(229, 62, 62, {alpha})',    # 빨간색  
            f'rgba(56, 161, 105, {alpha})',   # 초록색
            f'rgba(214, 158, 46, {alpha})',   # 노란색
            f'rgba(183, 121, 31, {alpha})',   # 금색
            f'rgba(102, 126, 234, {alpha})',  # 보라색
        ]
        return colors[index % len(colors)]
    
    def _generate_charts_javascript(self, radar_data: dict, matrix_data: dict, 
                                  timeline_data: dict, keyword_data: dict) -> str:
        """차트 JavaScript 코드 생성"""
        return f"""
        // 차트 데이터
        const radarData = {json.dumps(radar_data)};
        const matrixData = {json.dumps(matrix_data)};
        const timelineData = {json.dumps(timeline_data)};
        const keywordData = {json.dumps(keyword_data)};
        
        // 페이지 로드 시 차트 초기화
        document.addEventListener('DOMContentLoaded', function() {{
            initializeCharts();
        }});
        
        function initializeCharts() {{
            // 레이더 차트 초기화
            const radarCtx = document.getElementById('radarChart').getContext('2d');
            new Chart(radarCtx, {{
                type: 'radar',
                data: radarData,
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        r: {{
                            beginAtZero: true,
                            max: 100,
                            ticks: {{
                                stepSize: 20,
                                callback: function(value) {{
                                    return value + '%';
                                }}
                            }}
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            position: 'bottom'
                        }}
                    }}
                }}
            }});
            
            // 신뢰도 도넛 차트 초기화
            const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
            const confidenceValues = radarData.datasets.map(dataset => {{
                return Math.round(dataset.data.reduce((a, b) => a + b) / dataset.data.length);
            }});
            
            new Chart(confidenceCtx, {{
                type: 'doughnut',
                data: {{
                    labels: radarData.datasets.map(dataset => dataset.label),
                    datasets: [{{
                        data: confidenceValues,
                        backgroundColor: radarData.datasets.map((_, i) => 
                            ['#2C5282', '#E53E3E', '#38A169', '#D69E2E', '#B7791F'][i % 5]
                        ),
                        borderWidth: 3,
                        borderColor: '#FFFFFF'
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'bottom'
                        }}
                    }}
                }}
            }});
            
            // 히트맵 초기화
            initializeHeatmap();
            
            // 타임라인 차트 초기화
            const timelineCtx = document.getElementById('timelineChart').getContext('2d');
            new Chart(timelineCtx, {{
                type: 'line',
                data: {{
                    labels: timelineData.labels,
                    datasets: [{{
                        label: '신뢰도 (%)',
                        data: timelineData.data,
                        borderColor: '#2C5282',
                        backgroundColor: 'rgba(44, 82, 130, 0.1)',
                        fill: true,
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100,
                            ticks: {{
                                callback: function(value) {{
                                    return value + '%';
                                }}
                            }}
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            position: 'top'
                        }}
                    }}
                }}
            }});
        }}
        
        function initializeHeatmap() {{
            const container = document.getElementById('heatmapContainer');
            const size = 400;
            const cellSize = size / matrixData.matrix.length;
            
            const svg = d3.select(container)
                .append('svg')
                .attr('width', size + 100)
                .attr('height', size + 100);
            
            const colorScale = d3.scaleSequential(d3.interpolateRdYlGn)
                .domain([0, 1]);
            
            // 히트맵 셀 생성
            const cells = svg.selectAll('.heatmap-cell')
                .data(matrixData.matrix.flat().map((value, i) => ({{
                    value: value,
                    row: Math.floor(i / matrixData.matrix.length),
                    col: i % matrixData.matrix.length
                }})))
                .enter()
                .append('rect')
                .attr('class', 'heatmap-cell')
                .attr('x', d => d.col * cellSize + 50)
                .attr('y', d => d.row * cellSize + 50)
                .attr('width', cellSize - 2)
                .attr('height', cellSize - 2)
                .attr('fill', d => colorScale(d.value))
                .on('mouseover', function(event, d) {{
                    // 툴팁 표시
                    const tooltip = d3.select(container)
                        .append('div')
                        .attr('class', 'heatmap-tooltip')
                        .style('left', (event.offsetX + 10) + 'px')
                        .style('top', (event.offsetY - 10) + 'px')
                        .text(`${{matrixData.labels[d.row]}} ↔ ${{matrixData.labels[d.col]}}: ${{(d.value * 100).toFixed(1)}}%`);
                }})
                .on('mouseout', function() {{
                    d3.selectAll('.heatmap-tooltip').remove();
                }});
            
            // 라벨 추가
            svg.selectAll('.row-label')
                .data(matrixData.labels)
                .enter()
                .append('text')
                .attr('class', 'row-label')
                .attr('x', 45)
                .attr('y', (d, i) => i * cellSize + cellSize/2 + 55)
                .attr('text-anchor', 'end')
                .attr('dominant-baseline', 'middle')
                .style('font-size', '12px')
                .text(d => d.length > 15 ? d.substring(0, 15) + '...' : d);
            
            svg.selectAll('.col-label')
                .data(matrixData.labels)
                .enter()
                .append('text')
                .attr('class', 'col-label')
                .attr('x', (d, i) => i * cellSize + cellSize/2 + 50)
                .attr('y', 45)
                .attr('text-anchor', 'start')
                .attr('dominant-baseline', 'middle')
                .style('font-size', '12px')
                .attr('transform', (d, i) => `rotate(-45, ${{i * cellSize + cellSize/2 + 50}}, 45)`)
                .text(d => d.length > 15 ? d.substring(0, 15) + '...' : d);
        }}
        
        // 애니메이션 효과
        function animateElements() {{
            const elements = document.querySelectorAll('.fade-in');
            elements.forEach((element, index) => {{
                setTimeout(() => {{
                    element.style.opacity = '1';
                    element.style.transform = 'translateY(0)';
                }}, index * 200);
            }});
        }}
        
        // 페이지 로드 시 애니메이션 시작
        window.addEventListener('load', animateElements);
        """

# 사용 예시: 모의 데이터로 시각화 생성
def create_sample_visualization():
    """샘플 크로스 검증 시각화 생성"""
    
    # 모의 파일 지표 생성
    sample_metrics = [
        ValidationMetrics(
            file_id="file1",
            filename="main_recording.mp3",
            file_type="audio",
            content_completeness=95.0,
            keyword_accuracy=92.0,
            audio_quality=88.0,
            time_accuracy=96.0,
            confidence_score=0.93,
            cross_match_score=0.89,
            jewelry_terms_found=15,
            price_accuracy=94.0,
            technical_terms=8
        ),
        ValidationMetrics(
            file_id="file2", 
            filename="backup_recording.wav",
            file_type="audio",
            content_completeness=87.0,
            keyword_accuracy=85.0,
            audio_quality=76.0,
            time_accuracy=91.0,
            confidence_score=0.85,
            cross_match_score=0.89,
            jewelry_terms_found=12,
            price_accuracy=88.0,
            technical_terms=6
        ),
        ValidationMetrics(
            file_id="file3",
            filename="presentation.mp4", 
            file_type="video",
            content_completeness=91.0,
            keyword_accuracy=89.0,
            audio_quality=82.0,
            time_accuracy=93.0,
            confidence_score=0.88,
            cross_match_score=0.86,
            jewelry_terms_found=18,
            price_accuracy=91.0,
            technical_terms=10
        )
    ]
    
    # 크로스 검증 결과 생성
    validation_result = CrossValidationResult(
        session_id="session_123",
        session_name="2025 홍콩주얼리쇼 다이아몬드 세미나",
        overall_confidence=0.89,
        content_overlap=0.87,
        quality_improvement=0.23,
        file_metrics=sample_metrics,
        cross_matrix=[
            [1.0, 0.89, 0.86],
            [0.89, 1.0, 0.84], 
            [0.86, 0.84, 1.0]
        ],
        common_keywords=["다이아몬드", "4C", "GIA", "캐럿", "컬러", "클래리티"],
        unique_keywords={
            "main_recording.mp3": ["프린세스 컷", "도매가"],
            "backup_recording.wav": ["라운드 브릴리언트", "소매가"],
            "presentation.mp4": ["히트 트리트먼트", "인증서"]
        },
        jewelry_insights={
            "price_mentions": ["$8,500", "$25,000", "캐럿당 $3,000"],
            "quality_grades": ["4C", "GIA", "D-IF", "E-VS1"],
            "technical_terms": ["프린세스 컷", "라운드 브릴리언트", "히트 트리트먼트"]
        },
        validation_timestamp=datetime.now().isoformat(),
        processing_time=45.2
    )
    
    # 시각화 생성
    visualizer = CrossValidationVisualizer()
    html_content = visualizer.generate_visualization_html(validation_result)
    
    return html_content

if __name__ == "__main__":
    # 샘플 시각화 HTML 생성
    sample_html = create_sample_visualization()
    
    # 파일로 저장
    with open("cross_validation_visualization_demo.html", "w", encoding="utf-8") as f:
        f.write(sample_html)
    
    print("✅ 크로스 검증 시각화 시스템 완성!")
    print("📊 고급 차트 및 대시보드 생성 완료")
    print("💾 데모 파일: cross_validation_visualization_demo.html")
    print("🎯 특징:")
    print("   - 레이더 차트: 파일별 품질 비교")
    print("   - 히트맵: 파일 간 일치도 매트릭스") 
    print("   - 도넛 차트: 신뢰도 분포")
    print("   - 타임라인: 처리 시간 분석")
    print("   - 주얼리 특화 인사이트")
    print("   - 반응형 모바일 최적화")