#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 AI 인사이트 엔진 - 패턴 인식 및 미래 전망 시스템
AI Insights Engine for SOLOMOND AI Dual Brain System

핵심 기능:
1. 분석 패턴 탐지 (시간, 주제, 성공률 등)
2. 트렌드 예측 및 인사이트 생성
3. 개인화된 추천 시스템
4. 미래 계획 제안
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import streamlit as st
from collections import Counter, defaultdict
import re
from dataclasses import dataclass

@dataclass
class InsightPattern:
    """인사이트 패턴 데이터 구조"""
    pattern_type: str
    confidence: float
    description: str
    evidence: List[str]
    recommendation: str

class AIInsightsEngine:
    """AI 인사이트 엔진 - 듀얼 브레인의 핵심"""
    
    def __init__(self):
        self.history_dir = Path("analysis_history")
        self.insights_cache = {}
        self.load_analysis_data()
    
    def load_analysis_data(self) -> List[Dict]:
        """모든 분석 데이터 로드"""
        self.all_analyses = []
        
        if not self.history_dir.exists():
            return []
        
        # 메타데이터 로드
        metadata_file = self.history_dir / "analysis_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                # 각 분석의 상세 데이터 로드
                for analysis_meta in metadata.get("analyses", []):
                    analysis_file = self.history_dir / f"{analysis_meta['id']}_analysis.json"
                    if analysis_file.exists():
                        with open(analysis_file, 'r', encoding='utf-8') as f:
                            full_analysis = json.load(f)
                            self.all_analyses.append(full_analysis)
                            
            except Exception as e:
                st.warning(f"⚠️ 분석 데이터 로드 중 오류: {e}")
        
        return self.all_analyses
    
    def detect_temporal_patterns(self) -> List[InsightPattern]:
        """시간 기반 패턴 탐지"""
        patterns = []
        
        # 최적화: 데이터 부족 시에도 기본 패턴 탐지 (임계값 조정)
        if len(self.all_analyses) < 1:
            return patterns
        
        # 시간별 분석 활동 패턴
        hours = []
        weekdays = []
        success_by_hour = defaultdict(list)
        
        for analysis in self.all_analyses:
            try:
                timestamp = datetime.fromisoformat(analysis["timestamp"])
                hour = timestamp.hour
                weekday = timestamp.strftime("%A")
                
                hours.append(hour)
                weekdays.append(weekday)
                
                success_rate = analysis["success_count"] / analysis["total_files"]
                success_by_hour[hour].append(success_rate)
                
            except Exception as e:
                continue
        
        # 최고 활동 시간대 탐지 (최적화: 단일 데이터도 패턴으로 인식)
        if hours:
            most_active_hour = max(set(hours), key=hours.count)
            activity_count = hours.count(most_active_hour)
            
            if activity_count >= 1:  # 최적화: 임계값 2→1로 낮춤
                confidence = min(activity_count / len(hours), 0.9)
                
                # 해당 시간대 성공률 계산
                avg_success = np.mean(success_by_hour[most_active_hour]) if success_by_hour[most_active_hour] else 0
                
                pattern = InsightPattern(
                    pattern_type="temporal_peak",
                    confidence=confidence,
                    description=f"주로 {most_active_hour}시경에 분석 활동이 가장 활발합니다 (성공률: {avg_success*100:.1f}%)",
                    evidence=[f"{activity_count}/{len(hours)} 분석이 {most_active_hour}시에 수행됨"],
                    recommendation=f"🕐 {most_active_hour}시는 당신의 최고 집중 시간대입니다. 중요한 분석을 이 시간에 배치하세요."
                )
                patterns.append(pattern)
        
        # 주간 패턴 탐지
        if weekdays:
            most_active_day = max(set(weekdays), key=weekdays.count)
            day_count = weekdays.count(most_active_day)
            
            if day_count >= 2:
                confidence = min(day_count / len(weekdays), 0.85)
                
                pattern = InsightPattern(
                    pattern_type="weekly_pattern",
                    confidence=confidence,
                    description=f"{most_active_day}에 분석 활동이 집중됩니다",
                    evidence=[f"{day_count}/{len(weekdays)} 분석이 {most_active_day}에 수행됨"],
                    recommendation=f"📅 {most_active_day}는 당신의 주간 분석 데이 입니다. 정기 스케줄로 활용하세요."
                )
                patterns.append(pattern)
        
        return patterns
    
    def detect_content_patterns(self) -> List[InsightPattern]:
        """콘텐츠 및 주제 패턴 탐지"""
        patterns = []
        
        if len(self.all_analyses) < 2:
            return patterns
        
        # 컨퍼런스명에서 키워드 추출
        keywords = []
        industries = []
        success_by_industry = defaultdict(list)
        
        for analysis in self.all_analyses:
            try:
                pre_info = analysis.get("pre_info", {})
                conference_name = pre_info.get("conference_name", "").lower()
                industry = pre_info.get("industry_field", "").lower()
                
                # 키워드 추출 (간단한 방식)
                if conference_name:
                    words = re.findall(r'\b\w+\b', conference_name)
                    keywords.extend([w for w in words if len(w) > 2])
                
                if industry:
                    industries.append(industry)
                    success_rate = analysis["success_count"] / analysis["total_files"]
                    success_by_industry[industry].append(success_rate)
                    
            except Exception as e:
                continue
        
        # 주요 관심 분야 탐지
        if industries:
            top_industry = max(set(industries), key=industries.count)
            industry_count = industries.count(top_industry)
            
            if industry_count >= 2:
                confidence = min(industry_count / len(industries), 0.9)
                avg_success = np.mean(success_by_industry[top_industry])
                
                pattern = InsightPattern(
                    pattern_type="industry_focus",
                    confidence=confidence,
                    description=f"'{top_industry}' 분야에 지속적인 관심을 보입니다 (평균 성공률: {avg_success*100:.1f}%)",
                    evidence=[f"{industry_count}/{len(industries)} 분석이 {top_industry} 분야"],
                    recommendation=f"🎯 {top_industry} 분야 전문성이 높아지고 있습니다. 관련 심화 분석을 계획해보세요."
                )
                patterns.append(pattern)
        
        # 자주 등장하는 키워드 패턴
        if keywords:
            keyword_freq = Counter(keywords)
            top_keywords = keyword_freq.most_common(3)
            
            if top_keywords and top_keywords[0][1] >= 2:
                top_keyword, freq = top_keywords[0]
                confidence = min(freq / len(self.all_analyses), 0.8)
                
                pattern = InsightPattern(
                    pattern_type="keyword_trend",
                    confidence=confidence,
                    description=f"'{top_keyword}' 키워드가 자주 등장합니다 ({freq}회)",
                    evidence=[f"'{kw}': {count}회" for kw, count in top_keywords[:3]],
                    recommendation=f"🔍 '{top_keyword}' 관련 트렌드를 더 깊이 분석해보시는 것이 좋겠습니다."
                )
                patterns.append(pattern)
        
        return patterns
    
    def detect_performance_patterns(self) -> List[InsightPattern]:
        """성능 및 품질 패턴 탐지"""
        patterns = []
        
        if len(self.all_analyses) < 3:
            return patterns
        
        # 성공률 트렌드 분석
        success_rates = []
        file_counts = []
        timestamps = []
        
        for analysis in self.all_analyses:
            try:
                success_rate = analysis["success_count"] / analysis["total_files"]
                success_rates.append(success_rate)
                file_counts.append(analysis["total_files"])
                timestamps.append(datetime.fromisoformat(analysis["timestamp"]))
                
            except Exception as e:
                continue
        
        if len(success_rates) >= 3:
            # 성공률 개선 트렌드
            recent_rates = success_rates[-3:]
            earlier_rates = success_rates[:-3] if len(success_rates) > 3 else success_rates[:3]
            
            recent_avg = np.mean(recent_rates)
            earlier_avg = np.mean(earlier_rates)
            
            if recent_avg > earlier_avg + 0.1:  # 10% 이상 개선
                improvement = (recent_avg - earlier_avg) * 100
                confidence = min(improvement / 50, 0.95)  # 최대 95%
                
                pattern = InsightPattern(
                    pattern_type="performance_improvement",
                    confidence=confidence,
                    description=f"분석 성공률이 {improvement:.1f}% 향상되었습니다 ({earlier_avg*100:.1f}% → {recent_avg*100:.1f}%)",
                    evidence=[f"최근 3회 평균: {recent_avg*100:.1f}%", f"이전 평균: {earlier_avg*100:.1f}%"],
                    recommendation="🚀 분석 품질이 지속적으로 향상되고 있습니다. 현재 접근 방식을 유지하세요!"
                )
                patterns.append(pattern)
            
            # 파일 수와 성공률 관계 분석
            if len(file_counts) >= 5:
                correlation = np.corrcoef(file_counts, success_rates)[0, 1]
                
                if abs(correlation) > 0.5:
                    conf = min(abs(correlation), 0.9)
                    trend = "높을수록" if correlation > 0 else "낮을수록"
                    
                    pattern = InsightPattern(
                        pattern_type="volume_quality_relation",
                        confidence=conf,
                        description=f"파일 수가 {trend} 성공률이 높아지는 경향이 있습니다 (상관관계: {correlation:.2f})",
                        evidence=[f"파일 수 평균: {np.mean(file_counts):.1f}개", f"성공률 평균: {np.mean(success_rates)*100:.1f}%"],
                        recommendation="📊 최적의 파일 수를 찾아 분석 효율성을 높여보세요." if correlation < 0 else "📈 대용량 분석에서 더 좋은 결과를 얻고 있습니다."
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def predict_future_trends(self) -> List[Dict[str, Any]]:
        """미래 트렌드 예측"""
        predictions = []
        
        if len(self.all_analyses) < 4:
            return [{
                "type": "insufficient_data",
                "title": "데이터 축적 중",
                "description": "더 많은 분석 데이터가 축적되면 정확한 트렌드 예측이 가능합니다.",
                "confidence": 0.3,
                "timeline": "2-3회 분석 후"
            }]
        
        # 시간 기반 예측
        timestamps = []
        success_rates = []
        
        for analysis in self.all_analyses:
            try:
                timestamps.append(datetime.fromisoformat(analysis["timestamp"]))
                success_rates.append(analysis["success_count"] / analysis["total_files"])
            except:
                continue
        
        if len(timestamps) >= 4:
            # 분석 빈도 예측
            time_diffs = [(timestamps[i] - timestamps[i-1]).days for i in range(1, len(timestamps))]
            avg_interval = np.mean(time_diffs)
            
            next_analysis_date = timestamps[-1] + timedelta(days=avg_interval)
            
            predictions.append({
                "type": "next_analysis",
                "title": "다음 분석 예상 시점",
                "description": f"{next_analysis_date.strftime('%Y년 %m월 %d일')} 경에 새로운 분석을 수행할 가능성이 높습니다",
                "confidence": min(0.7, len(timestamps) / 10),
                "timeline": f"{avg_interval:.0f}일 후",
                "suggestion": "정기적인 분석 스케줄을 캘린더에 등록해보세요"
            })
            
            # 성공률 트렌드 예측
            if len(success_rates) >= 4:
                recent_trend = np.polyfit(range(len(success_rates)), success_rates, 1)[0]
                
                if abs(recent_trend) > 0.01:  # 1% 이상 변화
                    direction = "상승" if recent_trend > 0 else "하락"
                    future_rate = success_rates[-1] + recent_trend * 2  # 2회 분석 후 예상
                    
                    predictions.append({
                        "type": "success_trend",
                        "title": f"성공률 {direction} 트렌드",
                        "description": f"현재 트렌드가 지속되면 성공률이 {future_rate*100:.1f}%까지 {direction}할 것으로 예상됩니다",
                        "confidence": min(abs(recent_trend) * 50, 0.8),
                        "timeline": "다음 2회 분석",
                        "suggestion": "향상" if recent_trend > 0 else "개선책을 고려해보세요"
                    })
        
        return predictions
    
    def generate_personalized_recommendations(self) -> List[Dict[str, Any]]:
        """개인화된 추천 시스템"""
        recommendations = []
        
        if len(self.all_analyses) == 0:
            return [{
                "category": "시작하기",
                "priority": "높음",
                "title": "첫 번째 컨퍼런스 분석",
                "description": "AI 듀얼 브레인을 활성화하기 위해 첫 분석을 시작해보세요",
                "action": "모듈1에서 파일 업로드 후 분석 실행",
                "expected_benefit": "개인화된 인사이트 시작"
            }]
        
        # 분석 빈도 기반 추천
        total_analyses = len(self.all_analyses)
        
        if total_analyses < 3:
            recommendations.append({
                "category": "데이터 축적",
                "priority": "높음",
                "title": "더 많은 분석 수행",
                "description": f"현재 {total_analyses}회 분석 완료. 5회 이상 분석하면 고급 패턴 탐지가 가능합니다",
                "action": "다양한 컨퍼런스를 정기적으로 분석",
                "expected_benefit": "패턴 인식 정확도 대폭 향상"
            })
        
        # 최근 활동 기반 추천
        if self.all_analyses:
            last_analysis = max(self.all_analyses, key=lambda x: x["timestamp"])
            last_time = datetime.fromisoformat(last_analysis["timestamp"])
            days_since = (datetime.now() - last_time).days
            
            if days_since > 7:
                recommendations.append({
                    "category": "활동 재개",
                    "priority": "중간",
                    "title": "정기 분석 재개",
                    "description": f"마지막 분석 후 {days_since}일이 경과했습니다",
                    "action": "새로운 컨퍼런스 분석 시작",
                    "expected_benefit": "지속적인 인사이트 축적"
                })
        
        # 성공률 기반 추천
        success_rates = [a["success_count"] / a["total_files"] for a in self.all_analyses]
        avg_success = np.mean(success_rates)
        
        if avg_success < 0.8:
            recommendations.append({
                "category": "품질 개선",
                "priority": "높음",
                "title": "분석 품질 향상",
                "description": f"평균 성공률 {avg_success*100:.1f}%. 파일 전처리나 품질 확인이 필요할 수 있습니다",
                "action": "파일 형식과 품질을 사전에 확인",
                "expected_benefit": "85% 이상 성공률 달성"
            })
        elif avg_success > 0.95:
            recommendations.append({
                "category": "도전",
                "priority": "낮음", 
                "title": "고급 분석 도전",
                "description": f"뛰어난 성공률 {avg_success*100:.1f}%! 더 복잡한 분석에 도전해보세요",
                "action": "대용량 파일이나 다양한 형식 분석 시도",
                "expected_benefit": "분석 역량 확장"
            })
        
        # 구글 캘린더 연동 추천
        calendar_enabled = any("google_calendar_enabled" in str(a) for a in self.all_analyses)
        if total_analyses >= 3 and not calendar_enabled:
            recommendations.append({
                "category": "시스템 통합",
                "priority": "중간",
                "title": "구글 캘린더 연동 활성화",
                "description": "분석 이력을 캘린더에 자동 저장하여 더 체계적인 관리가 가능합니다",
                "action": "google_calendar_connector.py 실행 후 설정",
                "expected_benefit": "스케줄 최적화 및 패턴 시각화"
            })
        
        return recommendations
    
    def generate_comprehensive_insights(self) -> Dict[str, Any]:
        """종합 인사이트 생성"""
        temporal_patterns = self.detect_temporal_patterns()
        content_patterns = self.detect_content_patterns() 
        performance_patterns = self.detect_performance_patterns()
        predictions = self.predict_future_trends()
        recommendations = self.generate_personalized_recommendations()
        
        # 인사이트 점수 계산
        total_analyses = len(self.all_analyses)
        insight_maturity = min(total_analyses / 10, 1.0)  # 10회 분석이면 완전 성숙
        
        avg_success = np.mean([a["success_count"] / a["total_files"] for a in self.all_analyses]) if self.all_analyses else 0
        quality_score = avg_success
        
        pattern_count = len(temporal_patterns) + len(content_patterns) + len(performance_patterns)
        pattern_richness = min(pattern_count / 5, 1.0)  # 5개 패턴이면 풍부함
        
        overall_score = (insight_maturity * 0.4 + quality_score * 0.3 + pattern_richness * 0.3) * 100
        
        return {
            "metadata": {
                "total_analyses": total_analyses,
                "analysis_period": self._get_analysis_period(),
                "last_update": datetime.now().isoformat(),
                "insight_maturity": insight_maturity,
                "overall_score": overall_score
            },
            "patterns": {
                "temporal": temporal_patterns,
                "content": content_patterns,
                "performance": performance_patterns
            },
            "predictions": predictions,
            "recommendations": recommendations,
            "summary": self._generate_summary(temporal_patterns, content_patterns, performance_patterns, predictions)
        }
    
    def _get_analysis_period(self) -> str:
        """분석 기간 계산"""
        if not self.all_analyses:
            return "분석 없음"
        
        timestamps = [datetime.fromisoformat(a["timestamp"]) for a in self.all_analyses]
        start_date = min(timestamps)
        end_date = max(timestamps)
        
        period_days = (end_date - start_date).days
        
        if period_days == 0:
            return "1일"
        elif period_days < 7:
            return f"{period_days}일"
        elif period_days < 30:
            return f"{period_days // 7}주"
        else:
            return f"{period_days // 30}개월"
    
    def _generate_summary(self, temporal, content, performance, predictions) -> str:
        """인사이트 요약 생성"""
        if not self.all_analyses:
            return "아직 분석 데이터가 없어 패턴을 파악할 수 없습니다. 첫 분석을 시작해보세요!"
        
        summary_parts = []
        
        # 활동 패턴
        if temporal:
            time_pattern = temporal[0]
            summary_parts.append(f"⏰ {time_pattern.description}")
        
        # 주제 패턴  
        if content:
            content_pattern = content[0]
            summary_parts.append(f"🎯 {content_pattern.description}")
        
        # 성능 패턴
        if performance:
            perf_pattern = performance[0]
            summary_parts.append(f"📈 {perf_pattern.description}")
        
        # 예측
        if predictions and predictions[0]["type"] != "insufficient_data":
            pred = predictions[0]
            summary_parts.append(f"🔮 {pred['description']}")
        
        if summary_parts:
            return " | ".join(summary_parts)
        else:
            return f"총 {len(self.all_analyses)}회 분석 완료. 더 많은 데이터가 축적되면 상세한 패턴을 탐지할 수 있습니다."

def render_insights_dashboard():
    """인사이트 대시보드 렌더링"""
    st.title("🧠 AI 인사이트 대시보드")
    st.markdown("**듀얼 브레인의 핵심 - 패턴 인식 및 미래 전망**")
    
    # 인사이트 엔진 초기화
    engine = AIInsightsEngine()
    insights = engine.generate_comprehensive_insights()
    
    # 전체 점수 및 상태
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("AI 브레인 성숙도", f"{insights['metadata']['insight_maturity']*100:.0f}%")
    
    with col2:
        st.metric("총 분석 수", insights['metadata']['total_analyses'])
    
    with col3:
        st.metric("분석 기간", insights['metadata']['analysis_period'])
    
    with col4:
        st.metric("종합 점수", f"{insights['metadata']['overall_score']:.0f}/100")
    
    # 요약
    st.subheader("📋 AI 인사이트 요약")
    st.info(insights['summary'])
    
    # 탭으로 구성
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 패턴 분석", "🔮 미래 예측", "💡 개인화 추천", "📊 상세 데이터"])
    
    with tab1:
        st.header("🔍 발견된 패턴들")
        
        # 시간 패턴
        if insights['patterns']['temporal']:
            st.subheader("⏰ 시간 패턴")
            for pattern in insights['patterns']['temporal']:
                with st.expander(f"{'⭐' * int(pattern.confidence * 5)} {pattern.description}"):
                    st.write(f"**신뢰도**: {pattern.confidence*100:.0f}%")
                    st.write(f"**근거**: {', '.join(pattern.evidence)}")
                    st.success(f"💡 {pattern.recommendation}")
        
        # 콘텐츠 패턴
        if insights['patterns']['content']:
            st.subheader("🎯 콘텐츠 패턴") 
            for pattern in insights['patterns']['content']:
                with st.expander(f"{'⭐' * int(pattern.confidence * 5)} {pattern.description}"):
                    st.write(f"**신뢰도**: {pattern.confidence*100:.0f}%")
                    st.write(f"**근거**: {', '.join(pattern.evidence)}")
                    st.success(f"💡 {pattern.recommendation}")
        
        # 성능 패턴
        if insights['patterns']['performance']:
            st.subheader("📈 성능 패턴")
            for pattern in insights['patterns']['performance']:
                with st.expander(f"{'⭐' * int(pattern.confidence * 5)} {pattern.description}"):
                    st.write(f"**신뢰도**: {pattern.confidence*100:.0f}%")
                    st.write(f"**근거**: {', '.join(pattern.evidence)}")
                    st.success(f"💡 {pattern.recommendation}")
    
    with tab2:
        st.header("🔮 미래 트렌드 예측")
        
        for prediction in insights['predictions']:
            confidence_stars = '⭐' * int(prediction['confidence'] * 5)
            
            with st.expander(f"{confidence_stars} {prediction['title']}"):
                st.write(prediction['description'])
                st.write(f"**신뢰도**: {prediction['confidence']*100:.0f}%")
                st.write(f"**예상 시기**: {prediction['timeline']}")
                
                if 'suggestion' in prediction:
                    st.info(f"💡 제안: {prediction['suggestion']}")
    
    with tab3:
        st.header("💡 개인화된 추천")
        
        priority_order = {"높음": 0, "중간": 1, "낮음": 2}
        sorted_recommendations = sorted(insights['recommendations'], key=lambda x: priority_order.get(x['priority'], 3))
        
        for rec in sorted_recommendations:
            priority_color = {"높음": "🔴", "중간": "🟡", "낮음": "🟢"}.get(rec['priority'], "⚪")
            
            with st.expander(f"{priority_color} {rec['title']} ({rec['category']})"):
                st.write(rec['description'])
                st.write(f"**실행 방법**: {rec['action']}")
                st.success(f"🎯 기대 효과: {rec['expected_benefit']}")
    
    with tab4:
        st.header("📊 상세 데이터")
        
        if engine.all_analyses:
            # 분석 데이터프레임 생성
            df_data = []
            for analysis in engine.all_analyses:
                df_data.append({
                    "분석일": datetime.fromisoformat(analysis["timestamp"]).strftime("%Y-%m-%d %H:%M"),
                    "컨퍼런스명": analysis["pre_info"].get("conference_name", "Unknown"),
                    "업계": analysis["pre_info"].get("industry_field", "Unknown"),
                    "파일수": analysis["total_files"],
                    "성공수": analysis["success_count"],
                    "성공률": f"{analysis['success_count']/analysis['total_files']*100:.1f}%",
                    "파일유형": ", ".join(analysis["file_types"])
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # 원시 데이터 다운로드
            if st.button("📥 인사이트 데이터 다운로드"):
                insights_json = json.dumps(insights, ensure_ascii=False, indent=2, default=str)
                st.download_button(
                    "💾 JSON 파일 다운로드",
                    insights_json,
                    file_name=f"ai_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info("📊 분석 데이터가 없습니다. 먼저 컨퍼런스 분석을 수행해주세요.")

if __name__ == "__main__":
    st.set_page_config(
        page_title="AI 인사이트 엔진",
        page_icon="🧠",
        layout="wide"
    )
    
    render_insights_dashboard()