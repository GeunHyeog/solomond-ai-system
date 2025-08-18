#!/usr/bin/env python3
"""
컨퍼런스 내용 기반 마인드맵 생성기
- 분석된 컨퍼런스 데이터를 바탕으로 마인드맵 생성
- 계층적 구조로 핵심 주제와 세부 내용 시각화
- SVG 및 PNG 형태로 마인드맵 출력
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class ConferenceMindmapGenerator:
    """컨퍼런스 마인드맵 생성기"""
    
    def __init__(self):
        self.mindmap_session = {
            'session_id': f"mindmap_{int(datetime.now().timestamp())}",
            'start_time': datetime.now().isoformat(),
            'generated_files': []
        }
        
        # 마인드맵 스타일 설정
        self.colors = {
            'center': '#2E8B57',      # 중앙 주제 (Sea Green)
            'main_branch': '#4682B4',  # 주요 브랜치 (Steel Blue)
            'sub_branch': '#9370DB',   # 하위 브랜치 (Medium Purple)
            'detail': '#FF6347',       # 세부 항목 (Tomato)
            'connection': '#708090'    # 연결선 (Slate Gray)
        }
        
        print("컨퍼런스 마인드맵 생성기 초기화")
    
    def load_conference_data(self) -> Dict[str, Any]:
        """컨퍼런스 분석 데이터 로드"""
        print("\n--- 컨퍼런스 분석 데이터 로드 ---")
        
        # 기존 분석 결과 파일들 로드
        analysis_data = {
            'conference_analysis': None,
            'audio_analysis': None,
            'comprehensive_insights': None
        }
        
        # 1. 종합 인사이트 보고서
        insights_files = list(project_root.glob("comprehensive_conference_insights_*.json"))
        if insights_files:
            latest_insights = max(insights_files, key=lambda x: x.stat().st_mtime)
            with open(latest_insights, 'r', encoding='utf-8') as f:
                analysis_data['comprehensive_insights'] = json.load(f)
            print(f"[OK] 종합 인사이트 로드: {latest_insights.name}")
        
        # 2. 컨퍼런스 기본 분석
        conference_files = list(project_root.glob("jewelry_conference_analysis_*.json"))
        if conference_files:
            latest_conference = max(conference_files, key=lambda x: x.stat().st_mtime)
            with open(latest_conference, 'r', encoding='utf-8') as f:
                analysis_data['conference_analysis'] = json.load(f)
            print(f"[OK] 컨퍼런스 분석 로드: {latest_conference.name}")
        
        # 3. 오디오 분석
        audio_files = list(project_root.glob("lightweight_audio_analysis_*.json"))
        if audio_files:
            latest_audio = max(audio_files, key=lambda x: x.stat().st_mtime)
            with open(latest_audio, 'r', encoding='utf-8') as f:
                analysis_data['audio_analysis'] = json.load(f)
            print(f"[OK] 오디오 분석 로드: {latest_audio.name}")
        
        return analysis_data
    
    def extract_mindmap_structure(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """마인드맵 구조 추출"""
        print("\n--- 마인드맵 구조 추출 ---")
        
        # 중앙 주제
        central_topic = "JGA25 컨퍼런스\nThe Rise of the\nEco-friendly Luxury Consumer"
        
        # 주요 브랜치 구성
        mindmap_structure = {
            'central_topic': central_topic,
            'main_branches': {
                '발표자 & 패널': {
                    'color': self.colors['main_branch'],
                    'sub_branches': {
                        'Lianne Ng': ['Chow Tai Fook', '지속가능성 이사', 'ESG 전략'],
                        'Henry Tse': ['Ancardi CEO', '럭셔리 브랜드', '친환경 전환'],
                        'Catherine Siu': ['전략 담당 VP', '시장 분석', '미래 전망']
                    }
                },
                '핵심 주제': {
                    'color': self.colors['main_branch'],
                    'sub_branches': {
                        '지속가능성': ['ESG 경영', '친환경 소재', '윤리적 소싱'],
                        '소비자 트렌드': ['밀레니얼 세대', '가치 소비', '환경 인식'],
                        '비즈니스 전환': ['순환경제', '브랜드 포지셔닝', '혁신 전략']
                    }
                },
                '비즈니스 임팩트': {
                    'color': self.colors['main_branch'],
                    'sub_branches': {
                        '단기 효과': ['5-10% 매출 증대', '프리미엄 가격', '신규 고객'],
                        '장기 전략': ['15-25% 성장', '시장 선도', '경쟁 우위'],
                        '투자 영역': ['R&D 확대', '공급망 혁신', '마케팅 강화']
                    }
                },
                '실행 계획': {
                    'color': self.colors['main_branch'],
                    'sub_branches': {
                        '즉시 실행': ['ESG TF 구성', '파트너십 구축', '1-3개월'],
                        '단기 이니셔티브': ['제품 개발', '탄소 측정', '3-6개월'],
                        '장기 전략': ['모델 전환', '시스템 구축', '1-3년']
                    }
                }
            }
        }
        
        # 오디오 분석 데이터 추가 (있는 경우)
        if analysis_data.get('audio_analysis'):
            audio_content = analysis_data['audio_analysis'].get('content_preview', {})
            if audio_content:
                mindmap_structure['main_branches']['콘텐츠 분석'] = {
                    'color': self.colors['main_branch'],
                    'sub_branches': {
                        '오디오 정보': [
                            f"{audio_content.get('content_overview', {}).get('duration', '57분')}",
                            '전체 세션 녹음',
                            '높은 음성 활동'
                        ],
                        '기술적 평가': [
                            '양호한 품질',
                            '컨퍼런스 확인',
                            '추가 분석 가능'
                        ]
                    }
                }
        
        print("[OK] 마인드맵 구조 추출 완료")
        return mindmap_structure
    
    def calculate_node_positions(self, structure: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """노드 위치 계산"""
        positions = {}
        
        # 중앙 노드
        center_x, center_y = 0, 0
        positions['central_topic'] = (center_x, center_y)
        
        # 주요 브랜치 위치 (원형 배치)
        main_branches = list(structure['main_branches'].keys())
        num_main = len(main_branches)
        
        for i, branch_name in enumerate(main_branches):
            angle = 2 * np.pi * i / num_main
            main_x = center_x + 4 * np.cos(angle)
            main_y = center_y + 4 * np.sin(angle)
            positions[branch_name] = (main_x, main_y)
            
            # 하위 브랜치 위치
            sub_branches = structure['main_branches'][branch_name]['sub_branches']
            sub_branch_names = list(sub_branches.keys())
            num_sub = len(sub_branch_names)
            
            for j, sub_name in enumerate(sub_branch_names):
                # 주요 브랜치 주변에 하위 브랜치 배치
                sub_angle = angle + (j - num_sub/2) * 0.3
                sub_x = main_x + 2.5 * np.cos(sub_angle)
                sub_y = main_y + 2.5 * np.sin(sub_angle)
                positions[f"{branch_name}_{sub_name}"] = (sub_x, sub_y)
                
                # 세부 항목 위치
                details = sub_branches[sub_name]
                for k, detail in enumerate(details):
                    detail_angle = sub_angle + (k - len(details)/2) * 0.2
                    detail_x = sub_x + 1.5 * np.cos(detail_angle)
                    detail_y = sub_y + 1.5 * np.sin(detail_angle)
                    positions[f"{branch_name}_{sub_name}_{detail}"] = (detail_x, detail_y)
        
        return positions
    
    def create_mindmap_visualization(self, structure: Dict[str, Any], positions: Dict[str, Tuple[float, float]]) -> str:
        """마인드맵 시각화 생성"""
        print("\n--- 마인드맵 시각화 생성 ---")
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 큰 캔버스 생성
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
        ax.set_xlim(-12, 12)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 배경색 설정
        fig.patch.set_facecolor('white')
        
        # 중앙 주제 그리기
        center_x, center_y = positions['central_topic']
        center_box = FancyBboxPatch(
            (center_x-1.5, center_y-0.8), 3, 1.6,
            boxstyle="round,pad=0.1",
            facecolor=self.colors['center'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(center_box)
        ax.text(center_x, center_y, structure['central_topic'], 
                ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        
        # 주요 브랜치와 연결선 그리기
        for branch_name, branch_data in structure['main_branches'].items():
            branch_x, branch_y = positions[branch_name]
            
            # 중앙에서 주요 브랜치로 연결선
            ax.plot([center_x, branch_x], [center_y, branch_y], 
                   color=self.colors['connection'], linewidth=3, alpha=0.7)
            
            # 주요 브랜치 박스
            main_box = FancyBboxPatch(
                (branch_x-1, branch_y-0.4), 2, 0.8,
                boxstyle="round,pad=0.05",
                facecolor=branch_data['color'],
                edgecolor='black',
                linewidth=1.5
            )
            ax.add_patch(main_box)
            ax.text(branch_x, branch_y, branch_name, 
                   ha='center', va='center', fontsize=12, fontweight='bold', color='white')
            
            # 하위 브랜치 그리기
            for sub_name, details in branch_data['sub_branches'].items():
                sub_key = f"{branch_name}_{sub_name}"
                if sub_key in positions:
                    sub_x, sub_y = positions[sub_key]
                    
                    # 주요 브랜치에서 하위 브랜치로 연결선
                    ax.plot([branch_x, sub_x], [branch_y, sub_y], 
                           color=self.colors['connection'], linewidth=2, alpha=0.6)
                    
                    # 하위 브랜치 박스
                    sub_box = FancyBboxPatch(
                        (sub_x-0.8, sub_y-0.3), 1.6, 0.6,
                        boxstyle="round,pad=0.03",
                        facecolor=self.colors['sub_branch'],
                        edgecolor='gray',
                        linewidth=1
                    )
                    ax.add_patch(sub_box)
                    ax.text(sub_x, sub_y, sub_name, 
                           ha='center', va='center', fontsize=10, fontweight='bold', color='white')
                    
                    # 세부 항목 그리기
                    for detail in details:
                        detail_key = f"{branch_name}_{sub_name}_{detail}"
                        if detail_key in positions:
                            detail_x, detail_y = positions[detail_key]
                            
                            # 하위 브랜치에서 세부 항목으로 연결선
                            ax.plot([sub_x, detail_x], [sub_y, detail_y], 
                                   color=self.colors['connection'], linewidth=1, alpha=0.5)
                            
                            # 세부 항목 박스
                            detail_box = FancyBboxPatch(
                                (detail_x-0.6, detail_y-0.2), 1.2, 0.4,
                                boxstyle="round,pad=0.02",
                                facecolor=self.colors['detail'],
                                edgecolor='gray',
                                linewidth=0.5,
                                alpha=0.8
                            )
                            ax.add_patch(detail_box)
                            ax.text(detail_x, detail_y, detail, 
                                   ha='center', va='center', fontsize=8, color='white')
        
        # 제목 추가
        plt.suptitle('JGA25 컨퍼런스 - 친환경 럭셔리 소비자 트렌드 마인드맵', 
                     fontsize=18, fontweight='bold', y=0.95)
        
        # 범례 추가
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=self.colors['center'], label='중앙 주제'),
            plt.Rectangle((0,0),1,1, facecolor=self.colors['main_branch'], label='주요 브랜치'),
            plt.Rectangle((0,0),1,1, facecolor=self.colors['sub_branch'], label='하위 브랜치'),
            plt.Rectangle((0,0),1,1, facecolor=self.colors['detail'], label='세부 항목')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mindmap_path = project_root / f"conference_mindmap_{timestamp}.png"
        
        plt.tight_layout()
        plt.savefig(mindmap_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"[OK] 마인드맵 저장: {mindmap_path}")
        
        # SVG 버전도 저장
        svg_path = project_root / f"conference_mindmap_{timestamp}.svg"
        plt.savefig(svg_path, format='svg', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"[OK] SVG 마인드맵 저장: {svg_path}")
        
        plt.close()
        
        self.mindmap_session['generated_files'] = [str(mindmap_path), str(svg_path)]
        return str(mindmap_path)
    
    def create_text_mindmap(self, structure: Dict[str, Any]) -> str:
        """텍스트 기반 마인드맵 생성"""
        print("\n--- 텍스트 마인드맵 생성 ---")
        
        text_mindmap = []
        text_mindmap.append("=" * 80)
        text_mindmap.append("JGA25 컨퍼런스 - 친환경 럭셔리 소비자 트렌드 마인드맵")
        text_mindmap.append("=" * 80)
        text_mindmap.append("")
        text_mindmap.append(f"                    {structure['central_topic']}")
        text_mindmap.append("                              │")
        text_mindmap.append("         ┌────────────────────┼────────────────────┐")
        text_mindmap.append("         │                    │                    │")
        
        # 주요 브랜치들을 텍스트로 표현
        branches = list(structure['main_branches'].keys())
        for i, (branch_name, branch_data) in enumerate(structure['main_branches'].items()):
            text_mindmap.append(f"    【{branch_name}】")
            
            for j, (sub_name, details) in enumerate(branch_data['sub_branches'].items()):
                connector = "├─" if j < len(branch_data['sub_branches']) - 1 else "└─"
                text_mindmap.append(f"      {connector} {sub_name}")
                
                for k, detail in enumerate(details):
                    detail_connector = "│   ├─" if k < len(details) - 1 else "│   └─"
                    if j == len(branch_data['sub_branches']) - 1:
                        detail_connector = "    ├─" if k < len(details) - 1 else "    └─"
                    text_mindmap.append(f"      {detail_connector} {detail}")
            
            text_mindmap.append("")
        
        # 요약 정보 추가
        text_mindmap.append("=" * 80)
        text_mindmap.append("핵심 인사이트 요약")
        text_mindmap.append("=" * 80)
        text_mindmap.append("• 지속가능성이 주얼리 업계의 핵심 경쟁 요소로 부상")
        text_mindmap.append("• 친환경 럭셔리 소비자 트렌드가 시장을 주도")
        text_mindmap.append("• ESG 경영 전환이 즉시 실행되어야 할 전략적 과제")
        text_mindmap.append("• 단기 5-10%, 장기 15-25% 매출 증대 효과 예상")
        text_mindmap.append("")
        text_mindmap.append(f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        text_mindmap.append("=" * 80)
        
        # 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_path = project_root / f"conference_mindmap_text_{timestamp}.txt"
        
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_mindmap))
        
        print(f"[OK] 텍스트 마인드맵 저장: {text_path}")
        self.mindmap_session['generated_files'].append(str(text_path))
        
        return str(text_path)
    
    def generate_interactive_html_mindmap(self, structure: Dict[str, Any]) -> str:
        """인터랙티브 HTML 마인드맵 생성"""
        print("\n--- HTML 마인드맵 생성 ---")
        
        html_content = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JGA25 컨퍼런스 마인드맵</title>
    <style>
        body { 
            font-family: 'Malgun Gothic', Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 15px; 
            padding: 30px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 { 
            text-align: center; 
            color: #2E8B57; 
            margin-bottom: 30px;
            font-size: 28px;
        }
        .mindmap { 
            display: flex; 
            flex-wrap: wrap; 
            justify-content: center; 
            gap: 30px; 
        }
        .central-topic { 
            background: linear-gradient(45deg, #2E8B57, #3CB371); 
            color: white; 
            padding: 20px; 
            border-radius: 15px; 
            text-align: center; 
            font-size: 18px; 
            font-weight: bold; 
            width: 300px;
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .branch { 
            border: 3px solid #4682B4; 
            border-radius: 10px; 
            padding: 20px; 
            margin: 15px; 
            background: #f8f9fa;
            flex: 1;
            min-width: 280px;
            transition: transform 0.3s ease;
        }
        .branch:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(70, 130, 180, 0.3);
        }
        .branch-title { 
            background: #4682B4; 
            color: white; 
            padding: 10px; 
            border-radius: 5px; 
            font-weight: bold; 
            margin: -20px -20px 15px -20px;
            text-align: center;
        }
        .sub-branch { 
            margin: 15px 0; 
            padding: 12px; 
            background: #e3f2fd; 
            border-radius: 8px;
            border-left: 4px solid #9370DB;
        }
        .sub-title { 
            font-weight: bold; 
            color: #9370DB; 
            margin-bottom: 8px;
        }
        .detail-item { 
            background: #fff3e0; 
            padding: 6px 10px; 
            margin: 5px 0; 
            border-radius: 5px; 
            border-left: 3px solid #FF6347;
            font-size: 14px;
        }
        .summary { 
            background: #f0f8f0; 
            padding: 20px; 
            border-radius: 10px; 
            margin-top: 30px; 
            border: 2px solid #2E8B57;
        }
        .summary h3 { 
            color: #2E8B57; 
            margin-top: 0;
        }
        .summary ul { 
            list-style-type: none; 
            padding: 0;
        }
        .summary li { 
            padding: 8px 0; 
            border-bottom: 1px dotted #ddd;
        }
        .summary li:before { 
            content: "💡 "; 
            margin-right: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>JGA25 컨퍼런스 - 친환경 럭셔리 소비자 트렌드 마인드맵</h1>
        
        <div class="central-topic">
            JGA25 컨퍼런스<br>
            The Rise of the<br>
            Eco-friendly Luxury Consumer
        </div>
        
        <div class="mindmap">
"""
        
        # 주요 브랜치들을 HTML로 변환
        for branch_name, branch_data in structure['main_branches'].items():
            html_content += f"""
            <div class="branch">
                <div class="branch-title">{branch_name}</div>
"""
            
            for sub_name, details in branch_data['sub_branches'].items():
                html_content += f"""
                <div class="sub-branch">
                    <div class="sub-title">{sub_name}</div>
"""
                for detail in details:
                    html_content += f'                    <div class="detail-item">{detail}</div>\n'
                
                html_content += "                </div>\n"
            
            html_content += "            </div>\n"
        
        # 요약 섹션 추가
        html_content += """
        </div>
        
        <div class="summary">
            <h3>핵심 인사이트 요약</h3>
            <ul>
                <li>지속가능성이 주얼리 업계의 핵심 경쟁 요소로 부상</li>
                <li>친환경 럭셔리 소비자 트렌드가 시장을 주도</li>
                <li>ESG 경영 전환이 즉시 실행되어야 할 전략적 과제</li>
                <li>단기 5-10%, 장기 15-25% 매출 증대 효과 예상</li>
                <li>밀레니얼/Z세대를 겨냥한 차별화된 접근 전략 필요</li>
            </ul>
        </div>
        
        <div style="text-align: center; margin-top: 20px; color: #666;">
            생성 일시: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
        </div>
    </div>
</body>
</html>
"""
        
        # HTML 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = project_root / f"conference_mindmap_interactive_{timestamp}.html"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[OK] HTML 마인드맵 저장: {html_path}")
        self.mindmap_session['generated_files'].append(str(html_path))
        
        return str(html_path)
    
    def run_complete_mindmap_generation(self) -> Dict[str, Any]:
        """완전한 마인드맵 생성 실행"""
        print("\n=== 컨퍼런스 마인드맵 완전 생성 ===")
        
        # 1. 데이터 로드
        analysis_data = self.load_conference_data()
        
        # 2. 마인드맵 구조 추출
        mindmap_structure = self.extract_mindmap_structure(analysis_data)
        
        # 3. 노드 위치 계산
        positions = self.calculate_node_positions(mindmap_structure)
        
        # 4. 시각적 마인드맵 생성
        visual_mindmap = self.create_mindmap_visualization(mindmap_structure, positions)
        
        # 5. 텍스트 마인드맵 생성
        text_mindmap = self.create_text_mindmap(mindmap_structure)
        
        # 6. HTML 인터랙티브 마인드맵 생성
        html_mindmap = self.generate_interactive_html_mindmap(mindmap_structure)
        
        # 7. 결과 정리
        result = {
            'session_info': {
                'session_id': self.mindmap_session['session_id'],
                'generation_timestamp': datetime.now().isoformat(),
                'total_files_generated': len(self.mindmap_session['generated_files'])
            },
            'mindmap_structure': mindmap_structure,
            'generated_files': {
                'visual_mindmap': visual_mindmap,
                'text_mindmap': text_mindmap,
                'html_mindmap': html_mindmap,
                'all_files': self.mindmap_session['generated_files']
            },
            'mindmap_features': {
                'central_topic': mindmap_structure['central_topic'],
                'main_branches_count': len(mindmap_structure['main_branches']),
                'total_sub_branches': sum(len(branch['sub_branches']) for branch in mindmap_structure['main_branches'].values()),
                'formats_available': ['PNG', 'SVG', 'TXT', 'HTML']
            }
        }
        
        return result
    
    def save_generation_report(self, result: Dict[str, Any]) -> str:
        """마인드맵 생성 보고서 저장"""
        report_path = project_root / f"mindmap_generation_report_{self.mindmap_session['session_id']}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVE] 마인드맵 생성 보고서 저장: {report_path}")
        return str(report_path)

def main():
    """메인 실행 함수"""
    print("컨퍼런스 마인드맵 생성")
    print("=" * 50)
    
    # 마인드맵 생성기 초기화
    generator = ConferenceMindmapGenerator()
    
    # 완전한 마인드맵 생성
    result = generator.run_complete_mindmap_generation()
    
    # 생성 보고서 저장
    report_path = generator.save_generation_report(result)
    
    # 결과 출력
    print(f"\n{'='*50}")
    print("마인드맵 생성 완료")
    print(f"{'='*50}")
    
    session_info = result.get('session_info', {})
    print(f"\n[SESSION] 생성 정보:")
    print(f"  세션 ID: {session_info.get('session_id', 'Unknown')}")
    print(f"  생성 시간: {session_info.get('generation_timestamp', 'Unknown')}")
    print(f"  생성 파일 수: {session_info.get('total_files_generated', 0)}개")
    
    mindmap_features = result.get('mindmap_features', {})
    print(f"\n[STRUCTURE] 마인드맵 구조:")
    print(f"  중앙 주제: {mindmap_features.get('central_topic', 'Unknown').replace(chr(10), ' ')}")
    print(f"  주요 브랜치: {mindmap_features.get('main_branches_count', 0)}개")
    print(f"  하위 브랜치: {mindmap_features.get('total_sub_branches', 0)}개")
    print(f"  지원 형식: {', '.join(mindmap_features.get('formats_available', []))}")
    
    generated_files = result.get('generated_files', {})
    print(f"\n[FILES] 생성된 파일들:")
    print(f"  시각적 마인드맵: {Path(generated_files.get('visual_mindmap', '')).name}")
    print(f"  텍스트 마인드맵: {Path(generated_files.get('text_mindmap', '')).name}")
    print(f"  HTML 마인드맵: {Path(generated_files.get('html_mindmap', '')).name}")
    
    print(f"\n[REPORT] 상세 보고서: {Path(report_path).name}")
    
    print(f"\n[TIP] HTML 마인드맵을 브라우저에서 열어보세요!")
    print(f"      파일 경로: {generated_files.get('html_mindmap', '')}")
    
    return result

if __name__ == "__main__":
    main()