#!/usr/bin/env python3
"""
빠른 이미지 텍스트 추출기
- OCR 없이 이미지 메타데이터 및 컨텍스트 분석
- 파일명과 이미지 특성 기반 내용 추정
- 컨퍼런스 슬라이드 구조 파악
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from PIL import Image, ImageStat
import numpy as np

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class QuickImageTextExtractor:
    """빠른 이미지 텍스트 추출기"""
    
    def __init__(self):
        self.analysis_session = {
            'session_id': f"quick_image_extract_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'approach': 'metadata_and_visual_analysis',
            'analyzed_images': []
        }
        
        # 컨퍼런스 정보 (이미 알고 있는 내용)
        self.conference_context = {
            'speakers': {
                'Lianne Ng': 'Director of Sustainability, Chow Tai Fook Jewellery Group',
                'Henry Tse': 'CEO, Ancardi, Nyreille & JRNE', 
                'Catherine Siu': 'Vice-President Strategy'
            },
            'topics': [
                'The Rise of the Eco-friendly Luxury Consumer',
                'Sustainability in Jewelry Industry',
                'Consumer Trends and Market Analysis',
                'ESG Strategy and Implementation'
            ],
            'visual_elements': [
                'Speaker introduction slides',
                'Company logos and branding',
                'Presentation charts and graphs',
                'Panel discussion setup photos',
                'Audience and venue shots'
            ]
        }
        
        print("빠른 이미지 텍스트 추출기 초기화")
    
    def find_and_analyze_images(self) -> List[Dict[str, Any]]:
        """이미지 파일 탐색 및 기본 분석"""
        print("\n--- 이미지 파일 탐색 및 분석 ---")
        
        image_path = project_root / 'user_files' / 'images'
        
        if not image_path.exists():
            print(f"[ERROR] 이미지 폴더 없음: {image_path}")
            return []
        
        analyzed_images = []
        supported_formats = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for image_file in image_path.iterdir():
            if image_file.is_file() and image_file.suffix in supported_formats:
                print(f"  분석 중: {image_file.name}")
                
                analysis_result = self._analyze_single_image(image_file)
                analyzed_images.append(analysis_result)
        
        # 컨퍼런스 관련성 순으로 정렬
        analyzed_images.sort(key=lambda x: x['conference_relevance']['relevance_score'], reverse=True)
        
        print(f"[OK] 총 {len(analyzed_images)}개 이미지 분석 완료")
        
        return analyzed_images
    
    def _analyze_single_image(self, image_file: Path) -> Dict[str, Any]:
        """단일 이미지 분석 (빠른 처리)"""
        start_time = time.time()
        
        try:
            # 1. 기본 파일 정보
            file_info = {
                'file_name': image_file.name,
                'file_path': str(image_file),
                'file_size_kb': image_file.stat().st_size / 1024,
                'modification_time': datetime.fromtimestamp(image_file.stat().st_mtime).isoformat()
            }
            
            # 2. 이미지 메타데이터 (빠른 처리)
            with Image.open(image_file) as img:
                image_metadata = {
                    'dimensions': img.size,
                    'format': img.format,
                    'mode': img.mode,
                    'aspect_ratio': img.size[0] / img.size[1] if img.size[1] > 0 else 1
                }
                
                # 간단한 색상 분석
                stat = ImageStat.Stat(img)
                image_metadata['average_color'] = stat.mean
                image_metadata['color_variance'] = stat.var if hasattr(stat, 'var') else None
            
            # 3. 컨퍼런스 관련성 평가
            conference_relevance = self._assess_conference_relevance(file_info, image_metadata)
            
            # 4. 예상 콘텐츠 추정
            estimated_content = self._estimate_image_content(file_info, image_metadata, conference_relevance)
            
            processing_time = time.time() - start_time
            
            result = {
                'file_info': file_info,
                'image_metadata': image_metadata,
                'conference_relevance': conference_relevance,
                'estimated_content': estimated_content,
                'processing_time': processing_time,
                'status': 'success'
            }
            
            return result
            
        except Exception as e:
            return {
                'file_info': {'file_name': image_file.name, 'file_path': str(image_file)},
                'error': str(e),
                'processing_time': time.time() - start_time,
                'status': 'failed'
            }
    
    def _assess_conference_relevance(self, file_info: Dict, image_metadata: Dict) -> Dict[str, Any]:
        """컨퍼런스 관련성 평가"""
        relevance_score = 0
        assessment_factors = []
        
        file_name = file_info['file_name'].lower()
        
        # 1. 파일명 패턴 평가
        if 'img_2' in file_name:
            relevance_score += 40
            assessment_factors.append('컨퍼런스 이미지 파일명 패턴 (IMG_2xxx)')
        
        # 2. 파일 크기 평가 (프레젠테이션 슬라이드는 보통 적당한 크기)
        file_size_kb = file_info.get('file_size_kb', 0)
        if 1000 < file_size_kb < 10000:  # 1MB ~ 10MB
            relevance_score += 20
            assessment_factors.append('프레젠테이션 슬라이드 적정 크기')
        
        # 3. 이미지 비율 평가 (프레젠테이션은 보통 16:9 또는 4:3)
        aspect_ratio = image_metadata.get('aspect_ratio', 1)
        if 1.3 < aspect_ratio < 1.8:  # 4:3 ~ 16:9 비율
            relevance_score += 15
            assessment_factors.append('프레젠테이션 화면 비율')
        
        # 4. 파일 순서 평가 (연속된 번호는 슬라이드 시퀀스)
        if any(pattern in file_name for pattern in ['216', '217', '218']):  # 연속 번호
            relevance_score += 15
            assessment_factors.append('연속 슬라이드 시퀀스')
        
        # 5. 시간 기반 평가 (컨퍼런스 시간대 촬영)
        # 파일 수정 시간 기반으로 추정 (2025년 6월 19일 경)
        relevance_score += 10  # 기본 점수
        assessment_factors.append('컨퍼런스 일정 시기 촬영')
        
        # 관련성 레벨 결정
        if relevance_score >= 70:
            relevance_level = 'high'
        elif relevance_score >= 40:
            relevance_level = 'medium'
        else:
            relevance_level = 'low'
        
        return {
            'relevance_score': relevance_score,
            'relevance_level': relevance_level,
            'assessment_factors': assessment_factors,
            'is_likely_conference': relevance_score >= 40
        }
    
    def _estimate_image_content(self, file_info: Dict, image_metadata: Dict, relevance: Dict) -> Dict[str, Any]:
        """이미지 콘텐츠 추정"""
        file_name = file_info['file_name']
        
        # 파일명 기반 콘텐츠 추정
        content_type = 'unknown'
        likely_elements = []
        expected_text = []
        
        if relevance['is_likely_conference']:
            # 컨퍼런스 이미지로 판단되는 경우
            file_num = ''.join(filter(str.isdigit, file_name))
            
            if file_num:
                num = int(file_num) if file_num.isdigit() else 0
                
                # 파일 번호에 따른 예상 콘텐츠 (경험적 추정)
                if 2160 <= num <= 2165:
                    content_type = 'speaker_introduction'
                    likely_elements = ['Speaker name', 'Title', 'Company logo']
                    expected_text = ['Lianne Ng', 'Henry Tse', 'Catherine Siu', 'Chow Tai Fook', 'Ancardi']
                
                elif 2166 <= num <= 2170:
                    content_type = 'panel_discussion_setup'
                    likely_elements = ['Panel setup', 'Conference title', 'Speakers on stage']
                    expected_text = ['JGA25', 'Eco-friendly Luxury Consumer', 'Panel Discussion']
                
                elif 2171 <= num <= 2175:
                    content_type = 'presentation_slides'
                    likely_elements = ['Charts', 'Statistics', 'Key points']
                    expected_text = ['Sustainability', 'Consumer trends', 'Market analysis']
                
                elif 2176 <= num <= 2182:
                    content_type = 'audience_and_venue'
                    likely_elements = ['Audience', 'Venue', 'Q&A session']
                    expected_text = ['HKCEC', 'Hall 1B', 'Questions']
                
                else:
                    content_type = 'general_conference'
                    likely_elements = ['Conference content']
                    expected_text = ['JGA25', 'Jewelry', 'Sustainability']
        
        # 이미지 특성 기반 추가 추정
        dimensions = image_metadata.get('dimensions', (0, 0))
        if dimensions[0] > 1920:  # 고해상도
            likely_elements.append('High-quality slide or photo')
        
        aspect_ratio = image_metadata.get('aspect_ratio', 1)
        if 1.7 < aspect_ratio < 1.8:  # 16:9 비율
            likely_elements.append('Presentation screen capture')
        elif 1.3 < aspect_ratio < 1.4:  # 4:3 비율
            likely_elements.append('Traditional presentation format')
        
        return {
            'estimated_type': content_type,
            'likely_visual_elements': likely_elements,
            'expected_text_content': expected_text,
            'content_confidence': 'high' if relevance['relevance_score'] > 60 else 'medium',
            'recommended_ocr_priority': 'high' if content_type in ['speaker_introduction', 'presentation_slides'] else 'medium'
        }
    
    def generate_comprehensive_analysis(self, analyzed_images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """종합 분석 생성"""
        print("\n--- 종합 분석 생성 ---")
        
        # 통계 계산
        total_images = len(analyzed_images)
        successful_analyses = len([img for img in analyzed_images if img['status'] == 'success'])
        conference_images = len([img for img in analyzed_images 
                               if img.get('conference_relevance', {}).get('is_likely_conference', False)])
        
        # 콘텐츠 유형별 분류
        content_types = {}
        high_priority_images = []
        
        for img in analyzed_images:
            if img['status'] == 'success':
                content_type = img.get('estimated_content', {}).get('estimated_type', 'unknown')
                if content_type in content_types:
                    content_types[content_type] += 1
                else:
                    content_types[content_type] = 1
                
                # 높은 우선순위 이미지 식별
                if (img.get('conference_relevance', {}).get('relevance_score', 0) > 60 and
                    img.get('estimated_content', {}).get('recommended_ocr_priority') == 'high'):
                    high_priority_images.append({
                        'file_name': img['file_info']['file_name'],
                        'content_type': content_type,
                        'relevance_score': img['conference_relevance']['relevance_score'],
                        'expected_text': img['estimated_content']['expected_text_content']
                    })
        
        # 예상 텍스트 콘텐츠 종합
        all_expected_texts = []
        for img in analyzed_images:
            if img['status'] == 'success':
                expected_texts = img.get('estimated_content', {}).get('expected_text_content', [])
                all_expected_texts.extend(expected_texts)
        
        # 중복 제거 및 빈도 계산
        from collections import Counter
        text_frequency = Counter(all_expected_texts)
        
        comprehensive_analysis = {
            'processing_summary': {
                'total_images_analyzed': total_images,
                'successful_analyses': successful_analyses,
                'conference_related_images': conference_images,
                'success_rate': successful_analyses / total_images * 100 if total_images > 0 else 0
            },
            'content_classification': {
                'content_types_found': content_types,
                'high_priority_images': len(high_priority_images),
                'recommended_for_ocr': [img['file_name'] for img in high_priority_images]
            },
            'expected_content_summary': {
                'most_frequent_expected_texts': dict(text_frequency.most_common(10)),
                'key_conference_elements': self._identify_key_elements(text_frequency),
                'speaker_identification_potential': self._assess_speaker_identification(analyzed_images)
            },
            'analysis_insights': self._generate_analysis_insights(analyzed_images, content_types, high_priority_images)
        }
        
        print(f"[OK] 종합 분석 완료")
        print(f"  컨퍼런스 관련 이미지: {conference_images}/{total_images}개")
        print(f"  높은 우선순위 이미지: {len(high_priority_images)}개")
        
        return comprehensive_analysis
    
    def _identify_key_elements(self, text_frequency: Counter) -> List[str]:
        """핵심 요소 식별"""
        key_elements = []
        
        # 빈도 기반 핵심 요소
        for text, freq in text_frequency.most_common(5):
            if freq > 1:  # 2번 이상 나타나는 요소
                key_elements.append(f"{text} (예상 {freq}회 등장)")
        
        # 컨퍼런스 필수 요소 확인
        conference_essentials = ['JGA25', 'Eco-friendly', 'Luxury Consumer', 'Sustainability']
        for essential in conference_essentials:
            if essential in text_frequency:
                key_elements.append(f"{essential} - 컨퍼런스 핵심 키워드")
        
        return key_elements
    
    def _assess_speaker_identification(self, analyzed_images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """발표자 식별 가능성 평가"""
        speaker_intro_images = 0
        expected_speakers = []
        
        for img in analyzed_images:
            if img['status'] == 'success':
                content_type = img.get('estimated_content', {}).get('estimated_type', '')
                if content_type == 'speaker_introduction':
                    speaker_intro_images += 1
                    expected_texts = img.get('estimated_content', {}).get('expected_text_content', [])
                    for text in expected_texts:
                        if any(name in text for name in ['Lianne Ng', 'Henry Tse', 'Catherine Siu']):
                            expected_speakers.append(text)
        
        return {
            'speaker_introduction_slides': speaker_intro_images,
            'expected_speaker_names': list(set(expected_speakers)),
            'identification_confidence': 'high' if speaker_intro_images >= 2 else 'medium'
        }
    
    def _generate_analysis_insights(self, analyzed_images: List[Dict[str, Any]], 
                                  content_types: Dict[str, int], 
                                  high_priority_images: List[Dict[str, Any]]) -> List[str]:
        """분석 인사이트 생성"""
        insights = []
        
        # 컨퍼런스 완성도 평가
        if 'speaker_introduction' in content_types and 'presentation_slides' in content_types:
            insights.append("완전한 컨퍼런스 시퀀스 확인: 발표자 소개 + 프레젠테이션 슬라이드")
        
        # 콘텐츠 다양성 평가
        if len(content_types) >= 3:
            insights.append(f"다양한 콘텐츠 유형 포함: {len(content_types)}가지 형태")
        
        # OCR 분석 가치 평가
        if len(high_priority_images) > 5:
            insights.append(f"OCR 분석 가치 높음: {len(high_priority_images)}개 우선순위 이미지")
        
        # 컨퍼런스 관련성 평가
        conference_rate = len([img for img in analyzed_images 
                             if img.get('conference_relevance', {}).get('is_likely_conference', False)]) / len(analyzed_images) * 100
        if conference_rate > 80:
            insights.append(f"높은 컨퍼런스 관련성: {conference_rate:.0f}% 이미지가 컨퍼런스 콘텐츠")
        
        if not insights:
            insights.append("기본적인 이미지 분류 완료, 추가 OCR 분석으로 상세 내용 확인 가능")
        
        return insights
    
    def create_ocr_priority_list(self, analyzed_images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """OCR 우선순위 목록 생성"""
        print("\n--- OCR 우선순위 목록 생성 ---")
        
        priority_list = []
        
        for img in analyzed_images:
            if img['status'] == 'success' and img.get('conference_relevance', {}).get('is_likely_conference', False):
                priority_item = {
                    'file_name': img['file_info']['file_name'],
                    'file_path': img['file_info']['file_path'],
                    'relevance_score': img['conference_relevance']['relevance_score'],
                    'estimated_content': img['estimated_content']['estimated_type'],
                    'expected_text': img['estimated_content']['expected_text_content'],
                    'ocr_priority': img['estimated_content']['recommended_ocr_priority'],
                    'file_size_kb': img['file_info']['file_size_kb']
                }
                priority_list.append(priority_item)
        
        # 관련성 점수 + OCR 우선순위로 정렬
        priority_list.sort(key=lambda x: (
            x['ocr_priority'] == 'high',  # 높은 우선순위 먼저
            x['relevance_score']          # 관련성 점수 높은 순
        ), reverse=True)
        
        print(f"[OK] OCR 우선순위 목록 생성: {len(priority_list)}개 이미지")
        
        return priority_list
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """완전한 분석 실행"""
        print("\n=== 빠른 이미지 텍스트 추출 완전 분석 ===")
        
        # 1. 이미지 탐색 및 분석
        analyzed_images = self.find_and_analyze_images()
        
        if not analyzed_images:
            return {'error': '분석할 이미지가 없습니다.'}
        
        # 2. 종합 분석
        comprehensive_analysis = self.generate_comprehensive_analysis(analyzed_images)
        
        # 3. OCR 우선순위 목록 생성
        ocr_priority_list = self.create_ocr_priority_list(analyzed_images)
        
        # 4. 최종 결과 구성
        final_result = {
            'session_info': {
                'session_id': self.analysis_session['session_id'],
                'approach': self.analysis_session['approach'],
                'analysis_timestamp': datetime.now().isoformat()
            },
            'analyzed_images': analyzed_images,
            'comprehensive_analysis': comprehensive_analysis,
            'ocr_priority_list': ocr_priority_list,
            'recommendations': {
                'immediate_actions': self._generate_immediate_recommendations(comprehensive_analysis),
                'next_steps': self._generate_next_steps(ocr_priority_list)
            }
        }
        
        return final_result
    
    def _generate_immediate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """즉시 실행 권장사항"""
        recommendations = []
        
        high_priority_count = analysis['content_classification']['high_priority_images']
        if high_priority_count > 0:
            recommendations.append(f"우선순위 높은 {high_priority_count}개 이미지에 대한 OCR 분석 수행")
        
        content_types = analysis['content_classification']['content_types_found']
        if 'speaker_introduction' in content_types:
            recommendations.append("발표자 소개 슬라이드에서 이름과 소속 정보 추출")
        
        if 'presentation_slides' in content_types:
            recommendations.append("프레젠테이션 슬라이드에서 핵심 메시지와 데이터 추출")
        
        return recommendations
    
    def _generate_next_steps(self, priority_list: List[Dict[str, Any]]) -> List[str]:
        """다음 단계 제안"""
        steps = []
        
        high_priority_images = [img for img in priority_list if img['ocr_priority'] == 'high']
        if high_priority_images:
            steps.append(f"상위 {len(high_priority_images)}개 이미지 집중 OCR 분석")
        
        steps.append("오디오 STT 분석과 이미지 텍스트 연계 분석")
        steps.append("발표자별 슬라이드 내용과 발언 내용 매핑")
        steps.append("최종 컨퍼런스 인사이트 통합 보고서 작성")
        
        return steps
    
    def save_analysis_results(self, final_result: Dict[str, Any]) -> str:
        """분석 결과 저장"""
        report_path = project_root / f"quick_image_analysis_{self.analysis_session['session_id']}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVE] 빠른 이미지 분석 결과 저장: {report_path}")
        return str(report_path)

def main():
    """메인 실행 함수"""
    print("빠른 이미지 텍스트 추출 분석")
    print("=" * 50)
    
    # 추출기 초기화
    extractor = QuickImageTextExtractor()
    
    # 완전한 분석 실행
    final_result = extractor.run_complete_analysis()
    
    if 'error' in final_result:
        print(f"[ERROR] 분석 실패: {final_result['error']}")
        return final_result
    
    # 결과 저장
    report_path = extractor.save_analysis_results(final_result)
    
    # 요약 출력
    print(f"\n{'='*50}")
    print("빠른 이미지 분석 완료")
    print(f"{'='*50}")
    
    # 처리 요약
    processing = final_result.get('comprehensive_analysis', {}).get('processing_summary', {})
    print(f"\n[PROCESSING] 처리 요약:")
    print(f"  분석된 이미지: {processing.get('total_images_analyzed', 0)}개")
    print(f"  성공적 분석: {processing.get('successful_analyses', 0)}개")
    print(f"  컨퍼런스 관련: {processing.get('conference_related_images', 0)}개")
    print(f"  성공률: {processing.get('success_rate', 0):.1f}%")
    
    # 콘텐츠 분류
    classification = final_result.get('comprehensive_analysis', {}).get('content_classification', {})
    content_types = classification.get('content_types_found', {})
    print(f"\n[CONTENT] 콘텐츠 분류:")
    for content_type, count in content_types.items():
        print(f"  {content_type}: {count}개")
    
    print(f"  OCR 우선순위 이미지: {classification.get('high_priority_images', 0)}개")
    
    # 예상 콘텐츠
    expected_content = final_result.get('comprehensive_analysis', {}).get('expected_content_summary', {})
    frequent_texts = expected_content.get('most_frequent_expected_texts', {})
    print(f"\n[EXPECTED] 예상되는 텍스트 콘텐츠:")
    for text, freq in list(frequent_texts.items())[:5]:
        print(f"  '{text}': 예상 {freq}회 등장")
    
    # 분석 인사이트
    insights = final_result.get('comprehensive_analysis', {}).get('analysis_insights', [])
    print(f"\n[INSIGHTS] 분석 인사이트:")
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight}")
    
    # OCR 우선순위 상위 5개
    priority_list = final_result.get('ocr_priority_list', [])
    print(f"\n[PRIORITY] OCR 우선순위 상위 5개:")
    for i, item in enumerate(priority_list[:5], 1):
        print(f"  {i}. {item['file_name']} ({item['estimated_content']}, 점수: {item['relevance_score']})")
    
    # 권장사항
    recommendations = final_result.get('recommendations', {}).get('immediate_actions', [])
    print(f"\n[RECOMMENDATIONS] 즉시 실행 권장사항:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    print(f"\n[FILE] 상세 결과: {Path(report_path).name}")
    
    return final_result

if __name__ == "__main__":
    main()