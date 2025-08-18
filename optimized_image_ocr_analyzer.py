#!/usr/bin/env python3
"""
최적화된 이미지 OCR 분석기
- 타임아웃 방지를 위한 배치 처리
- 메모리 최적화 및 빠른 OCR 수행
- 컨퍼런스 키워드 중심 분석
"""

import os
import sys
import time
import json
import easyocr
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from PIL import Image
import numpy as np

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class OptimizedImageOCRAnalyzer:
    """최적화된 이미지 OCR 분석기"""
    
    def __init__(self):
        self.analysis_session = {
            'session_id': f"optimized_ocr_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'processed_images': [],
            'failed_images': [],
            'extracted_texts': {},
            'conference_keywords': {}
        }
        
        # 컨퍼런스 키워드 (이미 알고 있는 정보)
        self.known_keywords = {
            'speakers': [
                'lianne ng', 'henry tse', 'catherine siu', 'pui in catherine siu',
                'chow tai fook', 'ancardi', 'nyreille', 'jrne'
            ],
            'companies': [
                'chow tai fook', 'ancardi', 'nyreille', 'jrne', 'jga', 'hkcec'
            ],
            'topics': [
                'eco-friendly', 'luxury', 'consumer', 'sustainability', 'sustainable',
                'jewelry', 'jewellery', 'environment', 'green', 'ethical'
            ],
            'event_info': [
                'jga25', 'connecting', 'jewellery world', 'stage', 'hall 1b',
                '2:30pm', '3:30pm', '19/6/2025', 'thursday'
            ]
        }
        
        print("최적화된 이미지 OCR 분석기 초기화")
        self._initialize_ocr_reader()
    
    def _initialize_ocr_reader(self):
        """OCR 리더 초기화 (메모리 최적화)"""
        print("\n--- OCR 리더 초기화 ---")
        
        try:
            print("EasyOCR 리더 로딩 중... (한/영 모델)")
            # GPU 사용 안함, 메모리 절약 모드
            self.ocr_reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
            print("[OK] EasyOCR 리더 초기화 완료")
            return True
        except Exception as e:
            print(f"[ERROR] OCR 리더 초기화 실패: {e}")
            self.ocr_reader = None
            return False
    
    def find_conference_images(self) -> List[Dict[str, Any]]:
        """컨퍼런스 이미지 파일 탐색"""
        print("\n--- 컨퍼런스 이미지 탐색 ---")
        
        image_path = project_root / 'user_files' / 'images'
        
        if not image_path.exists():
            print(f"[ERROR] 이미지 폴더 없음: {image_path}")
            return []
        
        image_files = []
        supported_formats = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for image_file in image_path.iterdir():
            if image_file.is_file() and image_file.suffix in supported_formats:
                file_info = {
                    'file_path': str(image_file),
                    'file_name': image_file.name,
                    'file_size_kb': image_file.stat().st_size / 1024,
                    'is_conference_image': 'IMG_2' in image_file.name,  # 컨퍼런스 이미지 패턴
                    'priority': 'high' if 'IMG_2' in image_file.name else 'low'
                }
                image_files.append(file_info)
        
        # 컨퍼런스 이미지 우선, 파일명 순 정렬
        image_files.sort(key=lambda x: (not x['is_conference_image'], x['file_name']))
        
        print(f"[OK] 총 {len(image_files)}개 이미지 발견")
        
        conference_images = [img for img in image_files if img['is_conference_image']]
        print(f"[OK] 컨퍼런스 이미지: {len(conference_images)}개")
        
        return image_files
    
    def process_single_image(self, image_info: Dict[str, Any], batch_index: int, total_batches: int) -> Dict[str, Any]:
        """단일 이미지 OCR 처리 (최적화)"""
        file_name = image_info['file_name']
        file_path = image_info['file_path']
        
        print(f"  [{batch_index}/{total_batches}] {file_name} 처리 중...")
        
        start_time = time.time()
        
        try:
            # 1. 이미지 로드 및 전처리 (메모리 절약)
            with Image.open(file_path) as image:
                # 크기 최적화 (OCR 정확도 유지하면서 속도 향상)
                max_dimension = 1600  # 이전 2048에서 축소
                if max(image.size) > max_dimension:
                    ratio = max_dimension / max(image.size)
                    new_size = tuple(int(dim * ratio) for dim in image.size)
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                # numpy 배열로 변환
                image_array = np.array(image)
            
            # 2. EasyOCR 수행 (빠른 설정)
            ocr_results = self.ocr_reader.readtext(
                image_array, 
                width_ths=0.8,    # 더 빠른 처리를 위해 조정
                height_ths=0.8,
                paragraph=False,   # 단락 통합 비활성화로 속도 향상
                detail=1
            )
            
            # 3. 결과 처리 (신뢰도 필터링 강화)
            extracted_blocks = []
            all_text = ""
            
            for detection in ocr_results:
                bbox, text, confidence = detection
                
                # 신뢰도 임계값 높임 (더 정확한 텍스트만)
                if confidence > 0.5 and len(text.strip()) > 1:
                    cleaned_text = text.strip()
                    extracted_blocks.append({
                        'text': cleaned_text,
                        'confidence': float(confidence),
                        'bbox': [[float(p[0]), float(p[1])] for p in bbox]
                    })
                    all_text += cleaned_text + " "
            
            # 4. 컨퍼런스 키워드 매칭
            keyword_matches = self._match_conference_keywords(all_text)
            
            processing_time = time.time() - start_time
            
            result = {
                'image_info': image_info,
                'extracted_text': all_text.strip(),
                'text_blocks': extracted_blocks,
                'total_blocks': len(extracted_blocks),
                'average_confidence': np.mean([block['confidence'] for block in extracted_blocks]) if extracted_blocks else 0,
                'keyword_matches': keyword_matches,
                'processing_time': processing_time,
                'status': 'success'
            }
            
            print(f"    [OK] {len(extracted_blocks)}개 블록, {len(all_text)}자 추출 ({processing_time:.1f}초)")
            
            # 키워드 발견 시 출력
            if keyword_matches['total_matches'] > 0:
                print(f"    [KEYWORDS] {keyword_matches['total_matches']}개 키워드 발견")
            
            return result
            
        except Exception as e:
            error_result = {
                'image_info': image_info,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'status': 'failed'
            }
            
            print(f"    [ERROR] 처리 실패: {e}")
            return error_result
        
        finally:
            # 메모리 정리
            gc.collect()
    
    def _match_conference_keywords(self, text: str) -> Dict[str, Any]:
        """컨퍼런스 키워드 매칭"""
        text_lower = text.lower()
        matches = {}
        total_matches = 0
        
        for category, keywords in self.known_keywords.items():
            category_matches = []
            
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    category_matches.append(keyword)
                    total_matches += 1
            
            matches[category] = category_matches
        
        return {
            'matches_by_category': matches,
            'total_matches': total_matches,
            'is_conference_relevant': total_matches > 0,
            'relevance_score': min(total_matches * 10, 100)  # 최대 100점
        }
    
    def process_images_in_batches(self, image_files: List[Dict[str, Any]], batch_size: int = 5) -> Dict[str, Any]:
        """배치 단위로 이미지 처리"""
        print(f"\n--- 배치 처리 시작 (배치 크기: {batch_size}) ---")
        
        if not self.ocr_reader:
            return {'error': 'OCR 리더가 초기화되지 않았습니다.'}
        
        total_images = len(image_files)
        processed_results = []
        failed_results = []
        
        # 컨퍼런스 이미지 우선 처리
        conference_images = [img for img in image_files if img['is_conference_image']]
        other_images = [img for img in image_files if not img['is_conference_image']]
        
        print(f"처리 순서: 컨퍼런스 이미지 {len(conference_images)}개 → 기타 {len(other_images)}개")
        
        # 우선순위 이미지 처리 (컨퍼런스 이미지)
        for i, image_info in enumerate(conference_images, 1):
            if i > 15:  # 컨퍼런스 이미지도 너무 많으면 제한
                print(f"    [LIMIT] 컨퍼런스 이미지 처리 제한 (15개)")
                break
            
            result = self.process_single_image(image_info, i, min(len(conference_images), 15))
            
            if result['status'] == 'success':
                processed_results.append(result)
                self.analysis_session['processed_images'].append(image_info['file_name'])
                
                # 추출된 텍스트 저장
                self.analysis_session['extracted_texts'][image_info['file_name']] = result['extracted_text']
            else:
                failed_results.append(result)
                self.analysis_session['failed_images'].append(image_info['file_name'])
            
            # 메모리 정리 (매 5장마다)
            if i % 5 == 0:
                gc.collect()
                time.sleep(0.5)  # 시스템 부하 방지
        
        # 종합 키워드 분석
        comprehensive_keywords = self._analyze_all_keywords(processed_results)
        
        final_result = {
            'session_info': {
                'session_id': self.analysis_session['session_id'],
                'total_images_found': total_images,
                'images_processed': len(processed_results),
                'images_failed': len(failed_results),
                'success_rate': len(processed_results) / total_images * 100 if total_images > 0 else 0
            },
            'processing_results': processed_results,
            'failed_results': failed_results,
            'comprehensive_analysis': {
                'total_text_extracted': sum(len(r.get('extracted_text', '')) for r in processed_results),
                'total_blocks_found': sum(r.get('total_blocks', 0) for r in processed_results),
                'average_confidence': np.mean([r.get('average_confidence', 0) for r in processed_results]) if processed_results else 0,
                'keyword_analysis': comprehensive_keywords
            },
            'conference_insights': self._generate_conference_insights(processed_results, comprehensive_keywords)
        }
        
        return final_result
    
    def _analyze_all_keywords(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """전체 결과에서 키워드 종합 분석"""
        print("\n  --- 키워드 종합 분석 ---")
        
        all_category_matches = {}
        total_keyword_mentions = 0
        relevant_images = 0
        
        # 카테고리별 키워드 집계
        for category in self.known_keywords.keys():
            all_category_matches[category] = {}
        
        for result in results:
            if result['status'] == 'success':
                keyword_matches = result.get('keyword_matches', {})
                
                if keyword_matches.get('is_conference_relevant', False):
                    relevant_images += 1
                
                matches_by_category = keyword_matches.get('matches_by_category', {})
                
                for category, matches in matches_by_category.items():
                    for keyword in matches:
                        if keyword in all_category_matches[category]:
                            all_category_matches[category][keyword] += 1
                        else:
                            all_category_matches[category][keyword] = 1
                        total_keyword_mentions += 1
        
        # 가장 많이 발견된 키워드들
        top_keywords = []
        for category, keywords in all_category_matches.items():
            for keyword, count in keywords.items():
                top_keywords.append({'keyword': keyword, 'category': category, 'count': count})
        
        top_keywords.sort(key=lambda x: x['count'], reverse=True)
        
        analysis = {
            'total_keyword_mentions': total_keyword_mentions,
            'relevant_images_count': relevant_images,
            'category_analysis': all_category_matches,
            'top_keywords': top_keywords[:10],  # 상위 10개
            'conference_relevance_rate': relevant_images / len(results) * 100 if results else 0
        }
        
        print(f"  [OK] 총 {total_keyword_mentions}개 키워드 발견, {relevant_images}개 이미지 관련성 확인")
        
        return analysis
    
    def _generate_conference_insights(self, results: List[Dict[str, Any]], keyword_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """컨퍼런스 인사이트 생성"""
        # 발견된 주요 정보 분석
        speaker_mentions = keyword_analysis['category_analysis'].get('speakers', {})
        company_mentions = keyword_analysis['category_analysis'].get('companies', {})
        topic_mentions = keyword_analysis['category_analysis'].get('topics', {})
        
        insights = {
            'identified_speakers': list(speaker_mentions.keys()),
            'identified_companies': list(company_mentions.keys()),
            'main_topics_found': list(topic_mentions.keys()),
            'content_assessment': {
                'speaker_visibility': len(speaker_mentions) > 0,
                'company_branding_present': len(company_mentions) > 0,
                'topic_coverage': len(topic_mentions) > 0,
                'overall_conference_confirmation': keyword_analysis['conference_relevance_rate'] > 50
            },
            'key_findings': self._extract_key_findings(keyword_analysis),
            'recommended_focus_areas': self._recommend_focus_areas(keyword_analysis)
        }
        
        return insights
    
    def _extract_key_findings(self, keyword_analysis: Dict[str, Any]) -> List[str]:
        """핵심 발견사항 추출"""
        findings = []
        
        top_keywords = keyword_analysis.get('top_keywords', [])
        
        if top_keywords:
            most_mentioned = top_keywords[0]
            findings.append(f"가장 많이 언급된 키워드: '{most_mentioned['keyword']}' ({most_mentioned['count']}회)")
        
        speaker_count = len(keyword_analysis['category_analysis'].get('speakers', []))
        if speaker_count > 0:
            findings.append(f"{speaker_count}명의 발표자 이름이 이미지에서 확인됨")
        
        company_count = len(keyword_analysis['category_analysis'].get('companies', []))
        if company_count > 0:
            findings.append(f"{company_count}개 기업/기관명이 시각적으로 확인됨")
        
        relevance_rate = keyword_analysis.get('conference_relevance_rate', 0)
        if relevance_rate > 70:
            findings.append(f"이미지의 {relevance_rate:.0f}%가 컨퍼런스 관련 콘텐츠로 확인")
        
        if not findings:
            findings.append("기본적인 이미지 텍스트 추출 완료, 추가 분석 필요")
        
        return findings
    
    def _recommend_focus_areas(self, keyword_analysis: Dict[str, Any]) -> List[str]:
        """집중 분석 영역 추천"""
        recommendations = []
        
        # 발견된 카테고리별 추천
        speakers = keyword_analysis['category_analysis'].get('speakers', {})
        if speakers:
            recommendations.append(f"발표자별 슬라이드 내용 상세 분석 ({len(speakers)}명 확인)")
        
        topics = keyword_analysis['category_analysis'].get('topics', {})
        if 'sustainability' in ' '.join(topics.keys()).lower():
            recommendations.append("지속가능성 관련 슬라이드 집중 분석")
        
        if keyword_analysis.get('total_keyword_mentions', 0) > 10:
            recommendations.append("키워드 밀도가 높은 이미지들의 전체 텍스트 추출")
        
        if not recommendations:
            recommendations.append("오디오 분석과 연계하여 이미지-음성 내용 매칭")
        
        return recommendations
    
    def save_ocr_results(self, final_result: Dict[str, Any]) -> str:
        """OCR 결과 저장"""
        report_path = project_root / f"optimized_ocr_analysis_{self.analysis_session['session_id']}.json"
        
        # 결과를 세션에 추가
        self.analysis_session.update({
            'final_results': final_result,
            'completion_time': datetime.now().isoformat()
        })
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_session, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVE] 최적화된 OCR 분석 결과 저장: {report_path}")
        return str(report_path)

def main():
    """메인 실행 함수"""
    print("최적화된 이미지 OCR 전체 분석")
    print("=" * 50)
    
    # OCR 분석기 초기화
    analyzer = OptimizedImageOCRAnalyzer()
    
    # 이미지 파일 탐색
    image_files = analyzer.find_conference_images()
    
    if not image_files:
        print("분석할 이미지가 없습니다.")
        return
    
    # 배치 처리로 OCR 수행
    final_result = analyzer.process_images_in_batches(image_files)
    
    if 'error' in final_result:
        print(f"[ERROR] 분석 실패: {final_result['error']}")
        return final_result
    
    # 결과 저장
    report_path = analyzer.save_ocr_results(final_result)
    
    # 요약 출력
    print(f"\n{'='*50}")
    print("최적화된 OCR 분석 완료")
    print(f"{'='*50}")
    
    session_info = final_result.get('session_info', {})
    print(f"\n[PROCESSING] 처리 정보:")
    print(f"  발견된 이미지: {session_info.get('total_images_found', 0)}개")
    print(f"  성공적 처리: {session_info.get('images_processed', 0)}개")
    print(f"  실패한 처리: {session_info.get('images_failed', 0)}개")
    print(f"  성공률: {session_info.get('success_rate', 0):.1f}%")
    
    comprehensive = final_result.get('comprehensive_analysis', {})
    print(f"\n[EXTRACTION] 추출 결과:")
    print(f"  총 추출 텍스트: {comprehensive.get('total_text_extracted', 0)}자")
    print(f"  총 텍스트 블록: {comprehensive.get('total_blocks_found', 0)}개")
    print(f"  평균 신뢰도: {comprehensive.get('average_confidence', 0):.2f}")
    
    keyword_analysis = comprehensive.get('keyword_analysis', {})
    print(f"\n[KEYWORDS] 키워드 분석:")
    print(f"  총 키워드 발견: {keyword_analysis.get('total_keyword_mentions', 0)}개")
    print(f"  관련성 있는 이미지: {keyword_analysis.get('relevant_images_count', 0)}개")
    print(f"  컨퍼런스 관련성: {keyword_analysis.get('conference_relevance_rate', 0):.1f}%")
    
    # 상위 키워드 출력
    top_keywords = keyword_analysis.get('top_keywords', [])
    if top_keywords:
        print(f"\n[TOP KEYWORDS] 상위 발견 키워드:")
        for i, kw in enumerate(top_keywords[:5], 1):
            print(f"  {i}. '{kw['keyword']}' ({kw['category']}) - {kw['count']}회")
    
    # 핵심 발견사항
    insights = final_result.get('conference_insights', {})
    key_findings = insights.get('key_findings', [])
    print(f"\n[INSIGHTS] 핵심 발견사항:")
    for i, finding in enumerate(key_findings, 1):
        print(f"  {i}. {finding}")
    
    print(f"\n[FILE] 상세 결과: {Path(report_path).name}")
    
    return final_result

if __name__ == "__main__":
    main()