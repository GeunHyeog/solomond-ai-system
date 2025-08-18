#!/usr/bin/env python3
"""
컨퍼런스 이미지 OCR 텍스트 추출 시스템
- 주얼리 컨퍼런스 이미지 OCR 분석
- EasyOCR 기반 한/영 텍스트 추출
- 발표자 정보 및 키워드 추출
"""

import os
import sys
import time
import json
import easyocr
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from PIL import Image
import numpy as np

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class ConferenceImageOCRAnalyzer:
    """컨퍼런스 이미지 OCR 분석기"""
    
    def __init__(self):
        self.analysis_session = {
            'session_id': f"conference_ocr_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'target_images': [],
            'ocr_results': [],
            'extracted_info': {},
            'conference_insights': {}
        }
        
        # 컨퍼런스 관련 키워드
        self.conference_keywords = {
            'speakers': [
                'lianne ng', 'henry tse', 'catherine siu',
                'chow tai fook', 'ancardi', 'nyreille', 'jrne'
            ],
            'topics': [
                'eco-friendly', 'luxury', 'consumer', 'sustainability',
                'jewelry', 'jewellery', 'environment', 'green'
            ],
            'event_info': [
                'jga25', 'connecting', 'hkcec', 'stage', 'hall 1b',
                '2:30pm', '3:30pm', '19/6/2025'
            ]
        }
        
        print("컨퍼런스 이미지 OCR 분석 시스템 초기화")
        self._initialize_analyzer()
    
    def _initialize_analyzer(self):
        """분석 시스템 초기화"""
        print("=== OCR 분석 시스템 초기화 ===")
        
        # EasyOCR 리더 로드
        try:
            print("EasyOCR 모델 로딩 중...")
            self.ocr_reader = easyocr.Reader(['ko', 'en'], gpu=False)
            print("[OK] EasyOCR 모델: 한/영 모델 로드 완료")
        except Exception as e:
            print(f"[ERROR] EasyOCR 모델 로드 실패: {e}")
            self.ocr_reader = None
            return False
        
        # 이미지 파일 검색
        self._find_image_files()
        
        return True
    
    def _find_image_files(self):
        """이미지 파일 검색"""
        image_path = project_root / 'user_files' / 'images'
        
        if not image_path.exists():
            print(f"[ERROR] 이미지 폴더 없음: {image_path}")
            return
        
        image_files = []
        for image_file in image_path.glob('*.JPG'):
            file_info = {
                'file_path': str(image_file),
                'file_name': image_file.name,
                'file_size_kb': image_file.stat().st_size / 1024,
                'priority': 'high' if 'IMG_2' in image_file.name else 'medium'
            }
            image_files.append(file_info)
        
        # 파일명 순으로 정렬
        image_files.sort(key=lambda x: x['file_name'])
        
        self.analysis_session['target_images'] = image_files
        print(f"[OK] 이미지 파일 발견: {len(image_files)}개")
        
        for i, file_info in enumerate(image_files[:5], 1):  # 처음 5개만 표시
            print(f"  {i}. {file_info['file_name']} ({file_info['file_size_kb']:.1f}KB, {file_info['priority']})")
        
        if len(image_files) > 5:
            print(f"  ... 및 {len(image_files) - 5}개 파일 더")
    
    def analyze_all_images(self) -> Dict[str, Any]:
        """모든 이미지 OCR 분석"""
        if not self.analysis_session['target_images']:
            return {'error': '분석할 이미지 파일이 없습니다.'}
        
        print(f"\n--- 컨퍼런스 이미지 전체 OCR 분석 ---")
        print(f"총 {len(self.analysis_session['target_images'])}개 이미지 분석")
        
        analysis_start = time.time()
        all_results = []
        
        try:
            for i, image_info in enumerate(self.analysis_session['target_images'], 1):
                print(f"\n[{i}/{len(self.analysis_session['target_images'])}] {image_info['file_name']} 분석 중...")
                
                # 개별 이미지 OCR 수행
                ocr_result = self._perform_image_ocr(image_info['file_path'])
                
                if 'error' not in ocr_result:
                    # 키워드 매칭 및 정보 추출
                    extracted_info = self._extract_conference_info(ocr_result['text'])
                    
                    image_result = {
                        'image_info': image_info,
                        'ocr_result': ocr_result,
                        'extracted_info': extracted_info,
                        'processing_order': i,
                        'status': 'success'
                    }
                    
                    print(f"  [OK] 추출된 텍스트: {len(ocr_result['text'])}자")
                    if extracted_info['speaker_mentions']:
                        print(f"  발표자 언급: {extracted_info['speaker_mentions']}")
                    if extracted_info['topic_matches']:
                        print(f"  주제 키워드: {extracted_info['topic_matches'][:3]}")
                else:
                    image_result = {
                        'image_info': image_info,
                        'error': ocr_result['error'],
                        'processing_order': i,
                        'status': 'error'
                    }
                    print(f"  [ERROR] OCR 실패: {ocr_result['error']}")
                
                all_results.append(image_result)
                time.sleep(0.1)  # 시스템 부하 방지
            
            # 종합 정보 추출
            comprehensive_info = self._generate_comprehensive_insights(all_results)
            
            processing_time = time.time() - analysis_start
            
            final_result = {
                'total_images_processed': len(all_results),
                'successful_analyses': len([r for r in all_results if r['status'] == 'success']),
                'individual_results': all_results,
                'comprehensive_insights': comprehensive_info,
                'processing_time': processing_time,
                'analysis_timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }
            
            print(f"\n[OK] 전체 OCR 분석 완료 ({processing_time:.1f}초)")
            print(f"성공: {final_result['successful_analyses']}/{final_result['total_images_processed']}개")
            
            return final_result
            
        except Exception as e:
            error_result = {
                'error': str(e),
                'partial_results': all_results,
                'processing_time': time.time() - analysis_start,
                'analysis_timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
            
            print(f"[ERROR] OCR 분석 실패: {e}")
            return error_result
    
    def _perform_image_ocr(self, image_path: str) -> Dict[str, Any]:
        """개별 이미지 OCR 수행"""
        try:
            if not self.ocr_reader:
                return {'error': 'OCR 리더가 로드되지 않았습니다.'}
            
            # 이미지 로드 및 전처리
            image = Image.open(image_path)
            
            # 이미지 크기 최적화 (너무 큰 경우)
            max_size = 2048
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # 이미지를 numpy 배열로 변환
            image_array = np.array(image)
            
            # EasyOCR로 텍스트 추출
            ocr_results = self.ocr_reader.readtext(image_array)
            
            # 텍스트 블록 정리
            text_blocks = []
            full_text = ""
            
            for detection in ocr_results:
                bbox, text, confidence = detection
                
                # 신뢰도가 일정 수준 이상인 텍스트만 포함
                if confidence > 0.3:
                    text_blocks.append({
                        'text': text.strip(),
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    full_text += text.strip() + " "
            
            return {
                'text': full_text.strip(),
                'text_blocks': text_blocks,
                'total_blocks': len(text_blocks),
                'average_confidence': np.mean([block['confidence'] for block in text_blocks]) if text_blocks else 0,
                'image_size': image.size
            }
            
        except Exception as e:
            return {'error': f'이미지 OCR 실행 오류: {str(e)}'}
    
    def _extract_conference_info(self, text: str) -> Dict[str, Any]:
        """컨퍼런스 관련 정보 추출"""
        text_lower = text.lower()
        
        # 발표자 정보 추출
        speaker_mentions = []
        for speaker in self.conference_keywords['speakers']:
            if speaker in text_lower:
                speaker_mentions.append(speaker)
        
        # 주제 키워드 매칭
        topic_matches = []
        for topic in self.conference_keywords['topics']:
            if topic in text_lower:
                topic_matches.append(topic)
        
        # 이벤트 정보 추출
        event_info = []
        for info in self.conference_keywords['event_info']:
            if info in text_lower:
                event_info.append(info)
        
        # 시간 정보 추출 (정규식 사용)
        import re
        time_patterns = re.findall(r'\d{1,2}:\d{2}[ap]m', text_lower)
        date_patterns = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', text)
        
        return {
            'speaker_mentions': speaker_mentions,
            'topic_matches': topic_matches,
            'event_info': event_info,
            'time_info': time_patterns,
            'date_info': date_patterns,
            'relevance_score': len(speaker_mentions) * 3 + len(topic_matches) * 2 + len(event_info),
            'is_conference_relevant': len(speaker_mentions) > 0 or len(topic_matches) > 1 or len(event_info) > 1
        }
    
    def _generate_comprehensive_insights(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """종합 인사이트 생성"""
        successful_results = [r for r in all_results if r['status'] == 'success']
        
        # 전체 텍스트 집계
        all_speaker_mentions = []
        all_topic_matches = []
        all_event_info = []
        high_relevance_images = []
        
        for result in successful_results:
            extracted = result.get('extracted_info', {})
            all_speaker_mentions.extend(extracted.get('speaker_mentions', []))
            all_topic_matches.extend(extracted.get('topic_matches', []))
            all_event_info.extend(extracted.get('event_info', []))
            
            if extracted.get('relevance_score', 0) > 5:
                high_relevance_images.append({
                    'file_name': result['image_info']['file_name'],
                    'relevance_score': extracted['relevance_score'],
                    'key_info': {
                        'speakers': extracted.get('speaker_mentions', []),
                        'topics': extracted.get('topic_matches', [])
                    }
                })
        
        # 빈도 계산
        from collections import Counter
        speaker_counts = Counter(all_speaker_mentions)
        topic_counts = Counter(all_topic_matches)
        
        insights = {
            'conference_overview': {
                'total_relevant_images': len([r for r in successful_results 
                                            if r.get('extracted_info', {}).get('is_conference_relevant', False)]),
                'high_relevance_images': len(high_relevance_images),
                'speaker_detection_rate': len(speaker_counts) / len(successful_results) * 100 if successful_results else 0
            },
            'speaker_analysis': {
                'detected_speakers': dict(speaker_counts.most_common()),
                'most_mentioned_speaker': speaker_counts.most_common(1)[0] if speaker_counts else None,
                'speaker_appearance_count': len(speaker_counts)
            },
            'topic_analysis': {
                'detected_topics': dict(topic_counts.most_common()),
                'primary_topics': [topic for topic, count in topic_counts.most_common(5)],
                'topic_diversity': len(topic_counts)
            },
            'key_findings': self._extract_key_findings(successful_results),
            'image_quality_analysis': {
                'average_confidence': np.mean([
                    r['ocr_result']['average_confidence'] 
                    for r in successful_results 
                    if 'ocr_result' in r
                ]) if successful_results else 0,
                'high_quality_images': len([
                    r for r in successful_results 
                    if r.get('ocr_result', {}).get('average_confidence', 0) > 0.7
                ])
            },
            'recommended_focus_images': high_relevance_images[:5]  # 상위 5개
        }
        
        return insights
    
    def _extract_key_findings(self, successful_results: List[Dict[str, Any]]) -> List[str]:
        """핵심 발견사항 추출"""
        findings = []
        
        # 발표자 관련 발견사항
        all_speakers = []
        for result in successful_results:
            speakers = result.get('extracted_info', {}).get('speaker_mentions', [])
            all_speakers.extend(speakers)
        
        if 'henry tse' in all_speakers:
            findings.append("Henry Tse (Ancardi CEO) 발표 이미지 다수 포함")
        if 'lianne ng' in all_speakers:
            findings.append("Lianne Ng (Chow Tai Fook 지속가능성 이사) 발표 내용 확인")
        
        # 주제 관련 발견사항
        all_topics = []
        for result in successful_results:
            topics = result.get('extracted_info', {}).get('topic_matches', [])
            all_topics.extend(topics)
        
        if 'eco-friendly' in all_topics or 'sustainability' in all_topics:
            findings.append("친환경/지속가능성 관련 핵심 콘텐츠 다수 확인")
        if 'luxury' in all_topics and 'consumer' in all_topics:
            findings.append("럭셔리 소비자 트렌드 관련 주요 토론 내용 포함")
        
        # 이벤트 정보 발견사항
        event_mentions = []
        for result in successful_results:
            events = result.get('extracted_info', {}).get('event_info', [])
            event_mentions.extend(events)
        
        if 'jga25' in event_mentions:
            findings.append("JGA25 공식 컨퍼런스 이미지로 확인")
        if any('2:30pm' in str(events) or '3:30pm' in str(events) for events in event_mentions):
            findings.append("예정된 시간대 (2:30pm-3:30pm) 패널 토론 이미지")
        
        if not findings:
            findings.append("주얼리 업계 컨퍼런스 관련 이미지들로 추정")
        
        return findings
    
    def save_ocr_results(self, analysis_result: Dict[str, Any]) -> str:
        """OCR 분석 결과 저장"""
        self.analysis_session['ocr_results'].append(analysis_result)
        
        report_path = project_root / f"conference_ocr_analysis_{self.analysis_session['session_id']}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_session, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVE] OCR 분석 결과 저장: {report_path}")
        return str(report_path)

def main():
    """메인 실행 함수"""
    print("컨퍼런스 이미지 OCR 분석 시작")
    print("=" * 50)
    
    # OCR 분석기 초기화
    ocr_analyzer = ConferenceImageOCRAnalyzer()
    
    # 전체 이미지 OCR 분석
    ocr_result = ocr_analyzer.analyze_all_images()
    
    if ocr_result.get('status') == 'completed':
        # 결과 저장
        report_path = ocr_analyzer.save_ocr_results(ocr_result)
        
        # 요약 출력
        print(f"\n{'='*50}")
        print("OCR 분석 완료 요약")
        print(f"{'='*50}")
        
        insights = ocr_result.get('comprehensive_insights', {})
        overview = insights.get('conference_overview', {})
        speaker_analysis = insights.get('speaker_analysis', {})
        topic_analysis = insights.get('topic_analysis', {})
        
        print(f"처리된 이미지: {ocr_result.get('total_images_processed', 0)}개")
        print(f"성공적 분석: {ocr_result.get('successful_analyses', 0)}개")
        print(f"처리 시간: {ocr_result.get('processing_time', 0):.1f}초")
        print(f"컨퍼런스 관련 이미지: {overview.get('total_relevant_images', 0)}개")
        print(f"감지된 발표자: {speaker_analysis.get('speaker_appearance_count', 0)}명")
        print(f"주요 주제 키워드: {topic_analysis.get('topic_diversity', 0)}개")
        print(f"상세 보고서: {report_path}")
        
        # 핵심 발견사항 출력
        key_findings = insights.get('key_findings', [])
        if key_findings:
            print(f"\n핵심 발견사항:")
            for i, finding in enumerate(key_findings, 1):
                print(f"  {i}. {finding}")
        
        # 주요 발표자 출력
        detected_speakers = speaker_analysis.get('detected_speakers', {})
        if detected_speakers:
            print(f"\n감지된 발표자:")
            for speaker, count in list(detected_speakers.items())[:3]:
                print(f"  - {speaker}: {count}회 언급")
    
    else:
        print(f"OCR 분석 실패: {ocr_result.get('error', 'Unknown error')}")
    
    return ocr_result

if __name__ == "__main__":
    main()