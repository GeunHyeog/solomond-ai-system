#!/usr/bin/env python3
"""
첨부 이미지와 영상 슬라이드 매칭 및 시간 동기화 시스템
- 영상에서 추출된 슬라이드와 첨부 이미지 자동 매칭
- 시간 동기화된 프레젠테이션 타임라인 생성
- 5D 멀티모달 시스템과 완전 통합
"""

import os
import cv2
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import hashlib
from PIL import Image

# OCR for text matching
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

class SlideTimelineSynchronizer:
    """슬라이드 타임라인 동기화 시스템"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 매칭 설정
        self.similarity_threshold = 0.3  # 매칭 임계값
        self.text_weight = 0.4  # 텍스트 유사도 가중치
        self.image_weight = 0.6  # 이미지 유사도 가중치
        
        # 이미지 처리 설정
        self.target_size = (400, 300)  # 비교용 표준 크기
        
        # OCR 초기화
        self.ocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['en', 'ko'], gpu=False)
                print("+ OCR initialized for slide synchronization")
            except Exception as e:
                print(f"- OCR initialization failed: {e}")
    
    def _setup_logging(self):
        """로깅 설정"""
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def synchronize_slides_with_timeline(self, video_path: str, reference_slides: List[str], 
                                       transcript_text: str = None) -> Dict[str, Any]:
        """슬라이드와 타임라인 동기화"""
        
        print(f"SLIDE-TIMELINE SYNCHRONIZATION")
        print(f"Video: {os.path.basename(video_path)}")
        print(f"Reference slides: {len(reference_slides)}")
        
        start_time = time.time()
        
        try:
            # 1. 영상 정보 추출
            video_info = self._get_video_info(video_path)
            print(f"Video duration: {video_info['duration']:.1f}s")
            
            # 2. 참조 슬라이드 분석
            reference_analysis = self._analyze_reference_slides(reference_slides)
            print(f"Reference slides analyzed: {len(reference_analysis)}")
            
            # 3. 영상에서 키 시점 추출 (간소화)
            key_timepoints = self._extract_key_timepoints(video_path, video_info)
            print(f"Key timepoints extracted: {len(key_timepoints)}")
            
            # 4. 슬라이드 매칭
            matched_timeline = self._match_slides_to_timeline(key_timepoints, reference_analysis)
            print(f"Slides matched to timeline: {len(matched_timeline)}")
            
            # 5. 트랜스크립트와 동기화 (있는 경우)
            if transcript_text:
                transcript_sync = self._synchronize_with_transcript(matched_timeline, transcript_text)
                matched_timeline = transcript_sync
                print("Transcript synchronization completed")
            
            # 6. 최종 타임라인 생성
            final_timeline = self._create_synchronized_timeline(matched_timeline, video_info)
            
            processing_time = time.time() - start_time
            
            result = {
                "synchronization_type": "slide_timeline_sync",
                "video_path": video_path,
                "video_info": video_info,
                "reference_slides_count": len(reference_slides),
                "key_timepoints": key_timepoints,
                "matched_timeline": matched_timeline,
                "final_timeline": final_timeline,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"Synchronization completed in {processing_time:.1f}s")
            return result
            
        except Exception as e:
            print(f"Synchronization failed: {e}")
            return {"error": str(e)}
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """영상 정보 추출"""
        cap = cv2.VideoCapture(video_path)
        
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": 0
        }
        
        if info["fps"] > 0:
            info["duration"] = info["frame_count"] / info["fps"]
        
        cap.release()
        return info
    
    def _analyze_reference_slides(self, reference_slides: List[str]) -> List[Dict[str, Any]]:
        """참조 슬라이드 분석"""
        
        analyses = []
        
        for i, slide_path in enumerate(reference_slides):
            print(f"  Analyzing reference slide {i+1}/{len(reference_slides)}: {os.path.basename(slide_path)}")
            
            try:
                # 이미지 로드 및 전처리
                image = cv2.imread(slide_path)
                if image is None:
                    continue
                
                # 특징 추출
                features = self._extract_image_features(image)
                
                # OCR 텍스트 추출
                extracted_text = ""
                if self.ocr_reader:
                    try:
                        ocr_results = self.ocr_reader.readtext(slide_path)
                        extracted_text = " ".join([result[1] for result in ocr_results if result[2] > 0.5])
                    except Exception as e:
                        print(f"    OCR failed: {e}")
                
                # 슬라이드 타입 분류
                slide_type = self._classify_slide_content(extracted_text)
                
                analysis = {
                    "slide_index": i,
                    "slide_path": slide_path,
                    "slide_name": os.path.basename(slide_path),
                    "image_features": features,
                    "extracted_text": extracted_text,
                    "text_length": len(extracted_text),
                    "slide_type": slide_type,
                    "keywords": self._extract_keywords(extracted_text)
                }
                
                analyses.append(analysis)
                
            except Exception as e:
                print(f"    Failed to analyze {slide_path}: {e}")
        
        return analyses
    
    def _extract_image_features(self, image: np.ndarray) -> np.ndarray:
        """이미지 특징 추출"""
        
        # 크기 조정 및 그레이스케일 변환
        resized = cv2.resize(image, self.target_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # 히스토그램 특징
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist_norm = hist.flatten() / np.sum(hist)
        
        # 가장자리 특징
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 텍스처 특징 (간단한 그래디언트 통계)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        texture_features = [
            np.mean(np.abs(grad_x)),
            np.std(grad_x),
            np.mean(np.abs(grad_y)),
            np.std(grad_y)
        ]
        
        # 특징 벡터 결합
        features = np.concatenate([hist_norm, [edge_density], texture_features])
        
        return features
    
    def _classify_slide_content(self, text: str) -> str:
        """슬라이드 내용 분류"""
        text_lower = text.lower()
        
        # 키워드 기반 분류
        if any(word in text_lower for word in ['welcome', 'introduction', 'overview', 'agenda']):
            return 'introduction'
        elif any(word in text_lower for word in ['conclusion', 'summary', 'thank you', 'questions']):
            return 'conclusion'
        elif any(word in text_lower for word in ['data', 'analysis', 'research', 'results']):
            return 'content_data'
        elif any(word in text_lower for word in ['sustainable', 'eco-friendly', 'luxury', 'consumer']):
            return 'content_main'
        else:
            return 'content_general'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        if not text:
            return []
        
        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 기법 사용 가능)
        words = text.lower().split()
        
        # 중요 키워드 패턴
        important_keywords = []
        for word in words:
            if len(word) > 4 and word.isalpha():  # 4자 이상 알파벳만
                important_keywords.append(word)
        
        # 상위 5개 키워드 반환
        return list(set(important_keywords))[:5]
    
    def _extract_key_timepoints(self, video_path: str, video_info: Dict) -> List[Dict[str, Any]]:
        """영상에서 키 시점 추출 (샘플링 기반)"""
        
        cap = cv2.VideoCapture(video_path)
        fps = video_info["fps"]
        duration = video_info["duration"]
        
        # 균등 간격으로 샘플링 (참조 슬라이드 수에 맞춰)
        # 예: 23개 참조 슬라이드가 있다면 영상을 23등분
        num_samples = min(15, int(duration / 30))  # 최대 15개, 30초마다 1개
        
        timepoints = []
        
        for i in range(num_samples):
            # 시간 계산
            timestamp = (duration / num_samples) * (i + 0.5)  # 구간 중간점
            frame_idx = int(timestamp * fps)
            
            # 프레임 추출
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # 특징 추출
                features = self._extract_image_features(frame)
                
                timepoint = {
                    "timepoint_index": i,
                    "timestamp": timestamp,
                    "timestamp_formatted": self._format_timestamp(timestamp),
                    "frame_idx": frame_idx,
                    "image_features": features,
                    "matched_slide": None,  # 매칭 결과 저장용
                    "match_confidence": 0.0
                }
                
                timepoints.append(timepoint)
                
                if i % 5 == 0:
                    print(f"    Extracted timepoint {i+1}/{num_samples}: {timepoint['timestamp_formatted']}")
        
        cap.release()
        return timepoints
    
    def _match_slides_to_timeline(self, timepoints: List[Dict], reference_slides: List[Dict]) -> List[Dict]:
        """슬라이드를 타임라인에 매칭"""
        
        print("  Matching slides to timeline...")
        
        # 각 타임포인트에 대해 최적의 참조 슬라이드 찾기
        for timepoint in timepoints:
            best_match_idx = -1
            best_similarity = 0.0
            
            timepoint_features = timepoint["image_features"]
            
            # 모든 참조 슬라이드와 비교
            for ref_slide in reference_slides:
                ref_features = ref_slide["image_features"]
                
                # 이미지 유사도 계산 (코사인 유사도)
                img_similarity = self._calculate_cosine_similarity(timepoint_features, ref_features)
                
                # 종합 유사도 (현재는 이미지만 사용)
                total_similarity = img_similarity
                
                if total_similarity > best_similarity:
                    best_similarity = total_similarity
                    best_match_idx = ref_slide["slide_index"]
            
            # 매칭 결과 저장
            if best_match_idx >= 0 and best_similarity > self.similarity_threshold:
                timepoint["matched_slide"] = best_match_idx
                timepoint["match_confidence"] = best_similarity
                
                matched_slide = reference_slides[best_match_idx]
                print(f"    {timepoint['timestamp_formatted']} -> Slide {best_match_idx + 1} ({matched_slide['slide_name']}) - {best_similarity:.3f}")
            else:
                print(f"    {timepoint['timestamp_formatted']} -> No match (best: {best_similarity:.3f})")
        
        return timepoints
    
    def _calculate_cosine_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _synchronize_with_transcript(self, timeline: List[Dict], transcript_text: str) -> List[Dict]:
        """트랜스크립트와 동기화"""
        
        print("  Synchronizing with transcript...")
        
        # 트랜스크립트에서 시간 마커나 화자 전환점 감지
        transcript_markers = self._detect_transcript_markers(transcript_text)
        
        # 타임라인과 트랜스크립트 마커 연결
        for timepoint in timeline:
            timestamp = timepoint["timestamp"]
            
            # 해당 시점 근처의 트랜스크립트 내용 찾기
            relevant_transcript = self._find_relevant_transcript(timestamp, transcript_markers)
            
            if relevant_transcript:
                timepoint["transcript_context"] = relevant_transcript
        
        return timeline
    
    def _detect_transcript_markers(self, transcript_text: str) -> List[Dict[str, Any]]:
        """트랜스크립트 마커 감지"""
        
        markers = []
        lines = transcript_text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 화자 마커 감지
            if line.startswith(('Speaker', 'speaker', '발표자', '화자')):
                markers.append({
                    "line_number": i,
                    "type": "speaker_marker",
                    "content": line,
                    "estimated_position": i / len(lines)  # 상대적 위치
                })
            
            # 주제 전환 마커
            elif any(keyword in line.lower() for keyword in ['now', 'next', 'moving on', 'in conclusion']):
                markers.append({
                    "line_number": i,
                    "type": "topic_transition",
                    "content": line,
                    "estimated_position": i / len(lines)
                })
        
        return markers
    
    def _find_relevant_transcript(self, timestamp: float, markers: List[Dict], video_duration: float = 3600) -> str:
        """해당 시점의 관련 트랜스크립트 찾기"""
        
        # 타임스탬프를 트랜스크립트 상대 위치로 변환
        relative_position = timestamp / video_duration
        
        # 가장 가까운 마커 찾기
        closest_marker = None
        min_distance = float('inf')
        
        for marker in markers:
            distance = abs(marker["estimated_position"] - relative_position)
            if distance < min_distance:
                min_distance = distance
                closest_marker = marker
        
        return closest_marker["content"] if closest_marker else ""
    
    def _create_synchronized_timeline(self, matched_timeline: List[Dict], video_info: Dict) -> Dict[str, Any]:
        """동기화된 최종 타임라인 생성"""
        
        # 매칭된 슬라이드만 필터링
        matched_timepoints = [tp for tp in matched_timeline if tp["matched_slide"] is not None]
        
        # 타임라인 이벤트 생성
        timeline_events = []
        
        for i, timepoint in enumerate(matched_timepoints):
            # 다음 이벤트까지의 지속시간 계산
            if i < len(matched_timepoints) - 1:
                duration = matched_timepoints[i + 1]["timestamp"] - timepoint["timestamp"]
            else:
                duration = video_info["duration"] - timepoint["timestamp"]
            
            event = {
                "event_number": i + 1,
                "start_time": timepoint["timestamp"],
                "end_time": timepoint["timestamp"] + duration,
                "duration": duration,
                "start_time_formatted": timepoint["timestamp_formatted"],
                "end_time_formatted": self._format_timestamp(timepoint["timestamp"] + duration),
                "matched_slide_index": timepoint["matched_slide"],
                "match_confidence": timepoint["match_confidence"],
                "transcript_context": timepoint.get("transcript_context", "")
            }
            
            timeline_events.append(event)
        
        # 타임라인 통계
        timeline_stats = {
            "total_events": len(timeline_events),
            "total_duration": video_info["duration"],
            "average_event_duration": video_info["duration"] / len(timeline_events) if timeline_events else 0,
            "coverage_percentage": (len(timeline_events) * 100) / len(matched_timeline) if matched_timeline else 0,
            "high_confidence_matches": len([e for e in timeline_events if e["match_confidence"] > 0.7])
        }
        
        return {
            "timeline_events": timeline_events,
            "timeline_stats": timeline_stats,
            "synchronization_quality": "high" if timeline_stats["coverage_percentage"] > 60 else "medium"
        }
    
    def _format_timestamp(self, seconds: float) -> str:
        """타임스탬프 포맷팅"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def save_synchronization_result(self, result: Dict[str, Any], output_path: str = None) -> str:
        """동기화 결과 저장"""
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"slide_timeline_sync_{timestamp}.json"
        
        # JSON 직렬화 가능하도록 처리
        serializable_result = self._make_json_serializable(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"Synchronization results saved: {output_path}")
        return output_path
    
    def _make_json_serializable(self, obj):
        """JSON 직렬화 처리"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return float(obj)
        else:
            return obj

def main():
    """메인 실행 함수"""
    print("SLIDE-TIMELINE SYNCHRONIZATION SYSTEM")
    print("=" * 50)
    
    synchronizer = SlideTimelineSynchronizer()
    
    # 파일 경로 설정
    video_path = "C:/Users/PC_58410/solomond-ai-system/user_files/JGA2025_D1/IMG_0032.MOV"
    slides_folder = Path("C:/Users/PC_58410/solomond-ai-system/user_files/JGA2025_D1/")
    
    # 참조 슬라이드 (JPG 파일만)
    reference_slides = []
    for jpg_file in sorted(slides_folder.glob("IMG_21*.JPG"))[:10]:  # 처음 10개
        reference_slides.append(str(jpg_file))
    
    # 샘플 트랜스크립트
    transcript_text = """
    Speaker 1: Welcome everyone to today's presentation on The Rise of the Eco-Friendly Luxury Consumer.
    I'm excited to share our research findings with you.
    
    Speaker 2: Thank you for that introduction. Now, moving on to the market analysis.
    Our data shows a significant shift in consumer behavior over the past five years.
    
    Speaker 3: Building on that analysis, let me present our recommendations for luxury brands.
    In conclusion, the eco-friendly luxury consumer represents a fundamental shift in the market.
    """
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Reference slides: {len(reference_slides)}")
    print(f"Transcript: {len(transcript_text)} characters")
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        return
    
    try:
        # 동기화 실행
        result = synchronizer.synchronize_slides_with_timeline(
            video_path, reference_slides, transcript_text
        )
        
        if "error" in result:
            print(f"Synchronization failed: {result['error']}")
            return
        
        # 결과 출력
        timeline = result["final_timeline"]
        stats = timeline["timeline_stats"]
        
        print(f"\nSYNCHRONIZATION RESULTS:")
        print(f"Processing time: {result['processing_time']:.1f}s")
        print(f"Timeline events: {stats['total_events']}")
        print(f"Coverage: {stats['coverage_percentage']:.1f}%")
        print(f"Quality: {timeline['synchronization_quality']}")
        print(f"High confidence matches: {stats['high_confidence_matches']}")
        
        print(f"\nTIMELINE EVENTS:")
        for event in timeline["timeline_events"][:5]:  # 처음 5개
            confidence = event["match_confidence"]
            print(f"  Event {event['event_number']}: {event['start_time_formatted']}-{event['end_time_formatted']} "
                  f"(Slide {event['matched_slide_index'] + 1}, confidence: {confidence:.3f})")
        
        # 결과 저장
        output_file = synchronizer.save_synchronization_result(result)
        print(f"\nSlide-Timeline synchronization completed!")
        print(f"Results saved: {output_file}")
        
    except Exception as e:
        print(f"Synchronization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()