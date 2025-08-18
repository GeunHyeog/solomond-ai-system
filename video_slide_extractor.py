#!/usr/bin/env python3
"""
영상 내 PPT 슬라이드 자동 추출 및 분석 고도화 시스템
- 영상에서 슬라이드 전환 시점 자동 감지
- 슬라이드 프레임 자동 추출
- 첨부된 슬라이드 이미지와 매칭
- 시간 동기화된 슬라이드 타임라인 생성
"""

import os
import cv2
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import hashlib

# EasyOCR for text matching
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

class VideoSlideExtractor:
    """영상 내 슬라이드 자동 추출 시스템"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 슬라이드 감지 설정
        self.frame_sample_rate = 0.5  # 0.5초마다 1프레임
        self.similarity_threshold = 0.85  # 유사도 임계값
        self.min_slide_duration = 5.0  # 최소 슬라이드 지속시간 (초)
        
        # 이미지 처리 설정
        self.target_width = 800
        self.target_height = 600
        
        # OCR 설정
        self.ocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['en', 'ko'], gpu=False)
                print("+ EasyOCR initialized for slide matching")
            except Exception as e:
                print(f"- EasyOCR initialization failed: {e}")
        
        # 추출된 슬라이드 저장 폴더
        self.output_dir = Path("extracted_slides")
        self.output_dir.mkdir(exist_ok=True)
        
    def _setup_logging(self):
        """로깅 설정"""
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def extract_slides_from_video(self, video_path: str, reference_slides: List[str] = None) -> Dict[str, Any]:
        """영상에서 슬라이드 자동 추출"""
        
        print(f"Starting slide extraction from video: {os.path.basename(video_path)}")
        
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}
        
        start_time = time.time()
        
        try:
            # 1. 영상 정보 추출
            video_info = self._get_video_info(video_path)
            print(f"Video: {video_info['duration']:.1f}s, {video_info['fps']:.1f}fps")
            
            # 2. 프레임별 슬라이드 후보 추출
            slide_candidates = self._extract_slide_candidates(video_path, video_info)
            print(f"Found {len(slide_candidates)} slide candidates")
            
            # 3. 슬라이드 전환점 감지
            slide_transitions = self._detect_slide_transitions(slide_candidates)
            print(f"Detected {len(slide_transitions)} slide transitions")
            
            # 4. 참조 슬라이드와 매칭 (있는 경우)
            if reference_slides:
                matched_slides = self._match_with_reference_slides(slide_transitions, reference_slides)
                print(f"Matched {len(matched_slides)} slides with references")
            else:
                matched_slides = slide_transitions
            
            # 5. 타임라인 생성
            slide_timeline = self._create_slide_timeline(matched_slides, video_info)
            
            processing_time = time.time() - start_time
            
            result = {
                "video_path": video_path,
                "video_info": video_info,
                "slide_candidates": len(slide_candidates),
                "slide_transitions": slide_transitions,
                "matched_slides": matched_slides if reference_slides else slide_transitions,
                "slide_timeline": slide_timeline,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"Slide extraction completed in {processing_time:.1f}s")
            return result
            
        except Exception as e:
            print(f"Slide extraction failed: {e}")
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
    
    def _extract_slide_candidates(self, video_path: str, video_info: Dict) -> List[Dict[str, Any]]:
        """프레임별 슬라이드 후보 추출"""
        
        cap = cv2.VideoCapture(video_path)
        fps = video_info["fps"]
        duration = video_info["duration"]
        
        # 샘플링 간격 계산
        frame_interval = int(fps * self.frame_sample_rate)
        
        slide_candidates = []
        prev_frame_features = None
        
        print(f"  Sampling every {self.frame_sample_rate}s ({frame_interval} frames)")
        
        for frame_idx in range(0, video_info["frame_count"], frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            timestamp = frame_idx / fps
            
            # 프레임 전처리
            processed_frame = self._preprocess_frame(frame)
            
            # 특징 추출
            frame_features = self._extract_frame_features(processed_frame)
            
            # 이전 프레임과의 유사도 계산
            similarity = 0.0
            if prev_frame_features is not None:
                similarity = self._calculate_frame_similarity(prev_frame_features, frame_features)
            
            # 슬라이드 후보로 저장
            candidate = {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "frame_features": frame_features,
                "similarity_to_prev": similarity,
                "is_slide_transition": similarity < (1 - self.similarity_threshold),  # 큰 변화가 있으면 전환
                "frame_path": None  # 필요시 프레임 저장
            }
            
            # 전환점인 경우 프레임 저장
            if candidate["is_slide_transition"] or len(slide_candidates) == 0:
                frame_path = self._save_frame(processed_frame, frame_idx, timestamp)
                candidate["frame_path"] = frame_path
            
            slide_candidates.append(candidate)
            prev_frame_features = frame_features
            
            # 진행률 표시
            if frame_idx % (frame_interval * 20) == 0:  # 10초마다
                progress = (timestamp / duration) * 100
                print(f"    Progress: {progress:.1f}% ({timestamp:.1f}s)")
        
        cap.release()
        return slide_candidates
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임 전처리"""
        
        # 크기 조정
        resized = cv2.resize(frame, (self.target_width, self.target_height))
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 대비 개선
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def _extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """프레임 특징 추출"""
        
        # 히스토그램 특징
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / np.sum(hist)
        
        # 가장자리 특징
        edges = cv2.Canny(frame, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 텍스처 특징 (LBP 간소화 버전)
        texture_features = self._extract_texture_features(frame)
        
        # 특징 벡터 결합
        features = np.concatenate([
            hist_norm,
            [edge_density],
            texture_features
        ])
        
        return features
    
    def _extract_texture_features(self, frame: np.ndarray) -> np.ndarray:
        """텍스처 특징 추출 (간소화된 LBP)"""
        
        # 간단한 텍스처 특징 - 4방향 gradient
        grad_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient 통계
        texture_features = [
            np.mean(np.abs(grad_x)),
            np.std(grad_x),
            np.mean(np.abs(grad_y)),
            np.std(grad_y)
        ]
        
        return np.array(texture_features)
    
    def _calculate_frame_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """두 프레임 간 유사도 계산"""
        
        # 코사인 유사도 계산
        features1_norm = features1.reshape(1, -1)
        features2_norm = features2.reshape(1, -1)
        
        similarity = cosine_similarity(features1_norm, features2_norm)[0][0]
        return similarity
    
    def _save_frame(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> str:
        """프레임을 이미지 파일로 저장"""
        
        timestamp_str = f"{int(timestamp//60):02d}m{int(timestamp%60):02d}s"
        filename = f"slide_{frame_idx:06d}_{timestamp_str}.jpg"
        filepath = self.output_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        return str(filepath)
    
    def _detect_slide_transitions(self, candidates: List[Dict]) -> List[Dict[str, Any]]:
        """슬라이드 전환점 감지 및 정제"""
        
        transitions = []
        
        # 전환점 후보 필터링
        transition_candidates = [c for c in candidates if c["is_slide_transition"]]
        
        # 너무 가까운 전환점 제거 (최소 지속시간 적용)
        filtered_transitions = []
        last_timestamp = -self.min_slide_duration
        
        for candidate in transition_candidates:
            if candidate["timestamp"] - last_timestamp >= self.min_slide_duration:
                filtered_transitions.append(candidate)
                last_timestamp = candidate["timestamp"]
        
        # 슬라이드 정보 생성
        for i, transition in enumerate(filtered_transitions):
            
            # 다음 전환점까지의 지속시간 계산
            if i < len(filtered_transitions) - 1:
                duration = filtered_transitions[i + 1]["timestamp"] - transition["timestamp"]
            else:
                # 마지막 슬라이드는 영상 끝까지
                last_candidate = candidates[-1]
                duration = last_candidate["timestamp"] - transition["timestamp"]
            
            slide_info = {
                "slide_number": i + 1,
                "start_time": transition["timestamp"],
                "duration": duration,
                "end_time": transition["timestamp"] + duration,
                "frame_idx": transition["frame_idx"],
                "frame_path": transition["frame_path"],
                "confidence": 1.0 - transition["similarity_to_prev"],  # 변화량이 클수록 신뢰도 높음
                "extracted_text": None,  # OCR 결과 저장용
                "matched_reference": None  # 참조 슬라이드 매칭 결과
            }
            
            # OCR 텍스트 추출 (있는 경우)
            if self.ocr_reader and transition["frame_path"]:
                slide_info["extracted_text"] = self._extract_text_from_slide(transition["frame_path"])
            
            transitions.append(slide_info)
        
        return transitions
    
    def _extract_text_from_slide(self, image_path: str) -> str:
        """슬라이드에서 텍스트 추출"""
        
        try:
            results = self.ocr_reader.readtext(image_path)
            text = " ".join([result[1] for result in results if result[2] > 0.5])
            return text.strip()
        except Exception as e:
            print(f"    OCR failed for {os.path.basename(image_path)}: {e}")
            return ""
    
    def _match_with_reference_slides(self, transitions: List[Dict], reference_slides: List[str]) -> List[Dict]:
        """참조 슬라이드와 매칭"""
        
        print(f"  Matching {len(transitions)} extracted slides with {len(reference_slides)} references")
        
        # 참조 슬라이드 특징 추출
        reference_features = []
        reference_texts = []
        
        for ref_slide in reference_slides:
            # 이미지 특징 추출
            ref_image = cv2.imread(ref_slide, cv2.IMREAD_GRAYSCALE)
            if ref_image is not None:
                ref_processed = cv2.resize(ref_image, (self.target_width, self.target_height))
                ref_features = self._extract_frame_features(ref_processed)
                reference_features.append(ref_features)
                
                # OCR 텍스트 추출
                if self.ocr_reader:
                    ref_text = self._extract_text_from_slide(ref_slide)
                    reference_texts.append(ref_text)
                else:
                    reference_texts.append("")
            else:
                reference_features.append(None)
                reference_texts.append("")
        
        # 각 추출된 슬라이드를 참조 슬라이드와 매칭
        for transition in transitions:
            best_match_idx = -1
            best_similarity = 0.0
            best_text_similarity = 0.0
            
            if transition["frame_path"] and os.path.exists(transition["frame_path"]):
                # 추출된 슬라이드 특징
                extracted_image = cv2.imread(transition["frame_path"], cv2.IMREAD_GRAYSCALE)
                if extracted_image is not None:
                    extracted_features = self._extract_frame_features(extracted_image)
                    extracted_text = transition.get("extracted_text", "")
                    
                    # 각 참조 슬라이드와 비교
                    for i, (ref_features, ref_text) in enumerate(zip(reference_features, reference_texts)):
                        if ref_features is not None:
                            # 이미지 유사도
                            img_similarity = self._calculate_frame_similarity(extracted_features, ref_features)
                            
                            # 텍스트 유사도
                            text_similarity = self._calculate_text_similarity(extracted_text, ref_text)
                            
                            # 종합 유사도 (이미지 70%, 텍스트 30%)
                            combined_similarity = img_similarity * 0.7 + text_similarity * 0.3
                            
                            if combined_similarity > best_similarity:
                                best_similarity = combined_similarity
                                best_match_idx = i
                                best_text_similarity = text_similarity
            
            # 매칭 결과 저장
            if best_match_idx >= 0 and best_similarity > 0.3:  # 최소 유사도 임계값
                transition["matched_reference"] = {
                    "reference_index": best_match_idx,
                    "reference_path": reference_slides[best_match_idx],
                    "image_similarity": best_similarity,
                    "text_similarity": best_text_similarity,
                    "match_confidence": best_similarity
                }
                print(f"    Slide {transition['slide_number']} -> Reference {best_match_idx + 1} (similarity: {best_similarity:.3f})")
            else:
                print(f"    Slide {transition['slide_number']} -> No match found")
        
        return transitions
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산 (간단한 키워드 기반)"""
        
        if not text1 or not text2:
            return 0.0
        
        # 간단한 단어 기반 유사도
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _create_slide_timeline(self, transitions: List[Dict], video_info: Dict) -> Dict[str, Any]:
        """슬라이드 타임라인 생성"""
        
        timeline = {
            "total_slides": len(transitions),
            "total_duration": video_info["duration"],
            "average_slide_duration": video_info["duration"] / len(transitions) if transitions else 0,
            "timeline_events": []
        }
        
        for transition in transitions:
            event = {
                "slide_number": transition["slide_number"],
                "start_time": transition["start_time"],
                "end_time": transition["end_time"],
                "duration": transition["duration"],
                "start_time_formatted": self._format_timestamp(transition["start_time"]),
                "end_time_formatted": self._format_timestamp(transition["end_time"]),
                "confidence": transition["confidence"],
                "has_reference_match": transition["matched_reference"] is not None,
                "extracted_text_preview": (transition.get("extracted_text", "") or "")[:100]
            }
            timeline["timeline_events"].append(event)
        
        return timeline
    
    def _format_timestamp(self, seconds: float) -> str:
        """타임스탬프를 MM:SS 형식으로 변환"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def save_extraction_result(self, result: Dict[str, Any], output_path: str = None) -> str:
        """추출 결과 저장"""
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"video_slide_extraction_{timestamp}.json"
        
        # JSON 직렬화 가능하도록 처리
        serializable_result = self._make_json_serializable(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"Extraction results saved: {output_path}")
        return output_path
    
    def _make_json_serializable(self, obj):
        """JSON 직렬화 가능하게 변환"""
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
    print("VIDEO SLIDE EXTRACTOR - 영상 내 슬라이드 자동 추출")
    print("=" * 60)
    
    extractor = VideoSlideExtractor()
    
    # 파일 경로 설정
    video_path = "C:/Users/PC_58410/solomond-ai-system/user_files/JGA2025_D1/IMG_0032.MOV"
    
    # 참조 슬라이드 (첨부된 이미지들)
    slides_folder = Path("C:/Users/PC_58410/solomond-ai-system/user_files/JGA2025_D1/")
    reference_slides = []
    
    # JPG 파일만 참조 슬라이드로 사용
    for jpg_file in slides_folder.glob("IMG_21*.JPG"):
        reference_slides.append(str(jpg_file))
    
    reference_slides = sorted(reference_slides)[:10]  # 처음 10개만 테스트
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Reference slides: {len(reference_slides)} images")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    try:
        # 슬라이드 추출 실행
        result = extractor.extract_slides_from_video(video_path, reference_slides)
        
        if "error" in result:
            print(f"Extraction failed: {result['error']}")
            return
        
        # 결과 출력
        print(f"\nEXTRACTION RESULTS:")
        print(f"Processing time: {result['processing_time']:.1f}s")
        print(f"Slide candidates found: {result['slide_candidates']}")
        print(f"Slide transitions detected: {len(result['slide_transitions'])}")
        
        timeline = result['slide_timeline']
        print(f"\nSLIDE TIMELINE:")
        print(f"Total slides: {timeline['total_slides']}")
        print(f"Average duration: {timeline['average_slide_duration']:.1f}s")
        
        print(f"\nTIMELINE EVENTS:")
        for event in timeline['timeline_events'][:5]:  # 처음 5개만 표시
            match_status = "Matched" if event['has_reference_match'] else "No match"
            print(f"  Slide {event['slide_number']}: {event['start_time_formatted']}-{event['end_time_formatted']} ({match_status})")
        
        # 결과 저장
        output_file = extractor.save_extraction_result(result)
        print(f"\nSlide extraction completed! Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()