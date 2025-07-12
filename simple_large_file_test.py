#!/usr/bin/env python3
"""
고용량 파일 실전 테스트 - 간단 실행 버전
의존성 문제 최소화하여 바로 실행 가능한 버전

사용법:
  python simple_large_file_test.py
"""

import os
import sys
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

# 기본 라이브러리만 사용
try:
    import numpy as np
    import cv2
    import psutil
    print("✅ 필수 라이브러리 로드 성공")
except ImportError as e:
    print(f"❌ 필수 라이브러리 누락: {e}")
    print("설치 명령: pip install opencv-python numpy psutil")
    sys.exit(1)

class SimpleFileGenerator:
    """간단한 테스트 파일 생성기"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_simple_video(self, duration_seconds: int = 300) -> str:
        """간단한 테스트 비디오 생성 (5분)"""
        output_path = self.output_dir / "simple_test_video.mp4"
        
        if output_path.exists():
            print(f"✅ 기존 비디오 사용: {output_path}")
            return str(output_path)
        
        print(f"🎬 간단 비디오 생성 중... ({duration_seconds}초)")
        
        try:
            fps = 10  # 낮은 FPS로 빠른 생성
            width, height = 640, 480  # 작은 해상도
            total_frames = duration_seconds * fps
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            for frame_num in range(total_frames):
                # 간단한 프레임 생성
                current_seconds = frame_num / fps
                minutes = int(current_seconds // 60)
                seconds = int(current_seconds % 60)
                
                # 단색 배경
                color = int(255 * (frame_num % 100) / 100)
                frame = np.full((height, width, 3), [color, 100, 150], dtype=np.uint8)
                
                # 시간 텍스트
                time_text = f"{minutes:02d}:{seconds:02d}"
                cv2.putText(frame, time_text, (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                cv2.putText(frame, f"Frame {frame_num}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                writer.write(frame)
                
                # 진행률 표시 (10초마다)
                if frame_num % (fps * 10) == 0:
                    progress = (frame_num / total_frames) * 100
                    print(f"  비디오 생성: {progress:.1f}%")
            
            writer.release()
            
            file_size = os.path.getsize(output_path) / 1024 / 1024
            print(f"✅ 비디오 생성 완료: {file_size:.1f}MB")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ 비디오 생성 실패: {e}")
            return None
    
    def generate_simple_images(self, count: int = 5) -> List[str]:
        """간단한 테스트 이미지들 생성"""
        image_paths = []
        
        print(f"🖼️ 간단 이미지 {count}개 생성 중...")
        
        for i in range(count):
            filename = f"simple_image_{i+1:03d}.png"
            image_path = self.output_dir / filename
            
            if not image_path.exists():
                try:
                    # 간단한 이미지 생성
                    width, height = 800, 600
                    
                    # 랜덤 색상 배경
                    np.random.seed(i)
                    bg_color = np.random.randint(0, 256, 3)
                    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
                    
                    # 텍스트 추가
                    cv2.putText(img, f"Test Image #{i+1}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                    cv2.putText(img, f"Jewelry Analysis Test", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(img, f"Diamond Quality Check", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(img, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # 기하학적 패턴
                    cv2.circle(img, (400, 300), 50, (255, 0, 0), -1)
                    cv2.rectangle(img, (300, 200), (500, 400), (0, 255, 0), 3)
                    
                    cv2.imwrite(str(image_path), img)
                    
                except Exception as e:
                    print(f"⚠️ 이미지 {i+1} 생성 실패: {e}")
                    continue
            
            image_paths.append(str(image_path))
        
        print(f"✅ 이미지 {len(image_paths)}개 생성 완료")
        return image_paths

class SimpleProcessor:
    """간단한 파일 처리기"""
    
    def __init__(self):
        self.start_time = time.time()
        
    async def process_video_simple(self, video_path: str) -> Dict[str, Any]:
        """간단한 비디오 처리"""
        print("🎬 비디오 분석 중...")
        
        result = {
            "file_path": video_path,
            "file_size_mb": os.path.getsize(video_path) / 1024 / 1024,
            "processing_time": 0,
            "frame_count": 0,
            "duration_seconds": 0,
            "quality_score": 0.8,
            "status": "success"
        }
        
        start_time = time.time()
        
        try:
            # 비디오 정보 분석
            cap = cv2.VideoCapture(video_path)
            
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                result["frame_count"] = frame_count
                result["duration_seconds"] = duration
                result["fps"] = fps
                
                # 몇 개 프레임 샘플링
                sample_frames = min(10, frame_count)
                for i in range(sample_frames):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_count // sample_frames)
                    ret, frame = cap.read()
                    if ret:
                        # 간단한 품질 분석 (블러 감지)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                        result["quality_score"] = min(1.0, laplacian_var / 1000)
                    
                    # 진행률 표시
                    progress = (i + 1) / sample_frames * 100
                    print(f"\r  프레임 분석: {progress:.1f}%", end="")
                
                cap.release()
                print()  # 줄바꿈
                
            else:
                result["status"] = "error"
                result["error"] = "비디오 파일을 열 수 없음"
                
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            
        result["processing_time"] = time.time() - start_time
        return result
    
    async def process_images_simple(self, image_paths: List[str]) -> Dict[str, Any]:
        """간단한 이미지 처리"""
        print("🖼️ 이미지 분석 중...")
        
        result = {
            "total_images": len(image_paths),
            "processed_images": 0,
            "failed_images": 0,
            "total_size_mb": 0,
            "processing_time": 0,
            "image_details": [],
            "status": "success"
        }
        
        start_time = time.time()
        
        try:
            for i, img_path in enumerate(image_paths):
                try:
                    # 이미지 로드 및 분석
                    img = cv2.imread(img_path)
                    if img is not None:
                        height, width = img.shape[:2]
                        file_size = os.path.getsize(img_path)
                        
                        # 간단한 OCR 시뮬레이션 (실제로는 pytesseract 필요)
                        mock_ocr_text = f"Test Image #{i+1} - Jewelry Analysis - Quality Check"
                        
                        image_detail = {
                            "path": img_path,
                            "width": width,
                            "height": height,
                            "size_bytes": file_size,
                            "extracted_text": mock_ocr_text,
                            "has_text": True
                        }
                        
                        result["image_details"].append(image_detail)
                        result["total_size_mb"] += file_size / 1024 / 1024
                        result["processed_images"] += 1
                        
                    else:
                        result["failed_images"] += 1
                        
                except Exception as e:
                    print(f"⚠️ 이미지 처리 실패 {img_path}: {e}")
                    result["failed_images"] += 1
                
                # 진행률 표시
                progress = (i + 1) / len(image_paths) * 100
                print(f"\r  이미지 처리: {progress:.1f}%", end="")
            
            print()  # 줄바꿈
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            
        result["processing_time"] = time.time() - start_time
        return result

class SimpleMonitor:
    """간단한 시스템 모니터링"""
    
    def __init__(self):
        self.start_time = time.time()
        
    def get_system_status(self) -> Dict[str, Any]:
        """현재 시스템 상태"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                "memory_used_mb": (memory.total - memory.available) / 1024 / 1024,
                "memory_total_mb": memory.total / 1024 / 1024,
                "memory_percent": memory.percent,
                "cpu_percent": cpu_percent,
                "elapsed_time": time.time() - self.start_time
            }
        except Exception as e:
            return {"error": str(e)}

async def run_simple_test(test_dir: str = None):
    """간단한 실전 테스트 실행"""
    if test_dir is None:
        test_dir = tempfile.mkdtemp(prefix="simple_large_file_test_")
    
    test_dir = Path(test_dir)
    print(f"🧪 테스트 디렉토리: {test_dir}")
    
    # 컴포넌트 초기화
    generator = SimpleFileGenerator(str(test_dir))
    processor = SimpleProcessor()
    monitor = SimpleMonitor()
    
    test_results = {
        "test_info": {
            "start_time": datetime.now().isoformat(),
            "test_directory": str(test_dir),
            "test_type": "simple_large_file_test"
        },
        "system_info": monitor.get_system_status(),
        "file_generation": {},
        "processing_results": {},
        "performance": {}
    }
    
    try:
        print("🚀 간단 대용량 파일 테스트 시작!")
        print("=" * 50)
        
        # 1. 파일 생성
        print("1️⃣ 테스트 파일 생성")
        video_file = generator.generate_simple_video(duration_seconds=300)  # 5분
        image_files = generator.generate_simple_images(count=5)
        
        test_results["file_generation"] = {
            "video_file": video_file,
            "video_size_mb": os.path.getsize(video_file) / 1024 / 1024 if video_file else 0,
            "image_files": image_files,
            "image_count": len(image_files),
            "total_image_size_mb": sum(os.path.getsize(f) for f in image_files) / 1024 / 1024
        }
        
        print(f"✅ 파일 생성 완료:")
        print(f"   - 비디오: {test_results['file_generation']['video_size_mb']:.1f}MB")
        print(f"   - 이미지: {len(image_files)}개 ({test_results['file_generation']['total_image_size_mb']:.1f}MB)")
        
        # 2. 파일 처리
        print("\n2️⃣ 파일 처리 시작")
        
        # 비디오 처리
        if video_file:
            video_result = await processor.process_video_simple(video_file)
            test_results["processing_results"]["video"] = video_result
        
        # 이미지 처리
        if image_files:
            image_result = await processor.process_images_simple(image_files)
            test_results["processing_results"]["images"] = image_result
        
        # 3. 성능 측정
        print("\n3️⃣ 성능 측정")
        final_system_status = monitor.get_system_status()
        test_results["performance"] = {
            "initial_system": test_results["system_info"],
            "final_system": final_system_status,
            "total_processing_time": final_system_status.get("elapsed_time", 0)
        }
        
        # 4. 결과 요약
        print("\n4️⃣ 결과 요약")
        await print_test_summary(test_results)
        
        # 5. 결과 저장
        result_file = test_dir / "simple_test_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📋 결과 파일 저장: {result_file}")
        
        return test_results
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        test_results["error"] = str(e)
        return test_results

async def print_test_summary(test_results: Dict[str, Any]):
    """테스트 결과 요약 출력"""
    print("=" * 50)
    print("📋 테스트 결과 요약")
    print("=" * 50)
    
    # 파일 생성 결과
    file_gen = test_results.get("file_generation", {})
    total_size = file_gen.get("video_size_mb", 0) + file_gen.get("total_image_size_mb", 0)
    print(f"📁 처리된 데이터: {total_size:.1f}MB")
    
    # 처리 결과
    video_result = test_results.get("processing_results", {}).get("video", {})
    image_result = test_results.get("processing_results", {}).get("images", {})
    
    if video_result:
        print(f"🎬 비디오 처리:")
        print(f"   - 지속시간: {video_result.get('duration_seconds', 0):.1f}초")
        print(f"   - 프레임 수: {video_result.get('frame_count', 0)}")
        print(f"   - 품질 점수: {video_result.get('quality_score', 0):.2f}")
        print(f"   - 처리 시간: {video_result.get('processing_time', 0):.1f}초")
    
    if image_result:
        print(f"🖼️ 이미지 처리:")
        print(f"   - 처리 성공: {image_result.get('processed_images', 0)}개")
        print(f"   - 처리 실패: {image_result.get('failed_images', 0)}개")
        print(f"   - 총 크기: {image_result.get('total_size_mb', 0):.1f}MB")
        print(f"   - 처리 시간: {image_result.get('processing_time', 0):.1f}초")
    
    # 성능 결과
    performance = test_results.get("performance", {})
    if performance:
        final_sys = performance.get("final_system", {})
        print(f"⚡ 시스템 성능:")
        print(f"   - 메모리 사용: {final_sys.get('memory_used_mb', 0):.1f}MB ({final_sys.get('memory_percent', 0):.1f}%)")
        print(f"   - CPU 사용률: {final_sys.get('cpu_percent', 0):.1f}%")
        print(f"   - 총 처리 시간: {performance.get('total_processing_time', 0):.1f}초")
    
    # 전체 성공 여부
    has_error = test_results.get("error") is not None
    video_success = video_result.get("status") == "success" if video_result else True
    image_success = image_result.get("status") == "success" if image_result else True
    
    overall_success = not has_error and video_success and image_success
    
    print(f"\n🎯 전체 결과: {'✅ 성공' if overall_success else '❌ 실패'}")
    print("=" * 50)

async def main():
    """메인 함수"""
    print("🎯 간단 대용량 파일 실전 테스트 v2.1")
    print("=" * 50)
    
    try:
        result = await run_simple_test()
        
        if result.get("error"):
            print(f"❌ 테스트 실패: {result['error']}")
        else:
            print("✅ 테스트 완료!")
            
    except KeyboardInterrupt:
        print("\n⏹️ 사용자가 테스트를 중단했습니다.")
        
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")

if __name__ == "__main__":
    # 비동기 실행
    asyncio.run(main())
