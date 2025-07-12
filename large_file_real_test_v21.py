#!/usr/bin/env python3
"""
고용량 파일 실전 테스트 시스템 v2.1
실제 대용량 파일(1시간 영상 + 30개 이미지)로 성능 검증

주요 테스트:
- 메모리 사용량 모니터링 (최대 1GB 제한)
- 처리 속도 및 품질 측정
- 에러 복구 및 안정성 검증
- 실시간 진행률 표시
- 자동 샘플 파일 생성 (테스트용)
"""

import os
import sys
import asyncio
import time
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 필수 라이브러리 확인 및 설치
try:
    import cv2
    import librosa
    import soundfile as sf
    import psutil
    import aiofiles
    import whisper
    import pytesseract
    from moviepy.editor import VideoFileClip, AudioFileClip
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    print(f"⚠️ 필수 라이브러리 누락: {e}")
    print("설치 명령: pip install opencv-python librosa soundfile psutil aiofiles openai-whisper pytesseract moviepy pillow")
    sys.exit(1)

# 프로젝트 모듈 임포트
try:
    from core.ultra_large_file_processor_v21 import UltraLargeFileProcessor, ProcessingProgress
    from core.quality_analyzer_v21 import QualityAnalyzer
    from core.memory_optimizer_v21 import MemoryOptimizer
except ImportError as e:
    print(f"⚠️ 프로젝트 모듈 누락: {e}")
    print("core 디렉토리와 모듈들이 올바르게 설치되었는지 확인하세요.")

class SampleFileGenerator:
    """테스트용 샘플 파일 생성기"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_test_video(
        self, 
        duration_minutes: int = 60, 
        filename: str = "test_video_1hour.mp4"
    ) -> str:
        """테스트용 비디오 파일 생성 (1시간)"""
        output_path = self.output_dir / filename
        
        if output_path.exists():
            print(f"✅ 기존 테스트 비디오 사용: {output_path}")
            return str(output_path)
        
        print(f"🎬 테스트 비디오 생성 중... ({duration_minutes}분)")
        
        try:
            # 비디오 생성 (화면에 시간 표시)
            fps = 30
            width, height = 1280, 720
            total_frames = duration_minutes * 60 * fps
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # 진행률 표시
            frames_per_update = fps * 10  # 10초마다 업데이트
            
            for frame_num in range(total_frames):
                # 현재 시간 계산
                current_seconds = frame_num / fps
                minutes = int(current_seconds // 60)
                seconds = int(current_seconds % 60)
                
                # 프레임 생성 (그라데이션 배경 + 시간 텍스트)
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # 그라데이션 배경
                for y in range(height):
                    color = int(255 * (y / height))
                    frame[y, :] = [color // 3, color // 2, color]
                
                # 시간 텍스트 오버레이
                time_text = f"{minutes:02d}:{seconds:02d}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3
                thickness = 5
                text_size = cv2.getTextSize(time_text, font, font_scale, thickness)[0]
                text_x = (width - text_size[0]) // 2
                text_y = (height + text_size[1]) // 2
                
                cv2.putText(frame, time_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                
                # 추가 정보 텍스트
                info_text = f"Frame: {frame_num}/{total_frames} | Test Video for AI Processing"
                cv2.putText(frame, info_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                
                writer.write(frame)
                
                # 진행률 표시
                if frame_num % frames_per_update == 0:
                    progress = (frame_num / total_frames) * 100
                    print(f"  비디오 생성 진행률: {progress:.1f}%")
            
            writer.release()
            
            # 오디오 추가 (간단한 톤 생성)
            self._add_audio_to_video(str(output_path), duration_minutes)
            
            print(f"✅ 테스트 비디오 생성 완료: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f}MB)")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ 비디오 생성 실패: {e}")
            return None
    
    def _add_audio_to_video(self, video_path: str, duration_minutes: int):
        """비디오에 오디오 추가"""
        try:
            # 간단한 오디오 생성 (사인파 + 화이트 노이즈)
            sample_rate = 44100
            duration_seconds = duration_minutes * 60
            
            # 기본 톤 (440Hz A음)
            t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
            tone = 0.1 * np.sin(2 * np.pi * 440 * t)
            
            # 화이트 노이즈 추가 (배경음 시뮬레이션)
            noise = 0.02 * np.random.normal(0, 1, len(t))
            audio = tone + noise
            
            # 오디오 파일 저장
            audio_path = video_path.replace('.mp4', '_audio.wav')
            sf.write(audio_path, audio, sample_rate)
            
            # 비디오와 오디오 합치기
            import subprocess
            temp_video = video_path.replace('.mp4', '_temp.mp4')
            shutil.move(video_path, temp_video)
            
            cmd = [
                'ffmpeg', '-i', temp_video, '-i', audio_path,
                '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
                video_path, '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # 임시 파일 정리
            os.remove(temp_video)
            os.remove(audio_path)
            
        except Exception as e:
            print(f"⚠️ 오디오 추가 실패: {e}")
    
    def generate_test_images(self, count: int = 30) -> List[str]:
        """테스트용 이미지 파일들 생성 (30개)"""
        image_paths = []
        
        print(f"🖼️ 테스트 이미지 {count}개 생성 중...")
        
        for i in range(count):
            filename = f"test_image_{i+1:03d}.png"
            image_path = self.output_dir / filename
            
            if not image_path.exists():
                try:
                    # 이미지 생성 (다양한 패턴과 텍스트)
                    width, height = 1920, 1080
                    img = Image.new('RGB', (width, height), color=self._generate_gradient(i, width, height))
                    
                    draw = ImageDraw.Draw(img)
                    
                    # 큰 제목
                    try:
                        font = ImageFont.truetype("arial.ttf", 60)
                    except:
                        font = ImageFont.load_default()
                    
                    title = f"테스트 이미지 #{i+1}"
                    draw.text((100, 100), title, fill=(255, 255, 255), font=font)
                    
                    # 샘플 텍스트 (다국어)
                    sample_texts = [
                        f"Sample Text {i+1}",
                        f"주얼리 분석 테스트 #{i+1}",
                        "Diamond Quality Assessment",
                        "보석 품질 평가 시스템",
                        f"Test Image Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        "다이아몬드 감정서 OCR 테스트",
                        "Certificate Analysis Test",
                        f"Image Size: {width}x{height}"
                    ]
                    
                    y_pos = 250
                    for text in sample_texts:
                        draw.text((100, y_pos), text, fill=(255, 255, 255))
                        y_pos += 80
                    
                    # 기하학적 패턴 추가
                    self._add_geometric_patterns(draw, width, height, i)
                    
                    img.save(image_path, 'PNG')
                    
                except Exception as e:
                    print(f"⚠️ 이미지 {i+1} 생성 실패: {e}")
                    continue
            
            image_paths.append(str(image_path))
            
            if (i + 1) % 10 == 0:
                print(f"  이미지 생성 진행률: {(i+1)/count*100:.1f}%")
        
        print(f"✅ 테스트 이미지 {len(image_paths)}개 생성 완료")
        return image_paths
    
    def _generate_gradient(self, index: int, width: int, height: int) -> tuple:
        """그라데이션 색상 생성"""
        hue = (index * 37) % 360  # 다양한 색상
        base_color = self._hsv_to_rgb(hue, 70, 50)
        return base_color
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> tuple:
        """HSV를 RGB로 변환"""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h/360, s/100, v/100)
        return (int(r*255), int(g*255), int(b*255))
    
    def _add_geometric_patterns(self, draw, width: int, height: int, seed: int):
        """기하학적 패턴 추가"""
        np.random.seed(seed)
        
        # 원형 패턴
        for _ in range(5):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            r = np.random.randint(20, 100)
            color = tuple(np.random.randint(0, 256, 3))
            draw.ellipse([x-r, y-r, x+r, y+r], outline=color, width=3)
        
        # 직선 패턴
        for _ in range(3):
            x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
            x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
            color = tuple(np.random.randint(0, 256, 3))
            draw.line([x1, y1, x2, y2], fill=color, width=5)

class RealTimeMonitor:
    """실시간 성능 모니터링"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history = []
        self.memory_peaks = []
        self.processing_speeds = []
        
    async def monitor_progress(self, progress: ProcessingProgress):
        """진행률 및 성능 모니터링"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # 메트릭 수집
        metrics = {
            'timestamp': current_time,
            'elapsed_time': elapsed,
            'completed_chunks': progress.completed_chunks,
            'total_chunks': progress.total_chunks,
            'memory_usage_mb': progress.memory_usage_mb,
            'cpu_usage_percent': progress.cpu_usage_percent,
            'throughput_mb_per_sec': progress.throughput_mb_per_sec,
            'estimated_time_remaining': progress.estimated_time_remaining
        }
        
        self.metrics_history.append(metrics)
        
        # 메모리 피크 추적
        if progress.memory_usage_mb > 0:
            self.memory_peaks.append(progress.memory_usage_mb)
        
        # 처리 속도 추적
        if progress.throughput_mb_per_sec > 0:
            self.processing_speeds.append(progress.throughput_mb_per_sec)
        
        # 실시간 출력
        progress_pct = (progress.completed_chunks / progress.total_chunks * 100) if progress.total_chunks > 0 else 0
        
        print(f"\r🔄 진행률: {progress_pct:.1f}% "
              f"({progress.completed_chunks}/{progress.total_chunks}) | "
              f"메모리: {progress.memory_usage_mb:.1f}MB | "
              f"CPU: {progress.cpu_usage_percent:.1f}% | "
              f"속도: {progress.throughput_mb_per_sec:.1f}MB/s | "
              f"남은시간: {progress.estimated_time_remaining:.1f}초", end="")
        
        # 메모리 임계값 경고
        if progress.memory_usage_mb > 800:  # 800MB 이상
            print(f"\n⚠️ 메모리 사용량 높음: {progress.memory_usage_mb:.1f}MB")
    
    def generate_performance_report(self, output_dir: str) -> Dict[str, Any]:
        """성능 리포트 생성"""
        if not self.metrics_history:
            return {"error": "성능 데이터 없음"}
        
        report = {
            "summary": {
                "total_processing_time": time.time() - self.start_time,
                "peak_memory_usage_mb": max(self.memory_peaks) if self.memory_peaks else 0,
                "average_memory_usage_mb": np.mean(self.memory_peaks) if self.memory_peaks else 0,
                "average_processing_speed_mb_per_sec": np.mean(self.processing_speeds) if self.processing_speeds else 0,
                "peak_processing_speed_mb_per_sec": max(self.processing_speeds) if self.processing_speeds else 0,
                "total_metrics_collected": len(self.metrics_history)
            },
            "detailed_metrics": self.metrics_history[-50:],  # 최근 50개 메트릭
            "performance_analysis": self._analyze_performance()
        }
        
        # 차트 생성
        self._generate_performance_charts(output_dir)
        
        # 리포트 저장
        report_path = os.path.join(output_dir, "performance_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """성능 분석"""
        if not self.metrics_history:
            return {}
        
        memory_usage = [m['memory_usage_mb'] for m in self.metrics_history if m['memory_usage_mb'] > 0]
        cpu_usage = [m['cpu_usage_percent'] for m in self.metrics_history if m['cpu_usage_percent'] > 0]
        throughput = [m['throughput_mb_per_sec'] for m in self.metrics_history if m['throughput_mb_per_sec'] > 0]
        
        analysis = {
            "memory_stability": "안정" if np.std(memory_usage) < 50 else "불안정" if memory_usage else "데이터없음",
            "cpu_efficiency": "효율적" if np.mean(cpu_usage) < 80 else "높음" if cpu_usage else "데이터없음",
            "throughput_consistency": "일정" if np.std(throughput) < 1.0 else "변동" if throughput else "데이터없음",
            "performance_grade": self._calculate_performance_grade(memory_usage, cpu_usage, throughput)
        }
        
        return analysis
    
    def _calculate_performance_grade(self, memory_usage: List[float], cpu_usage: List[float], throughput: List[float]) -> str:
        """성능 등급 계산"""
        score = 100
        
        # 메모리 점수 (800MB 이하면 좋음)
        if memory_usage:
            avg_memory = np.mean(memory_usage)
            if avg_memory > 800:
                score -= 30
            elif avg_memory > 600:
                score -= 15
        
        # CPU 점수 (80% 이하면 좋음)
        if cpu_usage:
            avg_cpu = np.mean(cpu_usage)
            if avg_cpu > 90:
                score -= 20
            elif avg_cpu > 80:
                score -= 10
        
        # 처리량 점수 (1MB/s 이상이면 좋음)
        if throughput:
            avg_throughput = np.mean(throughput)
            if avg_throughput < 0.5:
                score -= 20
            elif avg_throughput < 1.0:
                score -= 10
        
        if score >= 90:
            return "A (우수)"
        elif score >= 80:
            return "B (양호)"
        elif score >= 70:
            return "C (보통)"
        elif score >= 60:
            return "D (개선필요)"
        else:
            return "F (불량)"
    
    def _generate_performance_charts(self, output_dir: str):
        """성능 차트 생성"""
        if not self.metrics_history:
            return
        
        try:
            plt.style.use('seaborn-v0_8')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            times = [(m['timestamp'] - self.start_time) / 60 for m in self.metrics_history]  # 분 단위
            
            # 1. 메모리 사용량
            memory_data = [m['memory_usage_mb'] for m in self.metrics_history]
            ax1.plot(times, memory_data, 'b-', linewidth=2)
            ax1.set_title('메모리 사용량 (MB)', fontsize=12)
            ax1.set_xlabel('시간 (분)')
            ax1.set_ylabel('메모리 (MB)')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=800, color='r', linestyle='--', alpha=0.7, label='임계값 (800MB)')
            ax1.legend()
            
            # 2. CPU 사용률
            cpu_data = [m['cpu_usage_percent'] for m in self.metrics_history]
            ax2.plot(times, cpu_data, 'g-', linewidth=2)
            ax2.set_title('CPU 사용률 (%)', fontsize=12)
            ax2.set_xlabel('시간 (분)')
            ax2.set_ylabel('CPU (%)')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='권장 임계값 (80%)')
            ax2.legend()
            
            # 3. 처리 속도
            throughput_data = [m['throughput_mb_per_sec'] for m in self.metrics_history]
            ax3.plot(times, throughput_data, 'purple', linewidth=2)
            ax3.set_title('처리 속도 (MB/초)', fontsize=12)
            ax3.set_xlabel('시간 (분)')
            ax3.set_ylabel('속도 (MB/s)')
            ax3.grid(True, alpha=0.3)
            
            # 4. 진행률
            progress_data = [(m['completed_chunks'] / m['total_chunks'] * 100) if m['total_chunks'] > 0 else 0 
                           for m in self.metrics_history]
            ax4.plot(times, progress_data, 'orange', linewidth=2)
            ax4.set_title('처리 진행률 (%)', fontsize=12)
            ax4.set_xlabel('시간 (분)')
            ax4.set_ylabel('진행률 (%)')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 100)
            
            plt.tight_layout()
            chart_path = os.path.join(output_dir, "performance_charts.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\n📊 성능 차트 저장: {chart_path}")
            
        except Exception as e:
            print(f"⚠️ 차트 생성 실패: {e}")

class LargeFileRealTest:
    """고용량 파일 실전 테스트 메인 클래스"""
    
    def __init__(self, test_dir: str = None):
        self.test_dir = Path(test_dir or tempfile.mkdtemp(prefix="large_file_test_"))
        self.test_dir.mkdir(exist_ok=True)
        
        # 컴포넌트 초기화
        self.sample_generator = SampleFileGenerator(str(self.test_dir))
        self.monitor = RealTimeMonitor()
        self.processor = UltraLargeFileProcessor(max_memory_mb=1000, max_workers=4)
        
        print(f"🧪 테스트 디렉토리: {self.test_dir}")
    
    async def run_full_test(self) -> Dict[str, Any]:
        """전체 테스트 실행"""
        print("🚀 대용량 파일 실전 테스트 시작!")
        print("=" * 60)
        
        test_results = {
            "test_info": {
                "start_time": datetime.now().isoformat(),
                "test_directory": str(self.test_dir),
                "system_info": self._get_system_info()
            },
            "file_generation": {},
            "processing_results": {},
            "performance_report": {},
            "test_summary": {}
        }
        
        try:
            # 1. 테스트 파일 생성
            print("1️⃣ 테스트 파일 생성 단계")
            video_file = self.sample_generator.generate_test_video(duration_minutes=60)
            image_files = self.sample_generator.generate_test_images(count=30)
            
            test_results["file_generation"] = {
                "video_file": video_file,
                "video_size_mb": os.path.getsize(video_file) / 1024 / 1024 if video_file else 0,
                "image_files_count": len(image_files),
                "total_image_size_mb": sum(os.path.getsize(f) for f in image_files) / 1024 / 1024,
                "generation_success": video_file is not None and len(image_files) > 0
            }
            
            if not test_results["file_generation"]["generation_success"]:
                raise Exception("테스트 파일 생성 실패")
            
            print(f"✅ 파일 생성 완료:")
            print(f"   - 비디오: {test_results['file_generation']['video_size_mb']:.1f}MB")
            print(f"   - 이미지: {len(image_files)}개 ({test_results['file_generation']['total_image_size_mb']:.1f}MB)")
            
            # 2. 프로세서 설정
            print("\n2️⃣ 프로세서 설정 단계")
            self.processor.set_progress_callback(self.monitor.monitor_progress)
            
            # 3. 실제 처리 실행
            print("\n3️⃣ 대용량 파일 처리 실행")
            print("처리 시작... (실시간 모니터링)")
            
            processing_start = time.time()
            
            processing_result = await self.processor.process_ultra_large_files(
                video_files=[video_file] if video_file else [],
                image_files=image_files,
                output_dir=str(self.test_dir / "results")
            )
            
            processing_end = time.time()
            processing_time = processing_end - processing_start
            
            print(f"\n✅ 처리 완료! 총 소요시간: {processing_time:.1f}초")
            
            test_results["processing_results"] = processing_result
            test_results["processing_results"]["actual_processing_time"] = processing_time
            
            # 4. 성능 리포트 생성
            print("\n4️⃣ 성능 분석 및 리포트 생성")
            performance_report = self.monitor.generate_performance_report(str(self.test_dir))
            test_results["performance_report"] = performance_report
            
            # 5. 테스트 요약
            test_results["test_summary"] = self._generate_test_summary(test_results)
            
            # 6. 최종 리포트 저장
            final_report_path = self.test_dir / "final_test_report.json"
            with open(final_report_path, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            
            print(f"\n📋 최종 리포트 저장: {final_report_path}")
            
            # 7. 결과 출력
            self._print_test_summary(test_results)
            
            return test_results
            
        except Exception as e:
            print(f"\n❌ 테스트 실패: {e}")
            test_results["error"] = str(e)
            return test_results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 수집"""
        try:
            return {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024**3,
                "memory_available_gb": psutil.virtual_memory().available / 1024**3,
                "disk_free_gb": psutil.disk_usage(str(self.test_dir)).free / 1024**3,
                "python_version": sys.version,
                "platform": sys.platform
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_test_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """테스트 요약 생성"""
        try:
            processing = test_results.get("processing_results", {})
            performance = test_results.get("performance_report", {})
            file_gen = test_results.get("file_generation", {})
            
            summary = {
                "overall_success": not test_results.get("error") and processing.get("processing_summary", {}).get("successful_chunks", 0) > 0,
                "total_data_processed_mb": file_gen.get("video_size_mb", 0) + file_gen.get("total_image_size_mb", 0),
                "processing_efficiency": processing.get("processing_summary", {}).get("successful_chunks", 0) / max(1, processing.get("processing_summary", {}).get("total_chunks", 1)) * 100,
                "memory_performance": "우수" if performance.get("summary", {}).get("peak_memory_usage_mb", 1000) < 800 else "주의",
                "speed_performance": "우수" if performance.get("summary", {}).get("average_processing_speed_mb_per_sec", 0) > 1.0 else "보통",
                "quality_metrics": {
                    "video_quality": processing.get("quality_metrics", {}).get("average_video_quality", 0),
                    "text_extraction_count": processing.get("quality_metrics", {}).get("total_text_extracted", 0),
                    "ocr_success_rate": len(processing.get("image_ocr_results", [])) / max(1, file_gen.get("image_files_count", 1)) * 100
                },
                "recommendations": self._generate_recommendations(test_results)
            }
            
            return summary
            
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        try:
            performance = test_results.get("performance_report", {})
            summary = performance.get("summary", {})
            
            # 메모리 관련 권장사항
            peak_memory = summary.get("peak_memory_usage_mb", 0)
            if peak_memory > 900:
                recommendations.append("⚠️ 메모리 사용량이 매우 높습니다. 청크 크기를 줄이거나 병렬 처리 수를 감소시키세요.")
            elif peak_memory > 800:
                recommendations.append("📊 메모리 사용량이 높습니다. 모니터링을 강화하세요.")
            else:
                recommendations.append("✅ 메모리 사용량이 안정적입니다.")
            
            # 처리 속도 관련 권장사항
            avg_speed = summary.get("average_processing_speed_mb_per_sec", 0)
            if avg_speed < 0.5:
                recommendations.append("🐌 처리 속도가 느립니다. 더 강력한 하드웨어나 병렬 처리 수 증가를 고려하세요.")
            elif avg_speed < 1.0:
                recommendations.append("⚡ 처리 속도가 보통입니다. 최적화 여지가 있습니다.")
            else:
                recommendations.append("🚀 처리 속도가 우수합니다.")
            
            # 품질 관련 권장사항
            processing = test_results.get("processing_results", {})
            success_rate = processing.get("processing_summary", {}).get("successful_chunks", 0) / max(1, processing.get("processing_summary", {}).get("total_chunks", 1))
            
            if success_rate < 0.8:
                recommendations.append("❌ 처리 성공률이 낮습니다. 에러 로그를 확인하고 품질 설정을 조정하세요.")
            elif success_rate < 0.95:
                recommendations.append("⚠️ 일부 청크 처리에 실패했습니다. 로그를 확인하세요.")
            else:
                recommendations.append("✅ 모든 청크가 성공적으로 처리되었습니다.")
            
        except Exception as e:
            recommendations.append(f"권장사항 생성 중 오류: {e}")
        
        return recommendations
    
    def _print_test_summary(self, test_results: Dict[str, Any]):
        """테스트 요약 출력"""
        print("\n" + "="*60)
        print("📋 최종 테스트 결과 요약")
        print("="*60)
        
        summary = test_results.get("test_summary", {})
        
        # 전체 성공 여부
        success = summary.get("overall_success", False)
        print(f"🎯 전체 테스트 결과: {'✅ 성공' if success else '❌ 실패'}")
        
        # 처리 데이터량
        total_mb = summary.get("total_data_processed_mb", 0)
        print(f"📊 처리된 데이터량: {total_mb:.1f}MB")
        
        # 처리 효율성
        efficiency = summary.get("processing_efficiency", 0)
        print(f"⚡ 처리 효율성: {efficiency:.1f}%")
        
        # 성능 평가
        memory_perf = summary.get("memory_performance", "알 수 없음")
        speed_perf = summary.get("speed_performance", "알 수 없음")
        print(f"🧠 메모리 성능: {memory_perf}")
        print(f"🚀 속도 성능: {speed_perf}")
        
        # 품질 메트릭
        quality = summary.get("quality_metrics", {})
        if quality:
            print(f"💎 비디오 품질: {quality.get('video_quality', 0):.2f}")
            print(f"📝 텍스트 추출: {quality.get('text_extraction_count', 0)}건")
            print(f"🔤 OCR 성공률: {quality.get('ocr_success_rate', 0):.1f}%")
        
        # 권장사항
        recommendations = summary.get("recommendations", [])
        if recommendations:
            print("\n💡 권장사항:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # 파일 경로
        print(f"\n📁 테스트 결과 위치: {self.test_dir}")
        print("="*60)

async def run_quick_test():
    """빠른 테스트 (5분 비디오 + 5개 이미지)"""
    print("🔥 빠른 테스트 모드 (5분 비디오 + 5개 이미지)")
    
    tester = LargeFileRealTest()
    
    # 빠른 테스트용 파일 생성
    video_file = tester.sample_generator.generate_test_video(duration_minutes=5, filename="quick_test_video.mp4")
    image_files = tester.sample_generator.generate_test_images(count=5)
    
    if video_file and image_files:
        try:
            processing_result = await tester.processor.process_ultra_large_files(
                video_files=[video_file],
                image_files=image_files,
                output_dir=str(tester.test_dir / "quick_results")
            )
            
            print("✅ 빠른 테스트 완료!")
            print(f"📊 결과: {processing_result['processing_summary']}")
            
        except Exception as e:
            print(f"❌ 빠른 테스트 실패: {e}")
    else:
        print("❌ 테스트 파일 생성 실패")

async def main():
    """메인 실행 함수"""
    print("🎯 대용량 파일 실전 테스트 시스템 v2.1")
    print("=" * 60)
    print("선택하세요:")
    print("1. 전체 테스트 (1시간 비디오 + 30개 이미지)")
    print("2. 빠른 테스트 (5분 비디오 + 5개 이미지)")
    print("3. 종료")
    
    try:
        choice = input("\n선택 (1-3): ").strip()
        
        if choice == "1":
            print("\n🚀 전체 테스트 시작...")
            tester = LargeFileRealTest()
            await tester.run_full_test()
            
        elif choice == "2":
            print("\n⚡ 빠른 테스트 시작...")
            await run_quick_test()
            
        elif choice == "3":
            print("👋 테스트를 종료합니다.")
            
        else:
            print("❌ 잘못된 선택입니다.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ 사용자가 테스트를 중단했습니다.")
        
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('large_file_test.log'),
            logging.StreamHandler()
        ]
    )
    
    # 비동기 실행
    asyncio.run(main())
