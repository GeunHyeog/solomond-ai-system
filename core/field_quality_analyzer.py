"""
솔로몬드 AI 시스템 - 현장 품질 분석 엔진
현장에서 촬영한 오디오/비디오의 노이즈 분석 및 품질 평가 모듈
"""

import numpy as np
import librosa
import scipy.signal
from typing import Dict, List, Optional, Tuple, Any
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os
from pathlib import Path
import json
from datetime import datetime

# 오디오 처리 라이브러리
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False

class FieldQualityAnalyzer:
    """현장 오디오/비디오 품질 분석 클래스"""
    
    def __init__(self):
        # 품질 평가 기준값들
        self.quality_thresholds = {
            "excellent": {"snr": 25, "clarity": 0.9, "noise": 0.1},
            "good": {"snr": 20, "clarity": 0.8, "noise": 0.2},
            "fair": {"snr": 15, "clarity": 0.7, "noise": 0.3},
            "poor": {"snr": 10, "clarity": 0.6, "noise": 0.4}
        }
        
        # 노이즈 유형별 특성
        self.noise_signatures = {
            "air_conditioning": {"freq_range": (100, 1000), "pattern": "constant"},
            "crowd_noise": {"freq_range": (200, 4000), "pattern": "variable"},
            "electronic_hum": {"freq_range": (50, 60), "pattern": "constant"},
            "traffic": {"freq_range": (50, 2000), "pattern": "variable"},
            "wind": {"freq_range": (10, 500), "pattern": "variable"},
            "microphone_handling": {"freq_range": (1, 200), "pattern": "burst"}
        }
        
        # VAD (Voice Activity Detection) 초기화
        self.vad = None
        if WEBRTCVAD_AVAILABLE:
            try:
                self.vad = webrtcvad.Vad(2)  # 중간 강도
            except Exception as e:
                logging.warning(f"WebRTC VAD 초기화 실패: {e}")
        
        # 스레드 풀 executor
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logging.info("현장 품질 분석 엔진 초기화 완료")
    
    async def analyze_audio_quality(self, 
                                  audio_data: bytes, 
                                  filename: str,
                                  sample_rate: int = None) -> Dict:
        """
        현장 오디오 품질 종합 분석
        
        Args:
            audio_data: 오디오 바이너리 데이터
            filename: 파일명
            sample_rate: 샘플링 레이트 (None이면 자동 감지)
            
        Returns:
            품질 분석 결과 딕셔너리
        """
        try:
            print(f"🔊 현장 오디오 품질 분석 시작: {filename}")
            
            # 오디오 데이터 로드
            audio_array, sr = await self._load_audio_data(audio_data, sample_rate)
            
            if audio_array is None:
                return {
                    "success": False,
                    "error": "오디오 데이터 로드 실패",
                    "filename": filename
                }
            
            # 병렬로 여러 분석 수행
            analysis_tasks = [
                self._analyze_snr(audio_array, sr),
                self._analyze_noise_characteristics(audio_array, sr),
                self._analyze_speech_clarity(audio_array, sr),
                self._detect_noise_types(audio_array, sr),
                self._analyze_volume_levels(audio_array, sr)
            ]
            
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # 결과 통합
            snr_result = results[0] if not isinstance(results[0], Exception) else {}
            noise_result = results[1] if not isinstance(results[1], Exception) else {}
            clarity_result = results[2] if not isinstance(results[2], Exception) else {}
            noise_types = results[3] if not isinstance(results[3], Exception) else {}
            volume_result = results[4] if not isinstance(results[4], Exception) else {}
            
            # 전체 품질 점수 계산
            overall_score = self._calculate_overall_quality_score(
                snr_result, noise_result, clarity_result, volume_result
            )
            
            # 개선 제안 생성
            recommendations = self._generate_improvement_recommendations(
                snr_result, noise_result, clarity_result, noise_types, volume_result
            )
            
            # 최종 결과 구성
            result = {
                "success": True,
                "filename": filename,
                "audio_info": {
                    "duration": len(audio_array) / sr,
                    "sample_rate": sr,
                    "channels": 1 if len(audio_array.shape) == 1 else audio_array.shape[1],
                    "file_size_mb": round(len(audio_data) / (1024 * 1024), 2)
                },
                "quality_analysis": {
                    "overall_score": overall_score,
                    "snr_analysis": snr_result,
                    "noise_analysis": noise_result,
                    "clarity_analysis": clarity_result,
                    "volume_analysis": volume_result,
                    "detected_noise_types": noise_types
                },
                "quality_assessment": self._assess_quality_level(overall_score),
                "recommendations": recommendations,
                "analyzed_at": datetime.now().isoformat()
            }
            
            print(f"✅ 품질 분석 완료: 전체 점수 {overall_score}/100")
            return result
            
        except Exception as e:
            logging.error(f"현장 오디오 품질 분석 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "filename": filename
            }
    
    async def _load_audio_data(self, audio_data: bytes, target_sr: int = None) -> Tuple[np.ndarray, int]:
        """오디오 데이터 로드 및 전처리"""
        try:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                # librosa로 오디오 로드
                audio_array, sr = librosa.load(temp_path, sr=target_sr, mono=True)
                
                # 정규화
                if np.max(np.abs(audio_array)) > 0:
                    audio_array = audio_array / np.max(np.abs(audio_array))
                
                return audio_array, sr
                
            finally:
                # 임시 파일 정리
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logging.error(f"오디오 데이터 로드 오류: {e}")
            return None, None
    
    async def _analyze_snr(self, audio: np.ndarray, sr: int) -> Dict:
        """신호 대 잡음비 (SNR) 분석"""
        try:
            # VAD를 사용하여 음성 구간과 비음성 구간 분리
            voice_segments, noise_segments = await self._separate_voice_noise(audio, sr)
            
            if len(voice_segments) == 0 or len(noise_segments) == 0:
                # VAD 실패 시 간단한 에너지 기반 분석
                return await self._analyze_snr_energy_based(audio, sr)
            
            # 음성 신호 전력 계산
            voice_power = np.mean([np.mean(segment**2) for segment in voice_segments])
            
            # 노이즈 신호 전력 계산
            noise_power = np.mean([np.mean(segment**2) for segment in noise_segments])
            
            # SNR 계산 (dB)
            if noise_power > 0:
                snr_db = 10 * np.log10(voice_power / noise_power)
            else:
                snr_db = 60  # 노이즈가 거의 없는 경우
            
            # SNR 품질 평가
            if snr_db >= 25:
                snr_quality = "excellent"
            elif snr_db >= 20:
                snr_quality = "good"
            elif snr_db >= 15:
                snr_quality = "fair"
            else:
                snr_quality = "poor"
            
            return {
                "snr_db": round(snr_db, 2),
                "voice_power": round(voice_power, 6),
                "noise_power": round(noise_power, 6),
                "quality_level": snr_quality,
                "voice_segments_count": len(voice_segments),
                "noise_segments_count": len(noise_segments)
            }
            
        except Exception as e:
            logging.error(f"SNR 분석 오류: {e}")
            return {"error": str(e)}
    
    async def _analyze_snr_energy_based(self, audio: np.ndarray, sr: int) -> Dict:
        """에너지 기반 간단한 SNR 분석"""
        try:
            # 프레임 단위로 에너지 계산
            frame_length = int(0.025 * sr)  # 25ms 프레임
            hop_length = int(0.01 * sr)     # 10ms 홉
            
            # 에너지 계산
            energy = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                energy.append(np.sum(frame**2))
            
            energy = np.array(energy)
            
            # 상위 30%를 음성, 하위 30%를 노이즈로 가정
            sorted_energy = np.sort(energy)
            top_30_percent = sorted_energy[int(0.7 * len(sorted_energy)):]
            bottom_30_percent = sorted_energy[:int(0.3 * len(sorted_energy))]
            
            voice_power = np.mean(top_30_percent)
            noise_power = np.mean(bottom_30_percent)
            
            if noise_power > 0:
                snr_db = 10 * np.log10(voice_power / noise_power)
            else:
                snr_db = 60
            
            return {
                "snr_db": round(snr_db, 2),
                "voice_power": round(voice_power, 6),
                "noise_power": round(noise_power, 6),
                "quality_level": "good" if snr_db >= 20 else "fair" if snr_db >= 15 else "poor",
                "method": "energy_based"
            }
            
        except Exception as e:
            logging.error(f"에너지 기반 SNR 분석 오류: {e}")
            return {"error": str(e)}
    
    async def _separate_voice_noise(self, audio: np.ndarray, sr: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """음성과 노이즈 구간 분리"""
        voice_segments = []
        noise_segments = []
        
        try:
            if self.vad and WEBRTCVAD_AVAILABLE:
                # WebRTC VAD 사용
                frame_duration = 30  # ms
                frame_length = int(sr * frame_duration / 1000)
                
                for i in range(0, len(audio) - frame_length, frame_length):
                    frame = audio[i:i + frame_length]
                    
                    # 16kHz, 16-bit PCM으로 변환
                    frame_16k = librosa.resample(frame, orig_sr=sr, target_sr=16000)
                    frame_bytes = (frame_16k * 32767).astype(np.int16).tobytes()
                    
                    try:
                        is_speech = self.vad.is_speech(frame_bytes, 16000)
                        if is_speech:
                            voice_segments.append(frame)
                        else:
                            noise_segments.append(frame)
                    except:
                        # VAD 실패 시 에너지 기반으로 분류
                        energy = np.sum(frame**2)
                        if energy > np.mean(audio**2):
                            voice_segments.append(frame)
                        else:
                            noise_segments.append(frame)
            else:
                # 간단한 에너지 기반 분류
                frame_length = int(0.03 * sr)  # 30ms 프레임
                threshold = np.mean(audio**2) * 2
                
                for i in range(0, len(audio) - frame_length, frame_length):
                    frame = audio[i:i + frame_length]
                    energy = np.sum(frame**2)
                    
                    if energy > threshold:
                        voice_segments.append(frame)
                    else:
                        noise_segments.append(frame)
        
        except Exception as e:
            logging.error(f"음성/노이즈 분리 오류: {e}")
        
        return voice_segments, noise_segments
    
    async def _analyze_noise_characteristics(self, audio: np.ndarray, sr: int) -> Dict:
        """노이즈 특성 분석"""
        try:
            # 스펙트로그램 계산
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            
            # 주파수별 노이즈 레벨 분석
            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
            noise_profile = np.mean(magnitude, axis=1)
            
            # 노이즈 집중 주파수 대역 찾기
            peak_freq_idx = np.argmax(noise_profile)
            peak_frequency = freq_bins[peak_freq_idx]
            
            # 노이즈 일관성 분석 (시간에 따른 변화)
            temporal_variance = np.var(magnitude, axis=1)
            consistency_score = 1 - (np.mean(temporal_variance) / np.max(temporal_variance))
            
            # 전체 노이즈 레벨
            overall_noise_level = np.mean(noise_profile)
            
            return {
                "peak_frequency": round(peak_frequency, 2),
                "overall_noise_level": round(overall_noise_level, 6),
                "consistency_score": round(consistency_score, 3),
                "frequency_distribution": {
                    "low_freq": round(np.mean(noise_profile[:len(noise_profile)//4]), 6),
                    "mid_freq": round(np.mean(noise_profile[len(noise_profile)//4:3*len(noise_profile)//4]), 6),
                    "high_freq": round(np.mean(noise_profile[3*len(noise_profile)//4:]), 6)
                }
            }
            
        except Exception as e:
            logging.error(f"노이즈 특성 분석 오류: {e}")
            return {"error": str(e)}
    
    async def _analyze_speech_clarity(self, audio: np.ndarray, sr: int) -> Dict:
        """음성 명확도 분석"""
        try:
            # 스펙트럼 중심점 계산 (음성 명확도 지표)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            mean_centroid = np.mean(spectral_centroids)
            
            # 스펙트럼 롤오프 계산
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            mean_rolloff = np.mean(spectral_rolloff)
            
            # 제로 크로싱 레이트 (음성 품질 지표)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            mean_zcr = np.mean(zcr)
            
            # MFCC 기반 음성 품질 평가
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_variance = np.var(mfcc, axis=1)
            speech_quality = 1 - (np.mean(mfcc_variance) / np.max(mfcc_variance))
            
            # 명확도 점수 계산 (0-1)
            clarity_score = min(1.0, (speech_quality + (1 - mean_zcr)) / 2)
            
            return {
                "clarity_score": round(clarity_score, 3),
                "spectral_centroid": round(mean_centroid, 2),
                "spectral_rolloff": round(mean_rolloff, 2),
                "zero_crossing_rate": round(mean_zcr, 4),
                "speech_quality": round(speech_quality, 3)
            }
            
        except Exception as e:
            logging.error(f"음성 명확도 분석 오류: {e}")
            return {"error": str(e)}
    
    async def _detect_noise_types(self, audio: np.ndarray, sr: int) -> Dict:
        """노이즈 유형 감지"""
        detected_noises = {}
        
        try:
            # 스펙트로그램 계산
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
            
            for noise_type, signature in self.noise_signatures.items():
                freq_range = signature["freq_range"]
                pattern = signature["pattern"]
                
                # 해당 주파수 대역의 에너지 추출
                freq_mask = (freq_bins >= freq_range[0]) & (freq_bins <= freq_range[1])
                if np.any(freq_mask):
                    band_energy = np.mean(magnitude[freq_mask], axis=0)
                    
                    # 패턴 분석
                    if pattern == "constant":
                        # 일정한 패턴인지 확인
                        variance = np.var(band_energy)
                        consistency = 1 - (variance / np.mean(band_energy) if np.mean(band_energy) > 0 else 1)
                        confidence = min(1.0, consistency * np.mean(band_energy) * 1000)
                    else:  # variable
                        # 변동하는 패턴인지 확인
                        variance = np.var(band_energy)
                        variability = variance / np.mean(band_energy) if np.mean(band_energy) > 0 else 0
                        confidence = min(1.0, variability * np.mean(band_energy) * 1000)
                    
                    if confidence > 0.3:  # 임계값 이상일 때만 감지로 판단
                        detected_noises[noise_type] = {
                            "confidence": round(confidence, 3),
                            "avg_energy": round(np.mean(band_energy), 6),
                            "frequency_range": freq_range
                        }
            
            return detected_noises
            
        except Exception as e:
            logging.error(f"노이즈 유형 감지 오류: {e}")
            return {"error": str(e)}
    
    async def _analyze_volume_levels(self, audio: np.ndarray, sr: int) -> Dict:
        """볼륨 레벨 분석"""
        try:
            # RMS 에너지 계산
            rms_energy = librosa.feature.rms(y=audio)[0]
            
            # dB로 변환
            rms_db = 20 * np.log10(rms_energy + 1e-6)  # 로그 계산 시 0 방지
            
            # 통계 계산
            mean_db = np.mean(rms_db)
            max_db = np.max(rms_db)
            min_db = np.min(rms_db)
            dynamic_range = max_db - min_db
            
            # 클리핑 감지
            clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
            
            # 볼륨 일관성
            volume_consistency = 1 - (np.std(rms_db) / abs(mean_db) if abs(mean_db) > 0 else 1)
            
            return {
                "mean_db": round(mean_db, 2),
                "max_db": round(max_db, 2),
                "min_db": round(min_db, 2),
                "dynamic_range": round(dynamic_range, 2),
                "clipping_ratio": round(clipping_ratio, 4),
                "volume_consistency": round(volume_consistency, 3)
            }
            
        except Exception as e:
            logging.error(f"볼륨 레벨 분석 오류: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_quality_score(self, 
                                       snr_result: Dict, 
                                       noise_result: Dict, 
                                       clarity_result: Dict, 
                                       volume_result: Dict) -> int:
        """전체 품질 점수 계산 (0-100)"""
        try:
            score = 0
            weight_sum = 0
            
            # SNR 점수 (가중치: 40%)
            if "snr_db" in snr_result:
                snr_score = min(100, max(0, (snr_result["snr_db"] + 10) * 2.5))  # -10dB~30dB -> 0~100
                score += snr_score * 0.4
                weight_sum += 0.4
            
            # 명확도 점수 (가중치: 30%)
            if "clarity_score" in clarity_result:
                clarity_score = clarity_result["clarity_score"] * 100
                score += clarity_score * 0.3
                weight_sum += 0.3
            
            # 볼륨 일관성 (가중치: 20%)
            if "volume_consistency" in volume_result:
                volume_score = volume_result["volume_consistency"] * 100
                score += volume_score * 0.2
                weight_sum += 0.2
            
            # 노이즈 레벨 (가중치: 10%)
            if "overall_noise_level" in noise_result:
                # 노이즈가 적을수록 높은 점수
                noise_score = max(0, 100 - (noise_result["overall_noise_level"] * 10000))
                score += noise_score * 0.1
                weight_sum += 0.1
            
            # 가중 평균 계산
            if weight_sum > 0:
                final_score = int(score / weight_sum)
            else:
                final_score = 50  # 기본값
            
            return max(0, min(100, final_score))
            
        except Exception as e:
            logging.error(f"전체 품질 점수 계산 오류: {e}")
            return 50
    
    def _assess_quality_level(self, score: int) -> Dict:
        """품질 점수를 레벨로 변환"""
        if score >= 85:
            return {
                "level": "excellent",
                "description": "매우 우수한 품질",
                "color": "green",
                "icon": "🟢"
            }
        elif score >= 70:
            return {
                "level": "good", 
                "description": "양호한 품질",
                "color": "lightgreen",
                "icon": "🟡"
            }
        elif score >= 50:
            return {
                "level": "fair",
                "description": "보통 품질",
                "color": "orange", 
                "icon": "🟠"
            }
        else:
            return {
                "level": "poor",
                "description": "품질 개선 필요",
                "color": "red",
                "icon": "🔴"
            }
    
    def _generate_improvement_recommendations(self, 
                                            snr_result: Dict,
                                            noise_result: Dict, 
                                            clarity_result: Dict,
                                            noise_types: Dict,
                                            volume_result: Dict) -> List[Dict]:
        """품질 개선 제안 생성"""
        recommendations = []
        
        try:
            # SNR 기반 제안
            if "snr_db" in snr_result:
                snr = snr_result["snr_db"]
                if snr < 15:
                    recommendations.append({
                        "type": "snr",
                        "priority": "high",
                        "issue": f"신호 대 잡음비가 낮습니다 ({snr:.1f}dB)",
                        "solution": "마이크를 화자에게 더 가까이 배치하거나 더 조용한 환경에서 녹음하세요",
                        "icon": "🎤"
                    })
                elif snr < 20:
                    recommendations.append({
                        "type": "snr",
                        "priority": "medium",
                        "issue": f"배경 노이즈가 다소 있습니다 ({snr:.1f}dB)",
                        "solution": "가능하면 더 조용한 환경에서 녹음하시기 바랍니다",
                        "icon": "🔇"
                    })
            
            # 노이즈 유형별 제안
            for noise_type, detection in noise_types.items():
                if detection["confidence"] > 0.5:
                    if noise_type == "air_conditioning":
                        recommendations.append({
                            "type": "noise",
                            "priority": "medium",
                            "issue": "에어컨 소음이 감지되었습니다",
                            "solution": "에어컨을 일시적으로 끄거나 더 멀리 떨어진 곳에서 녹음하세요",
                            "icon": "❄️"
                        })
                    elif noise_type == "crowd_noise":
                        recommendations.append({
                            "type": "noise",
                            "priority": "high",
                            "issue": "사람들의 대화 소음이 감지되었습니다",
                            "solution": "더 조용한 공간으로 이동하거나 지향성 마이크를 사용하세요",
                            "icon": "👥"
                        })
                    elif noise_type == "electronic_hum":
                        recommendations.append({
                            "type": "noise",
                            "priority": "medium",
                            "issue": "전자기기 험(hum) 노이즈가 감지되었습니다",
                            "solution": "전자기기에서 멀리 떨어져 녹음하거나 다른 전원을 사용하세요",
                            "icon": "⚡"
                        })
            
            # 명확도 기반 제안
            if "clarity_score" in clarity_result:
                clarity = clarity_result["clarity_score"]
                if clarity < 0.6:
                    recommendations.append({
                        "type": "clarity",
                        "priority": "high",
                        "issue": f"음성 명확도가 낮습니다 ({clarity:.2f})",
                        "solution": "화자가 더 크고 명확하게 발음하도록 요청하세요",
                        "icon": "🗣️"
                    })
            
            # 볼륨 기반 제안
            if "mean_db" in volume_result:
                mean_db = volume_result["mean_db"]
                if mean_db < -30:
                    recommendations.append({
                        "type": "volume",
                        "priority": "high",
                        "issue": f"녹음 볼륨이 너무 낮습니다 ({mean_db:.1f}dB)",
                        "solution": "마이크 게인을 높이거나 화자에게 더 가까이 가세요",
                        "icon": "🔊"
                    })
                elif volume_result.get("clipping_ratio", 0) > 0.01:
                    recommendations.append({
                        "type": "volume",
                        "priority": "medium",
                        "issue": "오디오 클리핑이 감지되었습니다",
                        "solution": "녹음 레벨을 낮추어 왜곡을 방지하세요",
                        "icon": "📉"
                    })
            
            # 일반적인 제안 (품질이 전반적으로 낮은 경우)
            if len(recommendations) >= 3:
                recommendations.append({
                    "type": "general",
                    "priority": "medium",
                    "issue": "전반적인 녹음 품질 개선이 필요합니다",
                    "solution": "조용한 환경에서 고품질 마이크로 재녹음을 권장합니다",
                    "icon": "🎯"
                })
            
            return recommendations[:5]  # 최대 5개 제안
            
        except Exception as e:
            logging.error(f"개선 제안 생성 오류: {e}")
            return [{
                "type": "error",
                "priority": "low",
                "issue": "개선 제안 생성 중 오류 발생",
                "solution": "수동으로 오디오 품질을 확인해 주세요",
                "icon": "⚠️"
            }]

# 전역 인스턴스
_field_quality_analyzer_instance = None

def get_field_quality_analyzer() -> FieldQualityAnalyzer:
    """전역 현장 품질 분석기 인스턴스 반환"""
    global _field_quality_analyzer_instance
    if _field_quality_analyzer_instance is None:
        _field_quality_analyzer_instance = FieldQualityAnalyzer()
    return _field_quality_analyzer_instance

# 편의 함수들
async def analyze_field_audio_quality(audio_data: bytes, filename: str, **kwargs) -> Dict:
    """현장 오디오 품질 분석 편의 함수"""
    analyzer = get_field_quality_analyzer()
    return await analyzer.analyze_audio_quality(audio_data, filename, **kwargs)

def check_quality_analyzer_support() -> Dict:
    """품질 분석기 지원 상태 확인"""
    return {
        "libraries": {
            "librosa": True,  # 필수 라이브러리
            "soundfile": SOUNDFILE_AVAILABLE,
            "webrtcvad": WEBRTCVAD_AVAILABLE,
            "numpy": True,
            "scipy": True
        },
        "features": {
            "snr_analysis": True,
            "noise_detection": True,
            "clarity_analysis": True,
            "volume_analysis": True,
            "vad_support": WEBRTCVAD_AVAILABLE
        },
        "noise_types": list(FieldQualityAnalyzer().noise_signatures.keys()),
        "quality_levels": ["excellent", "good", "fair", "poor"]
    }

if __name__ == "__main__":
    # 테스트 코드
    async def test_analyzer():
        print("현장 품질 분석 엔진 테스트")
        support_info = check_quality_analyzer_support()
        print(f"지원 상태: {support_info}")
        
        # 테스트용 더미 오디오 생성
        import numpy as np
        sample_rate = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_test = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz 사인파
        
        # 바이트로 변환 (더미 테스트)
        audio_bytes = (audio_test * 32767).astype(np.int16).tobytes()
        
        result = await analyze_field_audio_quality(audio_bytes, "test.wav")
        print(f"테스트 결과: {result.get('success', False)}")
    
    import asyncio
    asyncio.run(test_analyzer())
