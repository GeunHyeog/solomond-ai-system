#!/usr/bin/env python3
"""
🌍 모듈1 다국어 지원 및 포맷 확장 시스템
전 세계 언어 지원 및 다양한 파일 형식 처리

업데이트: 2025-01-30 - 다국어 지원 + 포맷 확장
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path
import easyocr
import whisper
try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False
    GoogleTranslator = None

import langdetect
from langdetect import detect
import subprocess
import tempfile
import os

class LanguageManager:
    """언어 관리 시스템"""
    
    def __init__(self):
        self.supported_languages = {
            'ko': '한국어 (Korean)',
            'en': '영어 (English)', 
            'ja': '일본어 (Japanese)',
            'zh': '중국어 (Chinese)',
            'zh-cn': '중국어 간체 (Simplified Chinese)',
            'zh-tw': '중국어 번체 (Traditional Chinese)',
            'fr': '프랑스어 (French)',
            'de': '독일어 (German)',
            'es': '스페인어 (Spanish)',
            'it': '이탈리아어 (Italian)',
            'ru': '러시아어 (Russian)',
            'pt': '포르투갈어 (Portuguese)',
            'ar': '아랍어 (Arabic)',
            'hi': '힌디어 (Hindi)',
            'th': '태국어 (Thai)',
            'vi': '베트남어 (Vietnamese)'
        }
        
        # EasyOCR 지원 언어 (주요 언어만)
        self.ocr_languages = [
            'ko', 'en', 'ja', 'zh', 'fr', 'de', 'es', 'it', 'ru', 'pt', 'ar', 'hi', 'th', 'vi'
        ]
        
        # Whisper 지원 언어
        self.whisper_languages = {
            'auto': '자동 감지',
            'ko': '한국어', 'en': '영어', 'ja': '일본어', 'zh': '중국어',
            'fr': '프랑스어', 'de': '독일어', 'es': '스페인어', 'it': '이탈리아어',
            'ru': '러시아어', 'pt': '포르투갈어', 'ar': '아랍어', 'hi': '힌디어'
        }
        
        self.translator_available = DEEP_TRANSLATOR_AVAILABLE
        self.setup_translator()
    
    def setup_translator(self):
        """번역기 설정"""
        if self.translator_available:
            st.sidebar.success("🌍 Deep Translator 활성화")
        else:
            st.sidebar.warning("⚠️ 번역 서비스 비활성화")
    
    def detect_language(self, text: str) -> str:
        """언어 자동 감지"""
        try:
            detected = detect(text)
            return detected
        except:
            return 'en'  # 기본값
    
    def translate_text(self, text: str, target_lang: str = 'ko', source_lang: str = 'auto') -> str:
        """텍스트 번역"""
        if not self.translator_available:
            return text
        
        try:
            if source_lang == 'auto':
                source_lang = self.detect_language(text)
            
            if source_lang == target_lang:
                return text
            
            # deep-translator 사용
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            result = translator.translate(text)
            return result
        except Exception as e:
            st.warning(f"번역 실패: {e}")
            return text
    
    def get_language_display_name(self, lang_code: str) -> str:
        """언어 코드를 표시명으로 변환"""
        return self.supported_languages.get(lang_code, lang_code)

class ExtendedFormatProcessor:
    """확장 포맷 처리 시스템"""
    
    def __init__(self):
        self.supported_formats = {
            # 오디오 포맷
            'audio': {
                '.mp3': 'MP3',
                '.wav': 'WAV',
                '.m4a': 'M4A',
                '.flac': 'FLAC',
                '.aac': 'AAC',
                '.ogg': 'OGG',
                '.wma': 'WMA',
                '.aiff': 'AIFF',
                '.au': 'AU',
                '.mp2': 'MP2'
            },
            # 비디오 포맷
            'video': {
                '.mp4': 'MP4',
                '.avi': 'AVI',
                '.mov': 'QuickTime MOV',
                '.mkv': 'Matroska MKV',  
                '.webm': 'WebM',
                '.flv': 'Flash Video',
                '.wmv': 'Windows Media',
                '.m4v': 'iTunes M4V',
                '.3gp': '3GP',
                '.ogv': 'Ogg Video',
                '.ts': 'MPEG Transport Stream',
                '.mts': 'AVCHD Video'
            },
            # 이미지 포맷
            'image': {
                '.jpg': 'JPEG',
                '.jpeg': 'JPEG',
                '.png': 'PNG',
                '.bmp': 'Bitmap',
                '.tiff': 'TIFF',
                '.tif': 'TIFF',
                '.gif': 'GIF',
                '.webp': 'WebP',
                '.svg': 'SVG',
                '.ico': 'Icon',
                '.psd': 'Photoshop',
                '.raw': 'RAW',
                '.cr2': 'Canon RAW',
                '.nef': 'Nikon RAW',
                '.arw': 'Sony RAW'
            }
        }
        
        # FFmpeg 사용 가능 여부 확인
        self.ffmpeg_available = self.check_ffmpeg()
    
    def check_ffmpeg(self) -> bool:
        """FFmpeg 사용 가능 여부 확인"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except:
            return False
    
    def get_file_type(self, filename: str) -> str:
        """파일 타입 확인"""
        ext = Path(filename).suffix.lower()
        
        for file_type, formats in self.supported_formats.items():
            if ext in formats:
                return file_type
        return 'unknown'
    
    def is_supported_format(self, filename: str) -> bool:
        """지원되는 포맷인지 확인"""
        return self.get_file_type(filename) != 'unknown'
    
    def convert_audio_format(self, input_file: str, output_format: str = 'wav') -> Optional[str]:
        """오디오 포맷 변환"""
        if not self.ffmpeg_available:
            return None
        
        try:
            output_file = tempfile.mktemp(suffix=f'.{output_format}')
            
            cmd = [
                'ffmpeg', '-i', input_file,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y', output_file
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            return output_file
            
        except Exception as e:
            st.error(f"오디오 변환 실패: {e}")
            return None
    
    def extract_audio_from_video(self, video_file: str, output_format: str = 'wav') -> Optional[str]:
        """비디오에서 오디오 추출"""
        if not self.ffmpeg_available:
            return None
        
        try:
            output_file = tempfile.mktemp(suffix=f'.{output_format}')
            
            cmd = [
                'ffmpeg', '-i', video_file,
                '-vn',  # 비디오 스트림 비활성화
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y', output_file
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            return output_file
            
        except Exception as e:
            st.error(f"오디오 추출 실패: {e}")
            return None
    
    def get_format_info(self, filename: str) -> Dict[str, str]:
        """파일 포맷 정보 반환"""
        ext = Path(filename).suffix.lower()
        file_type = self.get_file_type(filename)
        
        if file_type != 'unknown':
            format_name = self.supported_formats[file_type].get(ext, ext)
        else:
            format_name = ext
        
        return {
            'extension': ext,
            'type': file_type,
            'format_name': format_name,
            'supported': file_type != 'unknown'
        }

class MultilingualConferenceProcessor:
    """다국어 컨퍼런스 처리 시스템"""
    
    def __init__(self):
        self.language_manager = LanguageManager()
        self.format_processor = ExtendedFormatProcessor()
        self.ocr_readers = {}  # 언어별 OCR reader 캐시
        
    def setup_multilingual_ocr(self, languages: List[str]):
        """다국어 OCR 설정"""
        try:
            # 지원되는 언어만 필터링
            valid_languages = [lang for lang in languages if lang in self.language_manager.ocr_languages]
            
            if not valid_languages:
                valid_languages = ['ko', 'en']  # 기본값
            
            lang_key = ','.join(sorted(valid_languages))
            
            if lang_key not in self.ocr_readers:
                with st.spinner(f"다국어 OCR 모델 로딩 중... ({', '.join(valid_languages)})"):
                    self.ocr_readers[lang_key] = easyocr.Reader(
                        valid_languages, 
                        gpu=torch.cuda.is_available() if 'torch' in globals() else False,
                        verbose=False
                    )
            
            return self.ocr_readers[lang_key]
            
        except Exception as e:
            st.error(f"다국어 OCR 설정 실패: {e}")
            return None
    
    def process_multilingual_image(self, image_data: bytes, filename: str, 
                                 target_languages: List[str], 
                                 translate_to: str = 'ko') -> Dict[str, Any]:
        """다국어 이미지 처리"""
        try:
            # OCR 설정
            ocr_reader = self.setup_multilingual_ocr(target_languages)
            if not ocr_reader:
                return {'error': 'OCR 설정 실패'}
            
            # 이미지 처리
            from PIL import Image
            import numpy as np
            import io
            
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)
            
            start_time = time.time()
            results = ocr_reader.readtext(image_np)
            end_time = time.time()
            
            # 결과 처리
            text_blocks = []
            all_text = ""
            
            for bbox, text, confidence in results:
                if confidence > 0.3:
                    # 언어 감지
                    detected_lang = self.language_manager.detect_language(text)
                    
                    # 번역 (필요시)
                    translated_text = text
                    if translate_to != detected_lang:
                        translated_text = self.language_manager.translate_text(
                            text, target_lang=translate_to, source_lang=detected_lang
                        )
                    
                    block_info = {
                        'original_text': text,
                        'translated_text': translated_text,
                        'detected_language': detected_lang,
                        'confidence': confidence,
                        'bbox': bbox
                    }
                    
                    text_blocks.append(block_info)
                    all_text += f"{translated_text} "
            
            return {
                'text_blocks': text_blocks,
                'total_blocks': len(text_blocks),
                'processing_time': end_time - start_time,
                'combined_text': all_text.strip(),
                'detected_languages': list(set([block['detected_language'] for block in text_blocks])),
                'target_languages': target_languages,
                'translation_language': translate_to
            }
            
        except Exception as e:
            return {'error': f'다국어 이미지 처리 실패: {str(e)}'}
    
    def process_multilingual_audio(self, audio_data: bytes, filename: str,
                                 source_language: str = 'auto',
                                 translate_to: str = 'ko') -> Dict[str, Any]:
        """다국어 오디오 처리"""
        try:
            # 포맷 변환 (필요시)
            format_info = self.format_processor.get_format_info(filename)
            
            temp_path = None
            if format_info['extension'] not in ['.wav', '.mp3', '.m4a']:
                # FFmpeg로 변환
                with tempfile.NamedTemporaryFile(suffix=format_info['extension'], delete=False) as temp_input:
                    temp_input.write(audio_data)
                    temp_input_path = temp_input.name
                
                temp_path = self.format_processor.convert_audio_format(temp_input_path, 'wav')
                os.unlink(temp_input_path)
                
                if not temp_path:
                    return {'error': '오디오 포맷 변환 실패'}
            else:
                # 직접 사용
                with tempfile.NamedTemporaryFile(suffix=format_info['extension'], delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_path = temp_file.name
            
            # Whisper STT 처리
            import whisper
            model = whisper.load_model("base")
            
            start_time = time.time()
            
            # 언어 지정 또는 자동 감지
            whisper_language = None if source_language == 'auto' else source_language
            
            result = model.transcribe(
                temp_path,
                language=whisper_language,
                fp16=torch.cuda.is_available() if 'torch' in globals() else False
            )
            
            end_time = time.time()
            
            # 번역 (필요시)
            original_text = result['text']
            detected_language = result.get('language', 'unknown')
            
            translated_text = original_text
            if translate_to != detected_language:
                translated_text = self.language_manager.translate_text(
                    original_text, target_lang=translate_to, source_lang=detected_language
                )
            
            # 세그먼트별 번역
            translated_segments = []
            for segment in result.get('segments', []):
                segment_text = segment.get('text', '')
                if translate_to != detected_language and segment_text:
                    segment_translated = self.language_manager.translate_text(
                        segment_text, target_lang=translate_to, source_lang=detected_language  
                    )
                else:
                    segment_translated = segment_text
                
                translated_segments.append({
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0),
                    'original_text': segment_text,
                    'translated_text': segment_translated
                })
            
            # 임시 파일 정리
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            
            return {
                'original_text': original_text,
                'translated_text': translated_text,
                'detected_language': detected_language,
                'target_language': translate_to,
                'segments': translated_segments,
                'processing_time': end_time - start_time,
                'format_info': format_info
            }
            
        except Exception as e:
            # 임시 파일 정리
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            
            return {'error': f'다국어 오디오 처리 실패: {str(e)}'}
    
    def render_language_settings(self):
        """언어 설정 UI"""
        st.sidebar.markdown("### 🌍 언어 설정")
        
        # OCR 언어 선택
        selected_ocr_langs = st.sidebar.multiselect(
            "이미지 OCR 언어",
            options=self.language_manager.ocr_languages,
            default=['ko', 'en'],
            format_func=lambda x: self.language_manager.get_language_display_name(x),
            help="이미지에서 인식할 언어를 선택하세요"
        )
        
        # 음성 인식 언어
        selected_whisper_lang = st.sidebar.selectbox(
            "음성 인식 언어",
            options=list(self.language_manager.whisper_languages.keys()),
            format_func=lambda x: self.language_manager.whisper_languages[x],
            help="음성에서 인식할 언어를 선택하세요"
        )
        
        # 번역 대상 언어
        translate_to_lang = st.sidebar.selectbox(
            "번역 대상 언어",
            options=['ko', 'en', 'ja', 'zh', 'fr', 'de', 'es'],
            format_func=lambda x: self.language_manager.get_language_display_name(x),
            help="모든 텍스트를 이 언어로 번역합니다"
        )
        
        return {
            'ocr_languages': selected_ocr_langs,
            'whisper_language': selected_whisper_lang,
            'translate_to': translate_to_lang
        }
    
    def render_format_support_info(self):
        """지원 포맷 정보 표시"""
        with st.sidebar.expander("📁 지원 파일 형식", expanded=False):
            for file_type, formats in self.format_processor.supported_formats.items():
                st.markdown(f"**{file_type.upper()}**")
                format_list = list(formats.keys())
                # 3개씩 묶어서 표시
                for i in range(0, len(format_list), 3):
                    chunk = format_list[i:i+3]
                    st.text(" • " + " • ".join(chunk))
            
            if self.format_processor.ffmpeg_available:
                st.success("✅ FFmpeg 변환 지원")
            else:
                st.warning("⚠️ FFmpeg 없음 (일부 포맷 제한)")