#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ Unicode 안전성 보장 시스템 - SOLOMOND AI
Unicode Safety Enforcement System

핵심 기능:
1. 모든 텍스트 출력의 Unicode 안전성 보장
2. cp949 코덱 문제 완전 해결
3. 에러 메시지 안전 처리
4. 이모지 및 특수문자 폴백 시스템
5. 인코딩 오류 자동 복구
"""

import sys
import os
import logging
from typing import Any, Optional, Union
import unicodedata
import re

# 로깅 설정
logger = logging.getLogger(__name__)

class UnicodeSafetyManager:
    """Unicode 안전성 관리자"""
    
    def __init__(self):
        self.emoji_replacements = {
            '⚠️': '[주의]',
            '✅': '[완료]', 
            '❌': '[실패]',
            '🚀': '[시작]',
            '🔄': '[처리중]',
            '💡': '[팁]',
            '📊': '[통계]',
            '🎯': '[목표]',
            '🔍': '[검색]',
            '📁': '[폴더]',
            '🛡️': '[보안]',
            '⭐': '[중요]',
            '🎨': '[디자인]',
            '🔧': '[설정]',
            '📈': '[성장]',
            '🏆': '[성공]',
            '💎': '[주얼리]',
            '🎵': '[음악]',
            '🖼️': '[이미지]',
            '🎬': '[비디오]',
            '📄': '[문서]'
        }
        
        # 시스템 기본 인코딩 확인 및 설정
        self._setup_encoding_safety()
    
    def _setup_encoding_safety(self):
        """시스템 인코딩 안전성 설정"""
        try:
            # Windows cp949 환경 대응
            if sys.platform.startswith('win'):
                # 콘솔 출력 UTF-8 강제 설정
                if hasattr(sys.stdout, 'reconfigure'):
                    try:
                        sys.stdout.reconfigure(encoding='utf-8')
                        sys.stderr.reconfigure(encoding='utf-8')
                    except:
                        pass
                
                # 환경변수 설정
                os.environ['PYTHONIOENCODING'] = 'utf-8'
                
        except Exception as e:
            logger.warning(f"인코딩 설정 중 오류: {e}")
    
    def safe_text(self, text: Union[str, Any], fallback: str = "[텍스트 표시 불가]") -> str:
        """텍스트를 안전하게 처리"""
        if text is None:
            return fallback
        
        try:
            # 문자열로 변환
            if not isinstance(text, str):
                text = str(text)
            
            # Unicode 정규화
            text = unicodedata.normalize('NFC', text)
            
            # 이모지 교체
            for emoji, replacement in self.emoji_replacements.items():
                text = text.replace(emoji, replacement)
            
            # 문제가 될 수 있는 특수문자 제거/교체
            text = self._clean_problematic_chars(text)
            
            # cp949 호환성 테스트
            try:
                text.encode('cp949')
            except UnicodeEncodeError:
                # cp949로 인코딩할 수 없는 문자 교체
                text = text.encode('cp949', errors='replace').decode('cp949')
            
            return text
            
        except Exception as e:
            logger.error(f"텍스트 안전 처리 실패: {e}")
            return fallback
    
    def _clean_problematic_chars(self, text: str) -> str:
        """문제가 될 수 있는 문자들 정리"""
        # 제어 문자 제거 (개행, 탭 제외)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # 기타 문제 문자 교체
        replacements = {
            '…': '...',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '-',
            '•': '*',
            '→': '->',
            '←': '<-',
            '↑': '^',
            '↓': 'v'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def safe_error_message(self, error: Exception, context: str = "") -> str:
        """에러 메시지를 안전하게 처리"""
        try:
            error_str = str(error)
            if context:
                message = f"{context}: {error_str}"
            else:
                message = error_str
            
            return self.safe_text(message)
            
        except Exception:
            return f"[에러 처리 중 문제 발생] {context}" if context else "[에러 메시지 표시 불가]"
    
    def safe_format(self, format_string: str, *args, **kwargs) -> str:
        """포맷 문자열을 안전하게 처리"""
        try:
            # 모든 인자를 안전하게 변환
            safe_args = [self.safe_text(arg) for arg in args]
            safe_kwargs = {k: self.safe_text(v) for k, v in kwargs.items()}
            
            # 포맷 문자열도 안전하게 처리
            safe_format = self.safe_text(format_string)
            
            return safe_format.format(*safe_args, **safe_kwargs)
            
        except Exception as e:
            return f"[포맷 처리 실패: {self.safe_text(str(e))}]"
    
    def wrap_streamlit_display(self, display_func, text: str, **kwargs):
        """Streamlit 표시 함수를 안전하게 래핑"""
        try:
            safe_text = self.safe_text(text)
            return display_func(safe_text, **kwargs)
        except Exception as e:
            fallback_text = f"[표시 오류] {self.safe_error_message(e)}"
            return display_func(fallback_text, **kwargs)

# 전역 인스턴스
unicode_manager = UnicodeSafetyManager()

# 편의 함수들
def safe_text(text: Union[str, Any], fallback: str = "[텍스트 표시 불가]") -> str:
    """텍스트를 안전하게 처리하는 전역 함수"""
    return unicode_manager.safe_text(text, fallback)

def safe_error(error: Exception, context: str = "") -> str:
    """에러 메시지를 안전하게 처리하는 전역 함수"""
    return unicode_manager.safe_error_message(error, context)

def safe_format(format_string: str, *args, **kwargs) -> str:
    """포맷 문자열을 안전하게 처리하는 전역 함수"""
    return unicode_manager.safe_format(format_string, *args, **kwargs)

# Streamlit 안전 래퍼들
def safe_st_error(text: str):
    """안전한 st.error"""
    import streamlit as st
    return unicode_manager.wrap_streamlit_display(st.error, text)

def safe_st_warning(text: str):
    """안전한 st.warning"""
    import streamlit as st
    return unicode_manager.wrap_streamlit_display(st.warning, text)

def safe_st_info(text: str):
    """안전한 st.info"""
    import streamlit as st
    return unicode_manager.wrap_streamlit_display(st.info, text)

def safe_st_success(text: str):
    """안전한 st.success"""
    import streamlit as st
    return unicode_manager.wrap_streamlit_display(st.success, text)

def safe_st_write(text: str):
    """안전한 st.write"""
    import streamlit as st
    return unicode_manager.wrap_streamlit_display(st.write, text)

# 시스템 초기화
def initialize_unicode_safety():
    """Unicode 안전성 시스템 초기화"""
    try:
        # 시스템 인코딩 강제 설정
        if hasattr(sys, '_getframe'):
            # Python 환경에서 UTF-8 강제
            import locale
            try:
                locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
            except:
                try:
                    locale.setlocale(locale.LC_ALL, 'Korean_Korea.UTF-8')
                except:
                    pass
        
        logger.info("Unicode 안전성 시스템 초기화 완료")
        return True
        
    except Exception as e:
        logger.error(f"Unicode 안전성 시스템 초기화 실패: {e}")
        return False

# 자동 초기화
initialize_unicode_safety()

if __name__ == "__main__":
    # 테스트 코드
    print("🛡️ Unicode 안전성 시스템 테스트")
    
    test_texts = [
        "⚠️ 경고 메시지",
        "✅ 완료 상태",
        "🚀 시스템 시작",
        "테스트 텍스트",
        "한글 텍스트 처리"
    ]
    
    for text in test_texts:
        safe = safe_text(text)
        print(f"원본: {text} -> 안전: {safe}")
    
    print("테스트 완료!")