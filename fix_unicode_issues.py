#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛠️ Unicode 문제 일괄 수정 스크립트
자동으로 모든 파일의 Unicode 문제를 해결합니다.
"""

import os
import re
from pathlib import Path

def fix_unicode_in_file(file_path: Path) -> bool:
    """파일의 Unicode 문제 수정"""
    try:
        # 파일 읽기
        encodings = ['utf-8', 'cp949', 'euc-kr']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            print(f"❌ 인코딩 실패: {file_path}")
            return False
        
        # Unicode 문제 수정 매핑
        unicode_fixes = {
            # 이모지 교체
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
            '📄': '[문서]',
            'ℹ️': '[정보]',
            '🤖': '[AI]',
            
            # Streamlit 함수 안전화
            'st.error(f"❌': 'safe_st_error(f"[실패]',
            'st.warning(f"⚠️': 'safe_st_warning(f"[주의]',
            'st.info(f"ℹ️': 'safe_st_info(f"[정보]',
            'st.success(f"✅': 'safe_st_success(f"[완료]',
            'st.info("🚀': 'safe_st_info("[시작]',
            'st.error("❌': 'safe_st_error("[실패]',
            'st.warning("⚠️': 'safe_st_warning("[주의]',
            'st.success("✅': 'safe_st_success("[완료]',
        }
        
        # 패턴 기반 수정
        patterns = [
            # st.함수("이모지 텍스트") 패턴
            (r'st\.(error|warning|info|success)\("([⚠️✅❌🚀🔄💡📊🎯🔍📁🛡️⭐🎨🔧📈🏆💎🎵🖼️🎬📄ℹ️🤖]+)', 
             lambda m: f'safe_st_{m.group(1)}("[{get_emoji_replacement(m.group(2))}]'),
            
            # st.함수(f"이모지 텍스트 {변수}") 패턴  
            (r'st\.(error|warning|info|success)\(f"([⚠️✅❌🚀🔄💡📊🎯🔍📁🛡️⭐🎨🔧📈🏆💎🎵🖼️🎬📄ℹ️🤖]+)', 
             lambda m: f'safe_st_{m.group(1)}(f"[{get_emoji_replacement(m.group(2))}]'),
        ]
        
        original_content = content
        
        # 단순 교체 먼저 실행
        for old, new in unicode_fixes.items():
            content = content.replace(old, new)
        
        # 패턴 기반 교체
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # 안전 import 추가 확인
        if 'safe_st_' in content and 'from core.unicode_safety_system import' not in content:
            # 적절한 위치에 import 추가
            import_line = """
# 🛡️ Unicode 안전성 시스템
try:
    from core.unicode_safety_system import (
        safe_text, safe_error, safe_format,
        safe_st_error, safe_st_warning, safe_st_info, safe_st_success
    )
except ImportError:
    def safe_text(text, fallback="[텍스트 표시 불가]"):
        return str(text).encode('utf-8', errors='replace').decode('utf-8')
    def safe_error(error, context=""):
        return safe_text(str(error))
    def safe_st_error(text):
        return st.error(safe_text(text))
    def safe_st_warning(text):
        return st.warning(safe_text(text))
    def safe_st_info(text):
        return st.info(safe_text(text))
    def safe_st_success(text):
        return st.success(safe_text(text))
"""
            
            # import streamlit as st 다음에 추가
            if 'import streamlit as st' in content:
                content = content.replace(
                    'import streamlit as st',
                    'import streamlit as st' + import_line
                )
        
        # 변경사항이 있으면 파일 저장
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 수정 완료: {file_path}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ 수정 실패: {file_path} - {e}")
        return False

def get_emoji_replacement(emoji_text: str) -> str:
    """이모지 교체 텍스트 반환"""
    emoji_map = {
        '⚠️': '주의',
        '✅': '완료',
        '❌': '실패',
        '🚀': '시작',
        '🔄': '처리중',
        '💡': '팁',
        '📊': '통계',
        '🎯': '목표',
        '🔍': '검색',
        '📁': '폴더',
        '🛡️': '보안',
        '⭐': '중요',
        '🎨': '디자인',
        '🔧': '설정',
        '📈': '성장',
        '🏆': '성공',
        '💎': '주얼리',
        '🎵': '음악',
        '🖼️': '이미지',
        '🎬': '비디오',
        '📄': '문서',
        'ℹ️': '정보',
        '🤖': 'AI'
    }
    
    for emoji, replacement in emoji_map.items():
        if emoji in emoji_text:
            return replacement
    return '상태'

def main():
    """메인 실행 함수"""
    print("🛠️ Unicode 문제 일괄 수정 시작...")
    
    # 수정할 파일 확장자
    target_extensions = ['.py']
    
    # 현재 디렉토리 및 하위 디렉토리의 모든 파이썬 파일 처리
    current_dir = Path('.')
    fixed_count = 0
    total_count = 0
    
    for file_path in current_dir.rglob('*.py'):
        # 시스템 파일이나 가상환경 파일 제외
        if any(part.startswith('.') for part in file_path.parts):
            continue
        if any(exclude in str(file_path) for exclude in ['venv', '__pycache__', '.git']):
            continue
            
        total_count += 1
        if fix_unicode_in_file(file_path):
            fixed_count += 1
    
    print(f"\n📊 수정 완료:")
    print(f"   총 파일: {total_count}개")
    print(f"   수정된 파일: {fixed_count}개")
    print(f"   Unicode 안전성 확보!")

if __name__ == "__main__":
    main()