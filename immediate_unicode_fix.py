#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
즉시 Unicode 문제 해결 스크립트 (이모지 없음)
"""

import os
import sys
import re
from pathlib import Path

# 시스템 인코딩 강제 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'

def fix_critical_files():
    """핵심 파일들의 Unicode 문제 즉시 해결"""
    
    critical_files = [
        'conference_analysis_UNIFIED_COMPLETE.py',
        'semantic_connection_engine.py',
        'actionable_insights_extractor.py',
        'conference_story_generator.py',
        'holistic_conference_analyzer_supabase.py'
    ]
    
    # 이모지 -> 텍스트 매핑
    replacements = {
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
        '🤖': '[AI]'
    }
    
    fixed_count = 0
    
    for filename in critical_files:
        file_path = Path(filename)
        if not file_path.exists():
            continue
            
        try:
            # 파일 읽기 (UTF-8)
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            original_content = content
            
            # 이모지 교체
            for emoji, replacement in replacements.items():
                content = content.replace(emoji, replacement)
            
            # 변경사항이 있으면 저장
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Fixed: {filename}")
                fixed_count += 1
                
        except Exception as e:
            print(f"Error fixing {filename}: {e}")
    
    print(f"Total files fixed: {fixed_count}")
    return fixed_count > 0

if __name__ == "__main__":
    print("Fixing Unicode issues...")
    if fix_critical_files():
        print("Unicode issues fixed successfully!")
    else:
        print("No files needed fixing.")