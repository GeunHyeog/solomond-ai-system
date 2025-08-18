#!/usr/bin/env python3
"""
문서 처리 패키지 상태 확인 스크립트
"""

import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from core.document_processor import document_processor
    print("SUCCESS: document_processor imported")
    
    # 지원 형식 확인
    print(f"Supported formats: {document_processor.supported_formats}")
    
    # 설치 가이드 확인
    guide = document_processor.get_installation_guide()
    print(f"\nInstallation Guide:")
    print(f"- Supported formats: {guide['supported_formats']}")
    print(f"- Missing packages: {len(guide['missing_packages'])}")
    
    if guide['missing_packages']:
        print("\nMissing packages:")
        for pkg in guide['missing_packages']:
            print(f"  - {pkg['package']}: {pkg['command']} ({pkg['purpose']})")
        print(f"\nInstall all: {guide['install_all']}")
    else:
        print("\nAll packages are installed! ✅")
        
    # 개별 패키지 테스트
    print(f"\nPackage availability:")
    
    # PDF 테스트
    try:
        import fitz
        print("✅ PyMuPDF (PDF): Available")
    except ImportError:
        print("❌ PyMuPDF (PDF): Not available")
    
    # DOCX 테스트
    try:
        from docx import Document
        print("✅ python-docx (DOCX): Available")
    except ImportError:
        print("❌ python-docx (DOCX): Not available")
    
    # DOC 테스트
    try:
        import olefile
        print("✅ olefile (DOC): Available")
    except ImportError:
        print("❌ olefile (DOC): Not available")
        
except ImportError as e:
    print(f"ERROR: Import error: {e}")
except Exception as e:
    print(f"ERROR: Unexpected error: {e}")