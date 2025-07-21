#!/usr/bin/env python3
"""
테스트용 문서 파일 생성
"""

import os
from docx import Document

def create_test_docx():
    """테스트용 DOCX 파일 생성"""
    doc = Document()
    
    # 제목 추가
    title = doc.add_heading('솔로몬드 AI 시스템 테스트 문서', 0)
    
    # 본문 추가
    doc.add_paragraph('이것은 문서 처리 모듈 테스트를 위한 샘플 DOCX 파일입니다.')
    doc.add_paragraph('주요 기능:')
    
    # 목록 추가
    doc.add_paragraph('• 음성 파일 분석 (STT)', style='List Bullet')
    doc.add_paragraph('• 이미지 텍스트 추출 (OCR)', style='List Bullet')
    doc.add_paragraph('• AI 기반 요약 및 분석', style='List Bullet')
    doc.add_paragraph('• 다중 파일 배치 처리', style='List Bullet')
    
    # 섹션 추가
    doc.add_heading('기술 스택', level=1)
    doc.add_paragraph('• Python 3.13')
    doc.add_paragraph('• Streamlit UI')
    doc.add_paragraph('• Whisper STT')
    doc.add_paragraph('• EasyOCR')
    doc.add_paragraph('• Transformers')
    
    # 저장
    os.makedirs('test_files', exist_ok=True)
    doc.save('test_files/sample.docx')
    print("[SUCCESS] DOCX 테스트 파일 생성: test_files/sample.docx")

def create_test_pdf():
    """테스트용 PDF 파일 생성"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        os.makedirs('test_files', exist_ok=True)
        
        # PDF 생성
        c = canvas.Canvas('test_files/sample.pdf', pagesize=letter)
        width, height = letter
        
        # 제목
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Solomond AI System Test Document")
        
        # 본문
        c.setFont("Helvetica", 12)
        y_position = height - 80
        
        lines = [
            "This is a sample PDF file for testing document processing module.",
            "",
            "Key Features:",
            "- Audio file analysis (STT)",
            "- Image text extraction (OCR)", 
            "- AI-based summary and analysis",
            "- Multi-file batch processing",
            "",
            "Technology Stack:",
            "- Python 3.13",
            "- Streamlit UI",
            "- Whisper STT",
            "- EasyOCR",
            "- Transformers"
        ]
        
        for line in lines:
            c.drawString(50, y_position, line)
            y_position -= 20
        
        c.save()
        print("[SUCCESS] PDF 테스트 파일 생성: test_files/sample.pdf")
        
    except ImportError:
        print("[WARNING] ReportLab이 설치되지 않아 PDF 생성 불가")
        print("[INFO] 설치 명령: pip install reportlab")

if __name__ == "__main__":
    print("[INFO] 테스트 문서 파일 생성 시작")
    print("=" * 50)
    
    create_test_docx()
    create_test_pdf()
    
    print("=" * 50)
    print("[INFO] 테스트 문서 파일 생성 완료")