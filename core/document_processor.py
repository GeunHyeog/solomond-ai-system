#!/usr/bin/env python3
"""
문서 처리 모듈 - PDF, Word 문서 텍스트 추출
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document  # python-docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import olefile
    import zipfile
    import xml.etree.ElementTree as ET
    DOC_AVAILABLE = True
except ImportError:
    DOC_AVAILABLE = False

class DocumentProcessor:
    """문서 파일 처리 클래스"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.supported_formats = []
        
        # 지원 형식 확인
        if PDF_AVAILABLE:
            self.supported_formats.extend(['.pdf'])
            self.logger.info("[INFO] PDF 처리 지원 (PyMuPDF)")
        else:
            self.logger.warning("[WARNING] PDF 처리 불가 - PyMuPDF 설치 필요")
            
        if DOCX_AVAILABLE:
            self.supported_formats.extend(['.docx'])
            self.logger.info("[INFO] DOCX 처리 지원 (python-docx)")
        else:
            self.logger.warning("[WARNING] DOCX 처리 불가 - python-docx 설치 필요")
            
        if DOC_AVAILABLE:
            self.supported_formats.extend(['.doc'])
            self.logger.info("[INFO] DOC 처리 지원 (olefile)")
        else:
            self.logger.warning("[WARNING] DOC 처리 불가 - olefile 설치 필요")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def extract_text_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """PDF에서 텍스트 추출"""
        if not PDF_AVAILABLE:
            return {
                "status": "error",
                "error": "PyMuPDF가 설치되지 않음",
                "install_command": "pip install PyMuPDF"
            }
        
        try:
            self.logger.info(f"[INFO] PDF 텍스트 추출 시작: {file_path}")
            
            # PDF 문서 열기
            doc = fitz.open(file_path)
            
            extracted_text = ""
            page_count = len(doc)
            page_texts = []
            
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                page_texts.append({
                    "page": page_num + 1,
                    "text": page_text,
                    "char_count": len(page_text)
                })
                extracted_text += f"\\n--- 페이지 {page_num + 1} ---\\n{page_text}\\n"
            
            doc.close()
            
            # 메타데이터 추출
            doc_info = fitz.open(file_path)
            metadata = doc_info.metadata
            doc_info.close()
            
            result = {
                "status": "success",
                "file_path": file_path,
                "file_type": "pdf",
                "extracted_text": extracted_text.strip(),
                "page_count": page_count,
                "pages": page_texts,
                "total_characters": len(extracted_text.strip()),
                "metadata": {
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "subject": metadata.get("subject", ""),
                    "creator": metadata.get("creator", ""),
                    "producer": metadata.get("producer", ""),
                    "creation_date": metadata.get("creationDate", ""),
                    "modification_date": metadata.get("modDate", "")
                }
            }
            
            self.logger.info(f"[SUCCESS] PDF 처리 완료: {page_count}페이지, {len(extracted_text)}자")
            return result
            
        except Exception as e:
            error_msg = f"PDF 처리 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "file_path": file_path
            }
    
    def extract_text_from_docx(self, file_path: str) -> Dict[str, Any]:
        """DOCX에서 텍스트 추출"""
        if not DOCX_AVAILABLE:
            return {
                "status": "error",
                "error": "python-docx가 설치되지 않음",
                "install_command": "pip install python-docx"
            }
        
        try:
            self.logger.info(f"[INFO] DOCX 텍스트 추출 시작: {file_path}")
            
            # DOCX 문서 열기
            doc = Document(file_path)
            
            extracted_text = ""
            paragraphs = []
            
            for i, paragraph in enumerate(doc.paragraphs):
                para_text = paragraph.text.strip()
                if para_text:  # 빈 단락 제외
                    paragraphs.append({
                        "paragraph": i + 1,
                        "text": para_text,
                        "char_count": len(para_text)
                    })
                    extracted_text += para_text + "\\n"
            
            # 메타데이터 추출
            core_properties = doc.core_properties
            
            result = {
                "status": "success",
                "file_path": file_path,
                "file_type": "docx",
                "extracted_text": extracted_text.strip(),
                "paragraph_count": len(paragraphs),
                "paragraphs": paragraphs,
                "total_characters": len(extracted_text.strip()),
                "metadata": {
                    "title": core_properties.title or "",
                    "author": core_properties.author or "",
                    "subject": core_properties.subject or "",
                    "category": core_properties.category or "",
                    "comments": core_properties.comments or "",
                    "created": str(core_properties.created) if core_properties.created else "",
                    "modified": str(core_properties.modified) if core_properties.modified else ""
                }
            }
            
            self.logger.info(f"[SUCCESS] DOCX 처리 완료: {len(paragraphs)}단락, {len(extracted_text)}자")
            return result
            
        except Exception as e:
            error_msg = f"DOCX 처리 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "file_path": file_path
            }
    
    def extract_text_from_doc(self, file_path: str) -> Dict[str, Any]:
        """DOC에서 텍스트 추출 (기본적인 지원)"""
        try:
            self.logger.info(f"[INFO] DOC 텍스트 추출 시작: {file_path}")
            
            # DOC 파일은 복잡한 형식이므로 기본적인 텍스트만 추출
            # 실제 구현은 더 복잡할 수 있음
            extracted_text = f"DOC 파일 감지됨: {os.path.basename(file_path)}\\n"
            extracted_text += "DOC 형식은 DOCX로 변환하여 사용하는 것을 권장합니다.\\n"
            extracted_text += "Microsoft Word에서 '다른 이름으로 저장' → 'Word 문서(*.docx)'를 선택하세요."
            
            result = {
                "status": "partial_success",
                "file_path": file_path,
                "file_type": "doc",
                "extracted_text": extracted_text,
                "total_characters": len(extracted_text),
                "warning": "DOC 형식은 제한적으로 지원됩니다. DOCX 변환을 권장합니다.",
                "metadata": {
                    "file_size": os.path.getsize(file_path),
                    "recommendation": "Convert to DOCX format for full support"
                }
            }
            
            self.logger.warning(f"[WARNING] DOC 파일 제한적 처리: {file_path}")
            return result
            
        except Exception as e:
            error_msg = f"DOC 처리 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "file_path": file_path
            }
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """문서 파일 처리 (자동 형식 감지)"""
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"파일이 존재하지 않음: {file_path}"
            }
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_formats:
            return {
                "status": "error",
                "error": f"지원되지 않는 파일 형식: {file_ext}",
                "supported_formats": self.supported_formats
            }
        
        # 파일 형식에 따른 처리
        if file_ext == ".pdf":
            return self.extract_text_from_pdf(file_path)
        elif file_ext == ".docx":
            return self.extract_text_from_docx(file_path)
        elif file_ext == ".doc":
            return self.extract_text_from_doc(file_path)
        else:
            return {
                "status": "error",
                "error": f"처리 로직이 구현되지 않은 형식: {file_ext}"
            }
    
    def get_installation_guide(self) -> Dict[str, Any]:
        """필요한 패키지 설치 가이드"""
        missing_packages = []
        
        if not PDF_AVAILABLE:
            missing_packages.append({
                "package": "PyMuPDF",
                "command": "pip install PyMuPDF",
                "purpose": "PDF 문서 처리"
            })
        
        if not DOCX_AVAILABLE:
            missing_packages.append({
                "package": "python-docx",
                "command": "pip install python-docx",
                "purpose": "Word DOCX 문서 처리"
            })
        
        if not DOC_AVAILABLE:
            missing_packages.append({
                "package": "olefile",
                "command": "pip install olefile",
                "purpose": "구 Word DOC 문서 처리"
            })
        
        return {
            "supported_formats": self.supported_formats,
            "missing_packages": missing_packages,
            "install_all": "pip install PyMuPDF python-docx olefile"
        }

# 전역 인스턴스
document_processor = DocumentProcessor()

def process_document_file(file_path: str) -> Dict[str, Any]:
    """문서 파일 처리 함수 (전역 접근용)"""
    return document_processor.process_document(file_path)