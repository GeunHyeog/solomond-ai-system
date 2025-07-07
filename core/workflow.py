"""
솔로몬드 AI 시스템 - 워크플로우
4단계 처리 워크플로우 관리
"""

import asyncio
from typing import Dict, List, Optional
from .analyzer import get_analyzer
from .file_processor import get_file_processor

class WorkflowManager:
    """워크플로우 관리 클래스"""
    
    def __init__(self):
        self.analyzer = get_analyzer()
        self.file_processor = get_file_processor()
        
    async def execute_stt_workflow(self, 
                                 file_content: bytes,
                                 filename: str,
                                 language: str = "ko") -> Dict:
        """
        STT 4단계 워크플로우 실행
        
        1단계: 파일 업로드 및 검증
        2단계: AI 분석 (음성 인식)
        3단계: 결과 처리 (번역, 요약)
        4단계: 결과 반환
        """
        
        result = {
            "workflow_steps": [],
            "success": False,
            "final_result": None
        }
        
        try:
            # 1단계: 파일 처리
            step1 = await self._step1_file_processing(file_content, filename)
            result["workflow_steps"].append(step1)
            
            if not step1["success"]:
                return result
            
            # 2단계: AI 분석
            step2 = await self._step2_ai_analysis(file_content, filename, language)
            result["workflow_steps"].append(step2)
            
            if not step2["success"]:
                return result
            
            # 3단계: 결과 처리
            step3 = await self._step3_result_processing(step2["data"])
            result["workflow_steps"].append(step3)
            
            # 4단계: 최종 결과
            step4 = await self._step4_final_result(step2["data"], step3["data"])
            result["workflow_steps"].append(step4)
            
            result["success"] = True
            result["final_result"] = step4["data"]
            
            return result
            
        except Exception as e:
            result["workflow_steps"].append({
                "step": "error",
                "success": False,
                "error": str(e)
            })
            return result
    
    async def _step1_file_processing(self, file_content: bytes, filename: str) -> Dict:
        """
        1단계: 파일 업로드 및 검증
        """
        try:
            file_info = await self.file_processor.process_file(file_content, filename)
            
            return {
                "step": "file_processing",
                "success": file_info["success"],
                "data": file_info,
                "message": "파일 처리 완료"
            }
        except Exception as e:
            return {
                "step": "file_processing",
                "success": False,
                "error": str(e)
            }
    
    async def _step2_ai_analysis(self, file_content: bytes, filename: str, language: str) -> Dict:
        """
        2단계: AI 분석 (음성 인식)
        """
        try:
            analysis_result = await self.analyzer.analyze_uploaded_file(
                file_content=file_content,
                filename=filename,
                language=language
            )
            
            return {
                "step": "ai_analysis",
                "success": analysis_result["success"],
                "data": analysis_result,
                "message": "AI 분석 완료"
            }
        except Exception as e:
            return {
                "step": "ai_analysis",
                "success": False,
                "error": str(e)
            }
    
    async def _step3_result_processing(self, analysis_data: Dict) -> Dict:
        """
        3단계: 결과 처리 (번역, 요약)
        """
        try:
            processed_data = {
                "original_text": analysis_data.get("transcribed_text", ""),
                "translated_text": None,  # 미래 번역 기능
                "summary": None,         # 미래 요약 기능
                "keywords": []           # 미래 키워드 추출
            }
            
            # 기본 키워드 추출 (단순 버전)
            text = processed_data["original_text"]
            if text:
                words = text.split()
                processed_data["keywords"] = list(set([w for w in words if len(w) > 2]))[:10]
            
            return {
                "step": "result_processing",
                "success": True,
                "data": processed_data,
                "message": "결과 처리 완료"
            }
        except Exception as e:
            return {
                "step": "result_processing",
                "success": False,
                "error": str(e)
            }
    
    async def _step4_final_result(self, analysis_data: Dict, processed_data: Dict) -> Dict:
        """
        4단계: 최종 결과 생성
        """
        try:
            final_result = {
                # 기본 정보
                "filename": analysis_data.get("filename", ""),
                "file_size": analysis_data.get("file_size", ""),
                "processing_time": analysis_data.get("processing_time", 0),
                "detected_language": analysis_data.get("detected_language", "unknown"),
                
                # 분석 결과
                "transcribed_text": processed_data["original_text"],
                "keywords": processed_data["keywords"],
                
                # 메타데이터
                "confidence": analysis_data.get("confidence", 0.0),
                "segments_count": len(analysis_data.get("segments", [])),
                "workflow_version": "v3.0"
            }
            
            return {
                "step": "final_result",
                "success": True,
                "data": final_result,
                "message": "워크플로우 완료"
            }
        except Exception as e:
            return {
                "step": "final_result",
                "success": False,
                "error": str(e)
            }

# 전역 인스턴스
_workflow_instance = None

def get_workflow_manager() -> WorkflowManager:
    """전역 워크플로우 관리자 인스턴스 반환"""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = WorkflowManager()
    return _workflow_instance
