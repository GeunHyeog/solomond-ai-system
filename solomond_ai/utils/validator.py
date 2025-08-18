"""
검증 유틸리티
파일 및 결과 유효성 검증
"""

import os
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import logging

class FileValidator:
    """파일 유효성 검증 클래스"""
    
    def __init__(self):
        self.max_file_size = 10 * 1024 * 1024 * 1024  # 10GB
        self.supported_formats = {
            'audio': ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'],
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'],
            'video': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'],
            'text': ['.txt', '.md', '.json', '.csv', '.log', '.pdf', '.docx']
        }
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """단일 파일 유효성 검증"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        try:
            path_obj = Path(file_path)
            
            # 파일 존재 확인
            if not path_obj.exists():
                validation_result['valid'] = False
                validation_result['errors'].append('File does not exist')
                return validation_result
            
            # 파일 정보 수집
            stat_info = path_obj.stat()
            validation_result['file_info'] = {
                'size': stat_info.st_size,
                'size_mb': round(stat_info.st_size / (1024 * 1024), 2),
                'extension': path_obj.suffix.lower(),
                'name': path_obj.name
            }
            
            # 파일 크기 확인
            if stat_info.st_size > self.max_file_size:
                validation_result['valid'] = False
                validation_result['errors'].append(f'File too large (>{self.max_file_size/1024/1024/1024:.1f}GB)')
            
            # 파일 형식 확인
            file_type = self._detect_file_type(path_obj.suffix.lower())
            if file_type == 'unknown':
                validation_result['valid'] = False
                validation_result['errors'].append('Unsupported file format')
            else:
                validation_result['file_info']['type'] = file_type
            
            # 읽기 권한 확인
            if not os.access(file_path, os.R_OK):
                validation_result['valid'] = False
                validation_result['errors'].append('No read permission')
            
            # 경고 사항
            if stat_info.st_size > 100 * 1024 * 1024:  # 100MB
                validation_result['warnings'].append('Large file may take longer to process')
                
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Validation error: {str(e)}')
        
        return validation_result
    
    def _detect_file_type(self, extension: str) -> str:
        """파일 확장자로 타입 감지"""
        for file_type, extensions in self.supported_formats.items():
            if extension in extensions:
                return file_type
        return 'unknown'
    
    def validate_files(self, file_paths: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """다중 파일 유효성 검증"""
        valid_files = []
        validation_results = []
        
        for file_path in file_paths:
            result = self.validate_file(file_path)
            validation_results.append({
                'file_path': file_path,
                **result
            })
            
            if result['valid']:
                valid_files.append(file_path)
        
        return valid_files, validation_results

class ResultValidator:
    """분석 결과 유효성 검증 클래스"""
    
    def __init__(self):
        self.required_fields = {
            'audio': ['success', 'full_text', 'segments'],
            'image': ['success', 'text_blocks', 'full_text'],
            'video': ['success', 'metadata'],
            'text': ['success', 'content', 'basic_analysis'],
            'integration': ['consistency_score', 'common_keywords', 'insights']
        }
    
    def validate_engine_result(self, engine_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """엔진 결과 유효성 검증"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'completeness_score': 0
        }
        
        if engine_name not in self.required_fields:
            validation['warnings'].append(f'Unknown engine: {engine_name}')
            return validation
        
        required = self.required_fields[engine_name]
        present_fields = []
        
        # 필수 필드 확인
        for field in required:
            if field in result:
                present_fields.append(field)
            else:
                validation['errors'].append(f'Missing required field: {field}')
                validation['valid'] = False
        
        # 완성도 점수 계산
        validation['completeness_score'] = len(present_fields) / len(required) * 100
        
        # 엔진별 특화 검증
        if engine_name == 'audio' and result.get('success', False):
            self._validate_audio_result(result, validation)
        elif engine_name == 'image' and result.get('success', False):
            self._validate_image_result(result, validation)
        elif engine_name == 'integration':
            self._validate_integration_result(result, validation)
        
        return validation
    
    def _validate_audio_result(self, result: Dict[str, Any], validation: Dict[str, Any]):
        """오디오 결과 특화 검증"""
        # 텍스트 길이 확인
        full_text = result.get('full_text', '')
        if len(full_text) < 10:
            validation['warnings'].append('Very short transcription result')
        
        # 세그먼트 확인
        segments = result.get('segments', [])
        if not segments:
            validation['warnings'].append('No segments found')
        
        # 신뢰도 확인
        if segments:
            avg_confidence = sum(seg.get('confidence', 0) for seg in segments) / len(segments)
            if avg_confidence < 0.5:
                validation['warnings'].append('Low average confidence in transcription')
    
    def _validate_image_result(self, result: Dict[str, Any], validation: Dict[str, Any]):
        """이미지 결과 특화 검증"""
        # 텍스트 블록 확인
        text_blocks = result.get('text_blocks', [])
        if not text_blocks:
            validation['warnings'].append('No text blocks detected')
        
        # 평균 신뢰도 확인
        avg_confidence = result.get('average_confidence', 0)
        if avg_confidence < 0.7:
            validation['warnings'].append('Low OCR confidence')
    
    def _validate_integration_result(self, result: Dict[str, Any], validation: Dict[str, Any]):
        """통합 결과 특화 검증"""
        # 일관성 점수 확인
        consistency_score = result.get('consistency_score', 0)
        if consistency_score < 50:
            validation['warnings'].append('Low consistency score between engines')
        
        # 공통 키워드 확인
        common_keywords = result.get('common_keywords', [])
        if len(common_keywords) < 3:
            validation['warnings'].append('Few common keywords found across engines')
    
    def validate_batch_results(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """배치 결과 전체 유효성 검증"""
        batch_validation = {
            'overall_valid': True,
            'engine_validations': {},
            'summary': {
                'total_files': 0,
                'successful_files': 0,
                'failed_files': 0,
                'average_completeness': 0
            }
        }
        
        total_completeness = 0
        engine_count = 0
        
        for engine_name, engine_results in results.items():
            engine_validation = {
                'total_results': len(engine_results),
                'valid_results': 0,
                'completeness_scores': []
            }
            
            for result in engine_results:
                validation = self.validate_engine_result(engine_name, result)
                if validation['valid']:
                    engine_validation['valid_results'] += 1
                
                engine_validation['completeness_scores'].append(validation['completeness_score'])
                total_completeness += validation['completeness_score']
                engine_count += 1
            
            # 엔진별 평균 완성도
            if engine_validation['completeness_scores']:
                engine_validation['average_completeness'] = sum(engine_validation['completeness_scores']) / len(engine_validation['completeness_scores'])
            
            batch_validation['engine_validations'][engine_name] = engine_validation
            
            # 전체 통계 업데이트
            batch_validation['summary']['total_files'] += engine_validation['total_results']
            batch_validation['summary']['successful_files'] += engine_validation['valid_results']
            batch_validation['summary']['failed_files'] += (engine_validation['total_results'] - engine_validation['valid_results'])
        
        # 전체 평균 완성도
        if engine_count > 0:
            batch_validation['summary']['average_completeness'] = total_completeness / engine_count
        
        return batch_validation