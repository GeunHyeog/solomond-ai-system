#!/usr/bin/env python3
"""
🛡️ Defensive Model Loader - 연쇄 문제 방지 시스템
PyTorch 2.7.1 meta tensor 문제 완전 해결 및 안전한 모델 로딩
"""

import torch
import warnings
import traceback
from typing import Optional, Any, Union

def safe_model_load(load_function, *args, **kwargs):
    """
    안전한 모델 로딩 래퍼 - 모든 PyTorch 모델에 사용 가능
    """
    try:
        # 1차 시도: 원본 파라미터로 로딩
        return load_function(*args, **kwargs)
    except Exception as e:
        if "meta tensor" in str(e):
            print(f"⚠️ Meta tensor 문제 감지, 안전 모드로 전환: {e}")
            
            # 2차 시도: CPU 강제 로딩
            try:
                if 'device' in kwargs:
                    kwargs['device'] = 'cpu'
                elif len(args) > 1:
                    # whisper.load_model("base", device="cpu") 형태
                    args = list(args)
                    if len(args) == 1:
                        args.append('cpu')
                    else:
                        args[1] = 'cpu'
                
                model = load_function(*args, **kwargs)
                
                # 3차 시도: GPU로 안전하게 이동
                if torch.cuda.is_available():
                    try:
                        model = model.to('cuda')
                        print("✅ CPU 로딩 → GPU 이동 성공")
                    except:
                        print("⚠️ GPU 이동 실패, CPU 모드 유지")
                
                return model
                
            except Exception as e2:
                print(f"❌ 안전 모드도 실패: {e2}")
                raise e2
        else:
            raise e

def safe_whisper_load(model_size: str = "base", device: Optional[str] = None):
    """Whisper 전용 안전 로더"""
    import whisper
    
    def _load():
        if device:
            return whisper.load_model(model_size, device=device)
        else:
            return whisper.load_model(model_size)
    
    return safe_model_load(_load)

def safe_sentence_transformer_load(model_name: str, device: Optional[str] = None):
    """SentenceTransformer 전용 안전 로더"""
    try:
        from sentence_transformers import SentenceTransformer
        
        def _load():
            if device:
                return SentenceTransformer(model_name, device=device)
            else:
                return SentenceTransformer(model_name)
        
        return safe_model_load(_load)
    except ImportError:
        print("❌ SentenceTransformer 설치되지 않음")
        return None

# 전역 안전 모드 설정
def enable_defensive_mode():
    """
    전역적으로 안전 모드 활성화
    """
    # PyTorch 경고 필터링
    warnings.filterwarnings("ignore", category=UserWarning, message=".*meta tensor.*")
    
    # CUDA 메모리 최적화
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 메모리 할당 최적화
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

if __name__ == "__main__":
    # 테스트 코드
    enable_defensive_mode()
    
    print("Defensive Model Loader 테스트")
    
    # Whisper 테스트
    try:
        model = safe_whisper_load("base")
        print("OK Whisper 안전 로딩 성공")
    except Exception as e:
        print(f"ERROR Whisper 테스트 실패: {e}")
    
    # SentenceTransformer 테스트  
    try:
        embedder = safe_sentence_transformer_load('paraphrase-multilingual-MiniLM-L12-v2')
        if embedder:
            print("OK SentenceTransformer 안전 로딩 성공")
    except Exception as e:
        print(f"ERROR SentenceTransformer 테스트 실패: {e}")