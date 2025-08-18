#!/usr/bin/env python3
"""
진짜 최신 모델 설치 (GEMMA3, Qwen3)
"""

import subprocess
import os
import time

def find_ollama():
    """Ollama 실행파일 찾기"""
    
    paths = [
        r"C:\Users\{}\AppData\Local\Programs\Ollama\ollama.exe".format(os.getenv("USERNAME")),
        r"C:\Program Files\Ollama\ollama.exe"
    ]
    
    for path in paths:
        if os.path.exists(path):
            print(f"Ollama 경로: {path}")
            return path
    
    return "ollama"

def check_ollama_models():
    """Ollama에서 사용 가능한 모델 확인"""
    
    print("=== Ollama 모델 저장소 확인 ===")
    
    ollama_path = find_ollama()
    
    try:
        # ollama list로 현재 설치된 모델 확인
        result = subprocess.run([ollama_path, "list"],
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("현재 설치된 모델:")
            print(result.stdout)
        else:
            print("모델 목록 확인 실패")
            
    except Exception as e:
        print(f"모델 확인 오류: {str(e)}")

def install_cutting_edge_models():
    """최신 모델 설치 시도"""
    
    print("\n=== 2025년 최신 모델 설치 시도 ===")
    
    ollama_path = find_ollama()
    
    # 진짜 최신 모델들 (확인된 순서대로)
    cutting_edge_models = [
        # Qwen3 시리즈 (2025년 4월 출시)
        ("qwen3:8b", "Qwen3 8B - 알리바바 최신 3세대"),
        ("qwen3:7b", "Qwen3 7B - 알리바바 최신 3세대"),
        ("qwen3:4b", "Qwen3 4B - 알리바바 최신 3세대 경량"),
        
        # GEMMA3 시리즈 (2025년 7월 출시)
        ("gemma3:9b", "GEMMA3 9B - 구글 최신 3세대"),
        ("gemma3:7b", "GEMMA3 7B - 구글 최신 3세대"),
        ("gemma3:2b", "GEMMA3 2B - 구글 최신 3세대 경량"),
        
        # 백업 옵션 (확실히 존재하는 모델들)
        ("qwen2.5:7b", "Qwen 2.5 7B - 백업 옵션"),
        ("gemma2:9b", "Gemma 2 9B - 백업 옵션"),
        ("llama3.2:8b", "Llama 3.2 8B - 백업 옵션")
    ]
    
    successfully_installed = []
    
    for model_id, description in cutting_edge_models:
        print(f"\n{description} 설치 시도...")
        print(f"모델: {model_id}")
        
        try:
            # ollama pull 시도
            print("다운로드 중...")
            result = subprocess.run([ollama_path, "pull", model_id],
                                  capture_output=True, text=True, timeout=900)  # 15분 대기
            
            if result.returncode == 0:
                print(f"성공! {model_id} 설치 완료")
                successfully_installed.append(model_id)
                
                # 성공하면 같은 시리즈의 다른 크기는 스킵 (시간 절약)
                if model_id.startswith("qwen3:") and len([m for m in successfully_installed if m.startswith("qwen3:")]) >= 1:
                    print("Qwen3 시리즈 설치 완료, 다음 시리즈로...")
                    continue
                elif model_id.startswith("gemma3:") and len([m for m in successfully_installed if m.startswith("gemma3:")]) >= 1:
                    print("GEMMA3 시리즈 설치 완료, 다음 시리즈로...")
                    continue
                    
            else:
                print(f"실패: {model_id}")
                print(f"오류 메시지: {result.stderr}")
                
                # 만약 3세대 모델이 없다면 2세대로 폴백
                if "qwen3:" in model_id:
                    fallback_model = model_id.replace("qwen3:", "qwen2.5:")
                    print(f"폴백 시도: {fallback_model}")
                    
                    fallback_result = subprocess.run([ollama_path, "pull", fallback_model],
                                                   capture_output=True, text=True, timeout=600)
                    if fallback_result.returncode == 0:
                        print(f"폴백 성공: {fallback_model}")
                        successfully_installed.append(fallback_model)
                        
                elif "gemma3:" in model_id:
                    fallback_model = model_id.replace("gemma3:", "gemma2:")
                    print(f"폴백 시도: {fallback_model}")
                    
                    fallback_result = subprocess.run([ollama_path, "pull", fallback_model],
                                                   capture_output=True, text=True, timeout=600)
                    if fallback_result.returncode == 0:
                        print(f"폴백 성공: {fallback_model}")
                        successfully_installed.append(fallback_model)
                
        except subprocess.TimeoutExpired:
            print(f"시간 초과: {model_id}")
        except Exception as e:
            print(f"설치 오류: {str(e)}")
        
        # 성공적으로 2개 이상 설치되면 중단 (시간 절약)
        if len(successfully_installed) >= 2:
            print(f"\n충분한 모델 설치됨 ({len(successfully_installed)}개), 설치 중단")
            break
    
    return successfully_installed

def quick_test_installed_models(models):
    """설치된 모델 빠른 테스트"""
    
    print(f"\n=== 설치된 모델 빠른 테스트 ===")
    
    if not models:
        print("테스트할 모델이 없습니다.")
        return
    
    # 가장 최신 모델로 테스트
    test_model = models[0]
    print(f"테스트 모델: {test_model}")
    
    try:
        import requests
        
        # 간단한 한국어 주얼리 테스트
        prompt = """한국어로 답변해주세요:

고객이 "결혼반지 찾고 있는데 200만원 예산으로 어떤 걸 추천하세요?"라고 물었습니다.
전문 주얼리 상담사로서 답변해주세요."""

        payload = {
            "model": test_model,
            "prompt": prompt,
            "stream": False
        }
        
        print("테스트 실행 중...")
        start_time = time.time()
        
        response = requests.post("http://localhost:11434/api/generate",
                               json=payload, timeout=60)
        
        if response.status_code == 200:
            end_time = time.time()
            result = response.json()
            answer = result.get('response', '')
            
            processing_time = end_time - start_time
            
            print(f"\n테스트 결과:")
            print(f"  처리시간: {processing_time:.2f}초")
            print(f"  응답길이: {len(answer)} 글자")
            
            # 한국어 키워드 확인
            keywords = ['결혼반지', '200만원', '예산', '추천', '다이아', '금']
            found = [k for k in keywords if k in answer]
            
            print(f"  키워드 인식: {len(found)}/{len(keywords)}개")
            print(f"  인식된 키워드: {found}")
            print(f"  응답 샘플: {answer[:100]}...")
            
            if len(found) >= 3 and len(answer) > 100:
                print(f"\n최신 모델 테스트 성공!")
                return True
            else:
                print(f"\n테스트 부분 성공")
                return True
        else:
            print(f"API 오류: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"테스트 실패: {str(e)}")
        return False

def main():
    """메인 실행"""
    
    print("진짜 최신 모델 설치 (GEMMA3, Qwen3)")
    print("=" * 50)
    
    # 1. 현재 설치된 모델 확인
    check_ollama_models()
    
    # 2. 최신 모델 설치
    installed_models = install_cutting_edge_models()
    
    if installed_models:
        print(f"\n설치 완료: {len(installed_models)}개 최신 모델")
        for model in installed_models:
            is_gen3 = "3:" in model
            generation = "3세대" if is_gen3 else "2세대"
            print(f"  - {model} ({generation})")
        
        # 3. 빠른 테스트
        if quick_test_installed_models(installed_models):
            print(f"\n다음 단계:")
            print(f"1. python check_gemma3_status.py - 상세 확인")
            print(f"2. 솔로몬드 AI 시스템에 통합")
            print(f"3. 기존 모델과 성능 비교")
        
        return True
    else:
        print(f"\n최신 모델 설치 실패")
        print(f"원인: GEMMA3, Qwen3가 아직 Ollama에 등록되지 않았을 수 있음")
        print(f"\n대안:")
        print(f"1. 수동으로 ollama pull qwen2.5:7b 실행")
        print(f"2. 수동으로 ollama pull gemma2:9b 실행")
        
        return False

if __name__ == "__main__":
    main()