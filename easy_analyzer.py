#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=== AI 컨퍼런스 분석기 ===")
    
    # Ollama 연결
    try:
        from shared.ollama_interface import OllamaInterface
        ollama = OllamaInterface()
        print(f"AI 연결 성공: {len(ollama.available_models)}개 모델 사용 가능")
    except Exception as e:
        print(f"AI 연결 실패: {e}")
        return
    
    print("\n회의 내용을 입력하세요.")
    print("화자1:, 화자2: 형식으로 입력하고, 완료하면 'END'를 입력하세요.\n")
    
    lines = []
    while True:
        line = input("> ")
        if line.upper() == 'END':
            break
        if line.strip():
            lines.append(line)
    
    if not lines:
        print("입력된 내용이 없습니다.")
        return
    
    content = '\n'.join(lines)
    print(f"\n분석할 내용 ({len(lines)}줄):")
    print("-" * 40)
    for i, line in enumerate(lines, 1):
        print(f"{i:2d}. {line}")
    print("-" * 40)
    
    # AI 분석
    print("\nAI 분석 중...")
    try:
        response = ollama.generate_response(
            prompt=f"다음 한국어 회의를 분석해주세요:\n\n{content}\n\n분석 내용:\n1. 회의 주제\n2. 주요 안건\n3. 화자별 역할\n4. 결정사항\n5. 한 줄 요약",
            model="qwen2.5:7b"
        )
        
        print("\n=== AI 분석 결과 ===")
        # 한글 출력을 위해 인코딩 처리
        try:
            print(response)
        except UnicodeEncodeError:
            # 콘솔 출력 실패시 ASCII로 변환
            clean_response = response.encode('ascii', 'ignore').decode('ascii')
            print(clean_response)
        
        print("\n분석 완료!")
        
    except Exception as e:
        print(f"분석 실패: {e}")

if __name__ == "__main__":
    main()