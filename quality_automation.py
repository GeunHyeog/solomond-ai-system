#!/usr/bin/env python3
"""
솔로몬드 AI 자동화 품질 검사 스크립트
매일 실행되어 코드 품질을 모니터링하고 GitHub에 보고
"""

import sys
import subprocess
from pathlib import Path

def run_quality_checks():
    """품질 검사 실행"""
    
    print("🔍 솔로몬드 AI 품질 검사 시작...")
    
    checks = [
        ("Python 문법 검사", "python -m py_compile *.py"),
        ("Import 검사", "python -c 'import jewelry_stt_ui_v23_real'"),
        ("테스트 실행", "python -m pytest test_* -v"),
    ]
    
    results = []
    
    for name, command in checks:
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                results.append(f"✅ {name}: 성공")
            else:
                results.append(f"❌ {name}: 실패 - {result.stderr[:100]}")
        except Exception as e:
            results.append(f"⚠️ {name}: 오류 - {str(e)}")
    
    # 결과 저장
    with open("quality_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    
    print("📊 품질 검사 완료 - quality_report.txt 확인")

if __name__ == "__main__":
    run_quality_checks()
