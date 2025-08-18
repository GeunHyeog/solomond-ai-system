#!/usr/bin/env python3
import subprocess
import time
import webbrowser

def start_SOLOMOND_ai():
    print("솔로몬드 AI v3.0 자동 시작...")
    
    modules = [
        ("메인 대시보드", "solomond_ai_main_dashboard.py", 8500),
        ("컨퍼런스 분석", "modules/module1_conference/conference_analysis.py", 8501),
        ("웹 크롤러", "modules/module2_crawler/web_crawler_main.py", 8502),
        ("보석 분석", "modules/module3_gemstone/gemstone_analyzer.py", 8503),
        ("3D CAD", "modules/module4_3d_cad/image_to_cad.py", 8504)
    ]
    
    for name, file, port in modules:
        try:
            subprocess.Popen([
                "streamlit", "run", file, "--server.port", str(port)
            ])
            print(f"✓ {name} 시작됨 (포트 {port})")
            time.sleep(2)
        except:
            print(f"✗ {name} 시작 실패")
    
    time.sleep(5)
    webbrowser.open("http://localhost:8500")

if __name__ == "__main__":
    start_SOLOMOND_ai()
