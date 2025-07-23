#!/usr/bin/env python3
"""
MCP Server Health Checker
자동으로 MCP 서버 상태를 점검하고 문제를 진단하는 스크립트
"""

import json
import subprocess
import os
import sys
from pathlib import Path

def check_config_file():
    """Claude Desktop 설정 파일 유효성 검사"""
    config_path = Path.home() / ".config" / "claude" / "claude_desktop_config.json"
    
    if not config_path.exists():
        print(f"❌ 설정 파일이 존재하지 않습니다: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ JSON 설정 파일 구문 검사 통과")
        return config
    except json.JSONDecodeError as e:
        print(f"❌ JSON 구문 오류: {e}")
        return False
    except Exception as e:
        print(f"❌ 설정 파일 읽기 오류: {e}")
        return False

def test_mcp_server(name, server_config):
    """개별 MCP 서버 테스트"""
    print(f"\n테스트 중: {name} 서버...")
    
    command = server_config.get("command", "")
    args = server_config.get("args", [])
    env = server_config.get("env", {})
    cwd = server_config.get("cwd", None)
    
    # 환경 변수 설정
    test_env = os.environ.copy()
    test_env.update(env)
    
    try:
        # npx 명령어인 경우 패키지 존재 확인
        if command == "npx":
            pkg_name = args[0] if args else ""
            if pkg_name.startswith("-y"):
                pkg_name = args[1] if len(args) > 1 else ""
            
            # --help 플래그로 테스트
            test_cmd = [command] + args + ["--help"]
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                env=test_env,
                cwd=cwd
            )
            
            if result.returncode == 0:
                print(f"  [OK] {name}: 정상 작동")
                return True
            else:
                print(f"  [FAIL] {name}: 오류 발생")
                print(f"     stderr: {result.stderr[:200]}...")
                return False
                
        # Python 모듈인 경우
        elif command == "python":
            module_name = args[1] if len(args) > 1 and args[0] == "-m" else ""
            if module_name:
                test_cmd = [command, "-c", f"import {module_name}"]
                result = subprocess.run(
                    test_cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=test_env,
                    cwd=cwd
                )
                
                if result.returncode == 0:
                    print(f"  [OK] {name}: 모듈 임포트 성공")
                    return True
                else:
                    print(f"  [FAIL] {name}: 모듈 없음 - {module_name}")
                    return False
            
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {name}: 타임아웃 (30초)")
        return False
    except Exception as e:
        print(f"  [ERROR] {name}: 예상치 못한 오류 - {e}")
        return False

def main():
    """메인 실행 함수"""
    print("MCP Server Health Checker 시작")
    print("=" * 50)
    
    # 1. 설정 파일 검사
    config = check_config_file()
    if not config:
        sys.exit(1)
    
    # 2. MCP 서버들 개별 테스트
    mcp_servers = config.get("mcpServers", {})
    if not mcp_servers:
        print("❌ mcpServers 설정이 없습니다.")
        sys.exit(1)
    
    print(f"\n📋 총 {len(mcp_servers)}개 MCP 서버 발견:")
    for name in mcp_servers.keys():
        print(f"  - {name}")
    
    # 3. 각 서버 테스트 실행
    working_servers = []
    failed_servers = []
    
    for name, server_config in mcp_servers.items():
        if test_mcp_server(name, server_config):
            working_servers.append(name)
        else:
            failed_servers.append(name)
    
    # 4. 결과 요약
    print("\n" + "=" * 50)
    print("📊 최종 결과:")
    print(f"✅ 정상 작동: {len(working_servers)}개")
    for server in working_servers:
        print(f"  - {server}")
    
    if failed_servers:
        print(f"❌ 오류 발생: {len(failed_servers)}개")
        for server in failed_servers:
            print(f"  - {server}")
        
        print(f"\n💡 해결 방안:")
        print(f"1. 실패한 서버들을 설정에서 제거")
        print(f"2. 필요한 패키지 설치: npm install -g [패키지명]")
        print(f"3. Python 모듈의 경우 pip install 필요")
        print(f"4. Claude Desktop 재시작 필요")
    
    # 5. Claude CLI 상태 확인
    print(f"\n🔍 Claude CLI MCP 상태 확인:")
    try:
        result = subprocess.run(
            ["claude", "mcp", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(result.stdout)
    except Exception as e:
        print(f"❌ Claude CLI 실행 오류: {e}")

if __name__ == "__main__":
    main()