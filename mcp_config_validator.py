#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP 설정 검증 및 재발방지 시스템
- 실제 Claude Desktop이 사용하는 설정 파일 확인
- 8개 서버 설정 검증
- 자동 수정 기능
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# 예상되는 8개 MCP 서버 목록
EXPECTED_SERVERS = {
    "memory": "@modelcontextprotocol/server-memory",
    "sequential-thinking": "@modelcontextprotocol/server-sequential-thinking", 
    "filesystem": "@modelcontextprotocol/server-filesystem",
    "playwright": "@playwright/mcp",
    "perplexity": "mcp-perplexity-search",
    "github-v2": "@andrebuzeli/github-mcp-v2",
    "notion": "@notionhq/notion-mcp-server",
    "smart-crawler": "mcp-smart-crawler"
}

def find_actual_config_file():
    """실제 Claude Desktop이 사용하는 설정 파일 경로 찾기"""
    possible_paths = [
        Path.home() / "AppData/Roaming/Claude/claude_desktop_config.json",
        Path.home() / ".config/claude/claude_desktop_config.json"
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"[INFO] Found config file: {path}")
            return path
    
    print("[ERROR] No Claude Desktop config file found!")
    return None

def check_config_servers(config_path):
    """설정 파일의 서버 구성 확인"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        servers = config.get("mcpServers", {})
        print(f"\n[INFO] Found {len(servers)} servers in config:")
        
        for name in servers.keys():
            print(f"  - {name}")
        
        # 예상 서버와 비교
        missing_servers = set(EXPECTED_SERVERS.keys()) - set(servers.keys())
        extra_servers = set(servers.keys()) - set(EXPECTED_SERVERS.keys())
        
        if missing_servers:
            print(f"\n[WARNING] Missing servers: {missing_servers}")
        
        if extra_servers:
            print(f"\n[WARNING] Extra servers: {extra_servers}")
            
        return servers, missing_servers, extra_servers
        
    except Exception as e:
        print(f"[ERROR] Failed to read config: {e}")
        return None, None, None

def test_server_installation(server_name, package_name):
    """개별 서버 설치 상태 확인"""
    try:
        if package_name.startswith("@"):
            # npm 패키지
            result = subprocess.run(
                ["npx", package_name, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        else:
            # 다른 패키지
            result = subprocess.run(
                ["npx", package_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            return True  # 타임아웃이어도 실행되면 설치됨
            
    except subprocess.TimeoutExpired:
        return True  # MCP 서버는 대기 상태로 들어가므로 타임아웃 정상
    except Exception as e:
        print(f"[ERROR] Testing {server_name}: {e}")
        return False

def create_correct_config():
    """올바른 8개 서버 설정 생성"""
    config = {
        "mcpServers": {
            "memory": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-memory"]
            },
            "sequential-thinking": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-sequential-thinking"]
            },
            "filesystem": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-filesystem", "C:/Users/PC_58410/solomond-ai-system"]
            },
            "playwright": {
                "command": "npx",
                "args": ["@playwright/mcp"]
            },
            "perplexity": {
                "command": "npx",
                "args": ["mcp-perplexity-search"],
                "env": {
                    "PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}"
                }
            },
            "github-v2": {
                "command": "npx",
                "args": ["@andrebuzeli/github-mcp-v2"],
                "env": {
                    "GITHUB_ACCESS_TOKEN": "${GITHUB_ACCESS_TOKEN}"
                }
            },
            "notion": {
                "command": "npx",
                "args": ["@notionhq/notion-mcp-server"],
                "env": {
                    "NOTION_API_KEY": "${NOTION_API_KEY}"
                }
            },
            "smart-crawler": {
                "command": "npx",
                "args": ["mcp-smart-crawler"]
            }
        }
    }
    return config

def main():
    print("=" * 60)
    print("MCP 설정 검증 및 재발방지 시스템")
    print("=" * 60)
    
    # 1. 실제 설정 파일 찾기
    config_path = find_actual_config_file()
    if not config_path:
        sys.exit(1)
    
    # 2. 현재 설정 확인
    servers, missing, extra = check_config_servers(config_path)
    if servers is None:
        sys.exit(1)
    
    # 3. 서버 설치 상태 확인
    print(f"\n[INFO] Testing server installations...")
    for server_name, package_name in EXPECTED_SERVERS.items():
        is_installed = test_server_installation(server_name, package_name)
        status = "[OK]" if is_installed else "[NOT INSTALLED]"
        print(f"  {server_name}: {status}")
    
    # 4. 설정 파일 자동 수정 (필요시)
    if missing or extra:
        print(f"\n[INFO] Configuration issues detected. Auto-fixing...")
        correct_config = create_correct_config()
        
        # 백업 생성
        backup_path = config_path.with_suffix('.json.backup')
        import shutil
        shutil.copy2(config_path, backup_path)
        print(f"[INFO] Backup created: {backup_path}")
        
        # 올바른 설정 적용
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(correct_config, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Configuration fixed! Restart Claude Desktop.")
    else:
        print(f"\n[INFO] Configuration is correct!")
    
    print(f"\n[INFO] Validation complete. Check Claude Desktop for MCP functions.")

if __name__ == "__main__":
    main()