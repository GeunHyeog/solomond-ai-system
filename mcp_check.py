#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import subprocess
import os
import sys
from pathlib import Path

def check_config_file():
    config_path = Path.home() / ".config" / "claude" / "claude_desktop_config.json"
    
    if not config_path.exists():
        print(f"[FAIL] Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("[OK] JSON config file syntax check passed")
        return config
    except json.JSONDecodeError as e:
        print(f"[FAIL] JSON syntax error: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Config file read error: {e}")
        return False

def test_mcp_server(name, server_config):
    print(f"\nTesting: {name} server...")
    
    command = server_config.get("command", "")
    args = server_config.get("args", [])
    env = server_config.get("env", {})
    cwd = server_config.get("cwd", None)
    
    test_env = os.environ.copy()
    test_env.update(env)
    
    try:
        if command == "npx":
            pkg_name = args[0] if args else ""
            if pkg_name.startswith("-y"):
                pkg_name = args[1] if len(args) > 1 else ""
            
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
                print(f"  [OK] {name}: Working normally")
                return True
            else:
                print(f"  [FAIL] {name}: Error occurred")
                if result.stderr:
                    print(f"     stderr: {result.stderr[:100]}...")
                return False
                
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
                    print(f"  [OK] {name}: Module import successful")
                    return True
                else:
                    print(f"  [FAIL] {name}: Module not found - {module_name}")
                    return False
            
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {name}: Timeout (30s)")
        return False
    except Exception as e:
        print(f"  [ERROR] {name}: Unexpected error - {e}")
        return False

def main():
    print("MCP Server Health Checker")
    print("=" * 50)
    
    config = check_config_file()
    if not config:
        sys.exit(1)
    
    mcp_servers = config.get("mcpServers", {})
    if not mcp_servers:
        print("[FAIL] No mcpServers configuration found.")
        sys.exit(1)
    
    print(f"\nFound {len(mcp_servers)} MCP servers:")
    for name in mcp_servers.keys():
        print(f"  - {name}")
    
    working_servers = []
    failed_servers = []
    
    for name, server_config in mcp_servers.items():
        if test_mcp_server(name, server_config):
            working_servers.append(name)
        else:
            failed_servers.append(name)
    
    print("\n" + "=" * 50)
    print("Final Results:")
    print(f"[OK] Working: {len(working_servers)} servers")
    for server in working_servers:
        print(f"  - {server}")
    
    if failed_servers:
        print(f"[FAIL] Failed: {len(failed_servers)} servers")
        for server in failed_servers:
            print(f"  - {server}")
        
        print(f"\nSolutions:")
        print(f"1. Remove failed servers from config")
        print(f"2. Install required packages: npm install -g [package-name]")
        print(f"3. For Python modules: pip install [module-name]")
        print(f"4. Restart Claude Desktop")
    
    print(f"\nClaude CLI MCP status:")
    try:
        result = subprocess.run(
            ["claude", "mcp", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(result.stdout)
    except Exception as e:
        print(f"[ERROR] Claude CLI execution error: {e}")

if __name__ == "__main__":
    main()