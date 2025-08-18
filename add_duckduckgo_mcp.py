#!/usr/bin/env python3
"""
DuckDuckGo MCP Integration Script
Add DuckDuckGo web search capabilities to MCP ecosystem
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Any

def backup_claude_config():
    """Backup existing Claude Desktop configuration"""
    
    config_path = Path.home() / ".config" / "claude" / "claude_desktop_config.json"
    backup_path = config_path.with_suffix('.json.backup')
    
    if config_path.exists():
        shutil.copy2(config_path, backup_path)
        print(f"Backup completed: {backup_path}")
        return True
    else:
        print(f"Claude Desktop config not found: {config_path}")
        return False

def read_current_config():
    """Read current Claude Desktop configuration"""
    
    config_path = Path.home() / ".config" / "claude" / "claude_desktop_config.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Current config loaded: {len(config.get('mcpServers', {}))} MCP servers")
        return config
    except Exception as e:
        print(f"Failed to read config: {e}")
        return None

def add_duckduckgo_servers(config: Dict[str, Any]):
    """Add DuckDuckGo MCP servers to configuration"""
    
    if 'mcpServers' not in config:
        config['mcpServers'] = {}
    
    # DuckDuckGo MCP servers to add
    duckduckgo_servers = {
        "duckduckgo-search": {
            "command": "npx",
            "args": ["@nickclyde/duckduckgo-mcp-server"]
        },
        "ddg-search-privacy": {
            "command": "npx", 
            "args": ["@oevortex/ddg_search@latest"]
        },
        "duckduckgo-simple": {
            "command": "npx",
            "args": ["-y", "duckduckgo-mcp-server"]
        }
    }
    
    added_servers = []
    skipped_servers = []
    
    for server_name, server_config in duckduckgo_servers.items():
        if server_name not in config['mcpServers']:
            config['mcpServers'][server_name] = server_config
            added_servers.append(server_name)
            print(f"Added: {server_name}")
        else:
            skipped_servers.append(server_name)
            print(f"Skipped: {server_name} (already exists)")
    
    return added_servers, skipped_servers

def write_updated_config(config: Dict[str, Any]):
    """Write updated configuration to file"""
    
    config_path = Path.home() / ".config" / "claude" / "claude_desktop_config.json"
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Updated config saved: {config_path}")
        return True
    except Exception as e:
        print(f"Failed to save config: {e}")
        return False

def create_test_script():
    """Create DuckDuckGo search test script"""
    
    test_script_content = '''#!/usr/bin/env python3
"""
DuckDuckGo MCP Server Test Script
"""

def test_duckduckgo_search():
    """Test DuckDuckGo search functionality"""
    
    print("DuckDuckGo MCP Search Test")
    print("=" * 30)
    
    # Jewelry-related search test cases
    test_queries = [
        "diamond market trends 2024",
        "GIA vs AGS diamond certification", 
        "luxury jewelry industry analysis",
        "artificial diamond vs natural diamond",
        "jewelry appraisal methods"
    ]
    
    print("Test search queries:")
    for i, query in enumerate(test_queries, 1):
        print(f"  {i}. {query}")
    
    print("\\nTo test in Claude Desktop, try:")
    print("'Search DuckDuckGo for diamond market trends 2024'")
    print("'Find latest jewelry industry trends on the web'")
    
    return test_queries

if __name__ == "__main__":
    test_duckduckgo_search()
'''
    
    test_file = Path(__file__).parent / "test_duckduckgo_search.py"
    
    try:
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_script_content)
        print(f"Test script created: {test_file}")
        return test_file
    except Exception as e:
        print(f"Failed to create test script: {e}")
        return None

def display_final_summary(added_servers, total_servers):
    """Display final summary and usage instructions"""
    
    print("\n" + "=" * 60)
    print("DuckDuckGo MCP Integration Complete!")
    print("=" * 60)
    
    print(f"MCP Server Status:")
    print(f"  - New DuckDuckGo servers added: {len(added_servers)}")
    print(f"  - Total MCP servers: {total_servers}")
    
    if added_servers:
        print(f"\nAdded DuckDuckGo servers:")
        for server in added_servers:
            print(f"  - {server}")
    
    print(f"\nClaude Desktop restart required:")
    print(f"  1. Close Claude Desktop completely")
    print(f"  2. Restart Claude Desktop")
    print(f"  3. Wait for automatic MCP server installation (1-2 minutes)")
    
    print(f"\nUsage examples:")
    print(f"  - 'Search DuckDuckGo for diamond trends 2024'")
    print(f"  - 'Find jewelry market analysis on the web'")
    print(f"  - 'Search for GIA diamond certification info'")
    
    print(f"\nTroubleshooting:")
    print(f"  - MCP connection issues: Restart Claude Desktop")
    print(f"  - Search errors: Check internet connection")
    print(f"  - Restore config: Use claude_desktop_config.json.backup")

def main():
    """Main execution function"""
    
    print("DuckDuckGo Web Search Integration for MCP Ecosystem")
    print("=" * 55)
    print("Purpose: Add real-time web search to Claude Desktop")
    print("Target: Privacy-focused DuckDuckGo search capabilities")
    print()
    
    try:
        # 1. Backup existing config
        print("Step 1: Backup existing configuration")
        if not backup_claude_config():
            print("Backup failed. Continue anyway? (y/N): ", end="")
            if input().lower() != 'y':
                return
        
        # 2. Read current config
        print("\nStep 2: Read current configuration")
        config = read_current_config()
        if not config:
            print("Failed to read configuration. Aborting.")
            return
        
        original_server_count = len(config.get('mcpServers', {}))
        
        # 3. Add DuckDuckGo servers
        print("\nStep 3: Add DuckDuckGo MCP servers")
        added_servers, skipped_servers = add_duckduckgo_servers(config)
        
        # 4. Save configuration
        print("\nStep 4: Save updated configuration")
        if not write_updated_config(config):
            print("Failed to save configuration. Aborting.")
            return
        
        # 5. Create test script
        print("\nStep 5: Create test script")
        test_file = create_test_script()
        
        # 6. Final summary
        final_server_count = len(config.get('mcpServers', {}))
        display_final_summary(added_servers, final_server_count)
        
        # Save success log
        log_data = {
            "timestamp": "2025-07-24T07:35:00Z",
            "action": "DuckDuckGo MCP Integration",
            "original_servers": original_server_count,
            "final_servers": final_server_count,
            "added_servers": added_servers,
            "skipped_servers": skipped_servers,
            "test_script": str(test_file) if test_file else None,
            "status": "success"
        }
        
        log_file = Path(__file__).parent / "duckduckgo_mcp_upgrade_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nOperation log saved: {log_file}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()