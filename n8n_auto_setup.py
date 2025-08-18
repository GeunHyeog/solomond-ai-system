#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 n8n 자동 워크플로우 설정 스크립트
SOLOMOND AI용 워크플로우를 n8n API로 자동 생성
"""

import requests
import json
import time
from pathlib import Path

class N8nAutoSetup:
    """n8n 자동 설정"""
    
    def __init__(self):
        self.base_url = "http://localhost:5678/api/v1"
        self.session = requests.Session()
        
    def check_connection(self):
        """n8n 서버 연결 확인"""
        try:
            response = requests.get("http://localhost:5678/healthz", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_workflows_template(self):
        """워크플로우 템플릿 로드"""
        template_file = Path("solomond_n8n_workflows.json")
        if template_file.exists():
            with open(template_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def create_workflow_via_ui_guide(self):
        """UI를 통한 워크플로우 생성 가이드"""
        workflows = self.get_workflows_template()
        if not workflows:
            print("❌ 워크플로우 템플릿을 찾을 수 없습니다.")
            return
        
        print("🎯 n8n 워크플로우 생성 가이드")
        print("=" * 50)
        
        for i, (key, workflow) in enumerate(workflows["workflows"].items(), 1):
            print(f"\n📋 {i}. {workflow['name']}")
            print(f"설명: {workflow['description']}")
            print("\n🔧 생성 방법:")
            print("1. n8n 웹 인터페이스에서 'Add workflow' 클릭")
            print("2. 우측 상단 '⋮' 메뉴 → 'Import from JSON' 선택")
            print("3. 아래 JSON을 복사해서 붙여넣기:")
            print("-" * 30)
            
            # 개별 워크플로우 JSON 생성
            individual_workflow = {
                "name": workflow["name"],
                "nodes": workflow["nodes"],
                "connections": workflow["connections"],
                "active": True,
                "settings": {},
                "staticData": {}
            }
            
            print(json.dumps(individual_workflow, indent=2, ensure_ascii=False))
            print("-" * 30)
            print("4. 'Import' 클릭")
            print("5. 워크플로우 저장 (Ctrl+S)")
            print("6. 우측 상단 토글로 'Active' 상태로 변경")
            print("\n" + "="*50)
        
        print("\n✅ 모든 워크플로우 생성 완료 후:")
        print("- 각 워크플로우가 'Active' 상태인지 확인")
        print("- Webhook URL 확인:")
        for url_name, url in workflows["setup_instructions"]["webhook_urls"].items():
            print(f"  - {url_name}: {url}")
        
        return True
    
    def generate_simple_files(self):
        """사용자가 쉽게 import할 수 있도록 개별 JSON 파일 생성"""
        workflows = self.get_workflows_template()
        if not workflows:
            return False
        
        print("📁 개별 워크플로우 파일 생성 중...")
        
        for key, workflow in workflows["workflows"].items():
            # 개별 워크플로우 JSON 생성
            individual_workflow = {
                "name": workflow["name"],
                "nodes": workflow["nodes"],
                "connections": workflow["connections"],
                "active": True,
                "settings": {},
                "staticData": {}
            }
            
            # 파일로 저장
            filename = f"n8n_workflow_{key}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(individual_workflow, f, indent=2, ensure_ascii=False)
            
            print(f"✅ {filename} 생성됨")
        
        print("\n🎯 사용법:")
        print("1. n8n 웹에서 'Add workflow' 클릭")
        print("2. '⋮' → 'Import from JSON' 선택") 
        print("3. 생성된 JSON 파일 내용을 복사 붙여넣기")
        print("4. 'Import' → 저장 → 'Active' 설정")
        
        return True

def main():
    """메인 실행"""
    setup = N8nAutoSetup()
    
    print("🚀 SOLOMOND AI - n8n 자동 설정")
    print("=" * 40)
    
    # 연결 확인
    if not setup.check_connection():
        print("❌ n8n 서버에 연결할 수 없습니다.")
        print("💡 해결방법:")
        print("1. n8n이 실행 중인지 확인: http://localhost:5678")
        print("2. 방화벽 설정 확인")
        return
    
    print("✅ n8n 서버 연결 확인됨")
    
    # 워크플로우 템플릿 확인
    workflows = setup.get_workflows_template()
    if not workflows:
        print("❌ 워크플로우 템플릿이 없습니다.")
        return
    
    print(f"✅ {len(workflows['workflows'])}개 워크플로우 템플릿 로드됨")
    
    # 사용자 선택
    print("\n🎯 설정 방법을 선택하세요:")
    print("1. 단계별 가이드 보기 (추천)")
    print("2. 개별 JSON 파일 생성")
    
    try:
        choice = input("선택 (1 또는 2): ").strip()
        
        if choice == "1":
            setup.create_workflow_via_ui_guide()
        elif choice == "2":
            setup.generate_simple_files()
        else:
            print("❌ 잘못된 선택입니다.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자가 취소했습니다.")

if __name__ == "__main__":
    main()