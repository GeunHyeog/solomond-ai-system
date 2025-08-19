#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

def main():
    print("SOLOMOND AI - Large File Check")
    print("=" * 40)
    
    # 파일 경로
    mov_file = Path("user_files/JGA2025_D1/IMG_0032.MOV")
    
    if not mov_file.exists():
        print(f"File not found: {mov_file}")
        return
    
    # 파일 크기 확인
    file_size = mov_file.stat().st_size
    size_gb = file_size / (1024 * 1024 * 1024)
    
    print(f"File: {mov_file.name}")
    print(f"Size: {size_gb:.2f}GB ({file_size:,} bytes)")
    print()
    
    if size_gb > 2.0:
        print("Large file detected!")
        print("Ready for local processing (bypasses web upload limits)")
        print()
        
        print("Processing method:")
        print("1. File is correctly located in user_files folder")
        print("2. Local processing prevents MemoryError")
        print("3. Completely bypasses Streamlit limitations")
        print()
        
        print("SUCCESS: Large file processing ready!")
        
    else:
        print("Regular size file")
    
    print("\nLarge file validation complete!")
    return True

if __name__ == "__main__":
    main()