#!/usr/bin/env python3
"""
실제 OCR 테스트 스크립트
EasyOCR로 실제 이미지에서 텍스트 추출
"""

import easyocr
import os
import time
from pathlib import Path

def test_real_ocr(image_path: str):
    """실제 이미지 OCR 테스트"""
    
    print(f"🖼️ EasyOCR 실제 이미지 분석 시작")
    print(f"📁 파일: {os.path.basename(image_path)}")
    print("=" * 50)
    
    # 파일 존재 확인
    if not os.path.exists(image_path):
        print(f"❌ 파일을 찾을 수 없습니다: {image_path}")
        return None
    
    # 파일 크기 확인
    file_size = os.path.getsize(image_path) / (1024 * 1024)
    print(f"📏 파일 크기: {file_size:.2f} MB")
    
    try:
        # EasyOCR Reader 초기화
        print("🔄 EasyOCR 한/영 모델 로딩...")
        start_time = time.time()
        reader = easyocr.Reader(['ko', 'en'])
        load_time = time.time() - start_time
        print(f"✅ 모델 로드 완료 ({load_time:.1f}초)")
        
        # OCR 텍스트 추출
        print("📝 이미지에서 텍스트 추출 중...")
        ocr_start = time.time()
        
        results = reader.readtext(image_path)
        
        ocr_time = time.time() - ocr_start
        print(f"✅ OCR 완료 ({ocr_time:.1f}초)")
        
        # 결과 분석
        detected_texts = []
        total_confidence = 0
        
        print(f"\n📊 OCR 결과:")
        print(f"   📝 감지된 텍스트 블록: {len(results)}개")
        
        for i, (bbox, text, confidence) in enumerate(results, 1):
            detected_texts.append(text)
            total_confidence += confidence
            print(f"   {i}. [{confidence:.2f}] {text}")
        
        avg_confidence = total_confidence / len(results) if results else 0
        full_text = ' '.join(detected_texts)
        
        print(f"\n📄 추출된 전체 텍스트:")
        print("-" * 40)
        print(full_text)
        print("-" * 40)
        
        return {
            "file_name": os.path.basename(image_path),
            "file_size_mb": round(file_size, 2),
            "processing_time": round(ocr_time, 1),
            "blocks_detected": len(results),
            "average_confidence": round(avg_confidence, 3),
            "full_text": full_text,
            "detailed_results": results
        }
        
    except Exception as e:
        print(f"❌ OCR 중 오류: {str(e)}")
        return None

def main():
    """메인 실행 함수"""
    
    print("🚀 EasyOCR 실제 이미지 분석 테스트")
    print("💎 홍콩 세미나 이미지 → 텍스트 추출")
    print("=" * 60)
    
    # 테스트할 이미지 파일
    base_path = "/mnt/c/Users/PC_58410/Desktop/근혁/세미나/202506홍콩쇼/D1"
    
    test_images = [
        f"{base_path}/IMG_2160.JPG",
        # 추가 이미지가 있다면 여기에 추가
    ]
    
    results = []
    
    for i, image_path in enumerate(test_images, 1):
        print(f"\n🖼️ 테스트 {i}/{len(test_images)}")
        print("-" * 40)
        
        result = test_real_ocr(image_path)
        
        if result:
            results.append(result)
            print("✅ OCR 성공")
        else:
            print("❌ OCR 실패")
        
        print()
    
    # 최종 결과 요약
    print("📊 OCR 분석 최종 요약")
    print("=" * 60)
    print(f"✅ 성공적으로 분석된 이미지: {len(results)}/{len(test_images)}")
    
    if results:
        print("\n📝 OCR 결과:")
        for result in results:
            print(f"   📁 {result['file_name']}")
            print(f"      크기: {result['file_size_mb']}MB")
            print(f"      처리시간: {result['processing_time']}초")
            print(f"      텍스트 블록: {result['blocks_detected']}개")
            print(f"      평균 신뢰도: {result['average_confidence']}")
            print(f"      추출 텍스트: {result['full_text'][:100]}{'...' if len(result['full_text']) > 100 else ''}")
            print()

if __name__ == "__main__":
    main()