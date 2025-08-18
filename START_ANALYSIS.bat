@echo off
echo ========================================
echo SOLOMOND AI 자동 분석 시작
echo ========================================
echo.

cd /d "C:\Users\PC_58410\solomond-ai-system"

echo 1. 분석 파일 검색 중...
python -c "from pathlib import Path; files=list(Path('user_files').rglob('*.jpg'))+list(Path('user_files').rglob('*.wav'))+list(Path('user_files').rglob('*.m4a')); print(f'Found {len(files)} files')"

echo.
echo 2. AI 분석 시작...
python -c "
from pathlib import Path
import json
import time

print('Starting AI analysis...')
files = list(Path('user_files').rglob('*.jpg'))[:3]
results = []

try:
    import easyocr
    reader = easyocr.Reader(['en', 'ko'], gpu=False)
    
    for i, file in enumerate(files, 1):
        print(f'[{i}/{len(files)}] Processing: {file.name}')
        ocr_results = reader.readtext(str(file))
        text = ' '.join([r[1] for r in ocr_results if r[2] > 0.3])
        
        results.append({
            'file': file.name,
            'extracted_text': text[:200] + '...' if len(text) > 200 else text,
            'text_blocks': len(ocr_results),
            'status': 'success'
        })
        
except Exception as e:
    print(f'AI analysis failed: {e}')
    for file in files:
        results.append({
            'file': file.name,
            'status': 'basic_info_only',
            'size_mb': round(file.stat().st_size / (1024*1024), 2)
        })

# Save results
with open('quick_analysis_results.json', 'w', encoding='utf-8') as f:
    json.dump({
        'total_files': len(results),
        'results': results,
        'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S')
    }, f, indent=2, ensure_ascii=False)

print('Analysis complete!')
print('Results saved to: quick_analysis_results.json')
"

echo.
echo 3. 결과 파일 열기...
if exist quick_analysis_results.json (
    echo 분석 완료! quick_analysis_results.json 파일을 확인하세요.
    notepad quick_analysis_results.json
) else (
    echo 결과 파일 생성 실패
)

echo.
echo ========================================
echo 분석 완료!
echo ========================================
pause