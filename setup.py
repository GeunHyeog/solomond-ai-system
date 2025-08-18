"""
솔로몬드 AI 프레임워크 설치 스크립트
PyPI 배포 가능한 패키지 구성
"""

from setuptools import setup, find_packages
from pathlib import Path

# README 파일 읽기
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# 버전 정보
version = "1.0.0"

# 기본 의존성
install_requires = [
    # 핵심 AI 라이브러리
    "openai-whisper>=20230314",
    "easyocr>=1.7.0", 
    "transformers>=4.30.0",
    "torch>=2.0.0",
    
    # 파일 처리
    "opencv-python>=4.8.0",
    "Pillow>=10.0.0",
    "PyPDF2>=3.0.0",
    "python-docx>=0.8.11",
    
    # 웹 UI
    "streamlit>=1.28.0",
    "plotly>=5.15.0",
    
    # 유틸리티
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "requests>=2.31.0",
    
    # 오디오/비디오 처리
    "librosa>=0.10.0",
    "soundfile>=0.12.0",
    
    # 데이터 처리
    "pandas>=2.0.0",
    "numpy>=1.24.0"
]

# 선택적 의존성
extras_require = {
    'jewelry': [
        "gemstone-classifier>=1.0.0",  # 가상 패키지
        "jewelry-ocr-enhanced>=1.0.0"
    ],
    'medical': [
        "medical-ner>=1.0.0",  # 가상 패키지
        "hipaa-compliance>=1.0.0",
        "medical-image-processor>=1.0.0"
    ],
    'education': [
        "education-content-analyzer>=1.0.0",  # 가상 패키지
        "curriculum-mapper>=1.0.0"
    ],
    'dev': [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.7.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0"
    ],
    'all': [
        # 모든 도메인 패키지 포함
    ]
}

# 'all' 의존성 구성
all_deps = []
for deps in extras_require.values():
    if deps != extras_require['all']:
        all_deps.extend(deps)
extras_require['all'] = list(set(all_deps))

setup(
    name="solomond-ai-framework",
    version=version,
    author="SolomondAI Team",
    author_email="contact@solomond.ai",
    description="멀티모달 AI 분석을 위한 재사용 가능한 프레임워크",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GeunHyeog/solomond-ai-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Text Processing :: Linguistic"
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'solomond-ai=solomond_ai.cli:main',
            'solomond-create-project=solomond_ai.cli:create_project',
            'solomond-analyze=solomond_ai.cli:analyze_files',
        ],
    },
    include_package_data=True,
    package_data={
        'solomond_ai': [
            'templates/*.yaml',
            'templates/*.json',
            'static/*',
            'ui/themes/*'
        ],
    },
    zip_safe=False,
    keywords=[
        "AI", "machine learning", "multimodal", "speech recognition", 
        "OCR", "image analysis", "video processing", "natural language processing",
        "whisper", "easyocr", "transformers", "streamlit",
        "jewelry analysis", "medical analysis", "education"
    ],
    project_urls={
        "Documentation": "https://solomond-ai-docs.readthedocs.io/",
        "Source": "https://github.com/GeunHyeog/solomond-ai-system",
        "Bug Reports": "https://github.com/GeunHyeog/solomond-ai-system/issues",
        "Funding": "https://github.com/sponsors/GeunHyeog",
    }
)