"""
솔로몬드 AI 시스템 - API 계층
FastAPI 기반 REST API 모듈
"""

__version__ = "3.0.0"
__author__ = "전근혁 (솔로몬드 대표)"

from .app import *
from .routes import *
from .middleware import *

__all__ = [
    "app",
    "routes",
    "middleware"
]
