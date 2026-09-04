"""저장소 루트의 uq_paths.py 로 위임하는 shim.

스크립트는 자기 디렉터리 안에서 실행되므로 `import paths` 가 이 파일을 찾는다.
실제 경로 정의는 저장소 루트 uq_paths.py 한 곳에만 있다.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uq_paths import *  # noqa: F401,F403
from uq_paths import ensure_dirs  # noqa: F401
