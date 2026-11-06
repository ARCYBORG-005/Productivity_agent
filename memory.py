import os
from typing import Optional

ROOT = os.path.dirname(__file__)
MEM_DIR = os.path.join(ROOT, "memories")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_memory(scope: str, name: str, content: str) -> str:
    """Write memory content to `memories/{scope}/{name}` and return the path."""
    scope_dir = os.path.join(MEM_DIR, scope)
    _ensure_dir(scope_dir)
    file_path = os.path.join(scope_dir, name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path


def read_memory(scope: str, name: str) -> Optional[str]:
    file_path = os.path.join(MEM_DIR, scope, name)
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def list_memory(scope: str) -> list:
    dir_path = os.path.join(MEM_DIR, scope)
    if not os.path.exists(dir_path):
        return []
    return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
