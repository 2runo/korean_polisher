import os
from typing import Optional


def get_env(key: str, fallback: Optional[str]=None) -> str:
    val = os.environ.get(key) or fallback
    if val:
        return val
    raise Exception(f'key {key} not found on env')
