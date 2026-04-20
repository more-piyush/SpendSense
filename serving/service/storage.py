import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict


class JsonLineStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
