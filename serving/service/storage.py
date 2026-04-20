import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, Protocol

import boto3


class EventStore(Protocol):
    def write(self, record: Dict[str, Any], event_type: str = "event") -> None:
        ...


class JsonLineStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def write(self, record: Dict[str, Any], event_type: str = "event") -> None:
        payload = dict(record)
        payload.setdefault("event_type", event_type)
        line = json.dumps(payload, ensure_ascii=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


class S3EventStore:
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        endpoint_url: str | None = None,
        region_name: str = "us-east-1",
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region_name,
        )

    def write(self, record: Dict[str, Any], event_type: str = "event") -> None:
        timestamp = datetime.now(timezone.utc)
        payload = dict(record)
        payload.setdefault("event_type", event_type)
        payload.setdefault("recorded_at", timestamp.isoformat())
        key_parts = [
            self.prefix,
            event_type,
            timestamp.strftime("%Y/%m/%d/%H"),
            f"{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:12]}.json",
        ]
        key = "/".join(part for part in key_parts if part)
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
            ContentType="application/json",
        )


class CompositeEventStore:
    def __init__(self, stores: Iterable[EventStore]) -> None:
        self.stores = list(stores)

    def write(self, record: Dict[str, Any], event_type: str = "event") -> None:
        for store in self.stores:
            store.write(record, event_type=event_type)
