from dataclasses import dataclass, asdict
from datetime import datetime, timezone

@dataclass(frozen=True)
class DatasetProvenance:
    platform: str
    dataset_id: str
    revision: str | None
    license: str | None
    source_url: str
    processed_at: str
    target: str
    policy: str

    def to_dict(self):
        return asdict(self)

def now_iso():
    return datetime.now(timezone.utc).isoformat()
