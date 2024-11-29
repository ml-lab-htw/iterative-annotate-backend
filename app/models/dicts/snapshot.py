from dataclasses import dataclass
from typing import List

from app.models.database import ModelSnapshotORM, ImageBundleORM

@dataclass
class SnapshotListDict:
    snapshot: ModelSnapshotORM
    bundles: List[ImageBundleORM]
