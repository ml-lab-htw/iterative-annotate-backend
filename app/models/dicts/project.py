from dataclasses import dataclass
from app.models.database import ProjectORM

@dataclass
class ProjectDataDict:
    orm: ProjectORM
    bundle_cnt: int
    image_sum: int
    snapshot_cnt: int
