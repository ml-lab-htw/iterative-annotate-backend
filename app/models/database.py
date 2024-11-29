from operator import truediv

from peewee import Model, CharField, ForeignKeyField, DateTimeField, TextField, IntegerField, FloatField, BooleanField
from app.utils.database import db
from datetime import datetime
from app.enums.model_enum import BaseModelEnum
from app.enums.snapshot_enum import SnapshotStatusEnum
from app.enums.status_enum import BundleStatusEnum


class BaseModel(Model):
    class Meta:
        database = db


def create_tables():
    with db:
        db.create_tables([ProjectORM, ImageBundleORM, ModelSnapshotORM, SnapshotBundleLink, ImageEntryORM, AnnotationORM], safe=True)


class ProjectORM(BaseModel):
    """
    ORM model for projects.

    Attributes:
        name (CharField): The name of the project, up to 150 characters.
        description (TextField): An optional description of the project.
        base_model (IntegerField): The base model used for the project, with choices defined in BaseModelEnum.
        created_at (DateTimeField): The date and time when the project was created, defaults to the current time.
    """
    name = CharField(max_length=150)
    description = TextField(null=True)
    base_model = IntegerField(choices=BaseModelEnum, default=BaseModelEnum.SSD)
    created_at = DateTimeField(default=datetime.now)

class ImageBundleORM(BaseModel):
    """
    ORM model for image bundles.

    Attributes:
        project (ForeignKeyField): A reference to the associated ProjectORM.
        status (IntegerField): The status of the image bundle, with choices defined in BundleStatusEnum.
        uploaded_at (DateTimeField): The date and time when the image bundle was uploaded, defaults to the current time.
    """
    project = ForeignKeyField(ProjectORM, backref='bundles')
    status = IntegerField(choices=BundleStatusEnum, default=BundleStatusEnum.CREATED)
    uploaded_at = DateTimeField(default=datetime.now)

class ModelSnapshotORM(BaseModel):
    """
    ORM model for model snapshots.

    Attributes:
        project (ForeignKeyField): A reference to the associated ProjectORM.
        name (CharField): An optional name of the snapshot, up to 150 characters.
        created_at (DateTimeField): The date and time when the snapshot was created, defaults to the current time.
        path (CharField): The file system path where the snapshot is stored.
        is_selected (BooleanField): A flag indicating whether the snapshot is selected.
        state (IntegerField): The state of the snapshot, with choices defined in SnapshotStatusEnum.

    Methods:
        deselect_all: Class method to deselect all snapshots for a given project.
    """
    project = ForeignKeyField(ProjectORM, backref='snapshots')
    name = CharField(max_length=150, null=True)
    created_at = DateTimeField(default=datetime.now)
    path = CharField(default="")  # Path to the stored model file
    is_selected = BooleanField(default=False)
    state = IntegerField(choices=SnapshotStatusEnum, default=SnapshotStatusEnum.CREATED)
    loss = TextField()
    batch_size = IntegerField(null=True)
    learning_rate = FloatField(null=True)

    @classmethod
    def deselect_all(cls, project):
        cls.update(is_selected=False).where(cls.project == project) .execute()

class SnapshotBundleLink(BaseModel):
    """
    ORM model for the link between snapshots and image bundles.

    Attributes:
        snapshot (ForeignKeyField): A reference to the associated ModelSnapshotORM.
        bundle (ForeignKeyField): A reference to the associated ImageBundleORM.

    Meta:
        indexes: A tuple defining a unique composite index on snapshot and bundle.
    """
    snapshot = ForeignKeyField(ModelSnapshotORM, backref='bundle_link')
    bundle = ForeignKeyField(ImageBundleORM, backref='snapshot_link')
    class Meta:
        indexes = (
            (('snapshot', 'bundle'), True),     # only allow any index once per row
        )

class ImageEntryORM(BaseModel):
    """
    ORM model for image entries.

    Attributes:
        filename (CharField): The name of the file.
        path (CharField): The file system path to the image.
        bundle (ForeignKeyField): A reference to the associated ImageBundleORM.
        is_corrected (BooleanField): A flag indicating whether the image was reviewed and corrected by the user.
    """
    filename = CharField()
    path = CharField()
    bundle = ForeignKeyField(ImageBundleORM, backref='images')
    is_corrected = BooleanField(default=False)      # Tells if the image was reviewed and corrected by the user

class AnnotationORM(BaseModel):
    """
    ORM model for annotations.

    Attributes:
        image (ForeignKeyField): A reference to the associated ImageEntryORM.
        x (IntegerField): The X coordinate of the annotation.
        y (IntegerField): The Y coordinate of the annotation.
        width (IntegerField): The width of the annotation.
        height (IntegerField): The height of the annotation.
        label (CharField): The label of the annotation.
        label_id (IntegerField): The identifier of the label.
        score (FloatField): The confidence score of the annotation.
        from_inference (BooleanField): A flag indicating if the annotation was initially predicted by the model.
        original_state (BooleanField): A flag indicating if the annotation was later modified by the user.
        active (BooleanField): A flag indicating if the annotation is active (not deleted).
    """
    image = ForeignKeyField(ImageEntryORM, backref='annotations')
    x = IntegerField()
    y = IntegerField()
    width = IntegerField()
    height = IntegerField()
    label = CharField()
    label_id = IntegerField()
    score = FloatField()
    from_inference = BooleanField(default=True)     # Tells if the Annotation was initially predicted by the model or manually added by a user
    original_state = BooleanField(default=True)     # Tells if the Annotation was later modified by the user
    active = BooleanField(default=True)             # Tells if the Annotation is active (not deleted)
