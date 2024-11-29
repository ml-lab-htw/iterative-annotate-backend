
import enum

class SnapshotStatusEnum(enum.IntEnum):
    CREATED = 1
    STARTED = 2
    COMPLETED = 3
    ABORTED = -1

class SnapshotStatusString(str, enum.Enum):
    CREATED = "Snapshot created"
    STARTED = "Fine-tuning running"
    COMPLETED = "Fine-tuning completed"
    ABORTED = "Fine-tuning aborted"


class SnapshotStatusMap:
    @staticmethod
    def get_string_from_index(index: int) -> str:
        try:
            # Find the corresponding enum name for the given index
            enum_name = SnapshotStatusEnum(index).name
            # Return the string representation from the string enum
            return SnapshotStatusString[enum_name].value
        except (ValueError, KeyError):
            raise ValueError(f"Invalid enum index: {index}")


    @staticmethod
    def get_enum_from_string(representation: str) -> SnapshotStatusEnum:
        try:
            # Find the corresponding enum name for the given string representation
            enum_name = SnapshotStatusString(representation).name
            # Return the index value from the integer enum
            return SnapshotStatusEnum[enum_name]
        except (ValueError, KeyError):
            raise ValueError(f"Invalid enum representation: {representation}")

