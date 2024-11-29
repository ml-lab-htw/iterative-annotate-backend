import enum

class BundleStatusEnum(enum.IntEnum):
    CREATED = 0
    UPLOADED = 1
    ANNOTATED = 2
    REVIEWED = 3


class BundleStatusString(str, enum.Enum):
    CREATED = "CREATED"
    UPLOADED = "UPLOADED"
    ANNOTATED = "ANNOTATED"
    REVIEWED = "REVIEWED"


class BundleStatusMap:
    @staticmethod
    def get_string_from_index(index: int) -> str:
        try:
            # Find the corresponding enum name for the given index
            enum_name = BundleStatusEnum(index).name
            # Return the string representation from the string enum
            return BundleStatusString[enum_name].value
        except (ValueError, KeyError):
            raise ValueError(f"Invalid enum index: {index}")


    @staticmethod
    def get_enum_from_string(representation: str) -> BundleStatusEnum:
        try:
            # Find the corresponding enum name for the given string representation
            enum_name = BundleStatusString(representation).name
            # Return the index value from the integer enum
            return BundleStatusEnum[enum_name]
        except (ValueError, KeyError):
            raise ValueError(f"Invalid enum representation: {representation}")

