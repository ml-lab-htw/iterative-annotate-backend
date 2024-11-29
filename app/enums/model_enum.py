import enum


class BaseModelEnum(enum.IntEnum):
    SSD = 1
    FasterRCNN = 2
    YOLO = 3


class BaseModelString(str, enum.Enum):
    SSD = "Single Shot Detection"
    FasterRCNN = "Faster R-CNN"
    YOLO = "YOLO CNN"


class BaseModelMap:
    @staticmethod
    def get_string_from_index(index: int) -> str:
        try:
            # Find the corresponding enum name for the given index
            enum_name = BaseModelEnum(index).name
            # Return the string representation from the string enum
            return BaseModelString[enum_name].value
        except (ValueError, KeyError):
            raise ValueError(f"Invalid enum index: {index}")


    @staticmethod
    def get_enum_from_string(representation: str) -> BaseModelEnum:
        try:
            # Find the corresponding enum name for the given string representation
            enum_name = BaseModelString(representation).name
            # Return the index value from the integer enum
            return BaseModelEnum[enum_name]
        except (ValueError, KeyError):
            raise ValueError(f"Invalid enum representation: {representation}")

