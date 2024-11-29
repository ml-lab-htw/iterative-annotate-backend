import random
import string


def generate_random_filename(length: int = 16) -> str:
    """
    Generates a random filename with the specified length.

    Args:
        length (int, optional): The length of the filename. Defaults to 16.

    Returns:
        str: A string representing the random filename.
    """
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))