import json

import numpy as np
from fastapi import Response

def custom_encoder(obj):
    """
    Custom JSON encoder function that handles various types properly.

    Args:
        obj (any): The object to encode into JSON.

    Returns:
        int: If obj is a numpy integer.
        float: If obj is a numpy floating point.
        list: If obj is a numpy array.
        scalar: If obj is a numpy scalar.
        dict: If obj has a to_dict method.
        str: A string representation of obj for all other cases.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert numpy scalars to Python scalars
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()  # Handle custom objects with to_dict methodc
    return str(obj)  # Fallback to string for everything else

def json_response(object):
    """
    Encodes the given object into a JSON-formatted string and returns an HTTP response object with that JSON string as content.

    Args:
        object (any): The object to be encoded into JSON.

    Returns:
        Response: An HTTP response object containing the JSON-formatted string.
    """
    json_str = json.dumps(object, indent=4, default=custom_encoder)
    return Response(content=json_str, media_type='application/json')
