"""
JSON Utility Module

This module contains utility classes and functions for JSON operations,
including custom encoders for handling non-standard Python types.
"""
import json
from decimal import Decimal
from typing import Any


class DecimalEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle Decimal objects.
    """
    def default(self, obj: Any) -> Any:
        """
        Convert Decimal objects to float for JSON serialization.

        :param obj: Object to be serialized
        :return: Serializable representation of the object
        """
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)
