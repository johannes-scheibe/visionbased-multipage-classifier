from enum import Enum
from pydantic import BaseModel




class Bucket(Enum):
    Training = "training"
    Validation = "validation"
    Testing = "testing"