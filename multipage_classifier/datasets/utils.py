from enum import Enum
from pydantic import BaseModel

class Bucket(Enum):
    Training = "Training"
    Validation = "Validation"
    Testing = "Testing"