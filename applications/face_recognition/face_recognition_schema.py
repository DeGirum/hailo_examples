from lancedb.pydantic import LanceModel, Vector
from datetime import datetime
import numpy as np, uuid
from typing import List

# Define the Lance schema for face recognition
class FaceRecognitionSchema(LanceModel):
    id: str  # Unique identifier for each entry
    vector: Vector(512)  # Face embeddings, fixed size of 512
    image_path: str = "image_path"  # Default image path
    entity_name: str = "default"  # Default entity name
    bbox: Vector(4)  # Bounding box with 4 dimensions (x, y, width, height)
    source: int = 0  # Source , default is 0
    timestamp: datetime = datetime.now()  # Timestamp of the entry creation

    @classmethod
    def format_data(cls, result) -> List['FaceRecognitionSchema']:
        """Converts the result to a FaceRecognitionSchema instance.

        Args:
            result: A list of results containing embeddings and bounding box data.
            image_path: The path to the image associated with the entries.
            entity_name: Optional name for the entity; defaults to None.

        Returns:
            A list of FaceRecognitionSchema instances.
        """
        image_path, entity_name = result.info["image_path"], result.info["entity_name"]

        data = [
            cls(
                id=str(uuid.uuid4()),  # Generate a unique ID for each entry
                vector=np.array(res["embedding"], dtype=np.float32),  # Convert embedding to a NumPy array with float32 dtype
                image_path=image_path,  # Set the image path
                entity_name=entity_name,  # Set the entity name, or use the default
                bbox=np.array(res["bbox"], dtype=np.float32)  # Convert bounding box to a NumPy array with float32 dtype
            )
            for res in result.results 
            if "embedding" in res
        ]
        return data
