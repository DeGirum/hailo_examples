import argparse
import lancedb
from face_recognition_schema import FaceRecognitionSchema
from face_processor import FaceProcessor
from utils import image_generator, load_config
from typing import Dict
from pathlib import Path

class FaceDatabaseIndexer:
    """
    This class indexes the faces into a database.
    """

    def __init__(self, config: Dict[str, dict]):
        """
        Constructor.

        Args:
            config (dict): Configuration yaml file containing model and database parameters.
        """
        self.config: Dict[str, dict] = load_config(config)

        self.uri: str = self.config.get("database", {}).get("name", "default_database")
        self.table_name: str = self.config.get("table", {}).get("name", "default_table")
        self.db = lancedb.connect(uri=self.uri)

        
        # Initialize the face processing pipeline
        self.face_processor = FaceProcessor(
            self.config, most_centered_only=True
        )

    def index_faces(self, image_dir: str) -> None:
        """
        Index faces from images found in the directory.
        
        Arguments:
            image_dir (str): The path to the directory containing faces to be indexed.
        """
        # Check if the table exists, create if not
        if self.table_name not in self.db.table_names():
            tbl = self.db.create_table(self.table_name, schema=FaceRecognitionSchema)
        else:
            tbl = self.db.open_table(self.table_name)
            schema_fields = [field.name for field in tbl.schema]
            if schema_fields != list(FaceRecognitionSchema.model_fields.keys()):
                raise RuntimeError(f"Table {self.table_name} has a different schema.")

        num_entities = 0  # Count the number of entities indexed

        # Process images in batches
        for result in self.face_processor.predict_batch(image_generator(image_dir)):
            data = FaceRecognitionSchema.format_data(result)
            if len(data) > 0:
                tbl.add(data=data)
            num_entities += len(data)

        print(f"Successfully indexed {num_entities} faces in the {self.table_name} table")

    def index_single_face(self, image_path, identity_name: str):
        """
        Index a single face from an image.
        
        Arguments:
            image_path (str): The file path to a single image that will be indexed.
            identity_name (str): Identity name for a face.
        """
        # Check if the table exists, create if not
        if self.table_name not in self.db.table_names():
            tbl = self.db.create_table(self.table_name, schema=FaceRecognitionSchema)
        else:
            tbl = self.db.open_table(self.table_name)
            schema_fields = [field.name for field in tbl.schema]
            if schema_fields != list(FaceRecognitionSchema.model_fields.keys()):
                raise RuntimeError(f"Table {self.table_name} has a different schema.")

        num_entities = 0  # Count the number of entities indexed

        # Process images in batches
        for result in self.face_processor.predict_batch(image_generator(image_path, identity_name)):
            data = FaceRecognitionSchema.format_data(result)
            if len(data) > 0:
                tbl.add(data=data)
            num_entities += len(data)

        print(f"Successfully indexed {num_entities} faces in the {self.table_name} table")

def main():

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Index faces from an image source (directory/single image file) into a database.")
    
    # Add arguments for config file, image directory, and identity name
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file with Model and Database indexing parameters."
    )
    
    parser.add_argument(
        "--image_source",
        type=str,
        default="./assets/Friends_dataset",
        help="Path to the directory/folder of images or a single image."
    )
    
    parser.add_argument(
        "--identity_name",
        type=str,
        default="Unknown",
        help="Identity name."
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Print out the parsed arguments (for testing purposes)
    print(f"Config file: {args.config}")
    print(f"Image source: {args.image_source}")
    print(f"Identity name: {args.identity_name}")
    
    # Initialize the FaceDatabaseIndexer with the provided config
    try:
        indexer = FaceDatabaseIndexer(args.config)
        path = Path(args.image_source)
        if path.is_file():
            print ("Image source is a single image file")
            indexer.index_single_face(args.image_source, args.identity_name)
        elif path.is_dir():
            print ("Image source is a directory of images")
            indexer.index_faces(args.image_source)
        
    except Exception as e:
        print(f"Error during face indexing: {e}")

if __name__ == "__main__":
    main()