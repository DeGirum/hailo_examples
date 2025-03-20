import numpy as np
import lancedb
import cv2
import argparse
from degirum_tools.video_support import get_video_stream_properties
from degirum_tools.ui_support import Display
from face_recognition_schema import FaceRecognitionSchema
from face_processor import FaceProcessor
from utils import load_config, check_input_type
from typing import Dict, Optional


class FaceIdentifier:
    """
    This class recognizes the faces in the input video source with the indexed database.
    """
    def __init__(self, config: Dict[str, dict]):
        self.config: Dict[str, dict] = load_config(config)

        # Initialize the face processing pipeline
        self.face_processor: FaceProcessor = FaceProcessor(self.config)

    def load_database_for_reid(self):

        uri: str = self.config.get("database", {}).get("name", "default_database")
        table_name: str = self.config.get("table", {}).get("name", "default_table")
        search_params: Dict[str, dict] = self.config.get("search_params", {})

        # Connect to the LanceDB database
        db = lancedb.connect(uri=uri)

        # Check if the table exists, create if not
        if table_name in db.table_names():
            tbl = db.open_table(table_name)
            schema_fields = [field.name for field in tbl.schema]
            if schema_fields != list(FaceRecognitionSchema.model_fields.keys()):
                raise RuntimeError(f"Table {table_name} has a different schema.")

        return tbl, search_params
    
    def process_face_result(self, result: any, threshold: float):
        """Process the face result: perform database search, calculate distance, and assign label."""
        
        for i, res in enumerate(result.results):
            if "embedding" in res:
                tbl, search_params = self.load_database_for_reid()

                top_k: int = search_params.get("top_k", 1)
                field_name: str = search_params.get("field_name", "vector")
                metric_type: str = search_params.get("metric_type", "cosine")

                search_result = (
                    tbl.search(
                        np.array(res["embedding"]).astype(np.float32),
                        vector_column_name=field_name,
                    )
                    .metric(metric_type)
                    .limit(top_k)
                    .to_list()
                )
                # Assert the result from the database search is valid
                assert len(search_result) > 0
                assert "_distance" in search_result[0]

                # Calculate distance and assign label
                distance = round(1 - search_result[0]["_distance"], 2)
                
                entity_name = search_result[0]["entity_name"]
                res["label"] = entity_name if distance >= threshold else "Unknown"
                res["score"] = distance

            else:
                res["bbox"] = (0, 0, 0, 0)  # Default or empty bounding box
                continue  # Skip to next result

            result.results[i].pop("landmarks", None)
            result.results[i].pop("embeddings", None)

    def identify_faces(
        self,
        input_source: str,
        fps: Optional[int],
        threshold: float,
        display: bool,
        save_output: bool,
        output_path: str
    ) -> None:
        """
        Identify faces in the video source.
        
        Arguments:
            input_source (str): Video source id/image source - identifier of input video stream/input image.
            threshold (float): Cosine distance threshold.
            fps (Optional[int]): Frames per second (Optional to set custom FPS).
            display (bool): Enable annotated video display.
            save_output (bool): Enable saving output video.
            output_path (str): Path to save the annotated video/annotated image.
        """
        input_source_type = check_input_type(input_source)
        if display:
            win_name = f"Annotating {input_source}"
            display_instance = Display(win_name)
        if input_source_type == "video":
            if input_source.isdigit():
                input_source = int(input_source)

            w, h, video_fps = get_video_stream_properties(input_source)

            if fps:
                video_fps = fps

            if save_output:
                fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
                writer = cv2.VideoWriter(output_path + ".mp4", fourcc, video_fps, (w, h))

            for result in self.face_processor.predict_stream(input_source, fps=video_fps):
                
                self.process_face_result(result, threshold)
                image = result.image_overlay
                display_instance.show(image)

                if save_output:
                    writer.write(image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        elif input_source_type == "image":
            result = self.face_processor.predict(input_source)
            self.process_face_result(result, threshold)
            image = result.image_overlay
            display_instance.show(image)
            
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_output:
                cv2.imwrite(output_path + ".jpg", image)
                    
        else:
            print ("Incorrect input data source")

def main():

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Identify faces in the input video or image source.")

    # Add arguments for each option in the Click version
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file with Model and Database indexing parameters.",
    )
    
    parser.add_argument(
        "--input_source",
        type=str,
        default="./assets/test_short.mp4",
        help="Video source ID or Image file path for the input stream.",
    )

    parser.add_argument(
        "--display",
        type=bool,
        default=True,
        help="Enable visual display.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="annotated_output",
        help="Path to save the annotated video/annotated image",
    )

    parser.add_argument(
        "--save_output",
        action="store_true",
        help="Enable saving output video.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Cosine distance threshold",
    )

    parser.add_argument(
        "--fps",
        type=Optional[int],
        default=None,
        help="FPS (Frames per second)",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the FaceIdentifier with the provided arguments
    FaceIdentifier(args.config).identify_faces(
        args.input_source,
        args.fps,
        args.threshold,
        args.display,
        args.save_output,
        args.output_path,
    )

if __name__ == "__main__":
    main()