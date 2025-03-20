import numpy as np
import lancedb
import cv2
import argparse
from degirum_tools.video_support import get_video_stream_properties
from degirum_tools.ui_support import Display
from face_processor import FaceProcessor
from utils import load_config, check_input_type
from typing import Dict, Optional


class FacialAttributeAnalyzer:
    """
    This class recognizes the faces in the input video source with the indexed database.
    """
    def __init__(self, config: Dict[str, dict]):
        self.config: Dict[str, dict] = load_config(config)

        # Initialize the face processing pipeline
        self.face_processor: FaceProcessor = FaceProcessor(self.config)

    def process_face_result(self, result: any):
        """Process the face result: perform facial attribute analysis"""
        
        for i, res in enumerate(result.results):
            label_str = ""
            if "extra_results" in res:
                if "gender" in res["extra_results"]:
                    gender = res["extra_results"]["gender"]
                    label_str += gender

                if "age" in res["extra_results"]:
                    age = res["extra_results"]["age"]
                    label_str += str(age)   

                if "emotion" in res["extra_results"]:
                    emotion = res["extra_results"]["emotion"]
                    label_str += emotion

            else:
                res["bbox"] = (0, 0, 0, 0)  # Default or empty bounding box
                continue  # Skip to next result
            
            res["label"] = label_str
            result.results[i].pop("landmarks", None)
            result.results[i].pop("extra_results", None)
        return result

    def analyze(
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
                
                self.process_face_result(result)
                image = result.image_overlay
                display_instance.show(image)

                if save_output:
                    writer.write(image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        elif input_source_type == "image":
            result = self.face_processor.predict(input_source)
            self.process_face_result(result)
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
    FacialAttributeAnalyzer(args.config).analyze(
        args.input_source,
        args.fps,
        args.threshold,
        args.display,
        args.save_output,
        args.output_path,
    )

if __name__ == "__main__":
    main()