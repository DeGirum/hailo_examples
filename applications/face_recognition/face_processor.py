import cv2
import numpy as np
from skimage import transform as trans
from degirum_tools.video_support import video_source, open_video_stream
from typing import Generator, List, Dict, Tuple, Optional
from utils import load_model

class FaceProcessor:
    def __init__(self, config:  Dict[str, dict], most_centered_only: bool = False):
        """
        Initialize the FaceProcessor.

        Args:
            config (dict): Configuration containing task-specific models and task settings.
            most_centered_only (bool, optional): Whether to select only the most centered face. Defaults to False.
            indexing (bool, optional): Whether the indexing is done for face reidentification.
        """

        self.config = config
        self.most_centered_only = most_centered_only
        self.hw_location = self.config.get('hw_location')
        self.models = {'face_det_kypts_model': load_model(self.config, 'face_det_kypts_model', hw_location=self.hw_location), #Face detection plus keypoints model
                       'face_reid': load_model(self.config, 'face_reid_model', self.hw_location)} #Face Recognition model

    def align_and_crop(self, img: np.ndarray, landmarks: List[np.ndarray], image_size: int = 112) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align and crop the face from the image based on the given landmarks.

        Args:
            img (np.ndarray): The full image (not the cropped bounding box).
            landmarks (List[np.ndarray]): List of 5 keypoints (landmarks) as (x, y) coordinates.
            image_size (int, optional): The size to which the image should be resized. Defaults to 112.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The aligned face image and the transformation matrix.
        """
        _arcface_ref_kps = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        assert len(landmarks) == 5
        assert image_size % 112 == 0 or image_size % 128 == 0

        if image_size % 112 == 0:
            ratio = float(image_size) / 112.0
            diff_x = 0
        else:
            ratio = float(image_size) / 128.0
            diff_x = 8.0 * ratio

        dst = _arcface_ref_kps * ratio
        dst[:, 0] += diff_x
        tform = trans.SimilarityTransform()
        tform.estimate(np.array(landmarks), dst)
        M = tform.params[0:2, :]

        aligned_img = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)

        return aligned_img, M
    
    def select_most_centered_face(self, results: List[Dict], image_size: Tuple[int, int]) -> Optional[Dict]:
        """Select the most centered face from the detection results."""
        image_center = (image_size[0] / 2, image_size[1] / 2)
        min_distance = float("inf")
        most_centered_result = None

        for res in results:
            nose_kp = res["landmarks"][2]["landmark"]
            dist = (image_center[0] - nose_kp[0]) ** 2 + (
                image_center[1] - nose_kp[1]
            ) ** 2
            if dist < min_distance:
                min_distance = dist
                most_centered_result = res

        return most_centered_result

    def generate_aligned_faces(self, face_results, image):  
        if face_results:
            for result in face_results:
                landmarks = [landmark["landmark"] for landmark in result["landmarks"]]
                aligned_face, _ = self.align_and_crop(image, landmarks)
                yield aligned_face # Yield aligned face
        
    def process_face_results(self, detection_results):
        """Processes the face results from aligned and cropped face for the required tasks"""
        if not detection_results.results:
            return detection_results  # Return the detection results as is if no faces are detected

        h, w, _ = detection_results.image.shape
        assert h > 0 and w > 0, "Image dimensions are invalid, height and width must be greater than 0"
        face_results = detection_results.results

        #Using the Generator over the aligned faces
        if self.most_centered_only and face_results:
            face_results = [self.select_most_centered_face(face_results, (w, h))]

        aligned_faces = []
        for aligned_face in self.generate_aligned_faces(face_results, detection_results.image):
            aligned_faces.append(aligned_face)

        # Run batch predict on aligned faces and extract embeddings
        embeddings = self.models['face_reid'].predict_batch(aligned_faces)
        for face, face_embedding in zip(face_results, embeddings):   
            # Extract embedding from the result
            embedding = face_embedding.results[0]["data"]
            embedding = embedding.squeeze() if isinstance(embedding, np.ndarray) else embedding[0]
            face["embedding"] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

        return detection_results

    def predict_batch(self, image_generator: Generator):
        """Process a batch of images from the given generator and yield results."""
        for result in self.models["face_det_kypts_model"].predict_batch(image_generator):
            self.process_face_results(result)
            yield result

    def predict_stream(self, video_source_id: str, fps: Optional[int] = None):
        """Run a model on a video stream"""
        with open_video_stream(video_source_id) as stream:
            for result in self.models["face_det_kypts_model"].predict_batch(
                video_source(stream, fps=fps)
            ):  
                self.process_face_results(result)
                yield result

    def predict(self, image_input: str):
        """Process a single image from the given input and yield the results."""
        result = self.models["face_det_kypts_model"](image_input)
        self.process_face_results(result)
        return result

    def __call__(self, image_input: str):
        """Make the object callable and default to using the predict method."""
        return self.predict(image_input)
