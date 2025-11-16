import json
import numpy as np


class PostProcessor:
    """YOLO Postprocessor for DeGirum PySDK."""

    def __init__(self, json_config):
        """
        Initialize the YOLO postprocessor with configuration settings.

        Parameters:
            json_config (str): JSON string containing post-processing configuration.
        """
        config = json.loads(json_config)

        # Extract input image dimensions
        pre_process = config["PRE_PROCESS"][0]
        self.image_width = pre_process.get("InputW", 416)
        self.image_height = pre_process.get("InputH", 416)

        # Extract post-process configurations
        post_process = config.get("POST_PROCESS", [{}])[0]
        self.strides = np.array(post_process.get("Strides", [16, 32]), dtype=np.float32)
        self.anchors = np.array(post_process.get("Anchors", [])).reshape(-1, 2)
        self.conf_threshold = post_process.get("OutputConfThreshold", 0.3)
        self.nms_iou_thresh = post_process.get("OutputNMSThreshold", 0.6)
        self.num_classes = post_process.get("NumClasses", 80)

        # Load label dictionary
        label_path = post_process.get("LabelsPath", None)
        if label_path is None:
            raise ValueError("LabelsPath is required in POST_PROCESS configuration.")
        with open(label_path, "r") as json_file:
            self._label_dictionary = json.load(json_file)

    def forward(self, tensor_list, details_list):
        """
        Perform postprocessing on raw model outputs.

        Parameters:
            tensor_list (list): List of tensors from the model.
            details_list (list): Additional metadata for the tensors.

        Returns:
            str: JSON string containing processed inference results.
        """
        # Step 1: Dequantize tensors
        dequantized_tensors = []
        for data, tensor_info in zip(tensor_list, details_list):
            scale, zero_point = tensor_info["quantization"]
            dequantized_data = (data.astype(np.float32) - zero_point) * scale
            dequantized_tensors.append(dequantized_data)

        # Step 2: Decode detections
        detections = self.decode(dequantized_tensors)

        # Step 3: Format results with labels
        results = []
        for detection in detections:
            class_id = int(detection["class"])
            label = self._label_dictionary.get(str(class_id), f"class_{class_id}")
            # Scale the bounding box to the input image size
            scaled_bbox = [
                detection["box"][0] * self.image_width,  # x1
                detection["box"][1] * self.image_height,  # y1
                detection["box"][2] * self.image_width,  # x2
                detection["box"][3] * self.image_height,  # y2
            ]
            results.append(
                {
                    "bbox": scaled_bbox,
                    "category_id": class_id,
                    "label": label,
                    "score": float(detection["score"]),
                }
            )

        return results

    def decode(self, outputs):
        """Decode YOLO outputs into bounding boxes, classes, and scores."""
        decoded_detections = []
        num_anchors_per_scale = len(self.anchors) // len(self.strides)

        for scale_index, (stride, output) in enumerate(zip(self.strides, outputs)):
            grid_h, grid_w = int(self.image_height // stride), int(
                self.image_width // stride
            )
            grid_x, grid_y = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
            grid = (
                np.stack((grid_x, grid_y), axis=-1).reshape(-1, 1, 2).astype(np.float32)
            )

            output = output.reshape(grid_h * grid_w, num_anchors_per_scale, -1)
            scale_anchors = self.anchors[
                scale_index
                * num_anchors_per_scale : (scale_index + 1)
                * num_anchors_per_scale
            ]
            scale_anchors = scale_anchors / [self.image_width, self.image_height]

            box_xy = self._sigmoid(output[..., :2])
            box_xy = (box_xy + grid) * stride / [self.image_width, self.image_height]
            box_wh = np.exp(output[..., 2:4]) * scale_anchors

            box_x1y1 = box_xy - (box_wh / 2)
            box_x2y2 = box_xy + (box_wh / 2)
            boxes = np.concatenate([box_x1y1, box_x2y2], axis=-1)
            boxes = np.clip(boxes, 0, 1)

            objectness = self._sigmoid(output[..., 4])
            class_probs = self._sigmoid(output[..., 5:])
            scores = objectness * np.max(class_probs, axis=-1)
            valid_indices = scores > self.conf_threshold

            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
            classes = np.argmax(class_probs[valid_indices], axis=-1)

            for box, score, cls in zip(boxes, scores, classes):
                decoded_detections.append({"box": box, "score": score, "class": cls})

        return self._apply_nms(decoded_detections)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _apply_nms(self, detections):
        """Apply Non-Maximum Suppression to filter redundant detections."""
        if not detections:
            return []

        boxes = np.array([det["box"] for det in detections])
        scores = np.array([det["score"] for det in detections])
        classes = np.array([det["class"] for det in detections])

        final_detections = []
        for cls in np.unique(classes):
            cls_indices = np.where(classes == cls)[0]
            cls_boxes = boxes[cls_indices]
            cls_scores = scores[cls_indices]

            order = cls_scores.argsort()[::-1]
            while len(order) > 0:
                i = order[0]
                final_detections.append(detections[cls_indices[i]])
                if len(order) == 1:
                    break
                xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[order[1:], 0])
                yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[order[1:], 1])
                xx2 = np.minimum(cls_boxes[i, 2], cls_boxes[order[1:], 2])
                yy2 = np.minimum(cls_boxes[i, 3], cls_boxes[order[1:], 3])
                inter_area = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
                union_area = (
                    (cls_boxes[i, 2] - cls_boxes[i, 0])
                    * (cls_boxes[i, 3] - cls_boxes[i, 1])
                    + (cls_boxes[order[1:], 2] - cls_boxes[order[1:], 0])
                    * (cls_boxes[order[1:], 3] - cls_boxes[order[1:], 1])
                    - inter_area
                )
                iou = inter_area / union_area
                order = order[np.where(iou <= self.nms_iou_thresh)[0] + 1]

        return final_detections
