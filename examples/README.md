# **Examples for DeGirum PySDK with Hailo Hardware**  

This folder contains multiple Jupyter Notebook examples demonstrating various AI inference use cases using DeGirum PySDK and Hailo hardware. Each notebook showcases different functionalities, from basic inference to advanced AI pipelines.  

## **Getting Started**  

To run these examples, ensure that you have:  
1. Installed the required dependencies from `requirements.txt`.  
2. Set up your **virtual environment** and added it to Jupyter (as explained in the main [README](../README.md#2-create-a-virtual-environment)).  
3. Started Jupyter Notebook:  

   ```bash
   jupyter notebook
   ```

Then, navigate to the `examples` folder in Jupyter and open any of the notebooks.

---

## **List of Examples**  

Each notebook focuses on a specific AI application:

| Notebook | Description |
|----------|------------|
| **001_quick_start.ipynb** | A simple introduction to running inference using DeGirum PySDK. |
| **002_yolov8.ipynb** | Running different YOLOv8 models on images. |
| **003_face_detection.ipynb** | Detecting faces and keypoints in images. |
| **004_rtsp.ipynb** | Performing AI inference on an RTSP video stream. |
| **005_object_tracking.ipynb** | Tracking objects across video frames. |
| **006_multi_threading.ipynb** | Running multiple inference tasks in parallel with multi-threading. |
| **007_model_pipelining.ipynb** | Creating AI pipelines to process multiple models sequentially. |
| **008_object_detection_class_filtering.ipynb** | Filtering detected objects by class labels. |
| **009_zone_counting.ipynb** | Counting objects in specific zones of an image or video. |
| **010_emotion_recognition.ipynb** | Emotion recognition on images. |
| **011_obb_detection.ipynb** | Oriented bounding box (OBB) detection in images. |
| **012_blur_objects.ipynb** | Blur detected objects in images. |
| **013_overlay_effects.ipynb** | Overlay formatting on images. |
| **015_light_enhancement_with_custom_postprocessor.ipynb** | Low light enhancement on images. |
| **016_custom_video_source.ipynb** | Custom video generator. |
| **017_advanced_tiling_strategies.ipynb** | Tiling strategies on images. |
| **018_eager_batching.ipynb** | Impact of batching on single & multi-context models. |
| **019_warmup_inference.ipynb** | Running warm-up inference to reduce initial latency. |




---

## **How to Modify Inference Settings**  

Each notebook is **pre-configured** with default inference settings, including:  
- **Inference host address** (`@local`, `localhost`, `@cloud`)  
- **Model zoo location** (`degirum/hailo`, `../models`)  
- **Target hardware** (`HAILORT/HAILO8L`, `HAILORT/HAILO8`)  

These values can be modified inside the notebooks if needed.

---

## **Need Help?**  

- Refer to the [main README](../README.md) for installation and setup instructions.  
- Check the [DeGirum PySDK documentation](https://docs.degirum.com) for more details on AI inference.  
- For troubleshooting, feel free to [open an issue](https://github.com/DeGirum/hailo_examples/issues) in the repository or report on [Hailo Community](https://community.hailo.ai/).

---

