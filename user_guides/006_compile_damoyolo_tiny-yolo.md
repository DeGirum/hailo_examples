# Compile & Run DamoYOLO & Tiny‑YOLOv4 on Hailo with DeGirum PySDK

This guide is for ML developers who want to learn how to configure DeGirum PySDK to load a precompiled object detection model and run inference on a Hailo device. Similar to our last user guide, we’ll focus on the essential details and steps involved instead of a “just run this magic two line script” approach. We encourage the readers to start with User Guide 1–3 before proceeding with this guide. By the end of this guide, you’ll have a solid understanding of how to select and supply the correct end-node names (i.e., final convolutional layer names) for DamoYOLO and Tiny-YOLOv4 models.

Model compilation steps, including calibration and general compilation, are covered in [User Guide 1](https://community.degirum.com/t/hailo-guide-1-hailo-world-running-your-first-inference-on-a-hailo-device-using-degirum-pysdk/144), [User Guide 2](https://community.degirum.com/t/hailo-guide-2-running-your-first-object-detection-model-on-a-hailo-device-using-degirum-pysdk/145) and [User Guide 3](https://community.degirum.com/t/hailo-guide-3-simplifying-object-detection-on-a-hailo-device-using-degirum-pysdk/146). This document focuses specifically on choosing the final convolutional layer names.

---

## Why End-Node Names Matter

When you translate an ONNX or TFLite object detection model to Hailo HEF, the compiler needs to know which graph outputs correspond to the **final convolutional output tensors** that feed into the detection post-processor.
DeGirum PySDK relies on these outputs for NMS and bounding-box decoding.

Supplying the wrong end-node names will lead to missing or malformed detection results at runtime.

---

## Models Covered

> **Tip:** Open your exported ONNX/TFLite model in [Netron](https://netron.app), filter for the final `Conv` operations, and locate the last one or two nodes near the graph outputs. Provide those exact `Conv` layer names to `--end-node-names` for compilation.

---

## Supplying End‑Node Names to the Compiler

Whether you use the provided `hailo_compile_simple.py` script or your own compilation workflow, the key is to supply your identified convolutional output layer names via `--end-node-names` . The calibration file `calibration_data.npy` is assumed to be in the same directory, so you can omit its explicit path if it resides alongside your compile script.

```bash
python hailo_compile_simple.py \
  --model-path models/tiny_yolov4.onnx \
  --end-node-names Conv_263 Conv_264 \
  --calibration-npy-path calibration_data.npy \
  --device-type hailo8 \
  --output-path compiled_tiny_yolov4.hef
```

For a DamoYOLO variant:

```bash
python hailo_compile_simple.py \
  --model-path models/damoyolo_tinynasL20_S.onnx \
  --end-node-names Conv_465 Conv_469 \
  --calibration-npy-path calibration_data.npy \
  --device-type hailo8 \
  --output-path compiled_damoyolo.hef
```

---

### Bare-Minimum Compile Script

Below is a minimal Python script using `ClientRunner` that demonstrates how end‑node names and other essential flags are passed to Hailo's compiler. Skip other optional parameters as needed.

```python
import argparse
from hailo_sdk_client import ClientRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compile ONNX/TFLite to Hailo HEF')
    parser.add_argument('--device-type', choices=[
        'hailo8','hailo8r','hailo8l','hailo15h','hailo15m','hailo15l','hailo10h'
    ], default='hailo8', help='Target Hailo device')
    parser.add_argument('--model-path', required=True,
                        help='Path to .onnx or .tflite model')
    parser.add_argument('--output-path', default=None,
                        help='Output .hef filename')
    parser.add_argument('--end-node-names', nargs='+', required=True,
                        help='List of final convolutional layer names')
    parser.add_argument('--calibration-npy-path', default='calibration_data.npy',
                        help='Calibration .npy file')
    parser.add_argument('--optimization-level', type=int, default=0,
                        help='Model optimization level')
    parser.add_argument('--compression-level', type=int, default=0,
                        help='Compression level')
    parser.add_argument('--compiler-optimization-level', type=int, default=2,
                        help='Compiler optimization level (max=2)')

    args = parser.parse_args()

    runner = ClientRunner(hw_arch=args.device_type)
    ext = args.model_path.rsplit('.', 1)[-1].lower()
    model_name = args.model_path.split('/')[-1].split('.')[0]
    args.output_path = args.output_path or args.model_path.replace(f'.{ext}', '.hef')

    # Translate model
    if ext == 'onnx':
        runner.translate_onnx_model(
            model=args.model_path,
            net_name=model_name,
            end_node_names=args.end_node_names
        )
    else:
        runner.translate_tf_model(
            model_path=args.model_path,
            net_name=model_name,
            end_node_names=args.end_node_names
        )

    # Apply normalization and optimization ALLS lines
    alls_lines = [
        "normalization_in = normalization([0.0,0.0,0.0],[255.0,255.0,255.0])\n",
        f"model_optimization_flavor(optimization_level={args.optimization_level}, compression_level={args.compression_level}, batch_size=32)\n",
        f"performance_param(compiler_optimization_level={args.compiler_optimization_level})\n",
    ]
    runner.load_model_script(''.join(alls_lines))
    runner.optimize_full_precision(args.calibration_npy_path)
    runner.optimize(args.calibration_npy_path)
    hef_bytes = runner.compile()

    with open(args.output_path, 'wb') as f:
        f.write(hef_bytes)
```

---

## Summary

* Locate the final `Conv_…` layers in Netron.
* Supply those exact names to `--end-node-names` .
* Ensure the order matches your model’s detection heads.
* Compile, then verify outputs in PySDK.

With accurate end‑node names, DamoYOLO and Tiny‑YOLOv4 models will integrate seamlessly with Hailo devices and DeGirum PySDK.

---

## Python Post-Processor Implementations

Degirum provides ready-to-use Python post-processors for common detection models on GitHub:

* **YOLO family:** `HailoDetectorYOLOV3V4V5.py`
* **DamoYOLO variants:** `HailoDetectorDamoYOLO.py`

You can browse and download these from our GitHub repository:

[https://github.com/DeGirum/hailo_examples/tree/main/postprocessors](https://github.com/DeGirum/degirum-pysdk-postprocessors)

Once downloaded, reference either file in your model JSON under `PythonFile` .

---

## Configuring the Model JSON File

Your model JSON tells PySDK how to preprocess inputs, point to the compiled HEF, and perform post‑processing. A minimal example:

```json
{
  "ConfigVersion": 10,
  "DEVICE": [{
    "DeviceType": "HAILO8",
    "RuntimeAgent": "HAILORT",
    "SupportedDeviceTypes": "HAILORT/HAILO8"
  }],
  "PRE_PROCESS": [{
    "InputType": "Image",
    "InputN": 1,
    "InputH": 640,
    "InputW": 640,
    "InputC": 3,
    "InputPadMethod": "letterbox",
    "InputResizeMethod": "bilinear",
    "InputQuantEn": true
  }],
  "MODEL_PARAMETERS": [{
    "ModelPath": "your_model.hef"
  }],
  "POST_PROCESS": [{
    "OutputPostprocessType": "Detection",
    "PythonFile": "YourPostProcessor.py",
    "OutputNumClasses": 80,
    "LabelsPath": "labels.json",
    "OutputConfThreshold": 0.3
  }]
}
```

Adjust `InputH` /`InputW` to your model’s input size and point `ModelPath` at your `.hef` .

---

## Preparing the Model Zoo

To set up your model zoo, collect all assets in a single directory (or subdirectories) so PySDK can discover them automatically:

1. Create a top‑level folder, for example:

```bash
model_zoo/
```

2. For each model, add the following files:
  * **Compiled HEF:** `your_model.hef`
  * **JSON config:** `your_model.json`
  * **Post‑processor script:** `YourPostProcessor.py` (e.g. `HailoDetectorYOLOV3V4V5.py` or `HailoDetectorDamoYOLO.py` )
  * **Labels file:** `labels.json` (e.g. `labels_coco.json` )Your directory might look like:

```bash
model_zoo/
├── tiny-yolov4/
│   ├── tiny-yolov4.json
│   ├── tiny-yolov4.hef
│   ├── HailoDetectorYOLOV3V4V5.py
│   └── labels_coco.json
└── damoyolo_tinynasL20_S/
    ├── damoyolo_tinynasL20_S.json
    ├── damoyolo_tinynasL20_S.hef
    ├── HailoDetectorDamoYOLO.py
    └── labels_coco.json
```

3. (Optional) Use subdirectories per model for clarity and versioning.

When you call:

```python
model = dg.load_model(
  model_name="your_model",
  inference_host_address="@local",
  zoo_url="./model_zoo"
)
```

PySDK will recursively scan all subdirectories, locate each JSON config, and register the models for inference.

---

## Running Inference

In your Python code, simply load and run the model:

```python
import degirum as dg

model = dg.load_model(
    model_name="your_model",
    inference_host_address="@local",
    zoo_url="./model_zoo"
)

results = model("path/to/test_image.jpg")
print(results.results)
```

`results.results` is a list of dictionaries with `bbox` , `score` , `category_id` , and `label` .

---

## Visualizing the Output

PySDK provides an `image_overlay` attribute to draw boxes and labels:

```python
import cv2

cv2.imshow("Detections", results.image_overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

This shows an image with bounding boxes and class labels overlaid.

---

## Troubleshooting and Debug Tips

* **No detections**: Verify your `--end-node-names` match the final `Conv` layers.
* **Blank overlay**: Ensure `ModelPath` in JSON points to the correct `.hef` , and `LabelsPath` is valid.
* **Dimension mismatches**: Double‑check `InputH` /`InputW` against your model’s compiled input shape.
* **Incorrect classes**: Confirm your `labels.json` maps exactly to the class indices used in the model.

---

## Conclusion

You now know how to:

1. Identify final `Conv_…` layer names using Netron.
2. Supply those names to the Hailo compiler via `--end-node-names` .
3. Configure your PySDK JSON and organize a model zoo.
4. Run inference and visualize results with DeGirum PySDK.

With these steps, any exported DamoYOLO or Tiny‑YOLOv4 model can be compiled and integrated into your Hailo‑accelerated application.