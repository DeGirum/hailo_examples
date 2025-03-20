
# Face Recognition
Face recognition is a transformative application of computer vision that enables systems to identify or verify individuals based on their facial features. These systems operate through a pipeline of well-defined steps: detecting faces, aligning them for consistency, extracting unique facial features (embeddings), and comparing these features against a database of known individuals. This structured approach ensures accuracy, robustness, and scalability.

Modern face recognition systems rely heavily on deep learning models to perform these tasks. The pipeline begins with face detection, where specialized models identify faces and their key landmarks. These landmarks are used to align faces, ensuring that variations in pose, orientation, or size do not affect subsequent steps. The aligned faces are then passed through embedding models, which generate high-dimensional vectors representing unique facial features. Finally, these embeddings are compared using similarity metrics such as cosine similarity to identify or verify individuals.

By following this guide, you will gain practical knowledge and insights into implementing face recognition systems capable of handling real-world challenges like varying lighting, poses, and expressions.

## Quickstart 
1. Clone the repository:
      ```
      git clone https://github.com/DeGirum/hailo_examples.git
      cd hailo_examples/applications/face_recognition
      ```
2. Install the required dependencies:
      ```
      pip install -r requirements.txt
      ```
      
## Create a Database
Steps to follow to create a database of images : <br>
1. **Add a folder of images to the database** (image_source = Path to the database) <br>
    * Gather all the images you want to include in the database and place them in a single folder. <br>
    * Ensure that each image filename starts with the identity name,  followed by an underscore (“_”), and then include any integers or strings that you prefer. <br>
    * Most importantly, ensure that the filenames are unique, following the format 'uniqueName_1', where each 'uniqueName' should differ. <br>
    (If two people in the database has the same name, just add the lastname along with the uniquename, For example : 'uniqueNameLastName_1').

        Example of how the folder structure should look like,

        ```
        folder
        ├── John_1.jpg
        ├── John_2.jpg 
        ├── Michael_1.jpg
        ├── Michael_a.jpg
        ├── MichaelKeaton_1.jpg
        ├── MichaelKeaton_a.jpg
        ```
      **CLI**
        ```
        python index_faces.py --config config.yaml --image_source <path_to_the_database>
        ````

**OR**  

2. **Add a single image to the database** (image_source = Path to the single image file) <br>
    * Ensure that the image filename starts with the identity name,  followed by an underscore (“_”), and then include any integers or strings that you prefer. <br>
        ```
        For example:
            input_path = "John_1.jpg"
        ```

        **CLI**
        ```
        python index_faces.py --config config.yaml --image_source <path_to_the_single_image_file>
        ```

    **OR** <br>

    * Make sure to include the identity name along with the filename if it does not adhere to the above specified naming structure.
        ```
        For example:
            identity_name = "John"
            input_path = "image1.jpg"
        ```
        **CLI**
        ```
        python index_faces.py --config config.yaml  --image_source <path_to_the_single_image_file> --identity_name <identity_name>
        ```
## Video/Image Inference

Specify the following parameters:
  1. **config file** (Path to the config yaml)
  2. **input source** (Path to the input video source/image source)
  3. **threshold** (default= 0.35) 
  4. **fps** (optional)
  5. **display** (default=True)
  5. **save_output** (optional)
  6. **output path** (default = "annotated_output") <br>

**input source** can be of two types,<br>
  1. Video source
      - cv2.VideoCapture object, already opened by open_video_stream()
      - 0-based index for local cameras
      - IP camera URL in the format 'rtsp://<user>:<password>@<ip or hostname>'
      - Local path or URL to mp4 video file
      - YouTube video URL <br>
  2. Image source
      - Image object with extension .jpg,.png etc.

The output is displayed but in case you want to save the annotated output,
provide **--save_output** in the CLI to write the annotated video.<br>
You can also play around with the `threshold` parameter as this defines the cosine distance threshold. <br>

**Output Behavior**: <br>
By default, the output will be displayed on the screen. To save the annotated output, include the **--save_output** flag in the CLI. <br>

The extension of the **--output_path** is decided by the input source, if the **input source** is a video source, the default output path is "annotated_output.mp4", if its an image file, the output path is "annotated_output.jpg". <br>

**CLI**
  ```
  python identify_faces.py --config config.yaml --input_source <input_source>
  ```

**CONFIG**

There is a config yaml file (**config.yaml**) with all the parameters to index a Database.
  ```
  hw_location : "@cloud"
  face_det_kypts_model:
    model_zoo_url: degirum/hailo
    model_name: yolov8n_relu6_widerface_kpts--640x640_quant_hailort_hailo8_1
  face_reid_model:
    model_zoo_url: degirum/sandbox_shashi
    model_name : arcface_mobilefacenet--112x112_quant_hailort_hailo8_2
  database:
    name: face_recognition.db
  table:
    name: faces_hailo
  search_params:
      top_k: 1
      field_name: vector
      metric_type: cosine
  ```
Configurable Parameters:
* Database Name: Specify the name of the database in the config file. Default database name: "default_database" <br>
    For example: <br>
    ```
    database:
        name: "face_recognition.db"
    ```
* Table Name: Specify the name of the table in the config file. Default table name: "default_table" <br>
    For example: <br>
    ```
    collection:
        name: "faces_hailo"
    ```

## Models

**Face Recognition** is a three step process:
1. *Face Detection with Keypoints* - The face detection with keypoints model provides a bounding box for each detected face along with its 5 landmarks(Left Eye, Right Eye, Nose, Left Lip, Right Lip). 
2. The face is cropped and aligned based on the corresponding five landmarks using the *align_and_crop* method.
3. *Face Re-identification* - The face reid model provides a feature embedding for the above aligned and cropped face. <br>

Below are the default entries: <br>
  * Inference Host : @cloud <br>
  * Face Detection and Keypoints Model <br>
      * Model Zoo : degirum/hailo <br>
      * Model Name : yolov8n_relu6_widerface_kpts--640x640_quant_hailort_hailo8_1 <br>

  * Face Re-identification Model
      * Model Zoo : degirum/sandbox_shashi <br>
      * Model Name : arcface_mobilefacenet--112x112_quant_hailort_hailo8_2 <br>
