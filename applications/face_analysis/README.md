
# Facial Attribute Analysis
The Facial Attribute Analysis feature is designed to analyze various facial attributes, such as age, gender, and emotion, from images or video streams. By leveraging advanced deep learning models, this module can extract these attributes and provide valuable insights for applications in security, user interaction, and social media.

**Key Features**: <br>
Age Prediction: Estimates the approximate age of individuals in the image or video. <br>
Gender Classification: Classifies individuals' gender based on facial features. <br>
Emotion Recognition: Detects and classifies the emotion expressed on a person's face (e.g., happy, sad, angry, surprised, etc.). <br>


## Quickstart 
1. Clone the repository:
      ```
      git clone https://github.com/DeGirum/hailo_examples.git
      cd hailo_examples/applications/face_analysis
      ```
2. Install the required dependencies:
      ```
      pip install -r requirements.txt
      ```
      

**CLI**
  ```
  python facial_attribute_analysis.py --config config.yaml --input_source <input_source>
  ```
