
# **Using DeGirum PySDK, DeGirum Tools, and Hailo Hardware**  

This repository provides a comprehensive guide on using **DeGirum PySDK**, **DeGirum Tools**, and **Hailo hardware** for efficient AI inference. These tools simplify edge AI development by enabling seamless integration, testing, and deployment of AI models on multiple hardware platforms, including **Hailo-8** and **Hailo-8L**.  

---

## **Table of Contents**  

1. [Introduction](#introduction)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [Running and Configuring Jupyter Notebooks](#running-and-configuring-jupyter-notebooks) 
5. [Additional Resources](#additional-resources) 

---

## **Introduction**  

DeGirum provides a powerful suite of tools to simplify the development and deployment of edge AI applications:  

- [**DeGirum PySDK**](https://github.com/DeGirum/PySDKExamples): The core library for integrating AI inference capabilities into applications.  
- [**DeGirum Tools**](https://github.com/DeGirum/degirum_tools): Utilities for benchmarking, streaming, and interacting with DeGirum's model zoo.  

These tools are designed to be hardware-agnostic, enabling developers to build scalable, flexible solutions without being locked into a specific platform.  

---

## **Prerequisites**  

- **Hailo Tools Installed**: Ensure that Hailo's tools and SDK are properly installed and configured. Refer to [Hailo's documentation](https://hailo.ai/) for detailed setup instructions. Also, enable the HailoRT Multi-Process service as per HailoRT documentation:  

  ```bash
  sudo systemctl enable --now hailort.service  # for Ubuntu
  ```  

- **Hailo Runtime Compatibility**:  
  DeGirum PySDK supports **Hailo Runtime versions 4.19.0, 4.20.0, 4.21.0 and 4.22.0**. Ensure your Hailo environment is configured to use one of these versions.  

- **Python 3.9 or Later**: Ensure Python is installed on your system. You can check your Python version using:  

  ```bash
  python3 --version
  ```  

---

## **Installation**  

The best way to get started is to **clone this repository** and set up a virtual environment to keep dependencies organized. Follow these steps:  

### **1. Clone the Repository**  
```bash
git clone https://github.com/DeGirum/hailo_examples.git
cd hailo_examples
```  

### **2. Create a Virtual Environment**  
To keep the Python environment isolated, create a virtual environment:  

#### **Linux/macOS**  
```bash
python3 -m venv degirum_env
source degirum_env/bin/activate
```  

#### **Windows**  
```bash
python3 -m venv degirum_env
degirum_env\Scripts\activate
```  

### **3. Install Required Dependencies**  
Install all necessary packages from `requirements.txt`:  

```bash
pip install -r requirements.txt
```  

---

### **4. Add Virtual Environment to Jupyter**  

If you plan to use **Jupyter Notebooks**, ensure the virtual environment is available as a Jupyter kernel.  

#### **Step 1: Activate the Virtual Environment (if not already active)**  
If you are not already inside the virtual environment, activate it:  

**Linux/macOS:**  
```bash
source degirum_env/bin/activate
```  

**Windows:**  
```bash
degirum_env\Scripts\activate
```  

#### **Step 2: Ensure the Virtual Environment is Available in Jupyter**  
Since `notebook` and `ipykernel` are already installed via `requirements.txt`, simply run:  

```bash
python -m ipykernel install --user --name=degirum_env --display-name "Python (degirum_env)"
```  

This ensures that Jupyter recognizes the virtual environment as an available kernel.  

---

### **5. Verify Installation**  

To ensure that everything is set up correctly, run the provided test script:  

```bash
python test.py
```  

This script will:  
- Check system information.  
- Verify that Hailo hardware is recognized.  
- Load and run inference with a sample AI model.  

If the test runs successfully, your environment is properly configured.  


## **Running and Configuring Jupyter Notebooks**  

This repository includes an `examples` folder containing multiple use case examples demonstrating how to run AI inference using DeGirum PySDK and Hailo hardware. You can find detailed descriptions and usage instructions for each example in the [**Examples README**](examples/README.md).  

### **1. Start Jupyter Notebook**  
Now that the Jupyter environment is set up, you can start Jupyter Notebook:  

```bash
jupyter notebook
```  

This will open Jupyter in your web browser, allowing you to navigate to the `examples` folder and run the available notebooks.  

### **2. Ensure the Correct Kernel is Selected**  
When opening a notebook:  
- Go to **Kernel → Change Kernel**.  
- Select **Python (degirum_env)** to ensure the notebook runs inside the correct virtual environment.  


### **3. Default Notebook Settings and Customization**  
Each Jupyter Notebook in this repository is pre-configured with default inference settings, including the inference environment, model zoo location, and target hardware. However, you can modify these values if your setup requires different configurations.

Below are the default settings you will find in the notebooks, which you can adjust as needed:

#### **Select Inference Host Address**  
The `inference_host_address` determines where AI inference will be executed:  

```python
# Use local inference (e.g., when running on a device equipped with Hailo8/Hailo8L)
inference_host_address = "@local"

# Alternative: Specify a local server by IP or hostname
# inference_host_address = "localhost"

# Alternative: Use DeGirum AI Hub for cloud-based inference
# inference_host_address = "@cloud"
```  

#### **Choose Model Zoo Location**  
The `zoo_url` specifies where AI models are stored:  

```python
# Use DeGirum’s cloud model zoo (recommended for Hailo models)
zoo_url = "degirum/hailo"

# Alternative: Use a local directory containing models
# zoo_url = "../models"
```  

#### **Set Authentication Token**  
The `token` is required only for cloud inference with DeGirum AI Hub:  

```python
# No token needed for local inference
token = ''

# Alternative: Fetch token for cloud inference
# token = degirum_tools.get_token()  # Use this when running on AI Hub
```  

#### **Specify Target Hardware**  
The `device_type` defines the hardware used for inference:  

```python
# Default: Hailo8L device
device_type = "HAILORT/HAILO8L"

# Alternative: Hailo8 device (Note: Hailo8L models work on Hailo8, but not vice versa)
# device_type = "HAILORT/HAILO8"
```  
---
## Additional Resources

- [Hailo Model Zoo](./hailo_model_zoo.md): Explore the full list of models optimized for Hailo hardware.
- [DeGirum Documentation](https://docs.degirum.com)
- [Hailo Documentation](https://hailo.ai/)

