import yaml, os, degirum_tools, degirum as dg
from pathlib import Path
from typing import Generator, Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The configuration file '{config_path}' does not exist.")
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Error while parsing the YAML file: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred while loading config: {e}")
    
def load_model(config: Dict[str, Any], model_key: str, hw_location: str):
        """
        Load a specific task model based on the configuration.

        Args:
            config (dict): Configuration containing model information.
            model_key (str): Key for the specific model (e.g., 'face_reid_model', 'face_det_kypts_model').
            hw_location (str): Location for inference hardware.

        Returns:
            Optional[Any]: The loaded model or None if not available.
        """
        model_config = config.get(model_key)
        if not model_config:
            return None  # Return None if the model is not specified in the config

        model_name = model_config.get('model_name')
        model_zoo_url = model_config.get('model_zoo_url')

        if not model_name or not model_zoo_url:
            raise ValueError(f"Model name or URL is missing for {model_key} in the config.")

        return dg.load_model(model_name=model_name, inference_host_address=hw_location, zoo_url=model_zoo_url, token=degirum_tools.get_token())
    
def image_generator(input_path: str, identity_name=None) -> Generator[Dict, None, None]:
    """Generate image paths from a given directory or a single image file."""
    path = Path(input_path)
    valid_extensions = {".png", ".jpg", ".jpeg"}
    if path.is_file() and path.suffix.lower() in valid_extensions:
        # If the input path is a single file and it is an image
        entity_name = (
            identity_name if identity_name is not None else path.stem.split("_")[0]
        )
        yield str(path), {"image_path": str(path), "entity_name": entity_name}

    elif path.is_dir():
        # If the input path is a directory, yield all image files within
        for file in path.rglob("*"):
            if file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                entity_name = file.stem.split("_")[0]
                yield str(file), {"image_path": str(file), "entity_name": entity_name}
    else:
        raise ValueError(f"The input path '{input_path}' is neither a valid file nor directory.")

def check_input_type(input_source):
    # If input source is a file path, check the file extension
    if isinstance(input_source, str):
        _, file_extension = os.path.splitext(input_source.lower())
        
        # Remove leading dot (.) if present in extension
        file_extension = file_extension.lstrip('.')

        # List of known image extensions
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp']
        
        # List of known video extensions
        video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'webm', 'mpeg']
        
        if file_extension in image_extensions:
            return "image"
        elif file_extension in video_extensions or input_source.isdigit() or isinstance(input_source, str):
            return "video"
        else:
            return "Unknown file type"
        