import subprocess
import degirum as dg
import sys


def get_sys_info():
    try:
        # Run the CLI command and capture the output
        result = subprocess.run(
            ["degirum", "sys-info"], capture_output=True, text=True, check=True
        )
        print(result.stdout)  # Print the command output
    except subprocess.CalledProcessError as e:
        print("Error executing 'degirum sys-info':", e.stderr)
    except FileNotFoundError:
        print(
            "Error: 'degirum' command not found. Make sure DeGirum PySDK is installed."
        )
    except Exception as e:
        print(f"Unexpected error while getting system info: {e}")


if __name__ == "__main__":
    try:
        print("System information:")
        get_sys_info()

        # Check supported devices
        try:
            supported_devices = dg.get_supported_devices(
                inference_host_address="@local", "degirum/hailo"
            )
        except Exception as e:
            print(f"Error fetching supported devices: {e}")
            sys.exit(1)

        print("Supported RUNTIME/DEVICE combinations:", list(supported_devices))

        # Determine appropriate device_type
        if "HAILORT/HAILO8L" in supported_devices:
            device_type = "HAILORT/HAILO8L"
        elif "HAILORT/HAILO8" in supported_devices:
            device_type = "HAILORT/HAILO8"
        else:
            print(
                "Hailo device is NOT supported or NOT recognized properly. Please check the installation."
            )
            sys.exit(1)

        print(f"Using device type: {device_type}")

        print("Running inference on Hailo device")

        inference_host_address = "@local"
        zoo_url = "degirum/hailo"
        token = ""

        # Set model name and image source
        model_name = "yolov8n_coco--640x640_quant_hailort_multidevice_1"
        image_source = "assets/ThreePersons.jpg"

        # Load AI model
        try:
            model = dg.load_model(
                model_name=model_name,
                inference_host_address=inference_host_address,
                zoo_url=zoo_url,
                token=token,
                device_type=device_type,
            )
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            sys.exit(1)

        # Perform AI model inference
        print(
            f"Running inference using '{model_name}' on image source '{image_source}'"
        )
        try:
            inference_result = model(image_source)
            print(inference_result)
        except Exception as e:
            print(f"Error during inference: {e}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
