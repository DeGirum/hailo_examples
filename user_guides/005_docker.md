## Running HailoRT + DeGirum PySDK in Docker

This guide is for developers who want to containerize HailoRT and DeGirum PySDK using Docker. We’ll cover the essential steps to build a Docker image, run a container with hardware access, and verify your setup. By the end, you’ll have a reproducible Docker environment ready for Hailo + PySDK workloads.

---

## What You'll Need to Begin

1. **Host Machine Requirements**  
   - A Hailo device connected to your host machine.
   - A Docker installation.

2. **HailoRT `.deb` Package**  
   Download the **Ubuntu amd64 package (deb) for 4.20.0** from [Hailo Software Downloads](https://hailo.ai/developer-zone/software-downloads/)

3. **DeGirum PySDK**  
   Will be installed via `pip` inside the Docker image.

---

## Downloading the HailoRT Package

1. Navigate to **Hailo Software Downloads**.  
2. Select **HailoRT – Ubuntu package (deb) for amd64 4.20.0**.  
3. Save the file as:
   ```
   hailort_4.20.0_amd64.deb
   ```
4. Place it in your working directory alongside the Dockerfile you’ll create next.

---

## Creating the Dockerfile

Create a file named `Dockerfile` in your working directory with this content:

```dockerfile
# Simple DeGirum PySDK + HailoRT Dockerfile
FROM ubuntu:22.04

ARG HAILORT_DEB=hailort_4.20.0_amd64.deb

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev libusb-1.0-0 ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install HailoRT
COPY ${HAILORT_DEB} /tmp/hailort.deb
RUN ln -s /bin/true /usr/local/bin/systemctl && \
    apt-get update && \
    dpkg --unpack /tmp/hailort.deb && \
    DEBIAN_FRONTEND=noninteractive dpkg --configure -a || true && \
    rm -f /tmp/hailort.deb && \
    rm -rf /var/lib/apt/lists/*

# Install DeGirum PySDK
RUN pip install degirum

# Placeholder command
CMD ["bash"]
```  

---

## Building the Docker Image

From the directory containing your `Dockerfile` and `.deb` file, run:

```bash
docker build -t degirum/pysdk-hailo:latest \
    --build-arg HAILORT_DEB=hailort_4.20.0_amd64.deb .
```

This will produce an image tagged `degirum/pysdk-hailo:latest`.

---

## Running the Docker Container

Launch a container with access to your Hailo device and required volumes:

```bash
docker run -it --rm \
  --device=/dev/hailo0:/dev/hailo0 \
  -v /dev:/dev \
  -v /lib/firmware:/lib/firmware \
  -v /lib/udev/rules.d:/lib/udev/rules.d \
  -v /lib/modules:/lib/modules \
  degirum/pysdk-hailo:latest
```

You’ll be dropped into a shell inside the container, with the Hailo hardware exposed.

---

## Verifying the Installation

Inside the container, run:

   ```bash
   degirum sys-info
   ```
   You should see your connected Hailo device!

---

## Running the Container in the Background

By default, our placeholder `CMD ["bash"]` exits immediately in detached (background) mode. To keep the container alive when using `-d`, **change the image’s CMD/ENTRYPOINT**:

**Modify your Dockerfile**
```dockerfile
# Run a Python script:
CMD ["python", "app.py"]
```
Rebuild with:
```bash
docker build -t degirum/pysdk-hailo:latest .
```

**Run in detached mode**
```bash
docker run -d --name hailo-pysdk \
  --device=/dev/hailo0:/dev/hailo0 \
  -v /dev:/dev \
  -v /lib/firmware:/lib/firmware \
  -v /lib/udev/rules.d:/lib/udev/rules.d \
  -v /lib/modules:/lib/modules \
  degirum/pysdk-hailo:latest
```

Then you can:
- `docker logs -f hailo-pysdk`  
- `docker exec -it hailo-pysdk bash`  


## Conclusion

You now have a Docker-based environment for HailoRT and DeGirum PySDK that can be used as a foundation for developing and deploying your ML applications on Hailo accelerators.
