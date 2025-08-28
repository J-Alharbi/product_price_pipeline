# Use AWS Lambda Python 3.12 base image
FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies required for OpenCV and Pillow
RUN dnf install -y \
    gcc gcc-c++ git \
    libSM libXext libX11-devel libXrandr libXinerama libXcursor \
    mesa-libGL mesa-libGLU libpng zlib \
    && dnf clean all

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies into Lambda task root
RUN pip install --no-cache-dir \
    boto3 \
    ultralytics \
    opencv-python-headless \
    numpy \
    Pillow \
    google-cloud-vision \
    --target "${LAMBDA_TASK_ROOT}"

# Copy Lambda handler
COPY app.py ${LAMBDA_TASK_ROOT}

CMD ["app.lambda_handler"]
