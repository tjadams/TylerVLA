FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV and MuJoCo
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libglew-dev libglfw3 && \
    rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    numpy \
    mujoco \
    robot_descriptions \
    opencv-python-headless \
    pandas \
    pyarrow \
    pytest

COPY . .

CMD ["python", "-m", "pytest", "tests/real_robot/test_convert_lerobot.py", "-v"]
