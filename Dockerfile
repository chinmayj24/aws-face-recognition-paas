# ===== Base: AWS Lambda Python runtime (x86_64) =====
# If you prefer Python 3.9, switch to python:3.9
FROM public.ecr.aws/lambda/python:3.11

# Speed up pip & make logs clean
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# --- Copy dependency spec first to leverage Docker layer caching
COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt

# (Optional) Pin CPU-only Torch wheels via index URL to reduce size & avoid CUDA
# If you already pinned versions in requirements.txt you can drop the extra index.
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN python -m pip install --upgrade pip && \
    pip install --extra-index-url ${TORCH_INDEX_URL} -r ${LAMBDA_TASK_ROOT}/requirements.txt

# --- Copy application code (both handlers share one image)
# Put your model files here too if you need them at runtime for recognition.
COPY fd_lambda.py fr_lambda.py ${LAMBDA_TASK_ROOT}/
# If your recognition model files are part of the image, uncomment the next lines:
# COPY resnetV1.pt resnetV1_video_weights.pt ${LAMBDA_TASK_ROOT}/

# Default command tells the Lambda runtime which handler to use.
# You will override this per-function in the Lambda console:
#   - face-detection: fd_lambda.lambda_handler
#   - face-recognition: fr_lambda.lambda_handler
CMD ["fd_lambda.lambda_handler"]
