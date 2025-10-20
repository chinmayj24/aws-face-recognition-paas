# AWS Lambda Face Recognition (ECR + SQS)

A **serverless, container-based face recognition pipeline** built using **AWS Lambda**, **Elastic Container Registry (ECR)**, and **Amazon SQS**.  
This project implements face detection and recognition as a service for video frames sent by clients (e.g., IoT cameras).

---

## 🧠 Overview

This system runs entirely on **PaaS / serverless infrastructure**.  
It uses two Lambda functions—**Face Detection** and **Face Recognition**—packaged together in one Docker image stored on **Amazon ECR**.

<img width="800" height="313" alt="image" src="https://github.com/user-attachments/assets/fc263e97-86f6-40df-b76a-b099c37f5a31" />


### ⚙️ Architecture
1. **Client** uploads video frames (Base64 encoded) via an HTTP `POST` request to the **Face Detection Lambda**.
2. **Face Detection Lambda (`fd_lambda.py`)**
   - Detects faces in the frame using **MTCNN**.
   - Sends the cropped face image (Base64) and request ID to the **SQS Request Queue**.
3. **Face Recognition Lambda (`fr_lambda.py`)**
   - Triggered automatically by the SQS Request Queue.
   - Performs recognition using a **PyTorch ResNet (facenet-pytorch)** model.
   - Sends the results (`request_id` + `prediction`) to the **SQS Response Queue**.
4. **Client** polls the Response Queue to retrieve results.

---

## ☁️ AWS Resources

All resources must be created in **us-east-1**.

| Service | Resource | Example Name | Purpose |
|----------|-----------|--------------|----------|
| **ECR** | Container Repository | `face-recog-repo` | Stores Lambda Docker image |
| **SQS** | Request Queue | `<id>-req-queue` | From detection → recognition |
| **SQS** | Response Queue | `<id>-resp-queue` | From recognition → client |
| **Lambda** | face-detection | — | Handles frame uploads & detection |
| **Lambda** | face-recognition | — | Performs classification |
| **IAM Role** | lambda-exec-role | — | Grants SQS and CloudWatch permissions |

---

## 🧩 Project Structure

```
.
├── Dockerfile # Common Lambda image for both functions
├── requirements.txt # Python dependencies
├── fd_lambda.py # Face Detection Lambda handler
├── fr_lambda.py # Face Recognition Lambda handler
└── README.md
```


---

## 🐳 Building the Docker Image

1. **Authenticate with ECR**

   ```bash
   aws ecr get-login-password --region us-east-1 \
     | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com

2. **Build and Push**

    ```bash
    docker build -t face-recog-repo .
    docker tag face-recog-repo:latest <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/face-recog-repo:latest
    docker push <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/face-recog-repo:latest
    ```
3. **Confirm**

    ```bash
    aws ecr list-images --repository-name face-recog-repo
    ```

## ⚙️ Lambda Function Setup

### 🧩 Face Detection Function (face-detection)

- Runtime: Container image
- Image URI: <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/face-recog-repo:latest
- Handler: fd_lambda.lambda_handler
- Trigger: Function URL (HTTP POST)
- Env Vars:

  ```
  REGION=us-east-1
  SQS_REQUEST_QUEUE_URL=<your-request-queue-url>
  ```

### 🧠 Face Recognition Function (face-recognition)

- Runtime: Container image
- Image URI: same as above
- Handler: fr_lambda.lambda_handler
- Trigger: SQS → <id>-req-queue
- Env Vars:

  ```
  REGION=us-east-1
  SQS_RESPONSE_QUEUE_URL=<your-response-queue-url>
  MODEL_PATH=resnetV1.pt
  MODEL_WT_PATH=resnetV1_video_weights.pt
  ```

## 🚀 Running the Pipeline

1. **Send an image frame (Base64 encoded) to your Face Detection Lambda URL:**

    ```bash
    curl -X POST -H "Content-Type: application/json" \
    -d '{"request_id":"1","filename":"frame1.jpg","content":"<base64_image>"}' \
    https://<lambda-id>.lambda-url.us-east-1.on.aws/
    ```
2. **Detection Lambda**
- Extracts and encodes faces → sends to request SQS.
3. **Recognition Lambda**
- Triggered by the queue → recognizes faces → pushes result to response SQS.
4. **Client**
- Polls the response SQS queue to read predictions.

## 🧠 Tech Stack
- AWS Lambda
- AWS Elastic Container Registry (ECR)
- Amazon SQS
- Python 3.11
- PyTorch, facenet-pytorch, Pillow, boto3, awslambdaric
