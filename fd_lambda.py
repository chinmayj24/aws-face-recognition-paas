# fd_lambda.py (generalized for public repos)

import os
import json
import boto3
import base64
import tempfile
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from botocore.exceptions import BotoCoreError, ClientError

# ---------- Configuration via Environment Variables ----------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SQS_REQUEST_QUEUE_URL = os.getenv(
    "SQS_REQUEST_QUEUE_URL",
    "https://sqs.us-east-1.amazonaws.com/ACCOUNT_ID/your-req-queue"
)
# -------------------------------------------------------------

sqs = boto3.client("sqs", region_name=AWS_REGION)

class FaceDetection:
    def __init__(self):
        # CPU-only MTCNN is fine for Lambda
        self.mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

    def detect_face(self, input_image_path: str, output_dir: str) -> str | None:
        """
        Runs face detection on the file at input_image_path.
        Writes a normalized RGB face crop to output_dir and returns its path,
        or returns None if no face is detected.
        """
        img = Image.open(input_image_path).convert("RGB")
        img_np = np.array(img)
        img_pil = Image.fromarray(img_np)

        face_tensor, prob = self.mtcnn(img_pil, return_prob=True, save_path=None)
        if face_tensor is None:
            return None

        os.makedirs(output_dir, exist_ok=True)

        # Normalize to [0, 255] and convert to PIL RGB
        face_img = face_tensor - face_tensor.min()
        denom = face_img.max() - face_img.min()
        face_img = (face_img / denom * 255) if denom > 0 else face_img * 0
        face_arr = face_img.byte().permute(1, 2, 0).numpy()
        face_pil = Image.fromarray(face_arr, mode="RGB")

        key = os.path.splitext(os.path.basename(input_image_path))[0].split(".")[0]
        out_path = os.path.join(output_dir, f"{key}_face.jpg")
        face_pil.save(out_path)
        return out_path

detector = FaceDetection()

def _bad_request(message: str, status: int = 400):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": message})
    }

def lambda_handler(event, context):
    """
    Expects API Gateway/Lambda Function URL JSON body with keys:
      - request_id: str
      - filename: str (original file name, used only for output naming)
      - content: base64-encoded image bytes
    On success, enqueues to SQS request queue with {"request_id","face"} and returns 200.
    """
    try:
        # Body may already be a dict (Lambda Function URL) or a JSON string (API GW)
        body_raw = event.get("body")
        if body_raw is None:
            return _bad_request("Missing request body")

        body = body_raw if isinstance(body_raw, dict) else json.loads(body_raw)

        request_id = body.get("request_id")
        filename = body.get("filename")
        content_b64 = body.get("content")

        if not request_id or not filename or not content_b64:
            return _bad_request("Missing one or more required fields: request_id, filename, content")

        # Write incoming image to /tmp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir="/tmp") as in_file:
            in_file.write(base64.b64decode(content_b64))
            input_path = in_file.name

        # Detect face
        out_dir = tempfile.mkdtemp()
        face_img_path = detector.detect_face(input_path, out_dir)

        # Clean up input file regardless of detection outcome
        try:
            os.remove(input_path)
        except OSError:
            pass

        if not face_img_path:
            # No face found — return success with a helpful message
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"message": "No face detected."})
            }

        # Encode detected face and push to SQS
        with open(face_img_path, "rb") as f:
            face_b64 = base64.b64encode(f.read()).decode("utf-8")

        msg = {"request_id": request_id, "face": face_b64}

        try:
            sqs.send_message(QueueUrl=SQS_REQUEST_QUEUE_URL, MessageBody=json.dumps(msg))
        except (BotoCoreError, ClientError) as e:
            return _bad_request(f"Failed to enqueue message: {str(e)}", status=502)
        finally:
            # Best-effort cleanup
            try:
                os.remove(face_img_path)
            except OSError:
                pass

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"message": "Face detection complete and request enqueued."})
        }

    except json.JSONDecodeError:
        return _bad_request("Request body must be valid JSON")
    except Exception as e:
        # Avoid leaking internals; return generic error
        return _bad_request(f"Unhandled error: {str(e)}", status=500)
