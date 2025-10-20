import os
import json
import boto3
import base64
import tempfile
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

REGION = "us-east-1"
SQS_REQUEST_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/619071330649/1230031818-req-queue"
sqs = boto3.client("sqs", region_name=REGION)

class face_detection:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

    def face_detection_func(self, test_image_path, output_path):
        img = Image.open(test_image_path).convert("RGB")
        img = np.array(img)
        img = Image.fromarray(img)

        key = os.path.splitext(os.path.basename(test_image_path))[0].split(".")[0]
        face, prob = self.mtcnn(img, return_prob=True, save_path=None)

        if face is not None:
            os.makedirs(output_path, exist_ok=True)

            face_img = face - face.min()
            face_img = face_img / face_img.max()
            face_img = (face_img * 255).byte().permute(1, 2, 0).numpy()

            face_pil = Image.fromarray(face_img, mode="RGB")
            face_img_path = os.path.join(output_path, f"{key}_face.jpg")
            face_pil.save(face_img_path)

            return face_img_path
        else:
            print("No face is detected")
            return None

detector = face_detection()

def lambda_handler(event, context):
    body = json.loads(event["body"])
    request_id = body["request_id"]
    filename = body["filename"]
    content = base64.b64decode(body["content"])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir="/tmp") as ogFile:
        ogFile.write(content)
        input_path = ogFile.name

    opDir = tempfile.mkdtemp()
    face_img_path = detector.face_detection_func(input_path, opDir)

    if face_img_path:
        with open(face_img_path, "rb") as face_file:
            face_b64 = base64.b64encode(face_file.read()).decode("utf-8")

        msg = {
            "request_id": request_id,
            "face": face_b64
        }

        sqs.send_message(
            QueueUrl=SQS_REQUEST_QUEUE_URL,
            MessageBody=json.dumps(msg)
        )

        os.remove(face_img_path)
        os.remove(input_path)
    else:
        os.remove(input_path)

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Face detection complete."})
    }
