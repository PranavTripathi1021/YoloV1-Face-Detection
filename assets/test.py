
import cv2
import torch
import numpy as np

# Replace these imports with your actual model definition
from model_builder import YoloV1_model

# ---------------- CONFIGURATION ----------------

IMAGE_PATH = "images (1).jpeg"
CHECKPOINT_PATH = "../model/Epoch-86.pth"

IMG_SIZE = 448
S = 7

CONF_THRESHOLD = 0.15
NMS_THRESHOLD = 0.45

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- LOAD MODEL ----------------

model = YoloV1_model()  # Use your actual constructor

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

# Adjust this depending on how you save checkpoints
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()


# ---------------- IMAGE PREPROCESSING ----------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

original = image.copy()

H, W = image.shape[:2]

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

# Normalize exactly as during training
image = image.astype(np.float32) / 255.0

# HWC -> CHW
image = np.transpose(image, (2, 0, 1))

# Add batch dimension
image = torch.from_numpy(image).unsqueeze(0).to(device)


# ---------------- INFERENCE ----------------

with torch.inference_mode():
    predictions = model(image)

# Handle flattened or grid-shaped output
predictions = predictions.reshape(1, S, S, 11)

pred = predictions[0].detach().cpu().numpy()


# ---------------- DECODE BOUNDING BOXES ----------------

boxes = []
scores = []

for row in range(S):
    for col in range(S):

        class_prob = float(pred[row, col, 0])

        confidences = pred[row, col, 1:3]

        for b in range(2):

            confidence = float(confidences[b])
            score = class_prob * confidence

            if score < CONF_THRESHOLD:
                continue

            # Both confidence values come before coordinates
            start = 3 + b * 4

            x, y, w, h = pred[row, col, start:start + 4]

            # YOLOv1 convention:
            # x, y are relative to the grid cell.
            # w, h are relative to the entire image.

            cx = (col + x) / S * W
            cy = (row + y) / S * H

            bw = w * W
            bh = h * H

            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)

            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)

            # Clip to image boundaries
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2))
            y2 = max(0, min(H - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)


# ---------------- NON-MAXIMUM SUPPRESSION ----------------

if len(boxes) > 0:

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    if len(indices) > 0:
        indices = np.array(indices).flatten()

        for i in indices:

            x, y, w, h = boxes[i]
            score = scores[i]

            # Draw bounding box
            cv2.rectangle(
                original,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Draw confidence
            label = f"Face: {score:.3f}"

            cv2.putText(
                original,
                label,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        print("Detections:", len(indices))

    else:
        print("No faces detected.")

else:
    print("No faces detected.")


# ---------------- DISPLAY RESULT ----------------

cv2.imshow("YOLOv1 Face Detection", original)

cv2.waitKey(0)
cv2.destroyAllWindows()
