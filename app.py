from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "drowsiness_yolo.pt"   # put your model file here (same folder as app.py)
print("Loading model:", MODEL_PATH)
model = YOLO(MODEL_PATH)
print("Model classes:", model.names)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    # check file field
    if 'image' not in request.files:
        return jsonify({"error": "no file field named 'image'"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "empty filename"}), 400

    # save original file
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)
    print("Saved upload to", save_path)

    # run YOLO inference (lower conf threshold if needed)
    results = model.predict(source=save_path, imgsz=640, conf=0.25, verbose=False)

    # load image with OpenCV for drawing
    img = cv2.imread(save_path)
    if img is None:
        return jsonify({"error": "failed to read saved image"}), 500

    preds = []
    # parse results and draw boxes
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls.cpu().numpy())
            conf = float(box.conf.cpu().numpy())
            name = model.names.get(cls, str(cls))
            xyxy = box.xyxy.cpu().numpy().flatten().tolist()
            x1, y1, x2, y2 = map(int, xyxy[:4])

            preds.append({
                "class": name,
                "conf": round(conf, 3),
                "xyxy": [x1, y1, x2, y2]
            })

            # draw rectangle and label
            color = (0, 255, 0)  # green
            thickness = 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            label = f"{name} {conf:.2f}"
            # put label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

    # save annotated image
    annotated_name = "annotated_" + filename
    annotated_path = os.path.join(UPLOAD_FOLDER, annotated_name)
    cv2.imwrite(annotated_path, img)
    print("Annotated saved to:", annotated_path)

    return jsonify({
        "predictions": preds,
        "annotated_image": annotated_name
    })

# ------------------ Webcam frame prediction route ------------------
import numpy as np
import io
from flask import Response

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """
    Accepts a single JPEG/PNG image blob via FormData field 'image',
    decodes to OpenCV image, runs the same YOLO model and returns JSON predictions.
    This route does NOT save the image to disk (fast for webcam).
    """
    if 'image' not in request.files:
        return jsonify({"error": "no file field named 'image'"}), 400

    f = request.files['image']
    data = f.read()
    if not data:
        return jsonify({"error": "empty image data"}), 400

    # decode bytes to numpy array -> cv2 image
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "failed to decode image"}), 400

    # Run model on numpy image (ultralytics accepts numpy arrays)
    # You can tweak imgsz and conf for live performance/tolerance
    try:
        results = model.predict(source=img, imgsz=640, conf=0.12, iou=0.25, max_det=50, verbose=False)
    except Exception as e:
        # return server-side error so frontend can show it
        return jsonify({"error": f"model prediction failed: {str(e)}"}), 500

    preds = []
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls.cpu().numpy())
            conf = float(box.conf.cpu().numpy())
            name = model.names.get(cls, str(cls))
            xy = box.xyxy.cpu().numpy().flatten().tolist()
            x1, y1, x2, y2 = map(int, xy[:4])
            preds.append({"class": name, "conf": round(conf,3), "xyxy": [x1,y1,x2,y2]})

    return jsonify({"predictions": preds})
# --------------------------------------------------------------------

if __name__ == "__main__":
    # debug=True for development only
    app.run(host="0.0.0.0", port=5000, debug=True)
