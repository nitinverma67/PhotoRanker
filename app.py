from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

history_data = []

# Haarcascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ---------- IMAGE FEATURES ----------

def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray) / 255)

def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000)

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.3, 5)

def check_face_center(image, faces):
    if len(faces) == 0:
        return 0
    h, w, _ = image.shape
    (x, y, fw, fh) = faces[0]
    cx = x + fw/2
    cy = y + fh/2

    if abs(cx - w/2) < w*0.2 and abs(cy - h/2) < h*0.2:
        return 1
    return 0

# ---------- SCORING ----------

def calculate_score(image, mode):
    brightness = calculate_brightness(image)
    sharpness = calculate_sharpness(image)
    faces = detect_face(image)

    face_score = 1 if len(faces) > 0 else 0
    face_position = check_face_center(image, faces)

    if mode == "social":
        score = (0.3 * brightness +
                 0.3 * sharpness +
                 0.4 * face_score)
    else:
        score = (0.4 * sharpness +
                 0.3 * face_score +
                 0.3 * face_position)

    return float(round(score * 100, 2))   # ✅ FIXED

# ---------- ROUTES ----------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        files = request.files.getlist('images')
        mode = request.form.get('mode')

        print("FILES:", files)
        print("MODE:", mode)

        results = []

        for file in files:
            if file.filename == '':
                continue

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            print("Saved:", filepath)

            image = cv2.imread(filepath)

            if image is None:
                print("Image failed to load")
                continue

            score = calculate_score(image, mode)

            results.append({
                'filename': file.filename,
                'score': float(score)   # ✅ FIXED
            })

        print("RESULTS:", results)

        # sort results
        results.sort(key=lambda x: x['score'], reverse=True)

        # store history
        if results:
            history_data.append({
                'mode': mode,
                'results': results
            })

        print("HISTORY NOW:", history_data)

        return render_template('result.html', results=results, mode=mode)

    except Exception as e:
        print("ERROR OCCURRED:", e)
        return "ERROR: " + str(e)

# ---------- HISTORY ----------

@app.route('/history')
def history():
    print("HISTORY PAGE:", history_data)
    return render_template('history.html', history=history_data)

# ---------- SERVE IMAGES ----------

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ---------- RUN ----------

if __name__ == "__main__":
    print("Server starting...")
    app.run(debug=True, use_reloader=False)