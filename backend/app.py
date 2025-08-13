import base64
import os, sqlite3, uuid, time, cv2
import numpy as np
from datetime import datetime
from io import BytesIO
from PIL import Image
from flask import session, redirect, url_for

from flask import (
    Flask, request, jsonify,
    render_template, redirect, url_for, session, flash
)
from flask_login import (
    LoginManager, UserMixin, login_user,
    login_required, logout_user, current_user
)
import tensorflow as tf

# ── Flask init ──────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "super‑secret‑key"          # change in prod
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)

# User class for login
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username


# ── Database helper ────────────────────────────────────────
DB = "logs.db"
def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users(
         id INTEGER PRIMARY KEY AUTOINCREMENT,
         username TEXT UNIQUE, password TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS detections(
         id INTEGER PRIMARY KEY AUTOINCREMENT,
         user_id INTEGER,
         media_type TEXT,
         result TEXT,
         confidence REAL,
         frames INTEGER,
         duration REAL,
         ts TEXT)""")
    # demo user: admin / admin
    try:
        c.execute("INSERT INTO users (username,password) VALUES (?,?)",
                  ("admin","admin"))
    except sqlite3.IntegrityError:
        pass
    conn.commit(); conn.close()

init_db()

# ── Flask‑Login setup ───────────────────────────────────────
login_manager = LoginManager(app)

class User(UserMixin):
    def __init__(self, id_, name):
        self.id = id_
        self.name = name

@login_manager.user_loader
def load_user(user_id):
    conn = get_db()
    row = conn.execute("SELECT id,username FROM users WHERE id=?",
                       (user_id,)).fetchone()
    conn.close()
    return User(row["id"], row["username"]) if row else None


@app.route("/api/check-login")
def check_login():
    return jsonify(logged_in=current_user.is_authenticated)


# ── Load Model ──────────────────────────────────────────────
MODEL = tf.keras.models.load_model("backend/deepfake_detection_Jupiter_forbalanced_fakeframes.keras")

def preprocess_img(img_bgr):
    img = cv2.resize(img_bgr, (96, 96))
    img = img[..., ::-1] / 255.0  # BGR→RGB
    return np.expand_dims(img.astype("float32"), 0)

# ── Routes ─────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect-video")
@login_required
def detect_video_page():
    return render_template("detectvideo.html")


@app.route("/status")
@login_required
def status_page():
    if session.get("username") != "admin":
        return "Access denied", 403
    return render_template("status.html")



@app.route("/detect-image")
@login_required
def detect_image_page():
    return render_template("detectimage.html")

@app.route("/about")
def about_page():
    return render_template("aboutcontaxt.html")

# ---------- Auth ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]
        row = get_db().execute(
            "SELECT * FROM users WHERE username=? AND password=?", (u, p)
        ).fetchone()
        if row:
            session['username'] = row['username']
            login_user(User(row["id"], row["username"]))
            if session['username'] == "admin":
                return redirect("/status")  # redirect admin
            else:
                return redirect("/")        # redirect normal user
        return "Invalid credentials", 401
    return render_template("login.html")



@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# Register route
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        confirm  = request.form["confirm"]

        if password != confirm:
            flash("Passwords do not match", "error")
            return render_template("register.html")

        # Check if user already exists
        existing = get_db().execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if existing:
            flash("Username already taken", "error")
            return render_template("register.html")

        # Insert into DB
        db = get_db()
        db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        db.commit()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


# ---------- Detection API ----------
def record_log(media_type, result, conf, frames, dur):
    if not current_user.is_authenticated:
        user_id = None
    else:
        user_id = current_user.id
    conn = get_db()
    conn.execute("""INSERT INTO detections
        (user_id, media_type, result, confidence, frames, duration, ts)
        VALUES (?,?,?,?,?,?,?)""",
        (user_id, media_type, result, conf, frames, dur,
         datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

@app.post("/api/detect-image")
def detect_image():
    f = request.files.get("media")
    if not f:
        return jsonify(error="No file uploaded"), 400

    img_np = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="Invalid image file"), 400

    start = time.time()
    prob = float(MODEL.predict(preprocess_img(img))[0][0])
    duration = time.time() - start

    label = "Real" if prob >= 0.5 else "Fake"
    confidence = prob if label == "Real" else 1 - prob

    record_log("image", label, confidence, 1, duration)

    return jsonify(
        label=label,
        confidence=confidence,
        processing_time=duration,
        frames_analyzed=1
    )


@app.post("/api/detect-video")
def detect_video():
    f = request.files.get("media")
    if not f:
        return jsonify(error="No file"), 400

    # ---- Save temp video --------------------------------------------------
    fname = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.mp4")
    f.save(fname)

    cap   = cv2.VideoCapture(fname)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, total-1, num=min(30, total), dtype=int)

    preds          = []
    sample_frames  = []
    start          = time.time()

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()

        # Skip invalid frames
        if not ret or frame is None:
            continue

        # Predict
        preds.append(float(MODEL.predict(preprocess_img(frame))[0][0]))

        # Collect first 5 frames as thumbnails
        if len(sample_frames) < 5:
            ok, buff = cv2.imencode(".jpg", frame)
            if ok:
                b64 = base64.b64encode(buff).decode("utf-8")
                sample_frames.append(f"data:image/jpeg;base64,{b64}")

    cap.release()
    os.remove(fname)

    if not preds:
        return jsonify(error="Could not read frames"), 500

    # ---- Aggregate results -----------------------------------------------
    avg   = float(np.mean(preds))
    label = "Real" if avg >= 0.5 else "Fake"
    confidence = avg if label == "Real" else 1 - avg
    duration   = time.time() - start

    # Log to DB (if you have record_log function)
    record_log("video", label, confidence, len(preds), duration)

    # ---- Return JSON ------------------------------------------------------
    return jsonify(
        label            = label,
        confidence       = confidence,
        processing_time  = duration,
        frames_analyzed  = len(preds),
        frame_confidences= preds,
        sample_frames    = sample_frames
    )

# ---------- Stats ----------
@app.get("/api/stats")
@login_required
def stats():
    if session.get("username") != "admin":
        return "Access denied", 403

    conn = get_db()
    row  = conn.execute("""
        SELECT COUNT(*) as total,
               AVG(confidence) as avg_conf,
               SUM(result='Fake') as fake_cnt,
               SUM(result='Real') as real_cnt
        FROM detections
    """).fetchone()

    daily = conn.execute("""
        SELECT substr(ts,1,10) as day, COUNT(*) as uploads
        FROM detections GROUP BY day ORDER BY day
    """).fetchall()
    conn.close()

    return jsonify(
        total=row["total"],
        fake=row["fake_cnt"], real=row["real_cnt"],
        avg_conf=float(row["avg_conf"] or 0),
        daily=[dict(day=d["day"], uploads=d["uploads"]) for d in daily]
    )



# ---------- Error handler ----------
@app.errorhandler(404)
def page_not_found(_):
    return "Page not found", 404

# ── Run ────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
