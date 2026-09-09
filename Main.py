import os
import cv2
import argparse
import numpy as np
import onnxruntime as ort
import yt_dlp
import tempfile
import shutil
from collections import deque
import time
import sys

# ------------------------------
# YouTube download function
# ------------------------------
def download_youtube_video(url: str, target_res: str = "1080p", progress_callback=None):
    res_int = int(target_res.rstrip("p"))
    temp_dir = tempfile.gettempdir()
    
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.getcwd()
        
    ffmpeg_path = os.path.join(base_path, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_path):
        ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg.exe")
    if not os.path.exists(ffmpeg_path):
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            ffmpeg_path = system_ffmpeg

    output_template = os.path.join(temp_dir, 'yt_video_%(id)s_%(height)sp.%(ext)s')

    def hook(d):
        if d['status'] == 'downloading':
            percent_str = d.get('_percent_str', '').strip()
            speed = d.get('_speed_str', '').strip()
            eta = d.get('eta', '?')
            text = f"Downloading {percent_str} at {speed} (ETA: {eta}s)"
            numeric_percent = None
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            downloaded = d.get('downloaded_bytes', 0)
            if total and total > 0:
                numeric_percent = min(100.0, (downloaded / total) * 100.0)
            if progress_callback:
                progress_callback(text, numeric_percent)
        elif d['status'] == 'finished':
            if progress_callback:
                progress_callback("Download complete, finalizing file...", 100.0)

    ydl_opts = {
        'format': f'bestvideo[height<={res_int}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={res_int}]+bestaudio/best[height<={res_int}]/best',
        "ffmpeg_location": ffmpeg_path if os.path.exists(ffmpeg_path) else None,
        'merge_output_format': 'mp4',
        'outtmpl': output_template,
        'noplaylist': True,
        'quiet': True,
        'progress_hooks': [hook],
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Failed to download YouTube video: {url}")
    return filename


# ------------------------------
# GPU/CPU detection
# ------------------------------
def get_device(force_cpu=False):
    if force_cpu:
        print("  Forcing CPU mode...")
        return "CPUExecutionProvider"
    providers = ort.get_available_providers()
    print(f"  Available providers: {providers}")
    if "CUDAExecutionProvider" in providers:
        print(" Using GPU: CUDAExecutionProvider")
        return "CUDAExecutionProvider"
    if "DmlExecutionProvider" in providers:
        print(" Using GPU: DmlExecutionProvider (DirectML)")
        return "DmlExecutionProvider"
    print("  No GPU provider available, using CPU.")
    return "CPUExecutionProvider"

# ------------------------------
# Load ONNX model
# ------------------------------
def load_model(model_path, force_cpu=False):
    device = get_device(force_cpu)
    print(f"Loading model {model_path} on {device}")
    
    # Optimize session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4  # Adjust based on your CPU cores
    sess_options.inter_op_num_threads = 4
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    
    try:
        prov_list = [device] if device != "CPUExecutionProvider" else ["CPUExecutionProvider"]
        if "CPUExecutionProvider" not in prov_list:
            prov_list.append("CPUExecutionProvider")
        session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options,
            providers=prov_list
        )
    except Exception as e:
        print(f" Failed to load on {device}: {e}")
        print("Retrying on CPU...")
        session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
    try:
        ins = session.get_inputs()
        outs = session.get_outputs()
        for i in ins:
            print(f"  Input: {i.name} shape={i.shape}")
        for o in outs:
            print(f"  Output: {o.name} shape={o.shape}")
    except Exception as e:
        print(f"  Could not read IO metadata: {e}")
    return session

# ------------------------------
# Preprocess frame for YOLOv8
# ------------------------------
def preprocess_frame(frame, target_size=640):
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized

    inp = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    inp = inp.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))
    inp = np.expand_dims(inp, axis=0)
    return inp, scale, (dx, dy)

# ------------------------------
# Postprocess (multi-class & architecture-aware + NMS)
# ------------------------------
def postprocess_predictions(preds, conf_thresh=0.25, iou_thresh=0.45):
    out = np.array(preds[0])
    if out.ndim == 3:
        out = out[0]
    if out.ndim != 2:
        out = out.reshape(-1, out.shape[-1])

    num_rows, num_cols = out.shape

    # Check for End-to-End format (e.g. YOLOv10: shape (300, 6) with [x1, y1, x2, y2, score, class_id])
    if num_cols == 6 and num_rows <= 1000:
        boxes = []
        for p in out:
            conf = float(p[4])
            if conf >= conf_thresh:
                x1, y1, x2, y2 = float(p[0]), float(p[1]), float(p[2]), float(p[3])
                cls_id = int(p[5])
                boxes.append([x1, y1, x2, y2, conf, cls_id])
        return boxes

    # Transpose channels-first output (YOLOv8/v9/11: shape (4+C, N) where N > 4+C)
    if num_rows <= 512 and num_cols > num_rows:
        obs = out.T
        is_yolov8_style = True
    else:
        obs = out
        is_yolov8_style = False

    num_attrs = obs.shape[1]
    if num_attrs < 5:
        return []

    cx = obs[:, 0]
    cy = obs[:, 1]
    w = obs[:, 2]
    h = obs[:, 3]

    if num_attrs == 5:
        # 1 class, single score (no separate objectness)
        scores = obs[:, 4]
        cls_ids = np.zeros(len(scores), dtype=int)
    elif is_yolov8_style:
        # YOLOv8 / YOLOv9 / YOLO11: columns 4: are class probabilities directly
        cls_scores = obs[:, 4:]
        cls_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(len(cls_scores)), cls_ids]
    else:
        # YOLOv5 / YOLOv7: column 4 is objectness, columns 5: are class scores
        obj = obs[:, 4]
        cls_scores = obs[:, 5:]
        cls_ids = np.argmax(cls_scores, axis=1)
        scores = obj * cls_scores[np.arange(len(cls_scores)), cls_ids]

    mask = scores >= conf_thresh
    if not np.any(mask):
        return []

    cx = cx[mask]
    cy = cy[mask]
    w = w[mask]
    h = h[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    # Apply class-aware NMS using cv2.dnn.NMSBoxes
    cv_boxes = [[float(x1[i]), float(y1[i]), float(w[i]), float(h[i])] for i in range(len(scores))]
    confidences = [float(s) for s in scores]

    unique_classes = np.unique(cls_ids)
    final_boxes = []

    for c in unique_classes:
        c_indices = np.where(cls_ids == c)[0]
        c_cv_boxes = [cv_boxes[i] for i in c_indices]
        c_scores = [confidences[i] for i in c_indices]

        keep = cv2.dnn.NMSBoxes(c_cv_boxes, c_scores, score_threshold=conf_thresh, nms_threshold=iou_thresh)
        if len(keep) > 0:
            for k in np.array(keep).flatten():
                idx = c_indices[k]
                final_boxes.append([float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx]), float(scores[idx]), int(c)])

    return final_boxes

# ------------------------------
# Rescale boxes back to original frame
# ------------------------------
def rescale_boxes(boxes, scale, offset, orig_w, orig_h):
    dx, dy = offset
    for b in boxes:
        b[0] = (b[0] - dx) / scale
        b[1] = (b[1] - dy) / scale
        b[2] = (b[2] - dx) / scale
        b[3] = (b[3] - dy) / scale
        b[0] = max(0.0, min(float(b[0]), float(orig_w)))
        b[1] = max(0.0, min(float(b[1]), float(orig_h)))
        b[2] = max(0.0, min(float(b[2]), float(orig_w)))
        b[3] = max(0.0, min(float(b[3]), float(orig_h)))
    return boxes

# ------------------------------
# Resize image and adjust boxes (crop instead of squish)
# ------------------------------
def resize_and_adjust_boxes(frame, boxes, new_size):
    target_w, target_h = new_size
    orig_h, orig_w = frame.shape[:2]
    
    # Direct Center Crop (Pixel-perfect, no scaling)
    center_x, center_y = orig_w // 2, orig_h // 2
    half_w, half_h = target_w // 2, target_h // 2
    
    x1 = max(0, center_x - half_w)
    y1 = max(0, center_y - half_h)
    x2 = min(orig_w, center_x + half_w)
    y2 = min(orig_h, center_y + half_h)
    
    resized = frame[y1:y2, x1:x2]
    act_h, act_w = resized.shape[:2]
    
    new_boxes = []
    for b in boxes:
        bx1, by1, bx2, by2, conf, cls = b
        bx1 -= x1
        bx2 -= x1
        by1 -= y1
        by2 -= y1
        
        bx1 = max(0.0, min(bx1, float(act_w)))
        bx2 = max(0.0, min(bx2, float(act_w)))
        by1 = max(0.0, min(by1, float(act_h)))
        by2 = max(0.0, min(by2, float(act_h)))
        
        if bx2 > bx1 and by2 > by1:
            new_boxes.append([bx1, by1, bx2, by2, conf, cls])
            
    return resized, new_boxes

# ------------------------------
# IoU + Weighted Box Fusion (Ensemble merging)
# ------------------------------
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def merge_boxes(boxes, merge_iou=0.5):
    if not boxes:
        return []
    # Sort boxes descending by confidence
    sorted_boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    merged, used = [], set()

    for i, box in enumerate(sorted_boxes):
        if i in used:
            continue
        x1, y1, x2, y2, conf, cls = box
        total_w = conf
        sum_x1 = x1 * conf
        sum_y1 = y1 * conf
        sum_x2 = x2 * conf
        sum_y2 = y2 * conf
        max_conf = conf

        for j in range(i + 1, len(sorted_boxes)):
            if j in used:
                continue
            other = sorted_boxes[j]
            if cls != other[5]:
                continue
            if iou(box[:4], other[:4]) > merge_iou:
                w = other[4]
                sum_x1 += other[0] * w
                sum_y1 += other[1] * w
                sum_x2 += other[2] * w
                sum_y2 += other[3] * w
                total_w += w
                max_conf = max(max_conf, other[4])
                used.add(j)

        avg_x1 = sum_x1 / total_w
        avg_y1 = sum_y1 / total_w
        avg_x2 = sum_x2 / total_w
        avg_y2 = sum_y2 / total_w
        merged.append([avg_x1, avg_y1, avg_x2, avg_y2, max_conf, cls])
    return merged

# ------------------------------
# Perceptual Hash for duplicate detection
# ------------------------------
def compute_phash(frame, hash_size=8):
    """Compute perceptual difference hash (dHash) of a frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    val = 0
    for v in diff.flatten():
        val = (val << 1) | int(v)
    return val

def hamming_distance(hash1, hash2):
    """Calculate Hamming distance between two 64-bit perceptual hashes"""
    return (hash1 ^ hash2).bit_count()

def is_similar_to_recent(frame, recent_hashes, similarity_threshold=5):
    """
    Check if frame is similar to any recent frame.
    similarity_threshold: lower = stricter (0 = identical, 64 = completely different)
    """
    if not recent_hashes or similarity_threshold <= 0:
        return False
    current_hash = compute_phash(frame)
    for prev_hash in recent_hashes:
        if hamming_distance(current_hash, prev_hash) <= similarity_threshold:
            return True
    return False

# ------------------------------
# Save YOLO-format labels
# ------------------------------
def save_labels(boxes, img_w, img_h, save_path):
    lines = []
    if img_w > 0 and img_h > 0:
        for b in boxes:
            x1, y1, x2, y2, conf, cls = [float(v) for v in b]
            x_c = ((x1 + x2) / 2.0) / img_w
            y_c = ((y1 + y2) / 2.0) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            if w > 0 and h > 0:
                lines.append(f"{int(cls)} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    with open(save_path, "w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")

def annotate_frame(frame, boxes):
    h, w = frame.shape[:2]
    for b in boxes:
        x1, y1, x2, y2, conf, cls = [int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(b[4]), int(b[5])]
        area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        pct = (area / (h * w)) * 100 if h * w else 0
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{conf:.2f} | {pct:.1f}%"
        tsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - tsize[1] - 6), (x1 + tsize[0] + 6, y1), (0, 255, 0), -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

# ------------------------------
# Main with cancel + progress + duplicate detection
# ------------------------------
def main(args=None, progress_callback=None, detect_progress_callback=None, cancel_check=None, preview_callback=None, pause_check=None, save_callback=None, error_callback=None):

    def should_cancel():
        try:
            if cancel_check and cancel_check():
                return True
        except Exception:
            pass
        return False

    try:
        video_path = None  # Initialize early to prevent UnboundLocalError in finally block

        # Parse CLI if no args passed
        if args is None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--out-res", type=str, default="off")
            parser.add_argument("--video", type=str)
            parser.add_argument("--youtube", type=str)
            parser.add_argument("--yt-res", type=str, default="1080p")
            # (cookies args removed)
            parser.add_argument("--models", type=str, nargs="+", required=True)
            parser.add_argument("--out", type=str, required=True)
            parser.add_argument("--conf", type=float, default=0.25)
            parser.add_argument("--iou", type=float, default=0.45)
            parser.add_argument("--merge-iou", type=float, default=0.5)
            parser.add_argument("--frame-step", type=int, default=1)
            parser.add_argument("--input-size", type=int, default=640)
            parser.add_argument("--save-raw", action="store_true")
            parser.add_argument("--cpu", action="store_true")
            parser.add_argument("--similarity-threshold", type=int, default=5, 
                              help="Similarity threshold for duplicate detection (0-64, lower=stricter)")
            parser.add_argument("--history-size", type=int, default=30,
                              help="Number of recent frames to compare against")
            args = parser.parse_args()
        else:
            from types import SimpleNamespace
            args = SimpleNamespace(**args)
            # Set defaults for new parameters if not provided
            if not hasattr(args, 'similarity_threshold'):
                args.similarity_threshold = 5
            if not hasattr(args, 'history_size'):
                args.history_size = 30
            if not hasattr(args, 'input_size'):
                args.input_size = 640
            if not hasattr(args, 'save_raw'):
                args.save_raw = True
            if not hasattr(args, 'target_class_id'):
                args.target_class_id = -1

        # Source setup (prefer local video over YouTube)
        video_path = None
        if getattr(args, "video", None):
            video_path = args.video
            print(f"Using local video: {video_path}")
        elif getattr(args, "youtube", None):
            try:
                video_path = download_youtube_video(args.youtube, args.yt_res, progress_callback)
                print(f"Downloaded video: {video_path}")
            except Exception as e:
                msg = f"YouTube download failed: {e}"
                print(msg)
                if error_callback:
                    error_callback(msg)
                return
        else:
            raise ValueError("No video or YouTube URL provided.")

        try:
            os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
            os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)
            if getattr(args, 'save_annotated', True):
                os.makedirs(os.path.join(args.out, "annotated"), exist_ok=True)
        except Exception as e:
            msg = f"Cannot create output folders in {args.out}: {e}"
            print(msg)
            if error_callback:
                error_callback(msg)
            return

        models = []
        for m in args.models:
            if not m or not m.strip():
                continue
            try:
                models.append(load_model(m, force_cpu=args.cpu))
            except Exception as e:
                msg = f"Failed to load model {m}: {e}"
                print(msg)
                if error_callback:
                    error_callback(msg)
        if not models:
            msg = "No models loaded. Aborting."
            print(msg)
            if error_callback:
                error_callback(msg)
            return
        print(f"Loaded {len(models)} models.")
        print(f"Duplicate detection enabled: similarity_threshold={args.similarity_threshold}, history_size={args.history_size}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap or not cap.isOpened():
            msg = f"Cannot open video source: {video_path}"
            print(msg)
            if error_callback:
                error_callback(msg)
            return

        # Seek to start time if specified
        start_time = getattr(args, 'start_time', 0.0)
        end_time = getattr(args, 'end_time', 0.0)
        
        if start_time > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        
        # Adjust total frames for progress bar if range is set
        if end_time > 0 and end_time > start_time:
            total_frames = int((end_time - start_time) * fps)
        elif start_time > 0:
            total_frames -= int(start_time * fps)

        frame_idx = 0
        saved_count = 0
        skipped_duplicates = 0

        # Keep track of recent frame hashes
        recent_hashes = deque(maxlen=args.history_size)
        
        # Initialize ThreadPoolExecutor once
        from concurrent.futures import ThreadPoolExecutor, as_completed
        executor = ThreadPoolExecutor(max_workers=max(1, len(models)))
        last_progress_time = 0.0

        try:
            while True:
                if should_cancel():
                    print("Detection canceled by user (before reading).")
                    break
                    
                # Check end time
                if end_time > 0:
                    current_pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if current_pos_msec > end_time * 1000:
                        break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                while True:
                    if should_cancel():
                        print("Detection canceled by user (mid-loop).")
                        break
                    if pause_check and pause_check():
                        time.sleep(0.05)
                        continue
                    break
                if should_cancel():
                    break

                if frame_idx % args.frame_step != 0:
                    frame_idx += 1
                    continue

                if frame is None or not hasattr(frame, "shape"):
                    frame_idx += 1
                    continue
                
                img_h, img_w = frame.shape[:2]
                ensemble_boxes = []

                # Preprocess once for all models
                inp, scale, offset = preprocess_frame(frame, target_size=getattr(args, "input_size", 640))

                def run_model(model):
                    if should_cancel():
                        return []
                    try:
                        input_name = model.get_inputs()[0].name
                        pred = model.run(None, {input_name: inp})
                        boxes = postprocess_predictions(pred, conf_thresh=args.conf, iou_thresh=args.iou)
                        boxes = rescale_boxes(boxes, scale, offset, img_w, img_h)
                        return boxes
                    except Exception as e:
                        print(f" Model inference failed: {e}")
                        return []
                
                # Use thread pool for parallel inference
                futures = [executor.submit(run_model, model) for model in models]
                for future in as_completed(futures):
                    if should_cancel():
                        print("Detection canceled during model inference.")
                        break
                    ensemble_boxes.extend(future.result())

                if should_cancel():
                    break

                final_boxes = merge_boxes(ensemble_boxes, merge_iou=args.merge_iou)

                # Apply Class ID Corrector if enabled
                target_id = getattr(args, 'target_class_id', -1)
                if target_id >= 0:
                    for b in final_boxes:
                        b[5] = int(target_id)  # Override class ID

                if final_boxes or getattr(args, 'save_empty', False):
                    # Check if this frame is similar to recent frames
                    if is_similar_to_recent(frame, recent_hashes, args.similarity_threshold):
                        skipped_duplicates += 1
                        frame_idx += 1
                        continue
                    
                    # Add current frame hash to history
                    current_hash = compute_phash(frame)
                    recent_hashes.append(current_hash)

                    if args.out_res and args.out_res.lower() != "off":
                        try:
                            w, h = map(int, args.out_res.lower().split("x"))
                            frame, final_boxes = resize_and_adjust_boxes(frame, final_boxes, (w, h))
                            img_h, img_w = frame.shape[:2]
                        except Exception:
                            print("Invalid out_res format, expected WIDTHxHEIGHT")

                    annotated = annotate_frame(frame.copy(), final_boxes)
                    if preview_callback:
                        preview_callback(annotated)
                    img_name = f"{frame_idx:06d}.jpg"
                    img_path = os.path.join(args.out, "images", img_name)
                    label_name = f"{frame_idx:06d}.txt"
                    label_path = os.path.join(args.out, "labels", label_name)
                    ann_path = os.path.join(args.out, "annotated", img_name)

                    save_labels(final_boxes, img_w, img_h, label_path)
                
                    if getattr(args, 'save_annotated', True):
                        cv2.imwrite(ann_path, annotated)
                
                    if getattr(args, 'save_raw', True):
                        cv2.imwrite(img_path, frame)

                    if save_callback:
                        save_callback(img_path, label_path, ann_path)
                    saved_count += 1

                frame_idx += 1

                # Throttled progress bar update
                now = time.time()
                if detect_progress_callback and (now - last_progress_time >= 0.1 or frame_idx >= total_frames):
                    last_progress_time = now
                    percent = min(100.0, (frame_idx / total_frames) * 100.0)
                    remaining = max(0.0, (total_frames - frame_idx) / fps)
                    detect_progress_callback(percent, eta=remaining)

            cap.release()
            print(f"Finished. Saved {saved_count} frames with labels to {args.out}")
            print(f"Skipped {skipped_duplicates} duplicate/similar frames")

        except Exception as e:
            print(f"Error in detection loop: {e}")
            raise e

    finally:
        # Shutdown executor
        try:
            executor.shutdown(wait=False)
        except:
            pass
            
        # Cleanup temporary YouTube file
        if args and getattr(args, "youtube", None):
            try:
                if video_path and os.path.exists(video_path):
                    os.remove(video_path)
                    print(f"Deleted temporary video: {video_path}")
            except Exception as e:
                print(f"Warning: could not delete {video_path}: {e}")


if __name__ == "__main__":
    main()
