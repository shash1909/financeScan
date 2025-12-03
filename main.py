import os
import json
import torch
import gc
from PIL import Image
import easyocr
# transformers imports removed to keep this environment-light and avoid heavy TF/pyarrow deps
from huggingface_hub import login
from dotenv import load_dotenv
import sys
import io

# Fix Unicode encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables from .env file
load_dotenv()

# =========================================================
# CONFIGURATION
# =========================================================
DATA_DIR = "E:/Software/archive/SROIE2019/train"
IMG_DIR = os.path.join(DATA_DIR, "img")
ENTITY_DIR = os.path.join(DATA_DIR, "entities")
OUTPUT_DIR = os.path.join(DATA_DIR, "predictions")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# For testing, limit how many files to process. Set to None to process all.
# Set to None to process all files; otherwise limit for quick tests
MAX_FILES = None  # process all files

# =========================================================
# DEVICE SETUP
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Using device: {device}")

# Reduce memory footprint
torch.set_float32_matmul_precision('medium')
if device == "cuda":
    torch.cuda.empty_cache()

# =========================================================
# HUGGING FACE AUTHENTICATION
# =========================================================
token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    login(token=token)
    print("✅ Authenticated with Hugging Face")
else:
    print("⚠️ HUGGINGFACE_TOKEN not set - trying anonymous download")

# =========================================================
# INITIALIZE MODELS
# =========================================================
print("📦 Initializing OCR reader...")
try:
    # easyocr Reader accepts list of language codes and gpu flag
    reader = easyocr.Reader(['en'], gpu=(device == "cuda"))
    print("✅ Initialized EasyOCR reader")
except Exception as e:
    print(f"⚠️ Error initializing EasyOCR: {e}")
    print("Retrying EasyOCR without GPU...")
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        print("✅ Initialized EasyOCR reader (CPU)")
    except Exception as e2:
        print(f"❌ Failed to initialize EasyOCR: {e2}")
        raise

print("📦 LayoutLMv3 disabled — running OCR-only pipeline")
processor = None
model = None
MODEL_ENABLED = True

# Clear cache after loading
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()

# =========================================================
# OCR EXTRACTION
# =========================================================
def extract_ocr_data(image_path):
    """Extracts OCR text and bounding boxes using PaddleOCR."""
    # PaddleOCR API differs between versions. Try common methods and handle result shapes.
    words, boxes = [], []
    results = None
    # prefer `ocr` method if available
    if hasattr(reader, "ocr"):
        try:
            # cls=True enables text direction classification; safe default
            results = reader.ocr(image_path, cls=True)
        except TypeError:
            # some paddleocr versions expect different args
            results = reader.ocr(image_path)
    elif hasattr(reader, "readtext"):
        # older code path (kept for compatibility)
        results = reader.readtext(image_path, detail=1)
    else:
        raise RuntimeError("PaddleOCR instance has no supported OCR method (ocr/readtext)")

    # Normalize result formats to: list of (bbox, text, conf)
    normalized = []
    for item in results:
        # Format A: [bbox, (text, score)]
        if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple)):
            bbox = item[0]
            text = item[1][0]
            conf = item[1][1] if len(item[1]) > 1 else None
            normalized.append((bbox, text, conf))
        # Format B: (bbox, text, score)
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            bbox, text, conf = item
            normalized.append((bbox, text, conf))
        # Some versions return nested lists per line
        elif isinstance(item, (list, tuple)) and len(item) > 0:
            # try to extract plausible fields
            bbox = item[0]
            text = None
            conf = None
            # search for a (text, score) tuple in the rest
            for sub in item[1:]:
                if isinstance(sub, (list, tuple)) and len(sub) >= 1 and isinstance(sub[0], str):
                    text = sub[0]
                    conf = sub[1] if len(sub) > 1 else None
                    break
            if text is None:
                # fallback: join any strings found
                texts = [s for s in item if isinstance(s, str)]
                text = " ".join(texts) if texts else ""
            normalized.append((bbox, text, conf))
        else:
            # unknown format; skip
            continue

    confs = []
    for bbox, text, conf in normalized:
        if not text:
            continue
        t = text.strip()
        if not t:
            continue
        words.append(t)
        confs.append(conf)
        try:
            # bbox is usually 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x1, y1 = bbox[0]
            x2, y2 = bbox[2]
        except Exception:
            # if bbox is [x1,y1,x2,y2]
            try:
                x1, y1, x2, y2 = bbox
            except Exception:
                # as a last resort, compute bounding box from all points
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
        boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return words, boxes, confs

# =========================================================
# ENTITY PARSER
# =========================================================
def parse_entities(predictions):
    """Combine B-/I- tokens into structured entities."""
    entity_dict = {}
    for pred in predictions:
        label = pred["label"].replace("B-", "").replace("I-", "")
        if label not in entity_dict:
            entity_dict[label] = pred["token"]
        else:
            entity_dict[label] += " " + pred["token"]
    return entity_dict

# =========================================================
# MAIN LOOP: PROCESS ALL RECEIPTS
# =========================================================
processed_files = 0
for file in os.listdir(IMG_DIR):
    if not file.lower().endswith(".jpg"):
        continue

    image_path = os.path.join(IMG_DIR, file)
    entity_path = os.path.join(ENTITY_DIR, file.replace(".jpg", ".txt"))
    output_path = os.path.join(OUTPUT_DIR, file.replace(".jpg", "_pred.json"))

    print(f"\n🔍 Processing {file}...")

    try:
        # --- OCR ---
        words, boxes, confs = extract_ocr_data(image_path)
        if not words:
            print("⚠️ No OCR text detected — skipping file.")
            continue

        # --- Ground Truth (if available) ---
        gt_entities = {}
        if os.path.exists(entity_path):
            try:
                with open(entity_path, "r", encoding="utf8") as f:
                    gt_entities = json.load(f)
            except:
                gt_entities = {}

        # Model inference is disabled in OCR-only mode.
        structured_output = {}

        # --- Save Results ---
        out = {"ground_truth": gt_entities}
        # include OCR-only output (words/boxes/conf) so results are available
        ocr_items = []
        # We may have confidences for every OCR result; include them when available.
        for w, b, c in zip(words, boxes, confs):
            item = {"text": w, "box": b}
            if c is not None:
                try:
                    # ensure confidence is a float
                    item["conf"] = float(c)
                except Exception:
                    item["conf"] = c
            ocr_items.append(item)
        out["ocr"] = ocr_items

        if model is not None and structured_output:
            out["predicted"] = structured_output

        with open(output_path, "w", encoding="utf8") as f:
            json.dump(out, f, indent=2)

        print(f"✅ Saved results → {output_path}")

        # Clear memory after processing
        # encoding/outputs do not exist in OCR-only mode
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        processed_files += 1
        if MAX_FILES is not None and processed_files >= MAX_FILES:
            print(f"🛑 Reached MAX_FILES={MAX_FILES}; stopping early for test run")
            break

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
        continue

print("\n🎉 All receipts processed successfully!")
