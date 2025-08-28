import json
import os
import re
import boto3
import cv2
import numpy as np
from ultralytics import YOLO
from google.cloud import vision

# ---------------- Config ----------------
# Ensure /tmp exists
MODELS_DIR = "/tmp/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# S3 config
MODEL_BUCKET = "model-bucket-22"
s3 = boto3.client("s3")

# Model paths inside Lambda
PRODUCT_MODEL_PATH = os.path.join(MODELS_DIR, "product_best.pt")
PRICE_MODEL_PATH = os.path.join(MODELS_DIR, "pricetag_best.pt")
VISION_KEY_PATH = "/tmp/vision-api-key.json"


# Download models from S3 if not already present
def download_model_from_s3(model_name, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {model_name} from S3...")
        s3.download_file(MODEL_BUCKET, model_name, local_path)

download_model_from_s3("product_best.pt", PRODUCT_MODEL_PATH)
download_model_from_s3("pricetag_best.pt", PRICE_MODEL_PATH)
download_model_from_s3("vision-api-key.json", VISION_KEY_PATH)


# Load YOLO models
product_model = YOLO(PRODUCT_MODEL_PATH)
price_model = YOLO(PRICE_MODEL_PATH)

# Google Vision setup
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", VISION_KEY_PATH)
vision_client = vision.ImageAnnotatorClient()

# Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name="eu-west-1")

# Price regex
PRICE_REGEX = re.compile(
    r"(?i)\b(?:(SAR|SR)\s*([0-9]{1,4}(?:[.,][0-9]{1,2})?)|([0-9]{1,4}(?:[.,][0-9]{1,2})?)\s*(SAR|SR))\b"
)
# ---------------- Helpers ----------------
def extract_price_info(lines):
    for line in lines:
        m = PRICE_REGEX.search(line)
        if m:
            currency = (m.group(1) or m.group(4) or "").upper() or None
            value = (m.group(2) or m.group(3) or "").replace(",", ".") or None
            return {"price_value": value, "currency": currency, "raw_line": line}
    return {"price_value": None, "currency": None, "raw_line": None}

def run_vision(crop_bgr):
    ok, enc = cv2.imencode(".jpg", crop_bgr)
    if not ok:
        return {"text_lines": [], "text": "", "labels": [], "logos": []}
    image = vision.Image(content=enc.tobytes())
    response = vision_client.annotate_image({
        "image": image,
        "features": [
            {"type": vision.Feature.Type.DOCUMENT_TEXT_DETECTION},
            {"type": vision.Feature.Type.LABEL_DETECTION},
            {"type": vision.Feature.Type.LOGO_DETECTION},
        ],
    })
    full_text = getattr(response.full_text_annotation, "text", "") or ""
    lines = [ln for ln in full_text.splitlines() if ln.strip()]
    labels = [lab.description for lab in (response.label_annotations or [])]
    logos = [lg.description for lg in (response.logo_annotations or [])]
    return {"text_lines": lines, "text": "\n".join(lines), "labels": labels, "logos": logos}

def crop_and_vision(frame_bgr, boxes, want_price=False):
    out = []
    if boxes is None:
        return out
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame_bgr[y1:y2, x1:x2]
        vis = run_vision(crop)
        rec = {"bbox": [x1, y1, x2, y2], **vis}
        if want_price:
            price_info = extract_price_info(vis["text_lines"]) if vis["text_lines"] else {"price_value": None, "currency": None, "raw_line": None}
            rec.update(price_info)
        out.append(rec)
    return out

def iou_1d(a_min, a_max, b_min, b_max):
    inter = max(0, min(a_max, b_max) - max(a_min, b_min))
    denom = (a_max - a_min)
    return inter / denom if denom > 0 else 0

def bbox_match_products_to_prices_candidates(products, prices, vertical_thresh=250, min_overlap=0.1, max_candidates=3):
    matched = []
    for idx, prod in enumerate(products):
        px1, py1, px2, py2 = prod["bbox"]
        p_center_x = (px1 + px2) / 2
        p_bottom = py2
        candidates = []
        for price in prices:
            tx1, ty1, tx2, ty2 = price["bbox"]
            t_center_x = (tx1 + tx2) / 2
            vertical_dist = ty1 - p_bottom
            horizontal_dist = abs(p_center_x - t_center_x)
            overlap_ratio = iou_1d(px1, px2, tx1, tx2)
            if vertical_dist < -50 or vertical_dist > vertical_thresh or overlap_ratio < min_overlap:
                continue
            score = vertical_dist + 0.5 * horizontal_dist
            candidates.append({
                "score": score,
                "price_value": price["price_value"],
                "currency": price["currency"],
                "bbox": price["bbox"],
                "text": price["text"],
                "vertical_dist": vertical_dist,
                "horizontal_dist": horizontal_dist,
                "overlap_ratio": overlap_ratio
            })
        candidates = sorted(candidates, key=lambda c: c["score"])[:max_candidates]
        matched.append({
            "index": idx,
            "product_bbox": prod["bbox"],
            "product_text": prod["text"],
            "candidates": [
                {
                    "price_value": c["price_value"],
                    "currency": c["currency"],
                    "bbox": c["bbox"],
                    "price_text": c["text"],
                    "vertical_dist": c["vertical_dist"],
                    "horizontal_dist": c["horizontal_dist"],
                    "overlap_ratio": c["overlap_ratio"]
                } for c in candidates
            ]
        })
    return {"matched_products": matched}

def process_image(image_path: str):
    # optimized_path = flatten_image(image_path)
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image at {image_path}")
    product_results = product_model(image_path)
    price_results = price_model(image_path)
    product_boxes = product_results[0].boxes if product_results and len(product_results) else None
    price_boxes = price_results[0].boxes if price_results and len(price_results) else None
    product_data = crop_and_vision(frame, product_boxes, want_price=False)
    price_data = crop_and_vision(frame, price_boxes, want_price=True)
    matched = bbox_match_products_to_prices_candidates(product_data, price_data)
    return frame, matched

# ---------- Draw annotations ----------
def draw_matches(frame, matched):
    annotated = frame.copy()
    for item in matched.get("matched_products", []):
        prod_box = item.get("product_bbox")
        tag_box = item.get("price_bbox")
        if not prod_box or not tag_box:
            continue  # skip if missing
        px1, py1, px2, py2 = prod_box
        tx1, ty1, tx2, ty2 = tag_box
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
        cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)
        p_center = (int((px1 + px2) / 2), int((py1 + py2) / 2))
        t_center = (int((tx1 + tx2) / 2), int((ty1 + ty2) / 2))
        cv2.line(annotated, p_center, t_center, (255, 0, 0), 2)
    return annotated


def lambda_handler(event, context):
    # --- S3 input/output ---
    record = event["Records"][0]
    body = json.loads(record["body"])
    s3_info = body["Records"][0]["s3"]

    input_bucket = s3_info["bucket"]["name"]
    input_key = s3_info["object"]["key"]

    tmp_image_path = "/tmp/input_image.jpg"
    s3.download_file(input_bucket, input_key, tmp_image_path)

    # --- Process image ---
    frame, matched_json = process_image(tmp_image_path)

    # --- Prepare LLM prompt ---
    prompt = f'''
You are given a JSON list of products. Each product has:
- product_text: OCR text detected on the product
- product_bbox: bounding box of the product
- candidates: a list of candidate price tags, each with:
    - price_value
    - currency
    - bbox
    - price_text: OCR text detected on the price tag
    - vertical_dist: vertical distance from product
    - horizontal_dist: horizontal distance from product
    - overlap_ratio: horizontal overlap ratio with product

Your task:
1. For each product, choose the best price tag from the candidates using:
   - Semantic matching between product_text and price_text
   - Spatial closeness (vertical_dist, horizontal_dist, overlap_ratio)
   - Price/currency consistency
2. Extract item_name (Use the brand name in the item name as well as info from the pricetag), brand, and unit_of_measure (usually on price_text) from the product_text.
3. Preserve bounding boxes.
4. Ensure each price tag is assigned to only one product (remove duplicates), EACH PRICE TAG SHALL BE PRESENT ONCE.

Input JSON: {json.dumps(matched_json, ensure_ascii=False)}
Return only the JSON object "matched_products".
'''

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 15000,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    }

    response = bedrock.invoke_model(
        modelId="eu.anthropic.claude-sonnet-4-20250514-v1:0",
        body=json.dumps(payload)
    )
    output = json.loads(response['body'].read())

    # --- Extract JSON ---
    def clean_json_text(text):
        if not text:
            return None
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()
        return text

    parsed_json = None
    for item in output.get("content", []):
        if item.get("type") == "text":
            cleaned = clean_json_text(item["text"])
            try:
                parsed_json = json.loads(cleaned)
            except:
                start = cleaned.find("{")
                end = cleaned.rfind("}") + 1
                if start != -1 and end != -1:
                    parsed_json = json.loads(cleaned[start:end])

    if parsed_json is None:
        raise ValueError("Failed to parse JSON from LLM output")


    video_prefix = input_key.split("_")[0]  # e.g., "dairy4K"
    base_filename = os.path.basename(input_key).rsplit(".", 1)[0]  # strip extension
    frame_filename = f"{base_filename}.json"
    output_bucket = "video-analysis-results-1"
    output_key = f"{video_prefix}/json/{frame_filename}"

        # --- Annotate & upload image ---
    annotated = draw_matches(frame, parsed_json)
    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("Failed to encode annotated image")
    annotated_key = f"{video_prefix}/annotated/{base_filename}.jpg"
    s3.put_object(
        Bucket=output_bucket,
        Key=annotated_key,
        Body=encoded.tobytes(),
        ContentType="image/jpeg"
    )

    # --- Save result JSON to S3 ---
    s3.put_object(
        Bucket=output_bucket,
        Key=output_key,
        Body=json.dumps(parsed_json, ensure_ascii=False, indent=2),
        ContentType="application/json"
    )


    return {"status": "success", "output_key": output_key}
