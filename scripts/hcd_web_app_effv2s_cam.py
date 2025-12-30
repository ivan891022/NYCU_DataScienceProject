#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HCD EfficientNetV2-S Web App with Grad-CAM + LLM explanation (bright UI, 中文解釋)

- 使用你已訓練好的 checkpoint:
    outputs/effv2s_web_nostain_nocolor/tf_efficientnetv2_s_best.pth
- 首頁：
    * 上傳 .tif/.tiff/.png/.jpg 的 H&E patch
    * 顯示模型簡介與 Grad-CAM 說明
- 結果頁：
    * 顯示上傳 patch
    * 顯示 Grad-CAM overlay
    * 顯示 癌症機率 ＋ Positive / Negative
    * 「Generate AI Explanation (beta)」按鈕：
        - 呼叫 /llm_explain，請 LLM 用中文解釋熱力圖與提醒醫師注意的區域
        - 專有名詞保留英文，其餘使用自然繁體中文
        - 若 LLM 無法使用，退回 rule-based 中文說明

啟動方式（在 DataScienceProject 專案根目錄）：

    (hcd) uvicorn scripts.hcd_web_app_effv2s_cam:app --host 0.0.0.0 --port 8000 --reload

瀏覽器開啟：

    http://<你的 server IP>:8000
"""

import io
import os
import base64
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import torch
import albumentations as A
import albumentations.pytorch as AP
import timm

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse

# -------------------------------------------------------------------
# Google Gemini（透過 OpenAI 相容 API）用於中文解釋
#   - 需要：pip install --upgrade openai
#   - 金鑰請放在環境變數 GEMINI_API_KEY，不要寫在程式碼裡
# -------------------------------------------------------------------
try:
    from openai import OpenAI  # pip install openai
    _GEMINI_SDK_AVAILABLE = True
except Exception:
    OpenAI = None
    _GEMINI_SDK_AVAILABLE = False

LLM_MODEL_NAME = "gemini-2.5-flash"

_client = None
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if _GEMINI_SDK_AVAILABLE and _GEMINI_API_KEY:
    try:
        _client = OpenAI(
            api_key=_GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        print("Google Gemini Client initialized successfully.")
    except Exception:
        _client = None
        print("Failed to initialize Google Gemini Client.")
else:
    print("OpenAI SDK not available or GEMINI_API_KEY missing.")


# ---------- 路徑與基本設定 ----------
CKPT_PATH = Path("outputs/effv2s_web_nostain_nocolor/tf_efficientnetv2_s_best.pth")
IMG_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 驗證 / 推論的 transform，要跟訓練時一致
val_tf = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(
            mean=(0.67, 0.45, 0.69),
            std=(0.23, 0.21, 0.22),
        ),
        AP.ToTensorV2(),
    ]
)


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """PIL.Image -> (1,3,H,W) tensor (已 resize + normalize)."""
    img = pil_img.convert("RGB")
    arr = np.array(img)
    t = val_tf(image=arr)["image"]  # (3,H,W)
    t = t.unsqueeze(0)  # (1,3,H,W)
    return t


# ---------- 建 model 並載入 ckpt ----------
def build_model_for_inference(ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = timm.create_model(
        "tf_efficientnetv2_s",
        pretrained=False,
        in_chans=3,
        num_classes=1,
        drop_rate=0.2,
        drop_path_rate=0.2,
    )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    thr = float(state.get("thr", 0.5))
    return model, thr


model, BEST_THR = build_model_for_inference(CKPT_PATH, device)


def make_base64_from_array(arr: np.ndarray) -> str:
    """np.uint8(H,W,3) -> base64 PNG string."""
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def make_base64_from_pil(pil_img: Image.Image, max_side: int = 256) -> str:
    """PIL image -> base64 (會縮放到較小邊 max_side，適合顯示用)."""
    img = pil_img.convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def summarize_hotspot_region(cam_resized: np.ndarray):
    """
    根據 Grad-CAM map 粗略描述「熱區在 patch 的哪裡」以及「高 attention 區域大約占比」。
    回傳:
        region_en: e.g. "lower-right region"
        region_zh: e.g. "偏右下方區域"
        hotspot_ratio: 高於 0.6 的區域比例 (0~1)
    """
    h, w = cam_resized.shape
    if h == 0 or w == 0:
        return "unspecified region", "未明確局部區域", 0.0

    hotspot_mask = cam_resized >= 0.6
    hotspot_ratio = float(hotspot_mask.mean())

    total = cam_resized.sum()
    if total <= 1e-8:
        return "diffuse/low-intensity pattern", "整片皆為較低、分散的活化訊號", hotspot_ratio

    ys, xs = np.indices((h, w))
    cx = float((cam_resized * xs).sum() / total)
    cy = float((cam_resized * ys).sum() / total)

    # 英文方向
    horiz = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "central"
    vert = "upper" if cy < h / 3 else "lower" if cy > 2 * w / 3 else "central"

    # 中文方向
    horiz_zh = "左側" if horiz == "left" else "右側" if horiz == "right" else "中央"
    vert_zh = "上方" if vert == "upper" else "下方" if vert == "lower" else "中央"

    if vert == "central" and horiz == "central":
        region_en = "central region"
        region_zh = "中央區域"
    elif vert == "central":
        region_en = f"central-{horiz} region"
        region_zh = f"偏{horiz_zh}的中央區域"
    elif horiz == "central":
        region_en = f"{vert}-central region"
        region_zh = f"偏{vert_zh}的中央區域"
    else:
        region_en = f"{vert}-{horiz} region"
        region_zh = f"偏{vert_zh}{horiz_zh}的區域"

    return region_en, region_zh, hotspot_ratio


def gradcam_for_image(pil_img: Image.Image):
    """
    對單一 patch 做 Grad-CAM:
    - target layer: model.conv_head (EfficientNetV2-S 的最後 conv)
    - 輸出:
        prob (float), predicted_label (0/1),
        overlay_base64 (heatmap overlay),
        orig_base64 (縮放後原圖),
        hotspot_region_en,
        hotspot_region_zh,
        hotspot_ratio (0~1)
    """
    model.eval()

    x = preprocess_image(pil_img).to(device)  # (1,3,H,W)

    activations = {}
    gradients = {}

    target_layer = model.conv_head

    def fwd_hook(module, inp, out):
        activations["value"] = out  # (1,C,Hc,Wc)

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]  # (1,C,Hc,Wc)

    handle_fwd = target_layer.register_forward_hook(fwd_hook)
    handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    with torch.enable_grad():
        logits = model(x)  # (1,1)
        prob = torch.sigmoid(logits)[0, 0]
        score = logits[0, 0]

    score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    acts = activations["value"][0]       # (C,Hc,Wc)
    grads = gradients["value"][0]        # (C,Hc,Wc)

    weights = grads.mean(dim=(1, 2))     # (C,)

    cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))  # (Hc,Wc)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    cam_np = cam.detach().cpu().numpy()

    w, h = pil_img.size
    cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_CUBIC)

    hotspot_region_en, hotspot_region_zh, hotspot_ratio = summarize_hotspot_region(
        cam_resized
    )

    # 轉成 heatmap overlay
    heatmap = (cam_resized * 255.0).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # BGR
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    img_rgb = np.array(pil_img.convert("RGB"))
    overlay = 0.45 * heatmap_color + 0.55 * img_rgb
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    prob_val = float(prob.item())
    pred = 1 if prob_val >= BEST_THR else 0

    overlay_b64 = make_base64_from_array(overlay)
    orig_b64 = make_base64_from_pil(pil_img, max_side=256)

    return (
        prob_val,
        pred,
        overlay_b64,
        orig_b64,
        hotspot_region_en,
        hotspot_region_zh,
        hotspot_ratio,
    )


# ---------- 中文 rule-based & LLM explanation ----------

def rule_based_explanation(
    prob, threshold, pred_label, hotspot_region_zh, hotspot_ratio
):
    """沒有 LLM 或當作 fallback 時，一段中文說明。"""
    prob_pct = prob * 100.0
    area_pct = hotspot_ratio * 100.0

    if prob < threshold * 0.7:
        risk_txt = "模型估計此 patch 的轉移性腫瘤機率偏低"
    elif prob < threshold * 1.1:
        risk_txt = "模型估計此 patch 的轉移性腫瘤機率介於臨界值附近"
    else:
        risk_txt = "模型估計此 patch 的轉移性腫瘤機率偏高"

    if area_pct < 10:
        area_txt = "範圍相對較小的區域"
    elif area_pct < 35:
        area_txt = "中等範圍的區域"
    else:
        area_txt = "佔比較大的區域"

    pred_txt = (
        "在目前決策 threshold 下被判為 positive (cancer)"
        if pred_label == "positive"
        else "在目前決策 threshold 下被判為 negative (benign)"
    )

    return (
        f"模型輸出對應的癌症機率約為 {prob_pct:.1f}%，{pred_txt}，決策 threshold 約為 {threshold:.3f}。"
        f"{risk_txt}。Grad-CAM 顯示模型主要關注在 {hotspot_region_zh}，高活化區大約佔整張 patch 的 "
        f"{area_txt}（約 {area_pct:.1f}%）。這些被標示的區域代表模型認為較具轉移性腫瘤特徵的地方，"
        "可作為醫師放大檢視與標記時的輔助參考，但仍須搭配完整切片與臨床資訊綜合判讀。"
        "本說明僅供研究與教學使用，不能單獨作為任何醫療診斷結論。"
    )


def generate_llm_explanation(
    prob,
    threshold,
    pred_label,
    hotspot_region_en,
    hotspot_region_zh,
    hotspot_ratio,
):
    """
    嘗試呼叫 Gemini 產生中文說明；若 LLM 不可用，改用 rule-based 解釋。
    """
    base_text = rule_based_explanation(
        prob, threshold, pred_label, hotspot_region_zh, hotspot_ratio
    )

    if _client is None:
        return (
            base_text
            + "\n\n（目前伺服器未啟用 Gemini API，以上為規則式產生的範例說明。"
            "請確認已安裝 openai 套件並已在 GEMINI_API_KEY 設定金鑰。）"
        )

    try:
        user_prompt = f"""
你是一位協助病理科醫師解讀 AI 模型結果的助理。

模型資訊：
- Model: EfficientNetV2-S，針對 lymph node H&E patch 做 metastatic cancer (positive) vs benign (negative) 的二元分類。
- Cancer probability (0~1): {prob:.4f}
- Decision threshold: {threshold:.4f}
- Prediction label: {pred_label}
- Approximate Grad-CAM hotspot region (英文描述): {hotspot_region_en}
- Approximate Grad-CAM hotspot region (中文描述): {hotspot_region_zh}
- Approximate proportion of patch covered by high-attention area: {hotspot_ratio*100:.1f}%

請依照以上資訊，撰寫一段 3～5 句的說明，請注意：

1. 回覆請使用「繁體中文」。
2. 保留英文專有名詞（例如 EfficientNetV2-S、Grad-CAM、H&E、metastasis、lymph node、positive、negative 等），不要翻譯成中文。
3. 內容應該：
    - 一般性地說明這個機率與 Grad-CAM heatmap 代表的意義（不要對個案做確定診斷）。
    - 描述模型主要關注的區域位置，並提醒醫師在該區域可以特別放大檢視。
    - 強調這只是 AI 模型的輔助資訊，必須結合完整切片與臨床資訊，由醫師自行判讀。
4. 不要提出任何治療建議或預後評估。
5. 不要使用「診斷」或「確定」的語氣，只能說「模型顯示」、「可能暗示」、「建議進一步人工檢視」等。

最後再加上一句清楚的免責聲明，說明此說明僅供研究與教育使用，不能作為單獨診斷依據。
"""
        resp = _client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一個謹慎的醫療 AI 說明助手，只負責解釋模型輸出，不提供診斷或治療建議。"
                        "你會用中立、保守的語氣描述模型結果，並且一定會提醒使用者：這只是研究性質的工具，"
                        "不能單獨作為臨床診斷依據。"
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        explanation = resp.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        return base_text + f"\n\n（呼叫 Gemini API 失敗，訊息：{e}）"


# ---------- FastAPI app ----------
app = FastAPI(title="HCD EfficientNetV2-S (Grad-CAM + 中文 LLM explanation)")


@app.get("/", response_class=HTMLResponse)
async def index():
    html = f"""
    <html>
      <head>
        <title>HCD Lymph Node Cancer Assistant</title>
        <style>
          * {{
            box-sizing: border-box;
          }}
          body {{
            margin: 0;
            padding: 0;
            background: #f3f4f6;
            color: #111827;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }}
          .nav {{
            background: #ffffff;
            padding: 16px 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 10px rgba(15,23,42,0.08);
            position: sticky;
            top: 0;
            z-index: 20;
          }}
          .nav-left {{
            display: flex;
            align-items: center;
            gap: 10px;
          }}
          .logo-circle {{
            width: 32px;
            height: 32px;
            border-radius: 999px;
            background: radial-gradient(circle at 30% 30%, #38bdf8, #0ea5e9);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ecfeff;
            font-weight: 900;
            font-size: 18px;
          }}
          .nav-title {{
            font-size: 20px;
            font-weight: 700;
            letter-spacing: 0.02em;
            color: #0f172a;
          }}
          .badge {{
            font-size: 12px;
            padding: 4px 10px;
            border-radius: 999px;
            background: #e0f2fe;
            border: 1px solid #bae6fd;
            color: #0369a1;
            display: flex;
            align-items: center;
            gap: 6px;
          }}
          .chip-dot {{
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #22c55e;
          }}
          .container {{
            max-width: 980px;
            margin: 32px auto 40px auto;
            padding: 0 16px;
          }}
          .card {{
            background: #ffffff;
            border-radius: 24px;
            padding: 28px 32px 26px 32px;
            box-shadow: 0 20px 50px rgba(15,23,42,0.12);
            border: 1px solid #e5e7eb;
          }}
          h1 {{
            margin-top: 0;
            margin-bottom: 4px;
            font-size: 24px;
            font-weight: 700;
            color: #0f172a;
          }}
          .subtitle {{
            font-size: 14px;
            color: #4b5563;
            margin-bottom: 14px;
          }}
          .upload-area {{
            margin-top: 20px;
            padding: 20px 20px 16px 20px;
            border-radius: 18px;
            border: 1px dashed #cbd5f5;
            background: #f9fafb;
          }}
          .upload-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
          }}
          .upload-icon {{
            width: 26px;
            height: 26px;
            border-radius: 999px;
            background: #e0f2fe;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #0284c7;
          }}
          input[type=file] {{
            margin-top: 6px;
            font-size: 13px;
          }}
          .btn-primary {{
            margin-top: 18px;
            padding: 9px 22px;
            background: linear-gradient(135deg, #38bdf8, #0ea5e9);
            color: #f9fafb;
            border-radius: 999px;
            border: none;
            cursor: pointer;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.03em;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            position: relative;
            overflow: hidden;
          }}
          .btn-primary svg {{
            width: 18px;
            height: 18px;
          }}
          .btn-primary:hover {{
            filter: brightness(1.05);
          }}
          .info-grid {{
            display: grid;
            grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
            gap: 18px;
            margin-top: 22px;
          }}
          .info-card {{
            background: #f9fafb;
            border-radius: 18px;
            padding: 16px 18px;
            border: 1px solid #e5e7eb;
          }}
          .info-title {{
            font-size: 13px;
            font-weight: 600;
            color: #0f172a;
            margin-bottom: 6px;
          }}
          .info-text {{
            font-size: 13px;
            color: #4b5563;
            line-height: 1.6;
          }}
          .model-chip {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            background: #eff6ff;
            color: #1d4ed8;
            font-size: 12px;
            font-weight: 500;
            margin-top: 4px;
          }}
          .model-chip svg {{
            width: 14px;
            height: 14px;
          }}
          .disclaimer {{
            margin-top: 20px;
            font-size: 11px;
            color: #6b7280;
          }}
          .ripple-btn {{
            position: relative;
            overflow: hidden;
          }}
          .ripple {{
            position: absolute;
            border-radius: 50%;
            transform: scale(0);
            animation: ripple 600ms linear;
            background-color: rgba(255,255,255,0.55);
            pointer-events: none;
          }}
          @keyframes ripple {{
            to {{
              transform: scale(4);
              opacity: 0;
            }}
          }}
        </style>
      </head>
      <body>
        <div class="nav">
          <div class="nav-left">
            <div class="logo-circle">H</div>
            <div class="nav-title">HCD Lymph Node Cancer Assistant</div>
          </div>
          <div class="badge">
            <span class="chip-dot"></span>
            EfficientNetV2-S · Grad-CAM
          </div>
        </div>
        <div class="container">
          <div class="card">
            <h1>Upload a Histopathology Patch</h1>
            <p class="subtitle">
              Upload a <b>lymph node H&amp;E patch</b> (.tif / .tiff / .png / .jpg) to estimate cancer probability
              and visualize model attention using <b>Grad-CAM</b>.
            </p>

            <div class="upload-area">
              <div class="upload-header">
                <div class="upload-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M4 14a4 4 0 0 1 3.5-3.96A5 5 0 0 1 20 11a3 3 0 0 1-.5 5.96H7"></path>
                    <path d="M12 16V8"></path>
                    <path d="m9 11 3-3 3 3"></path>
                  </svg>
                </div>
                <div style="font-size:13px;color:#111827;font-weight:600;">
                  Choose a patch image
                </div>
              </div>
              <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept=".tif,.tiff,.png,.jpg,.jpeg" required />
                <br/>
                <button class="btn-primary ripple-btn" type="submit">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 5v14"></path>
                    <path d="M5 12h14"></path>
                  </svg>
                  Upload &amp; Run Analysis
                </button>
              </form>
            </div>

            <div class="info-grid">
              <div class="info-card">
                <div class="info-title">Model</div>
                <div class="info-text">
                  Patch-level binary classifier trained on the
                  <b>Histopathologic Cancer Detection</b> dataset to distinguish
                  metastatic vs. benign lymph node patches.
                  <div class="model-chip">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <rect x="3" y="4" width="18" height="12" rx="2" ry="2"></rect>
                      <path d="M7 20h10"></path>
                      <path d="M9 16v4"></path>
                      <path d="M15 16v4"></path>
                    </svg>
                    EfficientNetV2-S · ImageNet-1k init
                  </div>
                </div>
              </div>
              <div class="info-card">
                <div class="info-title">What the heatmap means</div>
                <div class="info-text">
                  Grad-CAM highlights areas that contribute most to a <b>cancer</b> prediction.
                  Warmer colors (red / yellow) indicate stronger model attention.
                  It does <b>not</b> draw precise tumor boundaries, but offers visual support
                  during slide review.
                </div>
              </div>
            </div>

            <p class="disclaimer">
              This is a research prototype and <b>not a diagnostic device</b>.
              Predictions and heatmaps are for educational and exploratory use only
              and must not be used as the sole basis for clinical decisions.
            </p>
          </div>
        </div>

        <script>
          document.addEventListener("DOMContentLoaded", function () {{
            const buttons = document.querySelectorAll(".ripple-btn");
            buttons.forEach((btn) => {{
              btn.addEventListener("click", function (e) {{
                const rect = btn.getBoundingClientRect();
                const diameter = Math.max(btn.clientWidth, btn.clientHeight);
                const radius = diameter / 2;
                const circle = document.createElement("span");
                circle.classList.add("ripple");
                circle.style.width = circle.style.height = diameter + "px";
                circle.style.left = e.clientX - rect.left - radius + "px";
                circle.style.top = e.clientY - rect.top - radius + "px";

                const existing = btn.getElementsByClassName("ripple")[0];
                if (existing) existing.remove();
                btn.appendChild(circle);
              }});
            }});
          }});
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as e:
        return HTMLResponse(
            content=f"<h3>Failed to read image: {e}</h3>", status_code=400
        )

    (
        prob,
        pred,
        overlay_b64,
        orig_b64,
        hotspot_region_en,
        hotspot_region_zh,
        hotspot_ratio,
    ) = gradcam_for_image(pil_img)

    label_str = "POSITIVE (CANCER)" if pred == 1 else "NEGATIVE (BENIGN)"
    pred_label_for_js = "positive" if pred == 1 else "negative"
    prob_pct = max(0.0, min(100.0, prob * 100.0))

    html = f"""
    <html>
      <head>
        <title>Prediction Result - HCD Assistant</title>
        <style>
          * {{
            box-sizing: border-box;
          }}
          body {{
            margin: 0;
            padding: 0;
            background: #f3f4f6;
            color: #111827;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }}
          .nav {{
            background: #ffffff;
            padding: 16px 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 10px rgba(15,23,42,0.08);
            position: sticky;
            top: 0;
            z-index: 20;
          }}
          .nav-left {{
            display: flex;
            align-items: center;
            gap: 10px;
          }}
          .logo-circle {{
            width: 32px;
            height: 32px;
            border-radius: 999px;
            background: radial-gradient(circle at 30% 30%, #38bdf8, #0ea5e9);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ecfeff;
            font-weight: 900;
            font-size: 18px;
          }}
          .nav-title {{
            font-size: 20px;
            font-weight: 700;
            letter-spacing: 0.02em;
            color: #0f172a;
          }}
          .badge {{
            font-size: 12px;
            padding: 4px 10px;
            border-radius: 999px;
            background: #e0f2fe;
            border: 1px solid #bae6fd;
            color: #0369a1;
          }}
          .container {{
            max-width: 1040px;
            margin: 32px auto 40px auto;
            padding: 0 16px;
          }}
          .card {{
            background: #ffffff;
            border-radius: 24px;
            padding: 26px 30px 26px 30px;
            box-shadow: 0 20px 50px rgba(15,23,42,0.12);
            border: 1px solid #e5e7eb;
          }}
          h2 {{
            margin-top: 0;
            font-size: 22px;
            font-weight: 700;
            color: #0f172a;
          }}
          .file-caption {{
            font-size: 13px;
            color: #6b7280;
            margin-bottom: 10px;
          }}
          .grid {{
            display: grid;
            grid-template-columns: minmax(0, 1.1fr) minmax(0, 1.1fr);
            gap: 24px;
            margin-top: 16px;
          }}
          .img-panel {{
            background: #f9fafb;
            border-radius: 18px;
            padding: 14px 14px 12px 14px;
            border: 1px solid #e5e7eb;
          }}
          .img-title {{
            font-size: 13px;
            margin-bottom: 8px;
            color: #4b5563;
          }}
          img {{
            max-width: 100%;
            border-radius: 12px;
            border: 1px solid #d1d5db;
          }}
          .result-main {{
            margin-top: 18px;
          }}
          .prob-label {{
            font-size: 13px;
            color: #6b7280;
            margin-bottom: 4px;
          }}
          .prob-value {{
            font-size: 24px;
            font-weight: 700;
            color: #0f172a;
          }}
          .prob-bar-bg {{
            width: 100%;
            height: 12px;
            border-radius: 999px;
            background: #e5e7eb;
            margin-top: 10px;
            overflow: hidden;
          }}
          .prob-bar-fill {{
            height: 100%;
            width: {prob_pct:.1f}%;
            background: linear-gradient(90deg, #22c55e, #facc15, #ef4444);
          }}
          .label-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            margin-top: 14px;
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            background: {"rgba(34,197,94,0.12)" if pred == 0 else "rgba(239,68,68,0.12)"};
            border: 1px solid {"#16a34a" if pred == 0 else "#f97373"};
            color: {"#166534" if pred == 0 else "#b91c1c"};
          }}
          .label-pill svg {{
            width: 16px;
            height: 16px;
          }}
          .bottom-row {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            margin-top: 20px;
          }}
          .model-note {{
            font-size: 12px;
            color: #6b7280;
          }}
          .btn-base {{
            padding: 9px 22px;
            border-radius: 999px;
            border: none;
            color: #ecfeff;
            font-size: 13px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            letter-spacing: 0.03em;
            font-weight: 600;
            position: relative;
            overflow: hidden;
          }}
          .btn-back {{
            background: #0ea5e9;
          }}
          .btn-explain {{
            background: #22c55e;
          }}
          .btn-base svg {{
            width: 16px;
            height: 16px;
          }}
          .btn-base:hover {{
            filter: brightness(1.05);
          }}
          .btn-base:disabled {{
            opacity: 0.6;
            cursor: default;
          }}
          /* 右下角浮動的「Back to Upload」按鈕 */
          .floating-back-btn {{
            position: fixed;
            right: 24px;
            bottom: 24px;
            padding: 10px 20px;
            border-radius: 999px;
            background: #0ea5e9;
            color: #ecfeff;
            font-size: 13px;
            font-weight: 600;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 18px 40px rgba(15,23,42,0.28);
            border: none;
          }}
          .floating-back-btn svg {{
            width: 16px;
            height: 16px;
          }}
          .disclaimer {{
            margin-top: 14px;
            font-size: 11px;
            color: #6b7280;
          }}
          .ripple-btn {{
            position: relative;
            overflow: hidden;
          }}
          .ripple {{
            position: absolute;
            border-radius: 50%;
            transform: scale(0);
            animation: ripple 600ms linear;
            background-color: rgba(255,255,255,0.6);
            pointer-events: none;
          }}
          @keyframes ripple {{
            to {{
              transform: scale(4);
              opacity: 0;
            }}
          }}
          a {{
            text-decoration: none;
          }}
          .explain-card {{
            margin-top: 22px;
            background: #f9fafb;
            border-radius: 18px;
            padding: 16px 18px;
            border: 1px solid #e5e7eb;
          }}
          .explain-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
          }}
          .explain-title {{
            font-size: 14px;
            font-weight: 600;
            color: #0f172a;
          }}
          .explain-status {{
            font-size: 11px;
            color: #6b7280;
          }}
          .explain-text {{
            font-size: 13px;
            color: #4b5563;
            line-height: 1.6;
            white-space: pre-wrap;
          }}
          .explain-disclaimer {{
            margin-top: 8px;
            font-size: 11px;
            color: #6b7280;
          }}
          /* ===== Loading 動畫樣式 ===== */
          .spinner {{
            width: 14px;
            height: 14px;
            border-radius: 999px;
            border: 2px solid rgba(248,250,252,0.35);
            border-top-color: #ecfeff;
            animation: spin 0.7s linear infinite;
            margin-right: 6px;
            display: none;
          }}
          .btn-loading .spinner {{
            display: inline-block;
          }}
          .btn-loading .btn-text {{
            opacity: 0.9;
          }}
          @keyframes spin {{
            to {{
              transform: rotate(360deg);
            }}
          }}
          .explain-loading {{
            display: none;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: #6b7280;
            margin-bottom: 6px;
          }}
          .dot-pulse {{
            display: inline-flex;
            gap: 4px;
          }}
          .dot-pulse span {{
            width: 6px;
            height: 6px;
            border-radius: 999px;
            background: #22c55e;
            opacity: 0.3;
            animation: dotPulse 1s ease-in-out infinite;
          }}
          .dot-pulse span:nth-child(2) {{
            animation-delay: 0.12s;
          }}
          .dot-pulse span:nth-child(3) {{
            animation-delay: 0.24s;
          }}
          @keyframes dotPulse {{
            0%, 100% {{
              transform: translateY(0);
              opacity: 0.3;
            }}
            50% {{
              transform: translateY(-3px);
              opacity: 1;
            }}
          }}
        </style>
      </head>
      <body>
        <div class="nav">
          <div class="nav-left">
            <div class="logo-circle">H</div>
            <div class="nav-title">HCD Lymph Node Cancer Assistant</div>
          </div>
          <div class="badge">Prediction &amp; Explainability</div>
        </div>
        <div class="container">
          <div class="card">
            <h2>Prediction Result</h2>
            <p class="file-caption">
              File: <span style="color:#111827;">{file.filename}</span>
            </p>

            <div class="grid">
              <div class="img-panel">
                <div class="img-title">Uploaded Patch (rescaled for display)</div>
                <img src="data:image/png;base64,{orig_b64}" alt="uploaded patch"/>
              </div>
              <div class="img-panel">
                <div class="img-title">Grad-CAM Heatmap (regions contributing most to cancer prediction)</div>
                <img src="data:image/png;base64,{overlay_b64}" alt="gradcam overlay"/>
              </div>
            </div>

            <div class="result-main">
              <div class="prob-label">Cancer probability (patch-level)</div>
              <div class="prob-value">{prob:.4f}</div>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill"></div>
              </div>
              <div class="label-pill">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                  <path d="M9 11l2 2 4-4"></path>
                </svg>
                {label_str}
              </div>
            </div>

            <div class="bottom-row">
              <div class="model-note">
                Model: <b>EfficientNetV2-S</b> (ImageNet-1k pretrained, fine-tuned on HCD) ·
                patch-level metastasis detection with Grad-CAM explainability.
              </div>
              <div style="display:flex; gap:8px; align-items:center;">
                <button id="explain-btn" class="btn-base btn-explain ripple-btn" type="button">
                  <span class="spinner" id="explain-spinner"></span>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 3v18"></path>
                    <path d="M5 12h14"></path>
                  </svg>
                  <span class="btn-text">Generate AI Explanation (beta)</span>
                </button>
              </div>
            </div>

            <div id="explain-card" class="explain-card" style="display:none;">
              <div class="explain-header">
                <span class="explain-title">AI 產生的中文說明</span>
                <span id="explain-status" class="explain-status">尚未產生說明</span>
              </div>
              <div id="explain-loading" class="explain-loading">
                <div class="dot-pulse">
                  <span></span><span></span><span></span>
                </div>
                <span class="explain-loading-text">大型語言模型正在產生中文說明…</span>
              </div>
              <p id="explain-text" class="explain-text">
點擊「Generate AI Explanation (beta)」後，系統會依照目前的機率與 Grad-CAM 熱力圖，自動產生一段中文說明與提醒重點。
              </p>
              <p class="explain-disclaimer">
                本段說明由大型語言模型產生，僅供研究與教學參考，不構成醫療診斷或治療建議，
                也不得單獨作為臨床決策的依據。
              </p>
            </div>

            <p class="disclaimer">
              Heatmaps 反映的是 <b>model attention</b>，並非精確的腫瘤邊界。
              解讀時請務必結合原始切片與完整臨床資訊，由合格醫師自行判讀。
            </p>
          </div>
        </div>

        <a href="/" class="floating-back-btn ripple-btn">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M15 18l-6-6 6-6"></path>
          </svg>
          <span>Back to Upload</span>
        </a>

        <script>
          const PRED_PROB = {prob:.6f};
          const PRED_THRESHOLD = {BEST_THR:.6f};
          const PRED_LABEL = "{pred_label_for_js}";
          const HOTSPOT_REGION_EN = "{hotspot_region_en}";
          const HOTSPOT_REGION_ZH = "{hotspot_region_zh}";
          const HOTSPOT_RATIO = {hotspot_ratio:.6f};

          document.addEventListener("DOMContentLoaded", function () {{
            const buttons = document.querySelectorAll(".ripple-btn");
            buttons.forEach((btn) => {{
              btn.addEventListener("click", function (e) {{
                const rect = btn.getBoundingClientRect();
                const diameter = Math.max(btn.clientWidth, btn.clientHeight);
                const radius = diameter / 2;
                const circle = document.createElement("span");
                circle.classList.add("ripple");
                circle.style.width = circle.style.height = diameter + "px";
                circle.style.left = e.clientX - rect.left - radius + "px";
                circle.style.top = e.clientY - rect.top - radius + "px";

                const existing = btn.getElementsByClassName("ripple")[0];
                if (existing) existing.remove();
                btn.appendChild(circle);
              }});
            }});

            const explainBtn = document.getElementById("explain-btn");
            const explainSpinner = document.getElementById("explain-spinner");
            const explainCard = document.getElementById("explain-card");
            const explainText = document.getElementById("explain-text");
            const explainStatus = document.getElementById("explain-status");
            const loadingRow = document.getElementById("explain-loading");

            if (explainBtn) {{
              explainBtn.addEventListener("click", async function () {{
                explainBtn.disabled = true;
                explainBtn.classList.add("btn-loading");
                if (explainSpinner) {{
                  explainSpinner.style.display = "inline-block";
                }}
                explainStatus.textContent = "正在產生說明…";
                if (explainCard) {{
                  explainCard.style.display = "block";
                }}
                if (loadingRow) {{
                  loadingRow.style.display = "flex";
                }}
                if (explainText) {{
                  explainText.textContent = "大型語言模型正在根據當前的機率與 Grad-CAM 熱力圖產生中文說明，請稍候…";
                }}

                try {{
                  const resp = await fetch("/llm_explain", {{
                    method: "POST",
                    headers: {{
                      "Content-Type": "application/json"
                    }},
                    body: JSON.stringify({{
                      probability: PRED_PROB,
                      threshold: PRED_THRESHOLD,
                      prediction_label: PRED_LABEL,
                      hotspot_region_en: HOTSPOT_REGION_EN,
                      hotspot_region_zh: HOTSPOT_REGION_ZH,
                      hotspot_ratio: HOTSPOT_RATIO
                    }})
                  }});
                  const data = await resp.json();
                  if (data.ok) {{
                    explainText.textContent = data.explanation;
                    explainStatus.textContent = "AI 已產生中文說明（僅供研究使用）";
                  }} else {{
                    explainText.textContent = data.error || "產生說明時發生錯誤。";
                    explainStatus.textContent = "Error";
                  }}
                }} catch (e) {{
                  explainText.textContent = "無法連線至說明服務。";
                  explainStatus.textContent = "Error";
                }} finally {{
                  explainBtn.disabled = false;
                  explainBtn.classList.remove("btn-loading");
                  if (explainSpinner) {{
                    explainSpinner.style.display = "none";
                  }}
                  if (loadingRow) {{
                    loadingRow.style.display = "none";
                  }}
                }}
              }});
            }}
          }});
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/api/predict", response_class=JSONResponse)
async def api_predict(file: UploadFile = File(...)):
    """純 JSON 版 API（回傳機率、Grad-CAM 與熱區資訊）。"""
    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents))
    (
        prob,
        pred,
        overlay_b64,
        _,
        hotspot_region_en,
        hotspot_region_zh,
        hotspot_ratio,
    ) = gradcam_for_image(pil_img)
    return {
        "filename": file.filename,
        "probability": prob,
        "threshold": BEST_THR,
        "prediction": int(pred),
        "prediction_label": "positive" if pred == 1 else "negative",
        "gradcam_overlay_base64": overlay_b64,
        "hotspot_region_en": hotspot_region_en,
        "hotspot_region_zh": hotspot_region_zh,
        "hotspot_ratio": hotspot_ratio,
    }


@app.post("/llm_explain", response_class=JSONResponse)
async def llm_explain(request: Request):
    """接收前端資訊並產生中文說明。"""
    data = await request.json()
    prob = float(data.get("probability", 0.0))
    threshold = float(data.get("threshold", BEST_THR))
    pred_label = str(data.get("prediction_label", "unknown"))
    hotspot_region_en = str(data.get("hotspot_region_en", "unspecified region"))
    hotspot_region_zh = str(data.get("hotspot_region_zh", "未明確局部區域"))
    hotspot_ratio = float(data.get("hotspot_ratio", 0.0))

    explanation = generate_llm_explanation(
        prob,
        threshold,
        pred_label,
        hotspot_region_en,
        hotspot_region_zh,
        hotspot_ratio,
    )
    return {"ok": True, "explanation": explanation}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("hcd_web_app_effv2s_cam:app", host="0.0.0.0", port=8000, reload=True)
