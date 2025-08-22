import os
import re
import sys
import time
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import html
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext

# --------------------------
# CONFIG (tune these)
# --------------------------
MODEL_DIR = "./blip2-opt-2.7b"
MAX_NEW_TOKENS = 40

# Keep CPU/Windows behavior identical to before
BATCH_SIZE_CPU = 2            # CPU batch size (Windows path unchanged)
NUM_WORKERS = 4               # threads for PIL load/resize
TARGET_THUMB = (480, 480)     # quick pre-resize; HF processor still resizes appropriately
SET_TORCH_THREADS = True      # CPU-only tuning

# macOS (Apple Silicon) tunables
USE_MPS_IF_AVAILABLE = True
BATCH_SIZE_MAC = 2            # try 3–4 if VRAM allows; start with 2 for safety

# --------------------------
# USER INPUT
# --------------------------
print("🟢 Welcome to Alt Tag Generator")
hotel_name = input("Enter Hotel Name: ").strip()
hotel_location = input("Enter Hotel Location: ").strip()
image_folder = input("Enter Image Folder Path: ").strip()
image_folder_path = Path(image_folder)

if not image_folder_path.exists() or not image_folder_path.is_dir():
    print("Invalid image folder path.")
    raise SystemExit(1)

OUTPUT_HTML = image_folder_path / f"alt_tags_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

# --------------------------
# DEVICE SELECTION
# --------------------------
is_mac = (sys.platform == "darwin")
use_mps = (is_mac and USE_MPS_IF_AVAILABLE and torch.backends.mps.is_available())

if use_mps:
    device = "mps"
    dtype = torch.float16          # fp16 weights on MPS for speed
    eff_batch_size = BATCH_SIZE_MAC
    print("🚀 Using Apple Silicon MPS (fp16)")
else:
    # Preserve CPU-only behavior (even if CUDA is present)
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        print("⚠️ Accelerators detected, but using CPU path for parity.")
    device = "cpu"
    dtype = torch.float32
    eff_batch_size = BATCH_SIZE_CPU
    print("🧠 Using CPU")

# CPU thread tuning (unchanged)
if device == "cpu" and SET_TORCH_THREADS:
    try:
        cores = os.cpu_count() or 4
        use_threads = max(1, cores - (1 if cores >= 6 else 0))
        torch.set_num_threads(use_threads)
        torch.set_num_interop_threads(max(1, use_threads // 2))
        print(f"🧵 PyTorch threads set: {use_threads} (interop {max(1, use_threads // 2)})")
    except Exception as e:
        print(f"Could not set PyTorch threads: {e}")

# --------------------------
# LOAD MODEL
# --------------------------
print("Loading BLIP2 (OPT-2.7B) from local folder...")
processor = Blip2Processor.from_pretrained(MODEL_DIR)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=dtype,   # fp16 on MPS, fp32 on CPU
    device_map=None
).to(device)
model.eval()
print("Model loaded successfully!\n")

# --------------------------
# HELPERS
# --------------------------
def sanitize_id(name):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

def is_stock_image(filename):
    name_lower = filename.lower()
    return name_lower.startswith("istock-") or name_lower.endswith("-st" + Path(filename).suffix.lower())

def sentence_case(text):
    return re.sub(r'(^\s*\w|(?<=[.!?]\s)\w)', lambda m: m.group().upper(), text)

def esc_attr(text: str) -> str:
    return html.escape(text, quote=True)

def load_and_resize(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(TARGET_THUMB, resample=Image.BILINEAR)
    return img

# --------------------------
# GATHER IMAGE FILES
# --------------------------
image_files = [f for f in sorted(image_folder_path.rglob("*"))
               if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]

if not image_files:
    print("No images found in the folder or subfolders.")
    raise SystemExit(1)

# --------------------------
# GROUP BY SUBFOLDER
# --------------------------
images_by_folder = {}
for f in image_files:
    rel_folder = f.parent.relative_to(image_folder_path)
    images_by_folder.setdefault(str(rel_folder), []).append(f)

# --------------------------
# PRELOAD & RESIZE (PARALLEL)
# --------------------------
print("Preloading & resizing images...")
preloaded = {}
start_pre = time.time()
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
    fut_to_path = {ex.submit(load_and_resize, p): p for p in image_files}
    for fut in tqdm(as_completed(fut_to_path),
                    total=len(fut_to_path),
                    desc="Preprocess",
                    leave=True,
                    dynamic_ncols=True):
        p = fut_to_path[fut]
        try:
            preloaded[p] = fut.result()
        except Exception as e:
            tqdm.write(f"⚠️ Skipping {p}: {e}")
end_pre = time.time()
print(f"Preprocess time: {end_pre - start_pre:.2f}s\n")

# --------------------------
# GENERATE ALT TAGS (BATCHED)
# --------------------------
results_by_folder = {}
start_time = time.time()

# Autocast:
# - CPU: keep the older API for parity with your fast path
# - MPS: disable autocast (already running fp16 end-to-end)
if device == "cpu":
    autocast_ctx = torch.cpu.amp.autocast(dtype=torch.bfloat16)  # may show FutureWarning; safe
else:
    autocast_ctx = nullcontext()  # MPS path

with torch.inference_mode():
    for folder, files in images_by_folder.items():
        folder_title = folder if folder != "." else "(root folder)"
        tqdm.write(f"Processing folder: {folder_title}")

        results = []
        num_batches = (len(files) + eff_batch_size - 1) // eff_batch_size
        with tqdm(total=num_batches,
                  desc=f"Images in {folder}",
                  leave=True,
                  dynamic_ncols=True) as pbar:
            for bstart in range(0, len(files), eff_batch_size):
                batch_paths = files[bstart:bstart + eff_batch_size]
                pil_batch = [preloaded[p] for p in batch_paths if p in preloaded]
                if not pil_batch:
                    pbar.update(1)
                    continue

                inputs = processor(images=pil_batch, return_tensors="pt")

                # CPU-only optimization: channels_last helps oneDNN
                if device == "cpu":
                    pixel_values = inputs["pixel_values"].to(device)
                    pixel_values = pixel_values.to(memory_format=torch.channels_last).contiguous()
                else:
                    # MPS: move as fp16 to avoid casts
                    pixel_values = inputs["pixel_values"].to(device, dtype=torch.float16)

                inputs["pixel_values"] = pixel_values

                t0 = time.time()
                with autocast_ctx:
                    out = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        use_cache=True
                    )
                t1 = time.time()

                # Decode per item
                for j, p in enumerate(batch_paths):
                    try:
                        caption = processor.decode(out[j], skip_special_tokens=True).strip()
                    except Exception:
                        caption = processor.decode(out[0], skip_special_tokens=True).strip()

                    caption = sentence_case(caption)
                    alt_text = caption if is_stock_image(p.name) else f"{caption}, {hotel_name}, {hotel_location}"
                    rel_path = os.path.relpath(p, image_folder_path)
                    results.append((p.name, rel_path, alt_text))

                tqdm.write(f"[Batch {bstart//eff_batch_size + 1}/{num_batches}] "
                           f"{', '.join(bp.name for bp in batch_paths)} in {t1 - t0:.2f}s")
                pbar.update(1)

                # Optional: trim MPS cache occasionally on long runs
                if device == "mps" and ((bstart // eff_batch_size) % 8 == 0):
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass

        results_by_folder[folder] = results

total_time = time.time() - start_time
num_imgs = sum(len(v) for v in results_by_folder.values())
avg_time = total_time / max(1, num_imgs)

# --------------------------
# CREATE HTML REPORT
# --------------------------
def esc(s): return html.escape(s)

html_lines = [
    "<!DOCTYPE html>",
    "<html lang='en'>",
    "<head>",
    "  <meta charset='UTF-8'>",
    f"  <title>Alt Tags for {esc(hotel_name)}</title>",
    "  <style>",
    "    :root { --sidebar-w: 220px; --gap: 20px; }",
    "    body { font-family: Arial, sans-serif; margin: 0; background: #f9fafc; color: #333; }",
    "    .container { display: flex; }",
    "    .sidebar { position: fixed; top: 0; left: 0; width: var(--sidebar-w); height: 100%; background: #145a32; color: white; overflow-y: auto; padding: 20px; box-sizing: border-box; transition: transform 0.3s ease; }",
    "    .sidebar.collapsed { transform: translateX(-100%); }",
    "    .sidebar h2 { font-size: 18px; margin-top: 0; color: #eafaf1; }",
    "    .sidebar a { display: block; color: #eafaf1; text-decoration: none; margin: 8px 0; font-size: 14px; }",
    "    .sidebar a:hover { text-decoration: underline; }",
    "    .content { margin-left: calc(var(--sidebar-w) + var(--gap)); padding: 20px; flex-grow: 1; transition: margin-left 0.3s ease; }",
    "    .content.fullwidth { margin-left: var(--gap); }",
    "    h1 { text-align: center; color: #145a32; }",
    "    h2 { margin-top: 40px; color: #1e8449; border-bottom: 2px solid #27ae60; padding-bottom: 5px; }",
    "    table { border-collapse: collapse; width: 100%; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }",
    "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: middle; }",
    "    thead th { background-color: #27ae60; color: white; position: sticky; top: 0; z-index: 10; }",
    "    img { width: 250px; height: 180px; object-fit: cover; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }",
    "    button { margin-top: 5px; padding: 4px 8px; font-size: 0.85rem; background: #27ae60; color: white; border: none; border-radius: 3px; cursor: pointer; min-width: 120px;}",
    "    button:hover { background: #1e8449; }",
    "    #toggleBtn { position: fixed; top: 45px; left: 12px; z-index: 1100; background: #27ae60; color: white; border: none; border-radius: 6px; padding: 8px 12px; cursor: pointer; font-size: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }",
    "    #toggleBtn:hover { background: #1e8449; }",
    "    body.sidebar-open #toggleBtn { left: calc(var(--sidebar-w) + 12px); }",
    "    td:first-child { width: 20%; text-align: center; }",
    "    td:nth-child(2) { width: 40%; }",
    "    td:nth-child(3) { width: 40%; }",
    "    @media (max-width: 768px) { body.sidebar-open #toggleBtn { left: 12px; } }",
    "  </style>",
    "</head>",
    "<body class='sidebar-open'>",
    "<button id='toggleBtn' aria-expanded='true' aria-controls='sidebar'>☰ Toggle Sidebar</button>",
    "<div class='container'>",
    "<div class='sidebar' id='sidebar'>",
    "  <h2>Table of Contents</h2>"
]

for folder in results_by_folder.keys():
    folder_title = folder if folder != "." else "(root folder)"
    html_lines.append(f"  <a href='#{sanitize_id(folder_title)}'>{esc(folder_title)}</a>")

html_lines.append("</div><div class='content' id='content'>")
html_lines.append(f"<h1>Alt Tags for {esc(hotel_name)}, {esc(hotel_location)}</h1>")

for folder, results in results_by_folder.items():
    folder_title = folder if folder != "." else "(root folder)"
    html_lines.append(f"<h2 id='{sanitize_id(folder_title)}'>Folder: {esc(folder_title)}</h2>")
    html_lines.append("<table>")
    html_lines.append("  <thead><tr><th>Image</th><th>File Name</th><th>ALT Text</th></tr></thead><tbody>")

    for img_name, rel_path, alt_text in results:
        escaped_path = esc(rel_path)
        escaped_alt = esc(alt_text)
        name_attr = esc_attr(img_name)
        alt_attr  = esc_attr(alt_text)
        path_attr = esc_attr(rel_path)

        html_lines.append("  <tr>")
        html_lines.append(f"    <td><img src='{escaped_path}' alt='{escaped_alt}'></td>")
        html_lines.append(
            "    <td>"
            f"{escaped_path}<br>"
            f"<button class='copy-btn' data-copy='{name_attr}'>Copy Name</button> "
            f"<button class='copy-btn copy-path' data-rel='{escaped_path}' data-copy='{path_attr}'>Copy Path</button>"
            "</td>"
        )
        html_lines.append(
            f"    <td>{escaped_alt}<br><button class='copy-btn' data-copy='{alt_attr}'>Copy ALT Text</button></td>"
        )
        html_lines.append("  </tr>")

    html_lines.append("</tbody></table>")

html_lines.append(f"<p><strong>Preprocess Time:</strong> {end_pre - start_pre:.2f} seconds</p>")
html_lines.append(f"<p><strong>Total Inference Time:</strong> {total_time:.2f} seconds</p>")
html_lines.append(f"<p><strong>Average Time per Image:</strong> {avg_time:.2f} seconds</p>")
html_lines.append("</div></div>")

html_lines.append("""
<script>
  const toggleBtn = document.getElementById('toggleBtn');
  const sidebar = document.getElementById('sidebar');
  const content = document.getElementById('content');
  const body = document.body;

  function updateAria() {
    const expanded = !sidebar.classList.contains('collapsed');
    toggleBtn.setAttribute('aria-expanded', expanded ? 'true' : 'false');
  }

  toggleBtn.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
    content.classList.toggle('fullwidth');
    body.classList.toggle('sidebar-open');
    updateAria();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key.toLowerCase() === 's' && !e.ctrlKey && !e.metaKey && !e.altKey) {
      const t = (e.target.tagName || '').toLowerCase();
      const typing = t === 'input' || t === 'textarea' || e.target.isContentEditable;
      if (!typing) toggleBtn.click();
    }
  });

  updateAria();
</script>
""")

html_lines.append("""
<script>
  document.addEventListener('click', async (e) => {
    const btn = e.target.closest('button.copy-btn');
    if (!btn) return;

    let toCopy = btn.getAttribute('data-copy') || '';

    if (btn.classList.contains('copy-path')) {
      const rel = btn.getAttribute('data-rel') || '';
      const reportDir = new URL('.', window.location.href);
      const absURL = new URL(rel, reportDir);
      toCopy = absURL.href;
      try {
        let p = decodeURIComponent(absURL.pathname);
        const isWindows = navigator.platform.toLowerCase().startsWith('win');
        if (isWindows) {
          if (p.startsWith('/')) p = p.slice(1);
          p = p.replace(/\\//g, '\\\\');
        }
        toCopy = p;
      } catch (_) {}
    }

    const old = btn.textContent;
    try {
      await navigator.clipboard.writeText(toCopy);
      btn.textContent = 'Copied!';
      setTimeout(() => btn.textContent = old, 900);
    } catch (err) {
      btn.textContent = 'Copy failed';
      setTimeout(() => btn.textContent = old, 1200);
    }
  });
</script>
""")

html_lines.append("</body></html>")

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write("\n".join(html_lines))

print(f"All done! Output saved to {OUTPUT_HTML}")
print(f"Preprocess time: {end_pre - start_pre:.2f} seconds")
print(f"Total inference time: {total_time:.2f} seconds")
print(f"Average time per image: {avg_time:.2f} seconds")

try:
    webbrowser.open(f"file://{OUTPUT_HTML.resolve()}")
except Exception as e:
    print(f"Could not open browser automatically: {e}")
Ma
