import os
import re
import time
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import html
import webbrowser

# --------------------------
# CONFIG
# --------------------------
MODEL_DIR = "./blip2-opt-2.7b"
MAX_NEW_TOKENS = 100  # adjust if you want longer captions

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
    exit(1)

# Place report inside the image folder itself
OUTPUT_HTML = image_folder_path / f"alt_tags_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

# --------------------------
# DEVICE + DTYPE SETUP
# --------------------------
if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
elif torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
    # CUDA perf knobs
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    device = "cpu"
    dtype = torch.float32

# --------------------------
# LOAD MODEL
# --------------------------
print("Loading BLIP2 (OPT-2.7B) from local folder...")
processor = Blip2Processor.from_pretrained(MODEL_DIR)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=dtype,
    device_map=None
).to(device)
model.eval()
print("Model loaded successfully!\n")

# --------------------------
# HELPER FUNCTIONS
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

# --------------------------
# GATHER IMAGE FILES RECURSIVELY
# --------------------------
image_files = [f for f in sorted(image_folder_path.rglob("*"))
               if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]

if not image_files:
    print("No images found in the folder or subfolders.")
    exit(1)

# --------------------------
# GROUP FILES BY SUBFOLDER
# --------------------------
images_by_folder = {}
for f in image_files:
    rel_folder = f.parent.relative_to(image_folder_path)
    images_by_folder.setdefault(str(rel_folder), []).append(f)

# --------------------------
# GENERATE ALT TAGS
# --------------------------
results_by_folder = {}
start_time = time.time()

with torch.inference_mode():
    for folder, files in images_by_folder.items():
        results = []
        print(f"Processing folder: {folder if folder != '.' else '(root folder)'}")
        for img_file in tqdm(files, desc=f"Images in {folder}", leave=False):
            img_start = time.time()

            # Open and resize image for faster processing
            image = Image.open(img_file).convert("RGB")
            image.thumbnail((480, 480), resample=Image.BILINEAR)

            # Prepare inputs on device
            inputs = processor(images=image, return_tensors="pt")
            if device == "cuda":
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generation (AMP for CUDA)
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            else:
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

            caption = processor.decode(out[0], skip_special_tokens=True)
            caption = sentence_case(caption.strip())

            if is_stock_image(img_file.name):
                alt_text = caption
            else:
                alt_text = f"{caption}, {hotel_name}, {hotel_location}"

            rel_path = os.path.relpath(img_file, image_folder_path)
            results.append((img_file.name, rel_path, alt_text))

            img_end = time.time()
            print(f"{img_file.name} processed in {img_end - img_start:.2f} seconds")

        results_by_folder[folder] = results

total_time = time.time() - start_time
avg_time = total_time / len(image_files)

# --------------------------
# CREATE HTML REPORT
# --------------------------
html_lines = [
    "<!DOCTYPE html>",
    "<html lang='en'>",
    "<head>",
    "  <meta charset='UTF-8'>",
    f"  <title>Alt Tags for {esc_attr(hotel_name)}</title>",
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

# Sidebar links
for folder in results_by_folder.keys():
    folder_title = folder if folder != "." else "(root folder)"
    html_lines.append(f"  <a href='#{sanitize_id(folder_title)}'>{html.escape(folder_title)}</a>")

html_lines.append("</div><div class='content' id='content'>")
html_lines.append(f"<h1>Alt Tags for {html.escape(hotel_name)}, {html.escape(hotel_location)}</h1>")

# Tables
for folder, results in results_by_folder.items():
    folder_title = folder if folder != "." else "(root folder)"
    html_lines.append(f"<h2 id='{sanitize_id(folder_title)}'>Folder: {html.escape(folder_title)}</h2>")
    html_lines.append("<table>")
    html_lines.append("  <thead><tr><th>Image</th><th>File Name</th><th>ALT Text</th></tr></thead><tbody>")

    for img_name, rel_path, alt_text in results:
        escaped_path = html.escape(rel_path)
        escaped_alt = html.escape(alt_text)

        # For attributes, escape quotes too
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

# Stats
html_lines.append(f"<p><strong>Total Time:</strong> {total_time:.2f} seconds</p>")
html_lines.append(f"<p><strong>Average Time per Image:</strong> {avg_time:.2f} seconds</p>")
html_lines.append("</div></div>")

# JavaScript for toggle (sticky, with a11y)
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

  // Keyboard shortcut: press "s" to toggle sidebar (ignore when typing)
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

# Unified copy handler for Name / Path / ALT (with "Copied!" feedback)
html_lines.append("""
<script>
  document.addEventListener('click', async (e) => {
    const btn = e.target.closest('button.copy-btn');
    if (!btn) return;

    let toCopy = btn.getAttribute('data-copy') || '';

    // If it's the Copy Path button, resolve relative -> absolute, OS-native path
    if (btn.classList.contains('copy-path')) {
      const rel = btn.getAttribute('data-rel') || '';
      const reportDir = new URL('.', window.location.href);
      const absURL = new URL(rel, reportDir);

      toCopy = absURL.href; // fallback to file URL
      try {
        let p = decodeURIComponent(absURL.pathname);
        const isWindows = navigator.platform.toLowerCase().startsWith('win');
        if (isWindows) {
          if (p.startsWith('/')) p = p.slice(1); // drop leading slash on Windows
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

# Save report
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write("\n".join(html_lines))

print(f"All done! Output saved to {OUTPUT_HTML}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per image: {avg_time:.2f} seconds")

# --------------------------
# AUTO-OPEN IN BROWSER
# --------------------------
try:
    webbrowser.open(f"file://{OUTPUT_HTML.resolve()}")
except Exception as e:
    print(f"Could not open browser automatically: {e}")
