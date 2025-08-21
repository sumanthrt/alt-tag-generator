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
# DEVICE SETUP
# --------------------------
if torch.backends.mps.is_available():
    dtype = torch.float16
elif torch.cuda.is_available():
    dtype = torch.float16
else:
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
).to("cuda" if torch.cuda.is_available() else "cpu")

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

for folder, files in images_by_folder.items():
    results = []
    print(f"Processing folder: {folder if folder != '.' else '(root folder)'}")
    for img_file in tqdm(files, desc=f"Images in {folder}"):
        img_start = time.time()

        # Open and resize image for faster processing
        image = Image.open(img_file).convert("RGB")
        image.thumbnail((480, 480))

        inputs = processor(images=image, return_tensors="pt")
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        out = model.generate(**inputs, max_new_tokens=50)
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
    f"  <title>Alt Tags for {hotel_name}</title>",
    "  <style>",
    "    body { font-family: Arial, sans-serif; margin: 0; background: #f9fafc; color: #333; }",
    "    .container { display: flex; }",
    "    .sidebar { position: fixed; top: 0; left: 0; width: 220px; height: 100%; background: #2c3e50; color: white; overflow-y: auto; padding: 20px; box-sizing: border-box; }",
    "    .sidebar h2 { font-size: 18px; margin-top: 0; color: #ecf0f1; }",
    "    .sidebar a { display: block; color: #ecf0f1; text-decoration: none; margin: 8px 0; font-size: 14px; }",
    "    .sidebar a:hover { text-decoration: underline; }",
    "    .content { margin-left: 240px; padding: 20px; flex-grow: 1; }",
    "    h1 { text-align: center; color: #2c3e50; }",
    "    h2 { margin-top: 40px; color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }",
    "    table { border-collapse: collapse; width: 100%; margin-bottom: 40px; table-layout: fixed; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }",
    "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }",
    "    thead th { background-color: #3498db; color: white; position: sticky; top: 0; z-index: 10; }",
    "    img { width: 300px; height: 200px; object-fit: cover; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }",
    "    button { margin-top: 5px; padding: 4px 8px; font-size: 0.85rem; background: #3498db; color: white; border: none; border-radius: 3px; cursor: pointer; }",
    "    button:hover { background: #2980b9; }",
    "    td:nth-child(1) { width: 320px; }",
    "    td:nth-child(2) { width: 40%; }",
    "    td:nth-child(3) { width: 40%; }",
    "  </style>",
    "</head>",
    "<body>",
    "<div class='container'>",
    "<div class='sidebar'>",
    "  <h2>Table of Contents</h2>"
]

# Sidebar links
for folder in results_by_folder.keys():
    folder_title = folder if folder != "." else "(root folder)"
    html_lines.append(f"  <a href='#{sanitize_id(folder_title)}'>{folder_title}</a>")

html_lines.append("</div><div class='content'>")
html_lines.append(f"<h1>Alt Tags for {hotel_name}, {hotel_location}</h1>")

# Tables
for folder, results in results_by_folder.items():
    folder_title = folder if folder != "." else "(root folder)"
    html_lines.append(f"<h2 id='{sanitize_id(folder_title)}'>Folder: {folder_title}</h2>")
    html_lines.append("<table>")
    html_lines.append("  <thead><tr><th>Image</th><th>File Name</th><th>Alt Text</th></tr></thead><tbody>")

    for img_name, rel_path, alt_text in results:
        escaped_path = html.escape(rel_path)
        escaped_alt = html.escape(alt_text)
        html_lines.append("  <tr>")
        html_lines.append(f"    <td><img src='{escaped_path}' alt='{escaped_alt}'></td>")
        html_lines.append(
            f"    <td>{escaped_path}<br>"
            f"<button onclick=\"navigator.clipboard.writeText('{img_name}')\">Copy Name</button> "
            f"<button onclick=\"navigator.clipboard.writeText('{escaped_path}')\">Copy Path</button></td>"
        )
        html_lines.append(f"    <td>{escaped_alt}<br><button onclick=\"navigator.clipboard.writeText('{escaped_alt}')\">Copy Alt</button></td>")
        html_lines.append("  </tr>")

    html_lines.append("</tbody></table>")

# Stats
html_lines.append(f"<p><strong>Total Time:</strong> {total_time:.2f} seconds</p>")
html_lines.append(f"<p><strong>Average Time per Image:</strong> {avg_time:.2f} seconds</p>")
html_lines.append("</div></div></body></html>")

# Save report
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write("\n".join(html_lines))

print(f"All done! Output saved to {OUTPUT_HTML}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per image: {avg_time:.2f} seconds")

# --------------------------
# AUTO-OPEN IN BROWSER
# --------------------------
webbrowser.open(f"file://{OUTPUT_HTML.resolve()}")
