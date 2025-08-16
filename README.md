# Alt Tag Generator

Generate SEO-friendly alt tags for images using **BLIP2-OPT-2.7B** with automatic handling of stock images, subfolders, and HTML output.

---

## **Features**

* Processes images recursively in all subfolders.
* Detects stock images (`istock-` prefix or `-st` suffix) and excludes hotel name/location.
* Generates alt tags in **sentence case**.
* Groups images by folder in separate HTML tables for easy viewing.
* Displays images, alt text, file path, and **copy buttons** in HTML.
* Supports **resized images** for faster processing (\~2–3× speedup).
* Tracks **total time** and **average time per image**.
* Works on **macOS/Linux** and **Windows**.

---

## **Requirements**

* Python 3.10+
* BLIP2-OPT-2.7B model (\~13GB)
* Dependencies (installed automatically by setup scripts):

  * `torch`
  * `transformers`
  * `Pillow`
  * `tqdm`
  * `accelerate`

---

## **Setup**

### macOS/Linux

```bash
chmod +x setup.sh run.sh
./setup.sh
```

This will:

1. Download the BLIP2 model if missing.
2. Install all Python dependencies.

---

### Windows

Double-click or run in Command Prompt:

```bat
setup.bat
```

---

## **Running the Generator**

### macOS/Linux

```bash
./run.sh
```

### Windows

```bat
run.bat
```

The script will prompt for:

1. **Hotel Name**
2. **Hotel Location**
3. **Image Folder Path** (drag-and-drop supported)

It will generate an HTML file like:

```
alt_tags_YYYYMMDD_HHMMSS.html
```

---

## **Output**

* **HTML file** with:

  * Separate tables for each subfolder.
  * Image previews.
  * File path and alt text.
  * Copy buttons for path and alt text.
* **Timing metrics** at the end:

  * Total time
  * Average time per image

---

## **Stock Image Rules**

* Image filename starts with `istock-` **OR** ends with `-st`.
* For stock images, the alt tag will **not** include hotel name or location.

---

## **Performance Tips**

* Images are resized to 480×480 pixels for faster processing.
* Processing time per image: \~10–15 seconds (depends on Mac specs).
* Using smaller images preserves alt text quality while speeding up generation.

---

## **Folder Structure Example**

```
project/
├─ alt_tag_generator.py
├─ setup.sh
├─ run.sh
├─ setup.bat
├─ run.bat
├─ blip2-opt-2.7b/       # downloaded model
└─ images/
   ├─ lobby/
   ├─ rooms/
   └─ restaurant/
```

Each subfolder will appear as a separate table in the HTML output.
