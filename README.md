# 3D-Print-Image-Vectorizer

A lightweight Python utility designed to convert raster images (PNG, JPG, etc.) into **CAD-ready SVG files**.

Unlike standard tracers, this tool specifically handles **contour winding orders** (CCW for parents, CW for holes) and uses the `nonzero` fill rule. This ensures that when you import the resulting SVG into CAD software like **Fusion 360, Blender, or OpenSCAD**, the holes are automatically recognized as cutouts rather than solid overlapping shapes.

---

## Key Features

* **Color Quantization:** Uses K-Means clustering to reduce complex images into a specific number of flat color layers.
* **CAD-Optimized Paths:** Automatically detects nested contours (holes) and applies correct winding directions to ensure clean Booleans in 3D software.
* **Noise Reduction:** Filter out small artifacts using a minimum area threshold.
* **Smoothing:** Adjustable Polygon Approximation to reduce "stair-stepping" from pixels without losing geometry.
* **Preview Mode:** See how the colors are grouped before generating the final file.

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/3D-Print-Image-Vectorizer.git
cd 3D-Print-Image-Vectorizer

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


*Dependencies include: `opencv-python`, `numpy`, and `svgwrite`.*

---

## Usage

Run the script from your terminal:

```bash
python img2cad.py input_image.png [options]

```

### Command Line Arguments

| Argument | Short | Default | Description |
| --- | --- | --- | --- |
| `input` | N/A | **Required** | Path to the source image. |
| `--output` | `-o` | `input.svg` | Path for the generated SVG file. |
| `--colors` | `-c` | `3` | Number of distinct color layers to extract. |
| `--smoothness` | `-s` | `0.001` | Higher values create smoother, more rounded edges. |
| `--threshold` | `-t` | `50` | Minimum area (in pixels) for a shape to be kept. |
| `--preview` | N/A | False | Pops up an OpenCV window to show the quantized color layers. |

### Example

To convert a logo into a 2-color SVG with high smoothing:

```bash
python img2cad.py logo.jpg -c 2 -s 0.005 -t 100 --preview

```

---

## Workflow for 3D Printing

1. **Prepare:** Use a high-contrast image for best results.
2. **Vectorize:** Run `img2cad.py` with the `--preview` flag to ensure your shapes are distinct.
3. **Import:** Drag the `.svg` into your CAD software.
4. **Extrude:** Select the profiles. Because of the winding order logic in this script, your "holes" should be pre-subtracted from the main body.

---

## License

This project is open-source and available under the MIT License.
