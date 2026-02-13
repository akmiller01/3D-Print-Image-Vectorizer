import argparse
import cv2
import numpy as np
import svgwrite
import os
import sys

def quantize_colors(image, k=4):
    data = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    return result.reshape((image.shape)), center

def ensure_winding(contour, want_clockwise):
    """
    Checks the winding direction of a contour and flips it if necessary.
    Uses the signed area method. 
    OpenCV contourArea(oriented=True) returns:
    - Positive if Counter-Clockwise (CCW)
    - Negative if Clockwise (CW)
    (Note: This depends on coordinate system, but relative opposition is what matters)
    """
    area = cv2.contourArea(contour, oriented=True)
    is_ccw = area > 0
    
    # If we want CW but it's CCW -> Flip
    if want_clockwise and is_ccw:
        return np.flip(contour, axis=0)
    
    # If we want CCW but it's CW -> Flip
    if not want_clockwise and not is_ccw:
        return np.flip(contour, axis=0)
        
    return contour

def convert_to_svg(input_path, output_path, num_colors, min_area, smoothness, show_preview):
    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' was not found.")
        sys.exit(1)

    print(f"Reading {input_path}...")
    img = cv2.imread(input_path)
    if img is None:
        print("Error: Could not decode image file.")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"Quantizing to {num_colors} colors...")
    quantized_img, distinct_colors = quantize_colors(img_rgb, k=num_colors)

    if show_preview:
        print("Displaying preview (Press any key to continue)...")
        preview_bgr = cv2.cvtColor(quantized_img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Quantized Preview', preview_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    h, w, _ = img.shape
    dwg = svgwrite.Drawing(output_path, profile='tiny', size=(f"{w}px", f"{h}px"))
    
    print(f"Tracing contours (Min Area: {min_area}, Smoothness: {smoothness})...")
    
    for i, color in enumerate(distinct_colors):
        color_hex = "#%02x%02x%02x" % tuple(color)
        
        # Create mask for current color
        lower = np.array(color, dtype="uint8")
        upper = np.array(color, dtype="uint8")
        mask = cv2.inRange(quantized_img, lower, upper)

        # RETR_CCOMP retrieves hierarchy: [Next, Prev, First_Child, Parent]
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None:
            continue
        
        hierarchy = hierarchy[0]
        # Use fill-rule="nonzero" which relies on winding order (Standard CAD preference)
        layer_group = dwg.g(id=f"color_{i}_{color_hex.replace('#', '')}", fill=color_hex, stroke="none")
        
        processed_indices = set()

        for idx, cnt in enumerate(contours):
            if idx in processed_indices:
                continue

            # Check if this contour is a top-level Parent (No parent)
            if hierarchy[idx][3] == -1:
                
                # 1. Process Parent (Outer Shape)
                epsilon = smoothness * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if cv2.contourArea(approx) < min_area:
                    continue

                # FORCE Parent to be Counter-Clockwise (CCW)
                approx = ensure_winding(approx, want_clockwise=False)

                points = approx.reshape(-1, 2)
                path_data = f"M {points[0][0]},{points[0][1]} "
                for p in points[1:]:
                    path_data += f"L {p[0]},{p[1]} "
                path_data += "Z "

                # 2. Process Children (Holes)
                child_idx = hierarchy[idx][2]
                
                while child_idx != -1:
                    child_cnt = contours[child_idx]
                    
                    # Process Hole
                    epsilon_child = smoothness * cv2.arcLength(child_cnt, True)
                    approx_child = cv2.approxPolyDP(child_cnt, epsilon_child, True)
                    
                    if cv2.contourArea(approx_child) > min_area:
                        # FORCE Child to be Clockwise (CW) - Opposite of Parent
                        approx_child = ensure_winding(approx_child, want_clockwise=True)
                        
                        c_points = approx_child.reshape(-1, 2)
                        
                        # Append hole geometry to the SAME path string
                        path_data += f"M {c_points[0][0]},{c_points[0][1]} "
                        for p in c_points[1:]:
                            path_data += f"L {p[0]},{p[1]} "
                        path_data += "Z "
                    
                    processed_indices.add(child_idx)
                    child_idx = hierarchy[child_idx][0] # Next sibling

                # Add path with nonzero rule (uses the winding order we just enforced)
                layer_group.add(dwg.path(d=path_data, fill_rule="nonzero"))

        dwg.add(layer_group)

    dwg.save()
    print(f"Done! SVG saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert raster to CAD-ready SVG.")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output SVG path")
    parser.add_argument("-c", "--colors", type=int, default=3, help="Number of colors (Default: 3)")
    parser.add_argument("-s", "--smoothness", type=float, default=0.001, help="Smoothing factor (Default: 0.001)")
    parser.add_argument("-t", "--threshold", type=int, default=50, help="Noise threshold (Default: 50 px)")
    parser.add_argument("--preview", action="store_true", help="Preview color groups")

    args = parser.parse_args()

    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        output_path = f"{base_name}.svg"
    else:
        output_path = args.output

    convert_to_svg(args.input, output_path, args.colors, args.threshold, args.smoothness, args.preview)

if __name__ == "__main__":
    main()

