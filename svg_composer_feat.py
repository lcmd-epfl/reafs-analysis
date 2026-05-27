import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union
from copy import deepcopy

# Register namespaces to avoid duplication
ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
ET.register_namespace("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
ET.register_namespace("dc", "http://purl.org/dc/elements/1.1/")


class SVGCanvas:
    """Simple SVG canvas for placing multiple SVGs at specific coordinates."""

    # SVG namespace
    SVG_NS = "http://www.w3.org/2000/svg"
    XLINK_NS = "http://www.w3.org/1999/xlink"

    def __init__(self, width: float = 1600, height: float = 1800):
        """Create a blank SVG canvas."""
        self.width = width
        self.height = height
        self.items = []  # List of (svg_element, x, y, width, height, label)

    @staticmethod
    def load_svg(filepath: Union[str, Path]) -> ET.Element:
        """Load an SVG file and return root element."""
        tree = ET.parse(filepath)
        return tree.getroot()

    @staticmethod
    def remove_matplotlib_patches(svg_elem: ET.Element):
        for child in svg_elem:
            if (
                child.tag.endswith("g")
                and "id" in child.attrib
                and child.attrib["id"].startswith("figure_")
            ):
                for subchild in list(child):
                    if (
                        subchild.tag.endswith("g")
                        and "id" in subchild.attrib
                        and subchild.attrib["id"].startswith("patch_")
                    ):
                        child.remove(subchild)

    def add_svg(
        self,
        filepath: Union[str, Path],
        x: float,
        y: float,
        remove_patches: bool = True,
    ):
        """
        Add an SVG to the canvas at specific coordinates (top-left corner).

        Args:
            filepath: Path to SVG file
            x, y: Coordinates where to place the SVG (no scaling)
            remove_patches: Remove matplotlib patches (default: True)
        """
        svg_elem = self.load_svg(filepath)

        if remove_patches:
            self.remove_matplotlib_patches(svg_elem)

        # Get viewBox to determine native size
        viewbox = svg_elem.get("viewBox")
        if viewbox:
            parts = viewbox.split()
            vb_x, vb_y, vb_w, vb_h = (
                float(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
            )
        else:
            vb_x, vb_y, vb_w, vb_h = 0, 0, 100, 100

        self.items.append(
            {
                "element": svg_elem,
                "x": x,
                "y": y,
                "vb": (vb_x, vb_y, vb_w, vb_h),
            }
        )

    def save(self, output_path: Union[str, Path]):
        """Save the composed SVG to a file."""
        # Create root SVG element
        root = ET.Element("{http://www.w3.org/2000/svg}svg")
        root.set("width", str(int(self.width)))
        root.set("height", str(int(self.height)))
        root.set("viewBox", f"0 0 {int(self.width)} {int(self.height)}")

        # Add white background
        # bg = ET.Element("{http://www.w3.org/2000/svg}rect")
        # bg.set("width", str(int(self.width)))
        # bg.set("height", str(int(self.height)))
        # bg.set("fill", "white")
        # root.append(bg)

        # Add each SVG as a group
        for item in self.items:
            group = ET.Element("{http://www.w3.org/2000/svg}g")

            svg_elem = item["element"]
            x, y = item["x"], item["y"]
            vb_x, vb_y, vb_w, vb_h = item["vb"]

            # Just translate to position, no scaling
            transform = f"translate({x - vb_x}, {y - vb_y})"
            group.set("transform", transform)

            # Deep copy all children from svg_elem
            for child in svg_elem:
                group.append(deepcopy(child))

            root.append(group)

        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        print(f"✓ Saved: {output_path}")


# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

# Canvas size (pixels)
CANVAS_WIDTH = 880
CANVAS_HEIGHT = 320

EXP = "lau"  # Used for file naming, e.g. "model_card_parity_plot_schoepfer_.svg"
ADDITIONAL_NAME = ""

# Output file
OUTPUT_FILE = Path(f"results/composed_{EXP}_{ADDITIONAL_NAME}.svg")

# List of SVGs to merge with their positions
# Format: (filepath, x, y)
SVG_FILES = [
    (Path(f"results/model_card_parity_plot_{EXP}_{ADDITIONAL_NAME}.svg"), 20, 0),
    (Path(f"results/model_card_pls_projection_{EXP}_{ADDITIONAL_NAME}.svg"), 584, 0),
    (Path(f"results/model_card_correlation_matrix_{EXP}_{ADDITIONAL_NAME}.svg"), 455, 17),
    (Path(f"results/model_card_validation_table_{EXP}_{ADDITIONAL_NAME}.svg"), 168, 20),
    (Path(f"results/model_card_vif_table_{EXP}_{ADDITIONAL_NAME}.svg"), 232, 160),
    (Path(f"results/model_card_equation_{EXP}_{ADDITIONAL_NAME}.svg"), 100, 185),
]

# ============================================================================
# EXECUTE - Don't change this
# ============================================================================

if __name__ == "__main__":
    # Create canvas
    canvas = SVGCanvas(width=CANVAS_WIDTH, height=CANVAS_HEIGHT)

    # Add each SVG
    print(f"Composing {len(SVG_FILES)} SVGs...\n")
    for filepath, x, y in SVG_FILES:
        if filepath.exists():
            print(f"  ✓ Adding {filepath.name} at ({x}, {y})")
            canvas.add_svg(filepath, x=x, y=y, remove_patches=True)
        else:
            print(f"  ✗ File not found: {filepath}")

    print(f"\nComposing canvas ({CANVAS_WIDTH}×{CANVAS_HEIGHT})...")
    canvas.save(OUTPUT_FILE)
    print(f"\n✅ Done! Open in Inkscape:\n   {OUTPUT_FILE}")
