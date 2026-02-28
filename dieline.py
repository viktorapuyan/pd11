"""
Carton Dieline Generator Module

This module generates RSC (Regular Slotted Container) carton dielines based on 
object dimensions. It automatically adds clearance to dimensions and creates a 
flat pattern layout suitable for die-cutting and folding.

Author: Python Software Engineer
Date: January 29, 2026
"""

from typing import Tuple, List, Dict, Any
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path


class DielinePanel:
    """Represents a single panel in the dieline with its coordinates."""
    
    def __init__(self, name: str, x: float, y: float, width: float, height: float):
        """
        Initialize a panel.
        
        Args:
            name: Panel identifier (e.g., 'front', 'left_flap_top')
            x: X-coordinate of bottom-left corner
            y: Y-coordinate of bottom-left corner
            width: Panel width
            height: Panel height
        """
        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def get_corners(self) -> List[Tuple[float, float]]:
        """Return the four corners of the panel as (x, y) tuples."""
        return [
            (self.x, self.y),                          # Bottom-left
            (self.x + self.width, self.y),             # Bottom-right
            (self.x + self.width, self.y + self.height),  # Top-right
            (self.x, self.y + self.height)             # Top-left
        ]
    
    def __repr__(self) -> str:
        return f"DielinePanel('{self.name}', x={self.x}, y={self.y}, w={self.width}, h={self.height})"


class CartonDieline:
    """
    Represents a complete RSC carton dieline with all panels and flaps.
    
    An RSC dieline consists of:
    - Bottom panel (base)
    - Four side panels (front, back, left, right)
    - Four top flaps
    - Four bottom flaps
    """
    
    def __init__(self, length: float, width: float, height: float):
        """
        Initialize the dieline structure.
        
        Args:
            length: Internal length of the carton (cm)
            width: Internal width of the carton (cm)
            height: Internal height of the carton (cm)
        """
        self.length = length
        self.width = width
        self.height = height
        self.panels: List[DielinePanel] = []
        self.fold_lines: List[Dict[str, Any]] = []
        self.cut_lines: List[Dict[str, Any]] = []
    
    def add_panel(self, panel: DielinePanel):
        """Add a panel to the dieline."""
        self.panels.append(panel)
    
    def add_fold_line(self, x1: float, y1: float, x2: float, y2: float):
        """Add a fold line to the dieline."""
        self.fold_lines.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    
    def add_cut_line(self, x1: float, y1: float, x2: float, y2: float):
        """Add a cut line to the dieline."""
        self.cut_lines.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Calculate the bounding box of the entire dieline.
        
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        if not self.panels:
            return (0, 0, 0, 0)
        
        all_x = []
        all_y = []
        for panel in self.panels:
            corners = panel.get_corners()
            all_x.extend([c[0] for c in corners])
            all_y.extend([c[1] for c in corners])
        
        return (min(all_x), min(all_y), max(all_x), max(all_y))
    
    def __repr__(self) -> str:
        return f"CartonDieline(L={self.length}, W={self.width}, H={self.height}, panels={len(self.panels)})"


def adjust_dimensions(length: float, width: float, height: float, 
                      clearance: float = 6.0) -> Tuple[float, float, float]:
    """
    Adjust raw object dimensions by adding clearance.
    
    Business Rule: Add 6 cm to each dimension for proper fit and handling.
    
    Args:
        length: Measured object length in cm
        width: Measured object width in cm
        height: Measured object height in cm
        clearance: Clearance to add to each dimension (default: 6.0 cm)
    
    Returns:
        Tuple of (adjusted_length, adjusted_width, adjusted_height)
    
    Example:
        >>> adjust_dimensions(30, 20, 15)
        (36.0, 26.0, 21.0)
    """
    adjusted_length = length + clearance
    adjusted_width = width + clearance
    adjusted_height = height + clearance
    
    return (adjusted_length, adjusted_width, adjusted_height)


def calculate_flap_size(box_width: float, flap_ratio: float = 0.5) -> float:
    """
    Calculate the flap size based on box width.
    
    Standard practice: Flaps are typically 50% of the box width.
    
    Args:
        box_width: Width of the box
        flap_ratio: Ratio of flap to width (default: 0.5 for 50%)
    
    Returns:
        Flap size in cm
    """
    return box_width * flap_ratio


def generate_rsc_dieline(length: float, width: float, height: float) -> CartonDieline:
    """
    Generate a complete RSC (Regular Slotted Container) dieline.
    
    The RSC layout consists of:
    - A continuous strip of panels: front, bottom, back, top
    - Side panels (left and right) attached to the bottom
    - Flaps attached to front, back, left, and right panels
    
    Layout (flat view):
    
           [Left Flap]     [Back Flap]    [Right Flap]   [Front Flap]
    [Left]   [Front]        [Bottom]        [Back]         [Top]      [Right]
           [Left Flap]     [Back Flap]    [Right Flap]   [Front Flap]
    
    Args:
        length: Internal carton length (cm)
        width: Internal carton width (cm)
        height: Internal carton height (cm)
    
    Returns:
        CartonDieline object containing all panels and coordinates
    """
    dieline = CartonDieline(length, width, height)
    
    # Calculate flap sizes
    # Top and bottom flaps extend from the length panels
    flap_length = calculate_flap_size(width)
    
    # Starting position for layout
    start_x = 0
    start_y = flap_length  # Offset to accommodate bottom flaps
    
    # ========== MAIN PANEL STRIP (horizontal) ==========
    # This creates the continuous strip: Left side -> Front -> Bottom -> Back -> Top
    
    current_x = start_x
    
    # 1. LEFT SIDE PANEL
    left_panel = DielinePanel('left_side', current_x, start_y, width, height)
    dieline.add_panel(left_panel)
    current_x += width
    
    # 2. FRONT PANEL (length x height)
    front_panel = DielinePanel('front', current_x, start_y, length, height)
    dieline.add_panel(front_panel)
    current_x += length
    
    # 3. BOTTOM PANEL (width x length) - rotated 90°
    bottom_panel = DielinePanel('bottom', current_x, start_y, width, height)
    dieline.add_panel(bottom_panel)
    current_x += width
    
    # 4. BACK PANEL (length x height)
    back_panel = DielinePanel('back', current_x, start_y, length, height)
    dieline.add_panel(back_panel)
    current_x += length
    
    # 5. TOP PANEL (width x length) - rotated 90°
    top_panel = DielinePanel('top', current_x, start_y, width, height)
    dieline.add_panel(top_panel)
    current_x += width
    
    # 6. RIGHT SIDE PANEL (with glue flap)
    right_panel = DielinePanel('right_side', current_x, start_y, width, height)
    dieline.add_panel(right_panel)
    
    # ========== FLAPS ==========
    
    # BOTTOM FLAPS (attached to left, front, back, right)
    # Left bottom flap
    left_bottom_flap = DielinePanel('left_bottom_flap', 
                                    start_x, start_y - flap_length, 
                                    width, flap_length)
    dieline.add_panel(left_bottom_flap)
    
    # Front bottom flaps (two half-width flaps)
    front_bottom_flap_1 = DielinePanel('front_bottom_flap_1',
                                       start_x + width, start_y - flap_length,
                                       length / 2, flap_length)
    dieline.add_panel(front_bottom_flap_1)
    
    front_bottom_flap_2 = DielinePanel('front_bottom_flap_2',
                                       start_x + width + length / 2, start_y - flap_length,
                                       length / 2, flap_length)
    dieline.add_panel(front_bottom_flap_2)
    
    # Back bottom flaps (two half-width flaps)
    back_x_start = start_x + width + length + width
    back_bottom_flap_1 = DielinePanel('back_bottom_flap_1',
                                      back_x_start, start_y - flap_length,
                                      length / 2, flap_length)
    dieline.add_panel(back_bottom_flap_1)
    
    back_bottom_flap_2 = DielinePanel('back_bottom_flap_2',
                                      back_x_start + length / 2, start_y - flap_length,
                                      length / 2, flap_length)
    dieline.add_panel(back_bottom_flap_2)
    
    # Right bottom flap
    right_x_start = start_x + width + length + width + length + width
    right_bottom_flap = DielinePanel('right_bottom_flap',
                                     right_x_start, start_y - flap_length,
                                     width, flap_length)
    dieline.add_panel(right_bottom_flap)
    
    # TOP FLAPS (attached to left, front, back, right)
    # Left top flap
    left_top_flap = DielinePanel('left_top_flap',
                                 start_x, start_y + height,
                                 width, flap_length)
    dieline.add_panel(left_top_flap)
    
    # Front top flaps (two half-width flaps)
    front_top_flap_1 = DielinePanel('front_top_flap_1',
                                    start_x + width, start_y + height,
                                    length / 2, flap_length)
    dieline.add_panel(front_top_flap_1)
    
    front_top_flap_2 = DielinePanel('front_top_flap_2',
                                    start_x + width + length / 2, start_y + height,
                                    length / 2, flap_length)
    dieline.add_panel(front_top_flap_2)
    
    # Back top flaps (two half-width flaps)
    back_top_flap_1 = DielinePanel('back_top_flap_1',
                                   back_x_start, start_y + height,
                                   length / 2, flap_length)
    dieline.add_panel(back_top_flap_1)
    
    back_top_flap_2 = DielinePanel('back_top_flap_2',
                                   back_x_start + length / 2, start_y + height,
                                   length / 2, flap_length)
    dieline.add_panel(back_top_flap_2)
    
    # Right top flap
    right_top_flap = DielinePanel('right_top_flap',
                                  right_x_start, start_y + height,
                                  width, flap_length)
    dieline.add_panel(right_top_flap)
    
    # ========== FOLD LINES ==========
    # Add vertical fold lines between main panels
    fold_x_positions = [
        start_x + width,                              # Between left and front
        start_x + width + length,                     # Between front and bottom
        start_x + width + length + width,             # Between bottom and back
        start_x + width + length + width + length,    # Between back and top
        start_x + width + length + width + length + width  # Between top and right
    ]
    
    for fold_x in fold_x_positions:
        dieline.add_fold_line(fold_x, start_y - flap_length, 
                             fold_x, start_y + height + flap_length)
    
    # Add horizontal fold lines for flaps
    dieline.add_fold_line(start_x, start_y, 
                         start_x + width + length + width + length + width + width, start_y)
    dieline.add_fold_line(start_x, start_y + height,
                         start_x + width + length + width + length + width + width, start_y + height)
    
    return dieline


def export_to_svg(dieline: CartonDieline, filename: str = 'carton_dieline.svg',
                  scale: float = 10.0) -> str:
    """
    Export the dieline to an SVG file.
    
    Args:
        dieline: CartonDieline object to export
        filename: Output filename (default: 'carton_dieline.svg')
        scale: Scaling factor for visualization (default: 10.0 pixels per cm)
    
    Returns:
        Path to the created SVG file
    """
    # Get bounding box
    min_x, min_y, max_x, max_y = dieline.get_bounding_box()
    
    # Add margin
    margin = 2  # cm
    canvas_width = (max_x - min_x + 2 * margin) * scale
    canvas_height = (max_y - min_y + 2 * margin) * scale
    
    # Offset for centering
    offset_x = margin - min_x
    offset_y = margin - min_y
    
    # Start building SVG
    svg_lines = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{canvas_width:.2f}" height="{canvas_height:.2f}" '
        f'viewBox="0 0 {canvas_width:.2f} {canvas_height:.2f}">',
        f'  <!-- Carton Dieline: L={dieline.length}cm W={dieline.width}cm H={dieline.height}cm -->',
        f'  <g id="dieline" transform="scale({scale})">',
    ]
    
    # Add panels as rectangles with labels
    svg_lines.append('    <!-- Panels -->')
    svg_lines.append('    <g id="panels" stroke="black" stroke-width="0.05" fill="none">')
    
    for panel in dieline.panels:
        x = panel.x + offset_x
        y = panel.y + offset_y
        svg_lines.append(
            f'      <rect x="{x:.2f}" y="{y:.2f}" '
            f'width="{panel.width:.2f}" height="{panel.height:.2f}" />'
        )
        # Add label at center of panel
        cx = x + panel.width / 2
        cy = y + panel.height / 2
        svg_lines.append(
            f'      <text x="{cx:.2f}" y="{cy:.2f}" '
            f'font-size="0.3" text-anchor="middle" fill="blue">{panel.name}</text>'
        )
    
    svg_lines.append('    </g>')
    
    # Add fold lines (dashed)
    svg_lines.append('    <!-- Fold Lines -->')
    svg_lines.append('    <g id="fold-lines" stroke="red" stroke-width="0.03" '
                    'stroke-dasharray="0.2,0.1">')
    
    for fold in dieline.fold_lines:
        x1 = fold['x1'] + offset_x
        y1 = fold['y1'] + offset_y
        x2 = fold['x2'] + offset_x
        y2 = fold['y2'] + offset_y
        svg_lines.append(f'      <line x1="{x1:.2f}" y1="{y1:.2f}" '
                        f'x2="{x2:.2f}" y2="{y2:.2f}" />')
    
    svg_lines.append('    </g>')
    
    # Close SVG
    svg_lines.append('  </g>')
    svg_lines.append('</svg>')
    
    # Write to file
    svg_content = '\n'.join(svg_lines)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    return filename


def create_carton_dieline(length: float, width: float, height: float,
                         export_svg: bool = False,
                         svg_filename: str = 'carton_dieline.svg') -> CartonDieline:
    """
    Main function to create a carton dieline from raw object dimensions.
    
    This function:
    1. Adjusts the input dimensions by adding 6 cm clearance
    2. Generates a complete RSC dieline with all panels and flaps
    3. Optionally exports the dieline to an SVG file
    
    Args:
        length: Measured object length in cm
        width: Measured object width in cm
        height: Measured object height in cm
        export_svg: Whether to export to SVG file (default: False)
        svg_filename: Output SVG filename (default: 'carton_dieline.svg')
    
    Returns:
        CartonDieline object containing the complete dieline
    
    Example:
        >>> dieline = create_carton_dieline(30, 20, 15, export_svg=True)
        >>> print(f"Created dieline with {len(dieline.panels)} panels")
        >>> bbox = dieline.get_bounding_box()
        >>> print(f"Bounding box: {bbox}")
    """
    # Step 1: Adjust dimensions according to business rule
    adjusted_length, adjusted_width, adjusted_height = adjust_dimensions(
        length, width, height
    )
    
    print(f"Original dimensions: L={length}cm, W={width}cm, H={height}cm")
    print(f"Adjusted dimensions: L={adjusted_length}cm, W={adjusted_width}cm, H={adjusted_height}cm")
    
    # Step 2: Generate the RSC dieline
    dieline = generate_rsc_dieline(adjusted_length, adjusted_width, adjusted_height)
    
    print(f"Generated RSC dieline with {len(dieline.panels)} panels")
    
    # Step 3: Calculate and display bounding box
    bbox = dieline.get_bounding_box()
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    print(f"Dieline dimensions: {bbox_width:.2f}cm x {bbox_height:.2f}cm")
    
    # Step 4: Optional SVG export
    if export_svg:
        svg_path = export_to_svg(dieline, svg_filename)
        print(f"Exported dieline to: {svg_path}")
    
    return dieline


def get_dieline_data(dieline: CartonDieline) -> Dict[str, Any]:
    """
    Convert dieline to a dictionary structure for serialization.
    
    Args:
        dieline: CartonDieline object
    
    Returns:
        Dictionary containing all dieline data
    """
    return {
        'dimensions': {
            'length': dieline.length,
            'width': dieline.width,
            'height': dieline.height
        },
        'panels': [
            {
                'name': panel.name,
                'x': panel.x,
                'y': panel.y,
                'width': panel.width,
                'height': panel.height,
                'corners': panel.get_corners()
            }
            for panel in dieline.panels
        ],
        'fold_lines': dieline.fold_lines,
        'bounding_box': dieline.get_bounding_box()
    }


def generate_dieline(width: float, height: float, length: float, 
                    show: bool = True, save_path: str = None) -> plt.Figure:
    """
    Generate and visualize a packaging dieline from measured dimensions.
    
    This function creates a complete RSC (Regular Slotted Container) dieline
    using matplotlib visualization. All dimensions are computed dynamically
    from the input parameters - no values are hard-coded.
    
    Args:
        width: Object width in cm (from camera measurements)
        height: Object height in cm (from camera measurements)
        length: Object length in cm (from camera measurements)
        show: Whether to display the plot (default: True)
        save_path: Optional path to save the figure (e.g., 'dieline.png')
    
    Returns:
        matplotlib.pyplot.Figure object containing the dieline visualization
    
    Example:
        >>> # Get measurements from cameras (in cm)
        >>> width, height, length = 20.0, 15.0, 30.0
        >>> fig = generate_dieline(width, height, length, show=True)
    """
    
    # Step 1: Add clearance for packaging (6 cm standard)
    clearance = 6.0
    adj_width = width + clearance
    adj_height = height + clearance
    adj_length = length + clearance
    
    print(f"Input dimensions: W={width}cm, H={height}cm, L={length}cm")
    print(f"Adjusted dimensions: W={adj_width}cm, H={adj_height}cm, L={adj_length}cm")
    
    # Step 2: Calculate flap size (typically 50% of width for RSC boxes)
    flap_height = adj_width * 0.5
    
    # Glue tab width (for manufacturer's joint)
    glue_tab_width = adj_width * 0.3
    
    # Step 3: Compute panel positions for RSC box layout
    # Horizontal layout: [Glue Tab] [Front] [Side] [Back] [Side]
    
    start_x = glue_tab_width + 2  # Add small gap after glue tab
    start_y = flap_height  # Offset for bottom flaps
    
    panels = []
    top_flaps = []
    bottom_flaps = []
    glue_tabs = []
    
    current_x = start_x
    
    # Main panel strip (4 panels)
    # 1. Front panel
    panels.append(('Front', current_x, start_y, adj_length, adj_height))
    current_x += adj_length
    
    # 2. Left Side panel
    panels.append(('Side', current_x, start_y, adj_width, adj_height))
    current_x += adj_width
    
    # 3. Back panel
    panels.append(('Back', current_x, start_y, adj_length, adj_height))
    current_x += adj_length
    
    # 4. Right Side panel
    panels.append(('Side', current_x, start_y, adj_width, adj_height))
    
    # Glue tab on the left (manufacturer's joint)
    glue_tabs.append(('Glue Tab', 0, start_y + adj_height * 0.15, glue_tab_width, adj_height * 0.7))
    
    # Top flaps (above each panel)
    top_flaps.append(('Top Flap', start_x, start_y + adj_height, adj_length, flap_height))
    top_flaps.append(('Top Flap', start_x + adj_length, start_y + adj_height, adj_width, flap_height))
    top_flaps.append(('Top Flap', start_x + adj_length + adj_width, start_y + adj_height, adj_length, flap_height))
    top_flaps.append(('Top Flap', start_x + 2 * adj_length + adj_width, start_y + adj_height, adj_width, flap_height))
    
    # Bottom flaps (below each panel)
    bottom_flaps.append(('Bottom Flap', start_x, start_y - flap_height, adj_length, flap_height))
    bottom_flaps.append(('Bottom Flap', start_x + adj_length, start_y - flap_height, adj_width, flap_height))
    bottom_flaps.append(('Bottom Flap', start_x + adj_length + adj_width, start_y - flap_height, adj_length, flap_height))
    bottom_flaps.append(('Bottom Flap', start_x + 2 * adj_length + adj_width, start_y - flap_height, adj_width, flap_height))
    
    # Step 4: Calculate total dimensions
    total_width = 2 * adj_length + 2 * adj_width + glue_tab_width
    total_height = adj_height + 2 * flap_height
    
    # Step 5: Draw the RSC carton box dieline using matplotlib
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_facecolor('white')
    
    # Draw main panels (white fill)
    for name, x, y, w, h in panels:
        rect = patches.Rectangle((x, y), w, h, linewidth=0, 
                                 facecolor='white', alpha=1.0)
        ax.add_patch(rect)
    
    # Draw flaps as simple rectangles (white fill)
    for name, x, y, w, h in top_flaps:
        rect = patches.Rectangle((x, y), w, h, linewidth=0, 
                                 facecolor='white', alpha=1.0)
        ax.add_patch(rect)
    
    for name, x, y, w, h in bottom_flaps:
        rect = patches.Rectangle((x, y), w, h, linewidth=0, 
                                 facecolor='white', alpha=1.0)
        ax.add_patch(rect)
    
    # Define cut line properties
    cut_line_color = 'blue'
    cut_line_width = 2.5
    
    top_edge_y = start_y + adj_height + flap_height
    bottom_edge_y = start_y - flap_height
    right_edge_x = start_x + 2 * adj_length + 2 * adj_width
    
    # Draw SIMPLE RECTANGULAR FLAPS - outer edges only (blue cut lines)
    # Top outer edge (straight across all flaps)
    ax.plot([start_x, right_edge_x], [top_edge_y, top_edge_y],
           color=cut_line_color, linestyle='-', linewidth=cut_line_width, solid_capstyle='butt')
    
    # Bottom outer edge (straight across all flaps)
    ax.plot([start_x, right_edge_x], [bottom_edge_y, bottom_edge_y],
           color=cut_line_color, linestyle='-', linewidth=cut_line_width, solid_capstyle='butt')
    
    # Left outer edges for top and bottom flaps (connecting them to glue tab area)
    # Top left flap outer edge
    ax.plot([glue_tab_width, glue_tab_width], [start_y + adj_height, top_edge_y],
           color=cut_line_color, linestyle='-', linewidth=cut_line_width, solid_capstyle='butt')
    # Bottom left flap outer edge
    ax.plot([glue_tab_width, glue_tab_width], [bottom_edge_y, start_y],
           color=cut_line_color, linestyle='-', linewidth=cut_line_width, solid_capstyle='butt')
    # Top left corner connector
    ax.plot([glue_tab_width, start_x], [top_edge_y, top_edge_y],
           color=cut_line_color, linestyle='-', linewidth=cut_line_width, solid_capstyle='butt')
    # Bottom left corner connector
    ax.plot([glue_tab_width, start_x], [bottom_edge_y, bottom_edge_y],
           color=cut_line_color, linestyle='-', linewidth=cut_line_width, solid_capstyle='butt')
    
    # Draw RECTANGULAR GLUE TAB on the left (blue cut lines)
    glue_tab_x = 0
    glue_tab_y = start_y
    glue_tab_rect_width = glue_tab_width
    glue_tab_rect_height = adj_height
    
    # Draw rectangle for glue tab
    ax.plot([glue_tab_x, glue_tab_x + glue_tab_rect_width], [glue_tab_y, glue_tab_y],
           color=cut_line_color, linestyle='-', linewidth=cut_line_width, solid_capstyle='butt')  # Bottom
    ax.plot([glue_tab_x + glue_tab_rect_width, glue_tab_x + glue_tab_rect_width], [glue_tab_y, glue_tab_y + glue_tab_rect_height],
           color=cut_line_color, linestyle='-', linewidth=cut_line_width, solid_capstyle='butt')  # Right
    ax.plot([glue_tab_x + glue_tab_rect_width, glue_tab_x], [glue_tab_y + glue_tab_rect_height, glue_tab_y + glue_tab_rect_height],
           color=cut_line_color, linestyle='-', linewidth=cut_line_width, solid_capstyle='butt')  # Top
    ax.plot([glue_tab_x, glue_tab_x], [glue_tab_y + glue_tab_rect_height, glue_tab_y],
           color=cut_line_color, linestyle='-', linewidth=cut_line_width, solid_capstyle='butt')  # Left
    
    # Fill the rectangle
    from matplotlib.patches import Rectangle
    glue_tab_patch = Rectangle((glue_tab_x, glue_tab_y), glue_tab_rect_width, glue_tab_rect_height,
                                facecolor='white', edgecolor='none')
    ax.add_patch(glue_tab_patch)
    
    # Draw connection line from glue tab to left panel (red dashed crease)
    glue_tab_center_y = glue_tab_y + glue_tab_rect_height / 2
    ax.plot([glue_tab_x + glue_tab_rect_width, start_x], [glue_tab_center_y, glue_tab_center_y],
           color='red', linestyle='--', linewidth=2.0, dashes=(8, 4), solid_capstyle='butt')
    
    # Left side edges connecting flaps to main body (blue cut lines)
    ax.plot([start_x, start_x], [bottom_edge_y, start_y],
           color=cut_line_color, linestyle='-', linewidth=cut_line_width, solid_capstyle='butt')
    ax.plot([start_x, start_x], [start_y + adj_height, top_edge_y],
           color=cut_line_color, linestyle='-', linewidth=cut_line_width, solid_capstyle='butt')
    
    # Right outer edge (straight)
    ax.plot([right_edge_x, right_edge_x], [bottom_edge_y, top_edge_y],
           color=cut_line_color, linestyle='-', linewidth=cut_line_width, solid_capstyle='butt')
    
    # Vertical cut lines between flaps (blue)
    flap_divider_positions = [
        start_x + adj_length,  # Between flaps
        start_x + adj_length + adj_width,
        start_x + 2 * adj_length + adj_width,
    ]
    
    for divider_x in flap_divider_positions:
        # Top flap dividers
        ax.plot([divider_x, divider_x], [start_y + adj_height, top_edge_y],
               color=cut_line_color, linestyle='-', linewidth=cut_line_width, solid_capstyle='butt')
        # Bottom flap dividers
        ax.plot([divider_x, divider_x], [bottom_edge_y, start_y],
               color=cut_line_color, linestyle='-', linewidth=cut_line_width, solid_capstyle='butt')
    
    # Draw SOFT CREASE LINES (Red dashed - vertical dividers between panels)
    crease_positions_vertical = [
        start_x + adj_length,  # Between front and side
        start_x + adj_length + adj_width,  # Between side and back
        start_x + 2 * adj_length + adj_width,  # Between back and side
    ]
    
    for crease_x in crease_positions_vertical:
        ax.plot([crease_x, crease_x], [start_y, start_y + adj_height],
               color='red', linestyle='--', linewidth=2.0, dashes=(8, 4), solid_capstyle='butt')
    
    # Horizontal soft crease lines (separating flaps from main panels)
    ax.plot([start_x, start_x + 2 * adj_length + 2 * adj_width], 
           [start_y, start_y], 
           color='red', linestyle='--', linewidth=2.0, dashes=(8, 4), solid_capstyle='butt')
    ax.plot([start_x, start_x + 2 * adj_length + 2 * adj_width], 
           [start_y + adj_height, start_y + adj_height], 
           color='red', linestyle='--', linewidth=2.0, dashes=(8, 4), solid_capstyle='butt')
    
    # Add DIMENSION LINES (Blue with arrows)
    dim_offset = 4
    
    # Convert cm to inches for display (1 cm = 0.393701 inches)
    cm_to_inch = 0.393701
    
    # Width dimension (right side, vertical)
    dim_x_right = start_x + 2 * adj_length + 2 * adj_width + dim_offset
    ax.annotate('', xy=(dim_x_right, start_y + adj_height), xytext=(dim_x_right, start_y),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax.text(dim_x_right + 1, start_y + adj_height/2, f'{adj_height * cm_to_inch:.0f}"', 
           ha='left', va='center', fontsize=10, color='blue', rotation=90)
    
    # Length dimension (top, first panel)
    dim_y_top = start_y + adj_height + flap_height + dim_offset
    ax.annotate('', xy=(start_x + adj_length, dim_y_top), xytext=(start_x, dim_y_top),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax.text(start_x + adj_length/2, dim_y_top + 1, f'{adj_length * cm_to_inch:.0f}"', 
           ha='center', va='bottom', fontsize=10, color='blue')
    
    # Width dimension (top, second panel)
    ax.annotate('', xy=(start_x + adj_length + adj_width, dim_y_top), 
                xytext=(start_x + adj_length, dim_y_top),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax.text(start_x + adj_length + adj_width/2, dim_y_top + 1, f'{adj_width * cm_to_inch:.0f}"', 
           ha='center', va='bottom', fontsize=10, color='blue')
    
    # Flap height dimension (right side)
    ax.annotate('', xy=(dim_x_right, start_y + adj_height + flap_height), 
                xytext=(dim_x_right, start_y + adj_height),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax.text(dim_x_right + 1, start_y + adj_height + flap_height/2, f'{flap_height * cm_to_inch:.0f}"', 
           ha='left', va='center', fontsize=9, color='blue', rotation=90)
    
    # Total width dimension (bottom)
    dim_y_bottom = start_y - flap_height - dim_offset
    total_display_width = 2 * adj_length + 2 * adj_width
    ax.annotate('', xy=(start_x + total_display_width, dim_y_bottom), 
                xytext=(start_x, dim_y_bottom),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax.text(start_x + total_display_width/2, dim_y_bottom - 1, f'{total_display_width * cm_to_inch:.1f}"', 
           ha='center', va='top', fontsize=10, color='blue')
    
    # Create legend box
    from matplotlib.patches import Rectangle as LegendRect
    from matplotlib.lines import Line2D
    
    legend_x = start_x + 1
    legend_y = start_y + 1
    legend_width = 25
    legend_height = 6
    
    # Legend background (white with black border)
    legend_bg = LegendRect((legend_x, legend_y), legend_width, legend_height,
                           facecolor='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(legend_bg)
    
    # Legend text and lines
    legend_items = [
        ('CUT LINES (Blue Line)', 'blue', '-', 2.5),
        ('Soft Crease (Red Line)', 'red', '--', 2.0),
    ]
    
    legend_text_x = legend_x + 0.5
    legend_line_start_x = legend_x + 14
    legend_line_end_x = legend_x + 19
    current_legend_y = legend_y + legend_height - 1.5
    
    for label, color, style, lw in legend_items:
        # Draw text
        ax.text(legend_text_x, current_legend_y, label, 
               fontsize=9, va='center', ha='left', color='black')
        # Draw sample line
        if style == '--':
            ax.plot([legend_line_start_x, legend_line_end_x], [current_legend_y, current_legend_y],
                   color=color, linestyle=style, linewidth=lw, dashes=(8, 4))
        else:
            ax.plot([legend_line_start_x, legend_line_end_x], [current_legend_y, current_legend_y],
                   color=color, linestyle=style, linewidth=lw)
        current_legend_y -= 2.5
    
    # Add secondary legend title for dimensions
    dim_legend_y = current_legend_y - 0.5
    ax.text(legend_text_x, dim_legend_y, 'DIMENSIONS (Blue line)', 
           fontsize=9, va='center', ha='left', color='blue', fontweight='bold')
    
    # Set axis properties
    margin = 12
    ax.set_xlim(-margin, total_width + margin + 8)
    ax.set_ylim(start_y - flap_height - margin - 2, start_y + adj_height + flap_height + margin + 2)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.axis('off')
    
    # Title with dimensions
    title = f'RSC Carton Shipping Box Dieline\n'
    title += f'Dimensions: L={adj_length:.1f}cm × W={adj_width:.1f}cm × H={adj_height:.1f}cm'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dieline saved to: {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    
    return fig


# ========== USAGE EXAMPLE ==========
if __name__ == '__main__':
    # Example: Create a dieline for an object measuring 30cm x 20cm x 15cm
    print("=" * 60)
    print("CARTON DIELINE GENERATOR")
    print("=" * 60)
    print()
    
    # Input dimensions (measured object)
    object_length = 30.0  # cm
    object_width = 20.0   # cm
    object_height = 15.0  # cm
    
    # Create the dieline with automatic adjustment and SVG export
    dieline = create_carton_dieline(
        object_length, 
        object_width, 
        object_height,
        export_svg=True,
        svg_filename='example_carton_dieline.svg'
    )
    
    print()
    print("=" * 60)
    print("PANEL DETAILS")
    print("=" * 60)
    
    for panel in dieline.panels:
        print(f"{panel.name:25s} -> Position: ({panel.x:.1f}, {panel.y:.1f}), "
              f"Size: {panel.width:.1f} x {panel.height:.1f} cm")
    
    print()
    print("=" * 60)
    print("Dieline generation complete!")
    print("=" * 60)
    
    # Test the new generate_dieline function
    print("\n" + "=" * 60)
    print("TESTING generate_dieline() FUNCTION")
    print("=" * 60)
    
    # Generate matplotlib dieline with measured dimensions
    fig = generate_dieline(width=object_width, height=object_height, length=object_length, 
                          show=False, save_path='dieline_matplotlib.png')
    print("Matplotlib dieline created successfully!")
