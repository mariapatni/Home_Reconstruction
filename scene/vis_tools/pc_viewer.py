"""
Point Cloud Visualization Tool
Visualize and compare point clouds with panoptic semantic coloring.

Features:
- Each semantic CLASS gets a unique color (dynamically generated)
- Multiple INSTANCES of the same class vary by opacity
- Background points are nearly transparent (10% opacity)
"""

import numpy as np
from pathlib import Path
import json
import colorsys
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional


# =============================================================================
# Dynamic Color Generation
# =============================================================================

def generate_class_colors(class_names: List[str]) -> Dict[str, Tuple[float, float, float]]:
    """
    Generate visually distinct colors for a list of class names.
    
    Uses golden ratio hue stepping for maximum distinction,
    with high saturation and value for vivid colors.
    
    Args:
        class_names: List of unique class names (excluding background)
        
    Returns:
        Dict mapping class_name -> (R, G, B) tuple in [0, 1]
    """
    colors = {}
    
    # Background is always gray
    colors["background"] = (0.5, 0.5, 0.5)
    
    # Filter out background if present
    classes = [c for c in class_names if c.lower() != "background"]
    n_classes = len(classes)
    
    if n_classes == 0:
        return colors
    
    # Golden ratio for optimal hue distribution
    golden_ratio = 0.618033988749895
    
    # Start hue (avoid pure red which can look like errors)
    start_hue = 0.1
    
    for i, class_name in enumerate(sorted(classes)):
        # Golden ratio stepping gives well-distributed hues
        hue = (start_hue + i * golden_ratio) % 1.0
        
        # High saturation and value for vivid, distinguishable colors
        # Slight variation to add visual interest
        saturation = 0.75 + (i % 3) * 0.08  # 0.75, 0.83, 0.91
        value = 0.85 + (i % 2) * 0.10       # 0.85, 0.95
        
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[class_name] = (r, g, b)
    
    return colors


def load_class_mapping(scene_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Load class mapping from scene directory.
    
    Returns:
        dict mapping object_id (int) -> {"class_name": str, "instance_id": int}
    """
    scene_path = Path(scene_path)
    
    # Try remapped version first (this is what we want after processing)
    mapping_file = scene_path / "class_mapping_remapped.json"
    if mapping_file.exists():
        with open(mapping_file) as f:
            raw = json.load(f)
        mapping = {}
        for k, v in raw.items():
            key = int(k)
            if isinstance(v, dict):
                mapping[key] = v
            else:
                mapping[key] = {"class_name": str(v), "instance_id": 0}
        print(f"✓ Loaded remapped class mapping: {len(mapping)} objects")
        return mapping
    
    # Fallback to original class_mapping.json
    mapping_file = scene_path / "class_mapping.json"
    if mapping_file.exists():
        with open(mapping_file) as f:
            raw = json.load(f)
        mapping = {}
        for k, v in raw.items():
            key = int(k)
            if isinstance(v, dict):
                mapping[key] = v
            else:
                mapping[key] = {"class_name": str(v), "instance_id": 0}
        print(f"✓ Loaded class mapping: {len(mapping)} objects")
        return mapping
    
    print("⚠ No class mapping file found")
    return {}


def build_class_instance_structure(
    class_mapping: Dict[int, Dict[str, Any]]
) -> Dict[str, List[int]]:
    """
    Build a structure grouping object IDs by their class name.
    
    Returns:
        {class_name: [obj_id_1, obj_id_2, ...]}
    """
    class_to_ids: Dict[str, List[int]] = defaultdict(list)
    
    for obj_id, info in class_mapping.items():
        if obj_id == 0:
            continue  # Skip background
        class_name = info.get("class_name", f"object_{obj_id}")
        class_to_ids[class_name].append(obj_id)
    
    # Sort IDs within each class by instance_id for consistent ordering
    for class_name in class_to_ids:
        class_to_ids[class_name].sort(
            key=lambda x: class_mapping.get(x, {}).get("instance_id", 0)
        )
    
    return dict(class_to_ids)


def get_object_label(obj_id: int, class_mapping: Dict[int, Dict[str, Any]]) -> str:
    """Get a human-readable label for an object ID."""
    if obj_id == 0:
        return "background"
    
    if obj_id in class_mapping:
        info = class_mapping[obj_id]
        class_name = info.get("class_name", f"object_{obj_id}")
        instance_id = info.get("instance_id", 0)
        return f"{obj_id}: {class_name} (inst {instance_id})"
    
    return f"object_{obj_id}"


def compute_panoptic_colors_and_opacities(
    object_ids: np.ndarray,
    class_mapping: Dict[int, Dict[str, Any]],
    background_opacity: float = 0.1,
    min_instance_opacity: float = 0.4,
    max_instance_opacity: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[float, float, float]]]:
    """
    Compute colors and opacities for panoptic visualization.
    
    Rules:
    - Each semantic CLASS gets a unique color (dynamically generated)
    - Background points get ~10% opacity
    - If only ONE instance of a class: full opacity
    - If MULTIPLE instances: vary opacity from max to min
    
    Returns:
        colors: (N, 3) RGB in [0, 1]
        opacities: (N,) in [0, 1]
        class_colors: dict mapping class_name -> (R, G, B)
    """
    n_points = len(object_ids)
    colors = np.zeros((n_points, 3), dtype=np.float32)
    opacities = np.ones(n_points, dtype=np.float32)
    
    # Build class -> object_ids mapping
    class_to_ids = build_class_instance_structure(class_mapping)
    
    # Generate colors dynamically for all classes found
    all_class_names = list(class_to_ids.keys())
    class_colors = generate_class_colors(all_class_names)
    
    # Pre-compute opacity for each object ID
    id_to_opacity: Dict[int, float] = {0: background_opacity}
    
    for class_name, obj_ids in class_to_ids.items():
        n_instances = len(obj_ids)
        
        if n_instances == 1:
            # Single instance: full opacity
            id_to_opacity[obj_ids[0]] = max_instance_opacity
        else:
            # Multiple instances: vary opacity
            for i, obj_id in enumerate(obj_ids):
                # Linear interpolation from max to min opacity
                t = i / max(1, n_instances - 1)
                opacity = max_instance_opacity - t * (max_instance_opacity - min_instance_opacity)
                id_to_opacity[obj_id] = opacity
    
    # Pre-compute color for each object ID
    id_to_color: Dict[int, Tuple[float, float, float]] = {0: class_colors["background"]}
    
    for class_name, obj_ids in class_to_ids.items():
        color = class_colors.get(class_name, (0.5, 0.5, 0.5))
        for obj_id in obj_ids:
            id_to_color[obj_id] = color
    
    # Apply colors and opacities
    unique_ids = np.unique(object_ids)
    
    for obj_id in unique_ids:
        mask = object_ids == obj_id
        
        # Color
        if obj_id in id_to_color:
            colors[mask] = id_to_color[obj_id]
        else:
            # Fallback: gray for unknown
            colors[mask] = (0.5, 0.5, 0.5)
        
        # Opacity
        if obj_id in id_to_opacity:
            opacities[mask] = id_to_opacity[obj_id]
        else:
            opacities[mask] = max_instance_opacity
    
    return colors, opacities, class_colors


def visualize_pointclouds(
    scene_path,
    ply_files: List[str],
    max_points: int = 150000,
    show_cameras: bool = False,
    title: Optional[str] = None,
    height: int = 900,
    color_by_semantics: bool = False,
    background_opacity: float = 0.1,
    min_instance_opacity: float = 0.4,
    max_instance_opacity: float = 1.0,
    prompts=None,  # DEPRECATED
):
    """
    Visualize point clouds with panoptic semantic coloring.
    
    Args:
        scene_path: Path to scene directory containing .ply files
        ply_files: LIST of .ply filenames (must be a list!)
        max_points: Maximum points to display per point cloud
        show_cameras: Whether to show camera positions
        title: Custom title for the plot
        height: Plot height in pixels
        color_by_semantics: Color points by semantic class
        background_opacity: Opacity for background points (default: 0.1)
        min_instance_opacity: Minimum opacity for instances (default: 0.4)
        max_instance_opacity: Maximum opacity for instances (default: 1.0)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    import open3d as o3d
    
    pio.renderers.default = "notebook_connected"
    
    scene_path = Path(scene_path)
    
    # Validate input
    if not isinstance(ply_files, list):
        raise TypeError(f"ply_files must be a list, got {type(ply_files)}. Use ['file.ply'] not 'file.ply'")
    
    if len(ply_files) == 0:
        raise ValueError("ply_files list is empty")
    
    # Load class mapping
    class_mapping = load_class_mapping(scene_path)
    
    print(f"\n{'='*60}")
    print(f"Loading {len(ply_files)} point cloud(s) from {scene_path.name}")
    if color_by_semantics:
        print(f"🎨 PANOPTIC coloring: class color + instance opacity")
        print(f"   Background opacity: {background_opacity:.0%}")
        print(f"   Instance opacity range: {min_instance_opacity:.0%} - {max_instance_opacity:.0%}")
    print(f"{'='*60}\n")
    
    # Load point clouds
    point_clouds = []
    all_class_colors = {}  # Will be populated during processing
    
    for ply_file in ply_files:
        ply_path = scene_path / ply_file
        
        if not ply_path.exists():
            raise FileNotFoundError(f"File not found: {ply_path}")
        
        print(f"Loading {ply_file}...")
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Load object IDs if semantic coloring requested
        object_ids = None
        semantic_colors = None
        opacities = None
        
        if color_by_semantics:
            ids_path = scene_path / "object_ids.npy"
            
            if ids_path.exists():
                object_ids = np.load(ids_path)
                unique_ids = np.unique(object_ids)
                print(f"  ✓ Loaded object IDs: {len(unique_ids)} unique objects")
                
                # Compute panoptic colors and opacities
                semantic_colors, opacities, class_colors = compute_panoptic_colors_and_opacities(
                    object_ids,
                    class_mapping,
                    background_opacity=background_opacity,
                    min_instance_opacity=min_instance_opacity,
                    max_instance_opacity=max_instance_opacity,
                )
                all_class_colors.update(class_colors)
                
                # Print class breakdown
                class_to_ids = build_class_instance_structure(class_mapping)
                print(f"\n  Classes ({len(class_to_ids)}):")
                for class_name in sorted(class_to_ids.keys()):
                    obj_ids_list = class_to_ids[class_name]
                    total_pts = sum(np.sum(object_ids == oid) for oid in obj_ids_list)
                    n_inst = len(obj_ids_list)
                    color = class_colors.get(class_name, (0.5, 0.5, 0.5))
                    print(f"    {class_name}: {n_inst} instance(s), {total_pts:,} pts")
            else:
                print(f"  ⚠ object_ids.npy not found, using RGB colors")
        
        # Validate colors if not using semantics
        if semantic_colors is None:
            if colors.size == 0 or colors.shape[0] != points.shape[0]:
                print(f"  ⚠ Warning: Could not read colors, using default gray")
                colors = np.ones((len(points), 3)) * 0.5
        
        print(f"  ✓ {len(points):,} points total\n")
        
        point_clouds.append({
            'name': ply_file,
            'points': points,
            'colors': colors,
            'object_ids': object_ids,
            'semantic_colors': semantic_colors,
            'opacities': opacities,
        })
    
    # Downsample for performance
    def downsample_for_plot(pc_data, max_pts):
        n = len(pc_data['points'])
        if n > max_pts:
            indices = np.random.choice(n, max_pts, replace=False)
            return {
                'points': pc_data['points'][indices],
                'colors': pc_data['colors'][indices] if pc_data['colors'] is not None else None,
                'object_ids': pc_data['object_ids'][indices] if pc_data['object_ids'] is not None else None,
                'semantic_colors': pc_data['semantic_colors'][indices] if pc_data['semantic_colors'] is not None else None,
                'opacities': pc_data['opacities'][indices] if pc_data['opacities'] is not None else None,
            }
        return pc_data
    
    # Prepare plot data
    plot_data = []
    
    for pc in point_clouds:
        original_count = len(pc['points'])
        pc_sampled = downsample_for_plot(pc, max_points)
        
        # Determine colors to use
        if pc_sampled['semantic_colors'] is not None:
            display_colors = pc_sampled['semantic_colors']
            display_opacities = pc_sampled['opacities']
        else:
            display_colors = pc_sampled['colors']
            display_opacities = np.ones(len(pc_sampled['points']))
        
        # Convert to rgba strings for plotly
        rgba_strings = []
        for i in range(len(display_colors)):
            r, g, b = display_colors[i]
            a = display_opacities[i] if display_opacities is not None else 1.0
            rgba_strings.append(f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.2f})')
        
        plot_data.append({
            'name': pc['name'],
            'points': pc_sampled['points'],
            'colors': rgba_strings,
            'object_ids': pc_sampled['object_ids'],
            'original_count': original_count,
        })
    
    # Create figure
    n_cols = len(plot_data)
    
    subplot_titles = [
        f"{data['name']}<br>({data['original_count']:,} points)" 
        for data in plot_data
    ]
    
    width = min(2400, 800 * n_cols)
    
    fig = make_subplots(
        rows=1, 
        cols=n_cols,
        specs=[[{'type': 'scatter3d'} for _ in range(n_cols)]],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        column_widths=[1.0/n_cols] * n_cols
    )
    
    # Add point clouds
    for col_idx, data in enumerate(plot_data, start=1):
        fig.add_trace(
            go.Scatter3d(
                x=data['points'][:, 0],
                y=data['points'][:, 1],
                z=data['points'][:, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color=data['colors'],
                ),
                name=data['name'],
                showlegend=False
            ),
            row=1, col=col_idx
        )
    
    # Add legend for semantic classes
    if color_by_semantics and class_mapping:
        class_to_ids = build_class_instance_structure(class_mapping)
        
        # Add background first
        bg_color = all_class_colors.get("background", (0.5, 0.5, 0.5))
        fig.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(
                    size=10, 
                    color=f'rgba({int(bg_color[0]*255)},{int(bg_color[1]*255)},{int(bg_color[2]*255)},{background_opacity:.2f})'
                ),
                name=f"background ({background_opacity:.0%} opacity)",
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add each class (sorted alphabetically)
        for class_name in sorted(class_to_ids.keys()):
            obj_ids_list = class_to_ids[class_name]
            color = all_class_colors.get(class_name, (0.5, 0.5, 0.5))
            n_inst = len(obj_ids_list)
            
            # Count total points for this class
            total_pts = 0
            for pc in point_clouds:
                if pc['object_ids'] is not None:
                    for oid in obj_ids_list:
                        total_pts += np.sum(pc['object_ids'] == oid)
            
            label = f"{class_name} ({n_inst} inst, {total_pts:,} pts)"
            
            fig.add_trace(
                go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(size=10, color=f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'),
                    name=label,
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # Update layout
    if title is None:
        if color_by_semantics:
            n_classes = len(build_class_instance_structure(class_mapping)) if class_mapping else 0
            title = f"Panoptic Point Cloud - {n_classes} classes"
        elif len(plot_data) == 1:
            title = f"{plot_data[0]['name']} - {plot_data[0]['original_count']:,} points"
        else:
            title = f"Point Cloud Comparison - {len(plot_data)} files"
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=16, color='#333')
        ),
        width=width,
        height=height,
        showlegend=color_by_semantics or show_cameras,
        legend=dict(
            x=1.02, 
            y=0.5,
            yanchor='middle',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.3)',
            borderwidth=1,
            font=dict(size=10),
        ),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # Update 3D scenes
    scene_config = dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        aspectmode='data',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
    
    for i in range(1, n_cols + 1):
        scene_key = 'scene' if i == 1 else f'scene{i}'
        fig.update_layout(**{scene_key: scene_config})
    
    fig.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for data in plot_data:
        displayed = len(data['points'])
        total = data['original_count']
        print(f"{data['name']:30s} {total:>10,} points ({displayed:>7,} displayed)")
    
    if color_by_semantics and class_mapping:
        class_to_ids = build_class_instance_structure(class_mapping)
        print(f"\n🎨 Class Legend ({len(class_to_ids)} classes + background):")
        print("-" * 60)
        
        # Background
        bg_color = all_class_colors.get("background", (0.5, 0.5, 0.5))
        print(f"  {'background':20s} color=({bg_color[0]:.2f},{bg_color[1]:.2f},{bg_color[2]:.2f})  opacity={background_opacity:.0%}")
        
        for class_name in sorted(class_to_ids.keys()):
            obj_ids_list = class_to_ids[class_name]
            color = all_class_colors.get(class_name, (0.5, 0.5, 0.5))
            n_inst = len(obj_ids_list)
            
            total_pts = 0
            for pc in point_clouds:
                if pc['object_ids'] is not None:
                    for oid in obj_ids_list:
                        total_pts += np.sum(pc['object_ids'] == oid)
            
            opacity_str = f"{max_instance_opacity:.0%}" if n_inst == 1 else f"{min_instance_opacity:.0%}-{max_instance_opacity:.0%}"
            print(f"  {class_name:20s} color=({color[0]:.2f},{color[1]:.2f},{color[2]:.2f})  {n_inst:2d} inst  {total_pts:>8,} pts  opacity={opacity_str}")
    
    print(f"{'='*60}\n")


# =============================================================================
# Convenience function for quick visualization
# =============================================================================
def quick_vis(scene_path: str, semantic: bool = True):
    """Quick visualization helper."""
    visualize_pointclouds(
        scene_path=scene_path,
        ply_files=["processed_semantic.ply"],
        color_by_semantics=semantic,
        max_points=200000,
    )