"""
Point Cloud Visualization Tool
Visualize and compare point clouds with optional semantic coloring
"""

import numpy as np
from pathlib import Path
import json


def visualize_pointclouds(scene_path, ply_files, max_points=150000, 
                         show_cameras=False, title=None, height=900,
                         color_by_semantics=False, prompts=None):
    """
    Visualize multiple point clouds side-by-side
    
    Args:
        scene_path: Path to scene directory containing .ply files
        ply_files: LIST of .ply filenames (must be a list!)
                   e.g., ["processed.ply"] or ["processed_semantic.ply"]
        max_points: Maximum points to display per point cloud (default: 150K)
        show_cameras: Whether to show camera positions (default: False)
        title: Custom title for the plot (default: auto-generated)
        height: Plot height in pixels (default: 900)
        color_by_semantics: Color points by object ID instead of RGB (default: False)
        prompts: List of prompt names for legend (e.g., ["bed", "dresser", "lamp"])
                 If None, will try to load from object_mapping.json
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    import open3d as o3d
    import matplotlib.pyplot as plt
    
    pio.renderers.default = "notebook_connected"
    
    scene_path = Path(scene_path)
    
    # ONLY accept list - no string conversion
    if not isinstance(ply_files, list):
        raise TypeError(
            f"ply_files must be a list, got {type(ply_files)}. "
            f"Use ['file.ply'] not 'file.ply'"
        )
    
    if len(ply_files) == 0:
        raise ValueError("ply_files list is empty")
    
    # Try to load prompts from object_mapping.json if not provided
    if color_by_semantics and prompts is None:
        mapping_path = scene_path / "EXR_RGBD" / "masks" / "object_mapping.json"
        if mapping_path.exists():
            with open(mapping_path) as f:
                mapping = json.load(f)
            prompts = mapping.get("prompts", [])
            print(f"Loaded prompts from mapping: {prompts}")
    
    print(f"\n{'='*60}")
    print(f"Loading {len(ply_files)} point cloud(s) from {scene_path.name}")
    if color_by_semantics:
        print(f"🎨 Coloring by SEMANTIC OBJECT ID")
    print(f"{'='*60}\n")
    
    # Load point clouds
    point_clouds = []
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
        if color_by_semantics:
            # Try to find object_ids.npy
            ids_path = scene_path / "object_ids.npy"
            if not ids_path.exists():
                # Try alternative locations
                ids_path = scene_path / ply_file.replace(".ply", "_ids.npy")
            
            if ids_path.exists():
                object_ids = np.load(ids_path)
                print(f"  ✓ Loaded object IDs: {len(np.unique(object_ids))} unique objects")
            else:
                print(f"  ⚠ object_ids.npy not found, using RGB colors")
        
        # Validate colors if not using semantics
        if object_ids is None:
            if colors.size == 0 or colors.shape[0] != points.shape[0]:
                print(f"  ⚠ Warning: Could not read colors, using default gray")
                colors = np.ones((len(points), 3)) * 0.5
        
        print(f"  ✓ {len(points):,} points")
        
        point_clouds.append({
            'name': ply_file,
            'points': points,
            'colors': colors,
            'object_ids': object_ids
        })
    
    print()
    
    # Create semantic colormap
    def get_semantic_colors(object_ids, num_objects=20):
        """Generate distinct colors for each object ID."""
        cmap = plt.cm.get_cmap('tab20', num_objects)
        
        colors = np.zeros((len(object_ids), 3))
        unique_ids = np.unique(object_ids)
        
        for obj_id in unique_ids:
            mask = object_ids == obj_id
            if obj_id == 0:
                # Background = gray
                colors[mask] = [0.5, 0.5, 0.5]
            else:
                colors[mask] = cmap(obj_id % 20)[:3]
        
        return colors, unique_ids
    
    # Downsample for performance
    def downsample_for_plot(points, colors, object_ids, max_pts):
        if len(points) > max_pts:
            indices = np.random.choice(len(points), max_pts, replace=False)
            new_ids = object_ids[indices] if object_ids is not None else None
            return points[indices], colors[indices], new_ids
        return points, colors, object_ids
    
    def colors_to_rgb_strings(colors):
        colors_clipped = np.clip(colors, 0, 1)
        colors_rgb = (colors_clipped * 255).astype(int)
        return [f'rgb({r},{g},{b})' for r, g, b in colors_rgb]
    
    # Prepare plot data
    plot_data = []
    all_unique_ids = set()
    
    for pc in point_clouds:
        pts = pc['points']
        cols = pc['colors']
        obj_ids = pc['object_ids']
        
        # Apply semantic coloring if available
        if obj_ids is not None:
            cols, unique_ids = get_semantic_colors(obj_ids)
            all_unique_ids.update(unique_ids)
        
        pts, cols, obj_ids = downsample_for_plot(pts, cols, obj_ids, max_points)
        rgb_strings = colors_to_rgb_strings(cols)
        
        plot_data.append({
            'name': pc['name'],
            'points': pts,
            'colors': rgb_strings,
            'object_ids': obj_ids,
            'original_count': len(pc['points'])
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
                    opacity=0.8
                ),
                name=data['name'],
                showlegend=False
            ),
            row=1, col=col_idx
        )
    
    # Add legend traces for semantic colors
    if color_by_semantics and len(all_unique_ids) > 0:
        cmap = plt.cm.get_cmap('tab20', 20)
        
        for obj_id in sorted(all_unique_ids):
            if obj_id == 0:
                color = 'rgb(128,128,128)'
                name = "background"
            else:
                c = cmap(obj_id % 20)[:3]
                color = f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'
                if prompts and obj_id <= len(prompts):
                    name = f"{obj_id}: {prompts[obj_id - 1]}"
                else:
                    name = f"object_{obj_id}"
            
            # Add invisible trace for legend
            fig.add_trace(
                go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=name,
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # Update layout
    if title is None:
        if color_by_semantics:
            title = f"Semantic Point Cloud - {len(all_unique_ids)} objects"
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
            borderwidth=1
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
    
    if color_by_semantics and len(all_unique_ids) > 0:
        print(f"\n🎨 Object Legend:")
        print("-" * 40)
        for obj_id in sorted(all_unique_ids):
            if obj_id == 0:
                name = "background"
            elif prompts and obj_id <= len(prompts):
                name = prompts[obj_id - 1]
            else:
                name = f"object_{obj_id}"
            
            # Count points with this ID
            total_pts = sum(
                np.sum(data['object_ids'] == obj_id) 
                for data in plot_data 
                if data['object_ids'] is not None
            )
            print(f"  ID {obj_id:2d}: {name:20s} ({total_pts:,} pts)")
    
    print(f"{'='*60}\n")