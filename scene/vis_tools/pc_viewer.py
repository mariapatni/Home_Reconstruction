"""
Point Cloud Visualization Tool
Visualize and compare any number of point clouds side-by-side
"""

import numpy as np
from pathlib import Path


def visualize_pointclouds(scene_path, ply_files, max_points=150000, 
                         show_cameras=False, title=None, height=900):
    """
    Visualize multiple point clouds side-by-side
    
    Args:
        scene_path: Path to scene directory containing .ply files
        ply_files: LIST of .ply filenames (must be a list!)
                   e.g., ["processed.ply"] or ["raw.ply", "filtered.ply"]
        max_points: Maximum points to display per point cloud (default: 150K)
        show_cameras: Whether to show camera positions (default: False)
        title: Custom title for the plot (default: auto-generated)
        height: Plot height in pixels (default: 900)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    import open3d as o3d
    
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
    
    print(f"\n{'='*60}")
    print(f"Loading {len(ply_files)} point cloud(s) from {scene_path.name}")
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
        
        print(f"  ✓ {len(points):,} points")
        
        point_clouds.append({
            'name': ply_file,
            'points': points,
            'colors': colors
        })
    
    print()
    
    # Downsample for performance
    def downsample_for_plot(points, colors, max_pts):
        if len(points) > max_pts:
            indices = np.random.choice(len(points), max_pts, replace=False)
            return points[indices], colors[indices]
        return points, colors
    
    def colors_to_rgb_strings(colors):
        colors_rgb = (colors * 255).astype(int)
        return [f'rgb({r},{g},{b})' for r, g, b in colors_rgb]
    
    # Prepare plot data
    plot_data = []
    for pc in point_clouds:
        pts, cols = downsample_for_plot(pc['points'], pc['colors'], max_points)
        rgb_strings = colors_to_rgb_strings(cols)
        
        plot_data.append({
            'name': pc['name'],
            'points': pts,
            'colors': rgb_strings,
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
                    opacity=0.6
                ),
                name=data['name'],
                showlegend=False
            ),
            row=1, col=col_idx
        )
    
    # Add cameras if requested
    if show_cameras:
        try:
            from scene.record3d_loader import Record3DScene
            scene = Record3DScene(scene_path=str(scene_path), model=None)
            
            train_positions = np.array([cam.camera_center.cpu().numpy() 
                                       for cam in scene.train_cameras])
            test_positions = np.array([cam.camera_center.cpu().numpy() 
                                      for cam in scene.test_cameras])
            
            for col_idx in range(1, n_cols + 1):
                fig.add_trace(
                    go.Scatter3d(
                        x=train_positions[:, 0],
                        y=train_positions[:, 1],
                        z=train_positions[:, 2],
                        mode='markers',
                        marker=dict(size=6, color='red', symbol='diamond'),
                        name='Train cameras',
                        showlegend=(col_idx == 1)
                    ),
                    row=1, col=col_idx
                )
                
                fig.add_trace(
                    go.Scatter3d(
                        x=test_positions[:, 0],
                        y=test_positions[:, 1],
                        z=test_positions[:, 2],
                        mode='markers',
                        marker=dict(size=6, color='green', symbol='diamond'),
                        name='Test cameras',
                        showlegend=(col_idx == 1)
                    ),
                    row=1, col=col_idx
                )
        
        except Exception as e:
            print(f"Warning: Could not load cameras: {e}")
    
    # Update layout
    if title is None:
        if len(plot_data) == 1:
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
        showlegend=show_cameras,
        legend=dict(
            x=0.5, 
            y=1.08, 
            xanchor='center', 
            orientation='h',
            bgcolor='rgba(255,255,255,0.8)'
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
    print(f"{'='*60}\n")