"""
ENHANCED DIAGNOSTIC - More detailed coordinate system analysis
"""

import torch
import numpy as np


def quick_diagnostic_test(scene, model):
    """
    Enhanced diagnostic with detailed coordinate system debugging
    """
    
    print("\n" + "=" * 70)
    print("QUICK PRE-TRAINING DIAGNOSTIC (ENHANCED)")
    print("=" * 70 + "\n")
    
    # Get parameters and move everything to CPU
    params = model.get_parameters_as_tensors()
    
    # Move all tensors to CPU
    params_cpu = {}
    for k, v in params.items():
        if torch.is_tensor(v):
            params_cpu[k] = v.cpu()
        else:
            params_cpu[k] = v
    
    # ==========================================
    # 1. COLOR CHECK
    # ==========================================
    print("1. COLOR INITIALIZATION")
    print("-" * 60)
    
    colors = params_cpu['color']
    color_std = colors.std().item()
    color_mean = colors.mean(0)
    
    print(f"   Color std: {color_std:.4f}")
    print(f"   Color mean: [{color_mean[0]:.3f}, {color_mean[1]:.3f}, {color_mean[2]:.3f}]")
    
    if color_std < 0.05:
        print("   FAIL: Colors too uniform")
    else:
        print("   PASS: Colors have variation")
    
    print()
    
    # ==========================================
    # 2. OPACITY CHECK
    # ==========================================
    print("2. OPACITY INITIALIZATION")
    print("-" * 60)
    
    opacities = torch.sigmoid(params_cpu['opacity_raw']).squeeze()
    opacity_mean = opacities.mean().item()
    visible = (opacities > 0.5).sum().item() / len(opacities)
    
    print(f"   Opacity mean: {opacity_mean:.3f}")
    print(f"   Visible (>0.5): {visible*100:.1f}%")
    
    if opacity_mean < 0.5:
        print("   FAIL: Most Gaussians are transparent")
    else:
        print("   PASS: Most Gaussians are opaque")
    
    print()
    
    # ==========================================
    # 3. DETAILED COORDINATE SYSTEM CHECK
    # ==========================================
    print("3. COORDINATE SYSTEM ANALYSIS (DETAILED)")
    print("=" * 70)
    
    cam = scene.getTrainCameras()[0]
    points = params_cpu['pos']
    
    # Point cloud stats
    pc_center = points.mean(0)
    pc_min = points.min(0)[0]
    pc_max = points.max(0)[0]
    
    print(f"\nPOINT CLOUD:")
    print(f"  Center: [{pc_center[0]:.2f}, {pc_center[1]:.2f}, {pc_center[2]:.2f}]")
    print(f"  Min:    [{pc_min[0]:.2f}, {pc_min[1]:.2f}, {pc_min[2]:.2f}]")
    print(f"  Max:    [{pc_max[0]:.2f}, {pc_max[1]:.2f}, {pc_max[2]:.2f}]")
    print(f"  Extent: [{pc_max[0]-pc_min[0]:.2f}, {pc_max[1]-pc_min[1]:.2f}, {pc_max[2]-pc_min[2]:.2f}]")
    
    # Camera info
    c2w = cam.c2w.cpu()
    cam_pos = c2w[:3, 3]
    
    print(f"\nCAMERA (ORIGINAL):")
    print(f"  Position: [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]")
    print(f"  Right:    [{c2w[0, 0]:.2f}, {c2w[0, 1]:.2f}, {c2w[0, 2]:.2f}]")
    print(f"  Up:       [{c2w[1, 0]:.2f}, {c2w[1, 1]:.2f}, {c2w[1, 2]:.2f}]")
    print(f"  Forward:  [{c2w[2, 0]:.2f}, {c2w[2, 1]:.2f}, {c2w[2, 2]:.2f}]")
    
    # Distance from camera to point cloud center
    dist_to_center = torch.norm(cam_pos - pc_center).item()
    print(f"  Distance to PC center: {dist_to_center:.2f}")
    
    # Check if camera has get_opencv_viewmat method
    has_opencv_method = hasattr(cam, 'get_opencv_viewmat')
    print(f"\nHAS get_opencv_viewmat(): {has_opencv_method}")
    
    # Original transformation (what you had before)
    print("\n" + "-" * 70)
    print("METHOD 1: Original (OpenGL style)")
    print("-" * 70)
    
    w2c_orig = torch.inverse(c2w)
    viewmat_orig = w2c_orig.T
    
    print(f"w2c matrix:")
    print(w2c_orig.numpy())
    
    points_hom = torch.cat([points, torch.ones(len(points), 1)], dim=1)
    points_cam_orig = (points_hom @ viewmat_orig)[:, :3]
    
    z_mean_orig = points_cam_orig[:, 2].mean().item()
    z_pos_orig = (points_cam_orig[:, 2] > 0).sum().item() / len(points_cam_orig)
    z_neg_orig = (points_cam_orig[:, 2] < 0).sum().item() / len(points_cam_orig)
    
    print(f"\nPoints in camera space:")
    print(f"  X range: [{points_cam_orig[:, 0].min():.2f}, {points_cam_orig[:, 0].max():.2f}]")
    print(f"  Y range: [{points_cam_orig[:, 1].min():.2f}, {points_cam_orig[:, 1].max():.2f}]")
    print(f"  Z range: [{points_cam_orig[:, 2].min():.2f}, {points_cam_orig[:, 2].max():.2f}]")
    print(f"  Z mean: {z_mean_orig:.2f}")
    print(f"  Z>0: {z_pos_orig*100:.1f}%  |  Z<0: {z_neg_orig*100:.1f}%")
    
    # Sample points
    print(f"\nSample points (first 3):")
    for i in range(min(3, len(points_cam_orig))):
        print(f"  [{points_cam_orig[i, 0]:.2f}, {points_cam_orig[i, 1]:.2f}, {points_cam_orig[i, 2]:.2f}]")
    
    # OpenCV conversion method
    if has_opencv_method:
        print("\n" + "-" * 70)
        print("METHOD 2: get_opencv_viewmat() conversion")
        print("-" * 70)
        
        viewmat_cv = cam.get_opencv_viewmat().cpu()
        
        print(f"OpenCV viewmat:")
        print(viewmat_cv.numpy())
        
        points_cam_cv = (points_hom @ viewmat_cv)[:, :3]
        
        z_mean_cv = points_cam_cv[:, 2].mean().item()
        z_pos_cv = (points_cam_cv[:, 2] > 0).sum().item() / len(points_cam_cv)
        z_neg_cv = (points_cam_cv[:, 2] < 0).sum().item() / len(points_cam_cv)
        
        print(f"\nPoints in camera space (OpenCV):")
        print(f"  X range: [{points_cam_cv[:, 0].min():.2f}, {points_cam_cv[:, 0].max():.2f}]")
        print(f"  Y range: [{points_cam_cv[:, 1].min():.2f}, {points_cam_cv[:, 1].max():.2f}]")
        print(f"  Z range: [{points_cam_cv[:, 2].min():.2f}, {points_cam_cv[:, 2].max():.2f}]")
        print(f"  Z mean: {z_mean_cv:.2f}")
        print(f"  Z>0: {z_pos_cv*100:.1f}%  |  Z<0: {z_neg_cv*100:.1f}%")
        
        print(f"\nSample points (first 3):")
        for i in range(min(3, len(points_cam_cv))):
            print(f"  [{points_cam_cv[i, 0]:.2f}, {points_cam_cv[i, 1]:.2f}, {points_cam_cv[i, 2]:.2f}]")
        
        # Show the GL->CV conversion matrix used
        print("\n" + "-" * 70)
        print("CONVERSION MATRIX ANALYSIS")
        print("-" * 70)
        
        gl_to_cv = torch.tensor([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=torch.float32)
        
        print("GL->CV conversion matrix:")
        print(gl_to_cv.numpy())
        
        c2w_cv = c2w @ gl_to_cv
        cam_pos_cv = c2w_cv[:3, 3]
        
        print(f"\nCamera position after conversion:")
        print(f"  Original: [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]")
        print(f"  OpenCV:   [{cam_pos_cv[0]:.2f}, {cam_pos_cv[1]:.2f}, {cam_pos_cv[2]:.2f}]")
        print(f"  Difference: [{(cam_pos_cv-cam_pos)[0]:.2f}, {(cam_pos_cv-cam_pos)[1]:.2f}, {(cam_pos_cv-cam_pos)[2]:.2f}]")
    
    # Manual GL->CV conversion attempt
    print("\n" + "-" * 70)
    print("METHOD 3: Manual GL->CV conversion")
    print("-" * 70)
    
    # Try flipping Y and Z of point cloud instead
    points_cv_manual = points.clone()
    points_cv_manual[:, 1] = -points[:, 1]
    points_cv_manual[:, 2] = -points[:, 2]
    
    points_cv_hom = torch.cat([points_cv_manual, torch.ones(len(points_cv_manual), 1)], dim=1)
    points_cam_manual = (points_cv_hom @ viewmat_orig)[:, :3]
    
    z_mean_manual = points_cam_manual[:, 2].mean().item()
    z_pos_manual = (points_cam_manual[:, 2] > 0).sum().item() / len(points_cam_manual)
    
    print(f"Flipped point cloud (Y and Z), original camera:")
    print(f"  Z mean: {z_mean_manual:.2f}")
    print(f"  Z>0: {z_pos_manual*100:.1f}%")
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 70)
    print("COORDINATE SYSTEM SUMMARY")
    print("=" * 70)
    
    print(f"\nMethod 1 (Original):     Z>0 = {z_pos_orig*100:.1f}%")
    if has_opencv_method:
        print(f"Method 2 (get_opencv):   Z>0 = {z_pos_cv*100:.1f}%")
    print(f"Method 3 (Flip PC):      Z>0 = {z_pos_manual*100:.1f}%")
    
    print("\nRECOMMENDATION:")
    
    best_method = max([
        ("Original", z_pos_orig),
        ("get_opencv", z_pos_cv if has_opencv_method else 0),
        ("Flip PC", z_pos_manual)
    ], key=lambda x: x[1])
    
    if best_method[1] > 0.8:
        print(f"  Use {best_method[0]} - {best_method[1]*100:.1f}% points in front")
    else:
        print(f"  ALL METHODS FAIL - coordinate system fundamentally wrong")
        print(f"  Best was {best_method[0]} with {best_method[1]*100:.1f}%")
        print(f"\n  Possible issues:")
        print(f"    - Camera poses are wrong")
        print(f"    - Point cloud reconstruction is wrong")
        print(f"    - Camera and PC are in completely different coordinate systems")
    
    print("=" * 70 + "\n")
    
    return best_method[1] > 0.8

"""
Training Degradation Diagnostic
Run this at different iterations to track what's happening
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def diagnose_training_degradation(model, save_path=None):
    """
    Diagnose why training degrades over time
    
    Run this at iter 100, 200, 400, etc. and compare results
    """
    
    print("\n" + "="*70)
    print("TRAINING DEGRADATION DIAGNOSTIC")
    print("="*70)
    
    params = model.get_parameters_as_tensors()
    
    # Get raw values
    opacities = torch.sigmoid(params['opacity_raw']).cpu().squeeze()
    scales = torch.exp(params['scale_raw']).cpu()
    colors = params['color'].cpu()
    positions = params['pos'].cpu()
    
    # ==========================================
    # 1. OPACITY ANALYSIS (KEY METRIC)
    # ==========================================
    print("\n1. OPACITY DISTRIBUTION")
    print("-" * 70)
    
    opacity_mean = opacities.mean().item()
    opacity_std = opacities.std().item()
    opacity_min = opacities.min().item()
    opacity_max = opacities.max().item()
    
    # Histogram bins
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(opacities.detach().numpy(), bins=bins)
    percentages = hist / len(opacities) * 100
    
    print(f"   Mean: {opacity_mean:.4f}")
    print(f"   Std:  {opacity_std:.4f}")
    print(f"   Min:  {opacity_min:.4f}")
    print(f"   Max:  {opacity_max:.4f}")
    print(f"\n   Distribution:")
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        bar = "█" * int(percentages[i] / 2)
        print(f"   [{low:.1f}-{high:.1f}]: {percentages[i]:5.1f}% {bar}")
    
    # DIAGNOSIS
    print(f"\n   DIAGNOSIS:")
    if opacity_std < 0.05:
        print(f"   ❌ PROBLEM: Opacities too uniform (std={opacity_std:.4f})")
        print(f"      All Gaussians contributing equally → blob")
    else:
        print(f"   ✓ OK: Opacities have variation (std={opacity_std:.4f})")
    
    if opacity_mean > 0.85:
        print(f"   ⚠️  WARNING: Mean opacity very high ({opacity_mean:.4f})")
        print(f"      Most Gaussians fully visible → blending issues")
    
    # ==========================================
    # 2. COLOR VARIATION
    # ==========================================
    print("\n2. COLOR VARIATION")
    print("-" * 70)
    
    color_std = colors.std().item()
    color_mean = colors.mean(0)
    
    print(f"   Color std: {color_std:.4f}")
    print(f"   Color mean: [{color_mean[0]:.3f}, {color_mean[1]:.3f}, {color_mean[2]:.3f}]")
    
    # Per-channel std
    color_std_r = colors[:, 0].std().item()
    color_std_g = colors[:, 1].std().item()
    color_std_b = colors[:, 2].std().item()
    
    print(f"   Per-channel std: R={color_std_r:.4f}, G={color_std_g:.4f}, B={color_std_b:.4f}")
    
    # DIAGNOSIS
    print(f"\n   DIAGNOSIS:")
    if color_std < 0.10:
        print(f"   ❌ PROBLEM: Colors converging to mean (std={color_std:.4f})")
        print(f"      Losing scene details")
    else:
        print(f"   ✓ OK: Colors still have variation (std={color_std:.4f})")
    
    # ==========================================
    # 3. SCALE ANALYSIS
    # ==========================================
    print("\n3. SCALE DISTRIBUTION")
    print("-" * 70)
    
    scale_mean = scales.mean().item()
    scale_std = scales.std().item()
    scale_max = scales.max().item()
    
    # Volume (scale_x * scale_y * scale_z)
    volumes = scales.prod(dim=-1)
    volume_mean = volumes.mean().item()
    volume_max = volumes.max().item()
    
    print(f"   Scale mean: {scale_mean:.4f}")
    print(f"   Scale std:  {scale_std:.4f}")
    print(f"   Scale max:  {scale_max:.4f}")
    print(f"\n   Volume mean: {volume_mean:.6f}")
    print(f"   Volume max:  {volume_max:.6f}")
    
    # DIAGNOSIS
    print(f"\n   DIAGNOSIS:")
    if scale_mean > 0.5:
        print(f"   ⚠️  WARNING: Scales very large (mean={scale_mean:.4f})")
        print(f"      Large Gaussians → blurring")
    else:
        print(f"   ✓ OK: Scales reasonable (mean={scale_mean:.4f})")
    
    # ==========================================
    # 4. POSITION SPREAD
    # ==========================================
    print("\n4. POSITION DISTRIBUTION")
    print("-" * 70)
    
    pos_std = positions.std(0)
    pos_mean = positions.mean(0)
    
    print(f"   Position mean: [{pos_mean[0]:.2f}, {pos_mean[1]:.2f}, {pos_mean[2]:.2f}]")
    print(f"   Position std:  [{pos_std[0]:.2f}, {pos_std[1]:.2f}, {pos_std[2]:.2f}]")
    
    # Check if Gaussians collapsed to center
    if (pos_std < 0.1).any():
        print(f"   ⚠️  WARNING: Positions collapsing")
    else:
        print(f"   ✓ OK: Positions well distributed")
    
    # ==========================================
    # 5. OVERALL HEALTH SCORE
    # ==========================================
    print("\n" + "="*70)
    print("OVERALL HEALTH SCORE")
    print("="*70)
    
    issues = []
    
    if opacity_std < 0.05:
        issues.append("Uniform opacities (CRITICAL)")
    if opacity_mean > 0.85:
        issues.append("Very high opacity mean")
    if color_std < 0.10:
        issues.append("Low color variation")
    if scale_mean > 0.5:
        issues.append("Large scales")
    
    if len(issues) == 0:
        print("✓ HEALTHY: No major issues detected")
    else:
        print(f"❌ UNHEALTHY: {len(issues)} issue(s) detected:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    
    print("="*70 + "\n")
    
    # ==========================================
    # 6. VISUALIZATION (Optional)
    # ==========================================
    if save_path:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Opacity histogram
        axes[0, 0].hist(opacities.detach().numpy(), bins=20, edgecolor='black')
        axes[0, 0].axvline(opacity_mean, color='r', linestyle='--', label=f'Mean: {opacity_mean:.3f}')
        axes[0, 0].set_xlabel('Opacity')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title(f'Opacity Distribution (std={opacity_std:.4f})')
        axes[0, 0].legend()
        
        # Color distribution (RGB)
        axes[0, 1].hist(colors[:, 0].detach().numpy(), bins=20, alpha=0.5, label='R', color='red')
        axes[0, 1].hist(colors[:, 1].detach().numpy(), bins=20, alpha=0.5, label='G', color='green')
        axes[0, 1].hist(colors[:, 2].detach().numpy(), bins=20, alpha=0.5, label='B', color='blue')
        axes[0, 1].set_xlabel('Color Value')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title(f'Color Distribution (std={color_std:.4f})')
        axes[0, 1].legend()
        
        # Scale distribution
        axes[1, 0].hist(scales.flatten().detach().numpy(), bins=30, edgecolor='black')
        axes[1, 0].axvline(scale_mean, color='r', linestyle='--', label=f'Mean: {scale_mean:.4f}')
        axes[1, 0].set_xlabel('Scale')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Scale Distribution')
        axes[1, 0].legend()
        
        # Volume distribution
        axes[1, 1].hist(volumes.detach().numpy(), bins=30, edgecolor='black')
        axes[1, 1].axvline(volume_mean, color='r', linestyle='--', label=f'Mean: {volume_mean:.6f}')
        axes[1, 1].set_xlabel('Volume (scale_x * scale_y * scale_z)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Volume Distribution')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved diagnostic plot to {save_path}")
        plt.close()
    
    return {
        'opacity_mean': opacity_mean,
        'opacity_std': opacity_std,
        'color_std': color_std,
        'scale_mean': scale_mean,
        'volume_mean': volume_mean,
        'healthy': len(issues) == 0,
        'issues': issues
    }


# Usage during training:
def track_degradation_over_time(model, iteration):
    """Call this at checkpoints during training"""
    result = diagnose_training_degradation(
        model, 
        save_path=f"diagnostic_iter_{iteration}.png"
    )
    return result


if __name__ == "__main__":
    print("Import and run: quick_diagnostic_test(scene, model)")