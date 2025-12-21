import sys
sys.path.insert(0, "/workspace/sam3")

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def generate_masks_sam3_image(
    scene_path,
    text_prompts,
    frame_step=20,
    device="cuda",
    score_threshold=0.3,
    save_visualizations=True,  # NEW: Save vis folder
    overlay_alpha=0.5,         # NEW: Transparency of mask overlay
):
    """
    Generate masks using SAM3 IMAGE model for selected frames.
    Also creates visualization folder with colored overlays and legends.
    """
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    scene_path = Path(scene_path)
    rgb_dir = scene_path / "EXR_RGBD" / "rgb"
    masks_dir = scene_path / "EXR_RGBD" / "masks"
    vis_dir = scene_path / "EXR_RGBD" / "vis"
    
    masks_dir.mkdir(parents=True, exist_ok=True)
    if save_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all frames and select every Nth
    rgb_files = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))
    rgb_files = sorted(rgb_files, key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
    selected_files = rgb_files[::frame_step]
    
    print(f"[SAM3] Total frames: {len(rgb_files)}")
    print(f"[SAM3] Processing every {frame_step}th frame: {len(selected_files)} frames")
    print(f"[SAM3] Prompts: {text_prompts}")
    
    # Create color map for objects
    cmap = plt.cm.get_cmap('tab20', max(len(text_prompts) + 1, 20))
    object_colors = {}
    object_colors[0] = (128, 128, 128)  # Background = gray
    for i, prompt in enumerate(text_prompts):
        rgb = cmap(i + 1)[:3]  # Skip 0 for background
        object_colors[i + 1] = tuple(int(c * 255) for c in rgb)
    
    # Build image model
    print("[SAM3] Loading model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("[SAM3] Model loaded!")
    
    # Process selected frames
    for rgb_path in tqdm(selected_files, desc="Generating masks"):
        frame_name = rgb_path.stem
        mask_output_path = masks_dir / f"{frame_name}.png"
        vis_output_path = vis_dir / f"{frame_name}.png"
        
        # Skip if already processed (check both mask and vis)
        if mask_output_path.exists() and (not save_visualizations or vis_output_path.exists()):
            continue
        
        image = Image.open(rgb_path)
        H, W = image.height, image.width
        
        # Initialize mask (0 = background)
        frame_mask = np.zeros((H, W), dtype=np.uint8)
        
        # Set image
        inference_state = processor.set_image(image)
        
        # Process each prompt
        detections = {}  # Track which objects were detected
        
        for prompt_idx, prompt in enumerate(text_prompts):
            object_id = prompt_idx + 1  # 0 is background
            
            try:
                output = processor.set_text_prompt(state=inference_state, prompt=prompt)
                
                masks_tensor = output.get("masks")
                scores_tensor = output.get("scores")
                
                if masks_tensor is None or len(masks_tensor) == 0:
                    continue
                
                masks_np = masks_tensor.cpu().numpy()
                scores_np = scores_tensor.cpu().numpy()
                
                # Find best mask above threshold
                for i in range(len(scores_np)):
                    if scores_np[i] >= score_threshold:
                        mask = masks_np[i].squeeze()
                        frame_mask[mask] = object_id
                        detections[object_id] = {
                            'name': prompt,
                            'score': float(scores_np[i]),
                            'pixels': int(mask.sum())
                        }
                        break
                        
            except Exception as e:
                print(f"Warning: '{prompt}' failed on frame {frame_name}: {e}")
                continue
        
        # Save mask
        Image.fromarray(frame_mask).save(mask_output_path)
        
        # Create and save visualization
        if save_visualizations:
            vis_image = create_mask_visualization(
                image, frame_mask, text_prompts, object_colors,
                overlay_alpha=overlay_alpha, detections=detections
            )
            vis_image.save(vis_output_path)
    
    # Save metadata
    metadata = {
        "prompts": text_prompts,
        "prompt_to_object_id": {prompt: i + 1 for i, prompt in enumerate(text_prompts)},
        "object_id_to_prompt": {i + 1: prompt for i, prompt in enumerate(text_prompts)},
        "num_objects": len(text_prompts) + 1,
        "num_frames": len(selected_files),
        "frame_step": frame_step,
        "colors": {str(k): list(v) for k, v in object_colors.items()},
    }
    
    with open(masks_dir / "object_mapping.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[SAM3] ✓ Complete!")
    print(f"  Masks: {masks_dir}")
    if save_visualizations:
        print(f"  Visualizations: {vis_dir}")
    print(f"  Frames processed: {len(selected_files)}")
    
    # Create summary visualization
    if save_visualizations:
        create_summary_legend(vis_dir, text_prompts, object_colors)
    
    return masks_dir


def create_mask_visualization(image, mask, prompts, colors, overlay_alpha=0.5, detections=None):
    """
    Create visualization of segmentation mask overlaid on image with legend.
    
    Args:
        image: PIL Image (original RGB)
        mask: numpy array of object IDs (H, W)
        prompts: list of prompt names
        colors: dict mapping object_id -> (R, G, B)
        overlay_alpha: transparency of overlay
        detections: dict of detected objects with scores
    
    Returns:
        PIL Image with overlay and legend
    """
    W, H = image.size
    
    # Convert to numpy for overlay
    img_array = np.array(image).astype(np.float32)
    
    # Create colored overlay
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    
    unique_ids = np.unique(mask)
    for obj_id in unique_ids:
        if obj_id == 0:
            continue  # Skip background
        obj_mask = mask == obj_id
        color = colors.get(obj_id, (255, 255, 255))
        overlay[obj_mask] = color
    
    # Blend overlay with original image
    mask_exists = mask > 0
    blended = img_array.copy()
    blended[mask_exists] = (
        img_array[mask_exists] * (1 - overlay_alpha) +
        overlay[mask_exists] * overlay_alpha
    )
    
    # Convert back to PIL
    vis_image = Image.fromarray(blended.astype(np.uint8))
    
    # Add legend
    vis_image = add_legend_to_image(vis_image, prompts, colors, detections)
    
    return vis_image


def add_legend_to_image(image, prompts, colors, detections=None):
    """
    Add a legend box to the top-right corner of the image.
    
    Args:
        image: PIL Image
        prompts: list of prompt names
        colors: dict mapping object_id -> (R, G, B)
        detections: dict of detected objects (optional, for showing checkmarks)
    
    Returns:
        PIL Image with legend
    """
    # Make a copy to draw on
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Try to get a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_bold = font
    
    # Legend dimensions
    padding = 10
    line_height = 22
    color_box_size = 16
    max_text_width = 150
    
    # Calculate legend size
    num_items = len(prompts) + 1  # +1 for header
    legend_height = padding * 2 + num_items * line_height
    legend_width = padding * 3 + color_box_size + max_text_width
    
    # Position (top-right corner with margin)
    margin = 15
    x0 = img.width - legend_width - margin
    y0 = margin
    x1 = img.width - margin
    y1 = y0 + legend_height
    
    # Draw semi-transparent background
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 180))
    
    # Composite overlay onto image
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Draw header
    header_y = y0 + padding
    draw.text((x0 + padding, header_y), "Objects", fill=(255, 255, 255), font=font_bold)
    
    # Draw legend items
    for i, prompt in enumerate(prompts):
        object_id = i + 1
        item_y = header_y + (i + 1) * line_height
        
        # Color box
        color = colors.get(object_id, (128, 128, 128))
        box_x0 = x0 + padding
        box_y0 = item_y + 2
        box_x1 = box_x0 + color_box_size
        box_y1 = box_y0 + color_box_size
        draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill=color, outline=(255, 255, 255))
        
        # Text label
        text_x = box_x1 + padding
        
        # Check if this object was detected
        if detections and object_id in detections:
            label = f"✓ {prompt}"
            text_color = (255, 255, 255)
        else:
            label = f"  {prompt}"
            text_color = (150, 150, 150)
        
        # Truncate if too long
        if len(label) > 18:
            label = label[:17] + "…"
        
        draw.text((text_x, item_y), label, fill=text_color, font=font)
    
    return img


def create_summary_legend(vis_dir, prompts, colors):
    """
    Create a standalone legend image summarizing all objects.
    """
    vis_dir = Path(vis_dir)
    
    # Create legend image
    padding = 20
    line_height = 30
    color_box_size = 24
    width = 400
    height = padding * 2 + (len(prompts) + 2) * line_height
    
    img = Image.new('RGB', (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
        font_bold = font
    
    # Header
    draw.text((padding, padding), "Semantic Segmentation Legend", fill=(255, 255, 255), font=font_bold)
    
    # Background entry
    y = padding + line_height * 1.5
    draw.rectangle([padding, y + 2, padding + color_box_size, y + 2 + color_box_size], 
                   fill=(128, 128, 128), outline=(255, 255, 255))
    draw.text((padding + color_box_size + 15, y), "0: background", fill=(180, 180, 180), font=font)
    
    # Object entries
    for i, prompt in enumerate(prompts):
        object_id = i + 1
        y = padding + line_height * (i + 2.5)
        color = colors.get(object_id, (128, 128, 128))
        
        draw.rectangle([padding, y + 2, padding + color_box_size, y + 2 + color_box_size], 
                       fill=color, outline=(255, 255, 255))
        draw.text((padding + color_box_size + 15, y), f"{object_id}: {prompt}", 
                  fill=(255, 255, 255), font=font)
    
    # Save
    legend_path = vis_dir / "legend.png"
    img.save(legend_path)
    print(f"  Saved legend: {legend_path}")


# # ============================================================
# # Usage
# # ============================================================

# scene_path = Path("/workspace/Home_Reconstruction/data_scenes/maria bedroom")

# # Clear existing masks/vis if you want to regenerate
# # import shutil
# # masks_dir = scene_path / "EXR_RGBD" / "masks"
# # vis_dir = scene_path / "EXR_RGBD" / "vis"
# # if masks_dir.exists(): shutil.rmtree(masks_dir)
# # if vis_dir.exists(): shutil.rmtree(vis_dir)

# # Generate masks WITH visualizations
# generate_masks_sam3_image(
#     scene_path=scene_path,
#     text_prompts=[
#         "nightstand", "dresser", "desk", "chair", "wardrobe",
#         "mirror", "lamp", "ceiling light", "pillow", "blanket",
#         "curtain", "rug", "plant", "picture frame", "window",
#         "door", "floor", "wall", "ceiling", "bed"
#     ],
#     frame_step=20,
#     save_visualizations=True,  # Creates /vis/ folder
#     overlay_alpha=0.5,         # Adjust transparency (0-1)
# )
# ```

# This creates:
# ```
# EXR_RGBD/
# ├── masks/
# │   ├── 0.png
# │   ├── 20.png
# │   ├── ...
# │   └── object_mapping.json
# └── vis/
#     ├── 0.png          # Colored overlay with legend
#     ├── 20.png
#     ├── ...
#     └── legend.png     # Standalone legend image