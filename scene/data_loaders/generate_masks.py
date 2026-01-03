"""
SAM3 Video Segmentation with Subprocess Isolation

Each prompt runs in a SEPARATE PROCESS that dies after completion.
This is the ONLY way to guarantee 100% VRAM release between prompts.

NEW:
- Optional visualization saving:
    save_vis=True will write overlays to: <scene_path>/vis/
  Overlays show the input RGB with colored masks (unique color per object id).
"""

from __future__ import annotations

from pathlib import Path
import shutil
import json
import gc
import re
import sys
import subprocess

import cv2
import numpy as np


# ============================================================================
# HELPER FUNCTIONS (used by both main process and subprocess)
# ============================================================================

def natural_sort_key(path):
    """Sort filenames numerically."""
    filename = path.name if hasattr(path, "name") else str(path)
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"([0-9]+)", filename)]


def create_rgb_step_folder(scene_path: str | Path, n: int) -> Path:
    """Create EXR_RGBD/rgb_step{n} containing every nth frame."""
    if n <= 0:
        raise ValueError("Step n must be > 0")

    scene_path = Path(scene_path)
    rgb_path = scene_path / "EXR_RGBD" / "rgb"

    if not rgb_path.exists():
        raise FileNotFoundError(rgb_path)

    step_folder = rgb_path.parent / f"rgb_step{n}"
    step_folder.mkdir(exist_ok=True)

    frames = sorted([p for p in rgb_path.iterdir() if p.is_file()], key=natural_sort_key)

    if not frames:
        raise RuntimeError("No frames found")

    print(f"[STEP] Source RGB: {rgb_path}")
    print(f"[STEP] Writing → {step_folder}")
    print(f"[STEP] Total frames: {len(frames)} | stride={n}")

    for idx, f in enumerate(frames):
        if idx % n == 0:
            shutil.copy2(f, step_folder / f.name)

    return step_folder


def get_frame_files(video_path: Path) -> list[Path]:
    """Get sorted list of frame files."""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    frame_files = sorted(
        [p for ext in exts for p in video_path.glob(ext)],
        key=natural_sort_key,
    )
    return frame_files


def load_object_id_masks(scene_path):
    """Load previously saved object ID masks and class mapping."""
    scene_path = Path(scene_path)
    masks_dir = scene_path / "object_id_masks"

    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks not found: {masks_dir}")

    mask_files = sorted(masks_dir.glob("frame_*.png"), key=natural_sort_key)

    masks = []
    for mask_file in mask_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        masks.append(mask)

    mapping_file = scene_path / "class_mapping.json"
    with open(mapping_file, "r") as f:
        class_mapping = {int(k): v for k, v in json.load(f).items()}

    print(f"[LOAD] Loaded {len(masks)} masks, {len(class_mapping)} objects")
    return masks, class_mapping


# ============================================================================
# SUBPROCESS WORKER SCRIPT
# ============================================================================

WORKER_SCRIPT = r'''
"""Worker script that runs in subprocess to process ONE prompt."""
import sys
import json
import gc
import re
from pathlib import Path

import torch
import cv2
import numpy as np

import sam3
from sam3.model_builder import build_sam3_video_predictor


def natural_sort_key(path):
    filename = path.name if hasattr(path, "name") else str(path)
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"([0-9]+)", filename)]


def move_to_cpu(out_dict):
    """Move all tensors to CPU numpy arrays."""
    cpu_out = {}
    for frame_idx, frame_data in out_dict.items():
        cpu_frame = {}
        for key, value in frame_data.items():
            if isinstance(value, torch.Tensor):
                cpu_frame[key] = value.detach().cpu().numpy()
            else:
                cpu_frame[key] = value
        cpu_out[frame_idx] = cpu_frame
    return cpu_out


def _id_to_color_bgr(obj_id: int) -> tuple:
    """
    Deterministic vivid color per object id.
    Returns BGR uint8 tuple.
    """
    # Simple hash -> HSV
    h = (obj_id * 37) % 180
    s = 200
    v = 255
    hsv = np.array([[[h, s, v]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def save_vis_overlays(frame_files, masks_dir, vis_dir, class_mapping, alpha=0.45, draw_legend=True):
    """
    Save per-frame overlay of *current global mask* onto the RGB input.
    Output filenames: vis/frame_XXXX.png (matching original frame numbers).
    """
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Build label strings for legend
    # mapping: {id: {class_name, instance_id}}
    legend_items = []
    for k, v in sorted(class_mapping.items(), key=lambda kv: int(kv[0])):
        obj_id = int(k)
        if obj_id == 0:
            continue
        name = v.get("class_name", str(obj_id))
        inst = v.get("instance_id", None)
        if inst is not None:
            legend_items.append((obj_id, f"{obj_id}: {name} (inst {inst})"))
        else:
            legend_items.append((obj_id, f"{obj_id}: {name}"))

    for frame_idx, frame_path in enumerate(frame_files):
        frame_path = Path(frame_path)
        orig_num = int("".join(filter(str.isdigit, frame_path.stem)) or frame_idx)

        img_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        mask_path = Path(masks_dir) / f"frame_{orig_num:04d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.dtype != np.uint16 and mask.dtype != np.int32:
            mask = mask.astype(np.uint16)

        overlay = img_bgr.copy()

        # Colorize each object id (skip 0)
        unique_ids = np.unique(mask)
        for obj_id in unique_ids:
            obj_id = int(obj_id)
            if obj_id == 0:
                continue
            color = _id_to_color_bgr(obj_id)
            m = (mask == obj_id)
            if not np.any(m):
                continue
            overlay[m] = (np.array(color, dtype=np.uint8) * 1.0)

        out = cv2.addWeighted(img_bgr, 1.0 - alpha, overlay, alpha, 0.0)

        # Optional: simple legend box (top-left)
        if draw_legend and legend_items:
            x0, y0 = 10, 10
            line_h = 18
            pad = 6
            max_lines = min(18, len(legend_items))
            box_w = 420
            box_h = pad * 2 + line_h * max_lines
            cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
            cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 1)

            for i in range(max_lines):
                obj_id, label = legend_items[i]
                cy = y0 + pad + (i + 1) * line_h - 5
                c = _id_to_color_bgr(int(obj_id))
                # color swatch
                cv2.rectangle(out, (x0 + 8, cy - 10), (x0 + 28, cy + 4), c, -1)
                cv2.putText(out, label, (x0 + 36, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = vis_dir / f"frame_{orig_num:04d}.png"
        cv2.imwrite(str(out_path), out)


def process_prompt(video_path, prompt, masks_dir, class_mapping_path, frame_files_data,
                   save_vis=False, vis_dir=None, vis_alpha=0.45):
    """Process a single prompt and save results."""
    video_path = Path(video_path)
    masks_dir = Path(masks_dir)

    # Load existing masks
    masks = {}
    frame_files = [Path(f) for f in frame_files_data]

    # Determine H,W from an existing mask if possible
    H, W = 960, 720
    for f in masks_dir.glob("frame_*.png"):
        m = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if m is not None:
            H, W = m.shape[:2]
            break

    for frame_idx, frame_file in enumerate(frame_files):
        orig_num = int("".join(filter(str.isdigit, frame_file.stem)) or frame_idx)
        mask_path = masks_dir / f"frame_{orig_num:04d}.png"
        if mask_path.exists():
            masks[frame_idx] = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED).astype(np.uint16)
        else:
            masks[frame_idx] = np.zeros((H, W), dtype=np.uint16)

    # Load class mapping
    with open(class_mapping_path, "r") as f:
        class_mapping = {int(k): v for k, v in json.load(f).items()}

    # Create predictor
    print("  Creating SAM3 predictor...")
    predictor = build_sam3_video_predictor(
        gpus_to_use=list(range(torch.cuda.device_count()))
    )

    try:
        # Start session
        resp = predictor.handle_request(dict(type="start_session", resource_path=str(video_path)))
        sid = resp["session_id"]
        print(f"  Session: {sid[:8]}...")

        predictor.handle_request(dict(type="reset_session", session_id=sid))

        print(f"  Running propagation for '{prompt}'...")
        predictor.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=prompt))

        out = {}
        for r in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
            out[r["frame_index"]] = r["outputs"]

        out = move_to_cpu(out)

        predictor.handle_request(dict(type="close_session", session_id=sid))

        # Find instances
        instance_ids = set()
        for _, fr in out.items():
            if "out_binary_masks" not in fr:
                continue
            frame_masks = np.asarray(fr["out_binary_masks"])
            if frame_masks.ndim == 2:
                instance_ids.add(0)
            else:
                for i in range(frame_masks.shape[0]):
                    instance_ids.add(i)

        if not instance_ids:
            print(f"  ⚠ No instances found for '{prompt}'")
            return 0

        # Assign global IDs
        next_id = max(int(k) for k in class_mapping.keys()) + 1 if class_mapping else 1
        instance_to_global = {}

        for inst_id in sorted(instance_ids):
            instance_to_global[inst_id] = next_id
            class_mapping[next_id] = {"class_name": prompt, "instance_id": int(inst_id)}
            next_id += 1

        id_range = f"{min(instance_to_global.values())}-{max(instance_to_global.values())}"
        print(f"  ✓ Found {len(instance_ids)} instances → IDs {id_range}")

        # Write to masks
        for frame_idx, fr in out.items():
            if frame_idx not in masks:
                continue
            if "out_binary_masks" not in fr:
                continue

            frame_masks = np.asarray(fr["out_binary_masks"])
            if frame_masks.ndim == 2:
                frame_masks = frame_masks[None]

            for inst_id, inst_mask in enumerate(frame_masks):
                binary = inst_mask > 0.5
                if binary.any():
                    global_id = instance_to_global.get(inst_id, 0)
                    masks[frame_idx][binary] = global_id

        # Save masks
        print("  Saving masks...")
        for frame_idx, mask in masks.items():
            orig_num = int("".join(filter(str.isdigit, frame_files[frame_idx].stem)) or frame_idx)
            mask_path = masks_dir / f"frame_{orig_num:04d}.png"
            cv2.imwrite(str(mask_path), mask.astype(np.uint16))

        # Save class mapping
        with open(class_mapping_path, "w") as f:
            json.dump({int(k): v for k, v in class_mapping.items()}, f, indent=2)

        # Save visuals (optional)
        if save_vis and vis_dir is not None:
            print(f"  Saving visualization overlays → {vis_dir}")
            save_vis_overlays(
                frame_files=frame_files,
                masks_dir=masks_dir,
                vis_dir=vis_dir,
                class_mapping=class_mapping,
                alpha=float(vis_alpha),
                draw_legend=True,
            )

        print(f"  ✓ Saved {len(masks)} masks")
        return len(instance_ids)

    finally:
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Args:
    # 1 video_path
    # 2 prompt
    # 3 masks_dir
    # 4 class_mapping_path
    # 5 frame_files_json
    # 6 save_vis (0/1)
    # 7 vis_dir
    # 8 vis_alpha
    video_path = sys.argv[1]
    prompt = sys.argv[2]
    masks_dir = sys.argv[3]
    class_mapping_path = sys.argv[4]
    frame_files_json = sys.argv[5]
    save_vis = bool(int(sys.argv[6])) if len(sys.argv) > 6 else False
    vis_dir = sys.argv[7] if len(sys.argv) > 7 else None
    vis_alpha = float(sys.argv[8]) if len(sys.argv) > 8 else 0.45

    with open(frame_files_json, "r") as f:
        frame_files_data = json.load(f)

    try:
        _ = process_prompt(
            video_path=video_path,
            prompt=prompt,
            masks_dir=masks_dir,
            class_mapping_path=class_mapping_path,
            frame_files_data=frame_files_data,
            save_vis=save_vis,
            vis_dir=vis_dir,
            vis_alpha=vis_alpha,
        )
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
'''


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def segment_and_save_incremental(
    text_prompts: list,
    scene_path: str | Path,
    n: int = 1,
    resume: bool = False,
    save_vis: bool = False,
    vis_alpha: float = 0.45,
):
    """
    Main segmentation pipeline with SUBPROCESS ISOLATION.

    Each prompt runs in a completely separate Python process.
    When the process dies, ALL GPU memory is guaranteed released.

    NEW:
      save_vis: if True, save overlays after EACH prompt to <scene_path>/vis/
      vis_alpha: overlay strength in [0..1]

    Args:
        text_prompts: List of class names to segment
        scene_path: Path to scene directory
        n: Frame stride
        resume: If True, continue from existing masks. If False, start fresh.
        save_vis: If True, save mask visualization overlays.
        vis_alpha: Overlay blend factor.

    Returns:
        masks_dir: Path to saved masks
        class_mapping: Final class mapping dict
    """
    scene_path = Path(scene_path)
    masks_dir = scene_path / "object_id_masks"
    class_mapping_path = scene_path / "class_mapping.json"
    vis_dir = scene_path / "vis"

    print(f"\n{'='*70}")
    print("INCREMENTAL SEGMENTATION (SUBPROCESS ISOLATION)")
    print(f"{'='*70}")
    print(f"Scene:     {scene_path}")
    print(f"Prompts:   {text_prompts}")
    print(f"Stride:    {n}")
    print(f"Resume:    {resume}")
    print(f"Save vis:  {save_vis} (alpha={vis_alpha})")
    print(f"{'='*70}\n")

    # Prepare video frames
    video_path = create_rgb_step_folder(scene_path, n)
    frame_files = get_frame_files(video_path)
    num_frames = len(frame_files)

    print(f"[INFO] Found {num_frames} frames")

    # Get frame dimensions from first frame
    first_frame = cv2.imread(str(frame_files[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {frame_files[0]}")
    H, W = first_frame.shape[:2]
    print(f"[INFO] Frame dimensions: {H} x {W}")

    masks_dir.mkdir(parents=True, exist_ok=True)

    if not resume:
        # Start fresh - create blank masks
        print("[INFO] Starting fresh (creating blank masks)...")
        for frame_file in frame_files:
            orig_num = int("".join(filter(str.isdigit, frame_file.stem)) or 0)
            mask_path = masks_dir / f"frame_{orig_num:04d}.png"
            blank = np.zeros((H, W), dtype=np.uint16)
            cv2.imwrite(str(mask_path), blank)

        class_mapping = {0: {"class_name": "background", "instance_id": 0}}
        with open(class_mapping_path, "w") as f:
            json.dump(class_mapping, f, indent=2)
    else:
        print("[INFO] Resuming from existing masks...")
        if not class_mapping_path.exists():
            class_mapping = {0: {"class_name": "background", "instance_id": 0}}
            with open(class_mapping_path, "w") as f:
                json.dump(class_mapping, f, indent=2)

    # Write worker script + frame list
    worker_script_path = scene_path / "_worker_script.py"
    with open(worker_script_path, "w") as f:
        f.write(WORKER_SCRIPT)

    frame_files_json = scene_path / "_frame_files.json"
    with open(frame_files_json, "w") as f:
        json.dump([str(p) for p in frame_files], f)

    # Process each prompt in a subprocess
    for prompt_idx, prompt in enumerate(text_prompts):
        print(f"\n{'='*70}")
        print(f"[{prompt_idx + 1}/{len(text_prompts)}] '{prompt}'")
        print(f"{'='*70}")

        cmd = [
            sys.executable,
            str(worker_script_path),
            str(video_path),
            prompt,
            str(masks_dir),
            str(class_mapping_path),
            str(frame_files_json),
            "1" if save_vis else "0",
            str(vis_dir),
            str(float(vis_alpha)),
        ]

        result = subprocess.run(
            cmd,
            cwd=str(scene_path),
            capture_output=False,
        )

        if result.returncode != 0:
            print(f"\n[ERROR] Subprocess failed for '{prompt}'")
            print("[ERROR] Previous prompts have been saved.")
            worker_script_path.unlink(missing_ok=True)
            frame_files_json.unlink(missing_ok=True)
            raise RuntimeError(f"Segmentation failed for prompt '{prompt}'")

        print("[OK] Subprocess completed and VRAM fully released")

    # Clean up temp files
    worker_script_path.unlink(missing_ok=True)
    frame_files_json.unlink(missing_ok=True)

    # Load final class mapping
    with open(class_mapping_path, "r") as f:
        class_mapping = {int(k): v for k, v in json.load(f).items()}

    # Summary
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"\nMasks:   {masks_dir}")
    print(f"Mapping: {class_mapping_path}")
    if save_vis:
        print(f"Vis:     {vis_dir}")

    print("\nObjects:")
    for obj_id, info in sorted(class_mapping.items(), key=lambda x: int(x[0])):
        print(f"  {obj_id}: {info['class_name']} (instance {info['instance_id']})")

    return masks_dir, class_mapping


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

def segment_all_objects(text_prompts, scene_path, n=1, vis=False, save_vis=True):
    """Legacy wrapper."""
    from collections import OrderedDict
    masks_dir, class_mapping = segment_and_save_incremental(
        text_prompts=text_prompts,
        scene_path=scene_path,
        n=n,
        resume=False,
        save_vis=bool(save_vis),
    )
    video_path = Path(scene_path) / "EXR_RGBD" / f"rgb_step{n}"
    frame_files = get_frame_files(video_path)

    frames = []
    for fp in frame_files:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    all_per_frame = OrderedDict((p, {}) for p in text_prompts)
    return all_per_frame, frames, frame_files


def process_and_save_masks(all_per_frame, video_frames, frame_files, scene_path, text_prompts):
    """Legacy wrapper."""
    masks, class_mapping = load_object_id_masks(scene_path)
    id_mapping = {}
    for obj_id, info in class_mapping.items():
        if obj_id != 0:
            id_mapping[(info["class_name"], info["instance_id"])] = obj_id
    return masks, id_mapping, class_mapping


if __name__ == "__main__":
    print(r"""
SUBPROCESS-ISOLATED SEGMENTATION

Each prompt runs in a separate process that completely dies after finishing.
This guarantees 100% VRAM release between prompts.

Usage:
    from generate_masks import segment_and_save_incremental

    masks_dir, class_mapping = segment_and_save_incremental(
        text_prompts=['chair', 'book', 'guitar'],
        scene_path='/path/to/scene',
        n=90,
        resume=False,
        save_vis=True,
        vis_alpha=0.45,
    )
""")
