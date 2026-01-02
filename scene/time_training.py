"""
Profiling utility for ObjectGS training loop.

Usage in Jupyter:
    from profile_training import profile
    profile(trainer)
    
Or with options:
    profile(trainer, num_iterations=100, verbose=True)
"""

import time
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_func


def profile(trainer, num_iterations=50, verbose=True):
    """
    Profile each component of the training loop.
    
    Args:
        trainer: GaussianTrainer instance
        num_iterations: Number of iterations to average over
        verbose: Whether to print detailed output
    
    Returns:
        dict with timing results
    """
    model = trainer.model
    camera = trainer.train_cameras[0]
    gt_image = camera._gt_image_gpu
    gt_mask = camera._gt_mask_gpu
    device = trainer.device

    # Warm-up
    if verbose:
        print("Warming up...")
    for _ in range(5):
        params = model.get_parameters_as_tensors()
    torch.cuda.synchronize()

    # Initialize timings
    timings = {
        'get_params': [],
        'render': [],
        'rgb_loss': [],
        'semantic_loss': [],
        'backward': [],
        'optimizer_step': [],
    }

    if verbose:
        print(f"Profiling {num_iterations} iterations...")
    
    for i in range(num_iterations):
        trainer.optimizer.zero_grad()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        params = model.get_parameters_as_tensors(camera_center=camera.camera_center.to(device))
        torch.cuda.synchronize()
        timings['get_params'].append(time.perf_counter() - t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rgb, sem, alpha, info = trainer.render_with_semantics(camera, params)
        torch.cuda.synchronize()
        timings['render'].append(time.perf_counter() - t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        l1_loss = F.l1_loss(rgb, gt_image)
        ssim_val = ssim_func(rgb.unsqueeze(0), gt_image.unsqueeze(0), data_range=1.0, size_average=True)
        torch.cuda.synchronize()
        timings['rgb_loss'].append(time.perf_counter() - t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sem_loss = trainer.compute_semantic_loss(sem, gt_mask) if gt_mask is not None else torch.tensor(0.0, device=device)
        torch.cuda.synchronize()
        timings['semantic_loss'].append(time.perf_counter() - t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = l1_loss + 0.3 * (1 - ssim_val) + 0.05 * sem_loss
        loss.backward()
        torch.cuda.synchronize()
        timings['backward'].append(time.perf_counter() - t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        trainer.optimizer.step()
        torch.cuda.synchronize()
        timings['optimizer_step'].append(time.perf_counter() - t0)

    # Compute stats
    results = {}
    total = 0
    for name, times in timings.items():
        avg_ms = 1000 * sum(times) / len(times)
        std_ms = 1000 * (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5
        results[name] = {'avg_ms': avg_ms, 'std_ms': std_ms}
        total += avg_ms

    for name in results:
        results[name]['pct'] = 100 * results[name]['avg_ms'] / total

    results['_total_ms'] = total
    results['_iterations_per_sec'] = 1000 / total

    # Print results
    if verbose:
        print("\n" + "=" * 70)
        print(f"TIMING BREAKDOWN (averaged over {num_iterations} iterations)")
        print("=" * 70)
        
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if not k.startswith('_')],
            key=lambda x: -x[1]['avg_ms']
        )
        
        for name, stats in sorted_results:
            bar = "█" * int(stats['pct'] / 2)
            print(f"{name:20s}: {stats['avg_ms']:7.2f} ± {stats['std_ms']:5.2f} ms ({stats['pct']:5.1f}%)  {bar}")
        
        print("-" * 70)
        print(f"{'TOTAL':20s}: {total:7.2f} ms  ({results['_iterations_per_sec']:.1f} it/s)")
        print("=" * 70)
        
        # Insights
        print("\nINSIGHTS:")
        top = sorted_results[0]
        print(f"  • Biggest bottleneck: {top[0]} ({top[1]['pct']:.1f}%)")
        
        forward = sum(results[k]['avg_ms'] for k in ['get_params', 'render', 'rgb_loss', 'semantic_loss'])
        backward = sum(results[k]['avg_ms'] for k in ['backward', 'optimizer_step'])
        print(f"  • Forward: {forward:.1f} ms ({100*forward/total:.1f}%)")
        print(f"  • Backward: {backward:.1f} ms ({100*backward/total:.1f}%)")

    return results