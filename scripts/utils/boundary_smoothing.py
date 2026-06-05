# """
# Boundary Smoothing Utility
# ===========================
# Post-processes generated traces to eliminate phase boundary discontinuities.

# Strategy: Linear blending across boundary windows

# Usage:
#     from boundary_smoothing import smooth_phase_boundaries
    
#     # After generating trace
#     smooth_trace = smooth_phase_boundaries(raw_trace, window_size=5)
# """

# import numpy as np


# def smooth_phase_boundaries(trace, phase_boundaries=None, window_size=5):
#     """
#     Smooth discontinuities at phase boundaries using linear blending.
    
#     Args:
#         trace: (T, M) array - time series trace
#         phase_boundaries: list of boundary timesteps (default: [96, 180, 300, 420, 600])
#         window_size: number of timesteps to blend on each side of boundary
    
#     Returns:
#         Smoothed trace (T, M)
    
#     Example:
#         >>> trace = np.random.rand(715, 10)
#         >>> smooth_trace = smooth_phase_boundaries(trace, window_size=5)
#     """
#     if phase_boundaries is None:
#         # Default S27/S29 boundaries at 5-second intervals
#         phase_boundaries = [96, 180, 300, 420, 600]
    
#     trace = trace.copy()  # Don't modify original
#     T, M = trace.shape
    
#     for boundary in phase_boundaries:
#         if boundary <= window_size or boundary >= T - window_size:
#             continue  # Skip boundaries too close to edges
        
#         # Define blending window
#         start_idx = boundary - window_size
#         end_idx = boundary + window_size
        
#         # Get values before and after boundary
#         before_val = trace[start_idx, :]  # (M,)
#         after_val = trace[end_idx, :]     # (M,)
        
#         # Create linear interpolation weights
#         blend_length = end_idx - start_idx + 1
#         weights = np.linspace(0, 1, blend_length).reshape(-1, 1)  # (blend_length, 1)
        
#         # Blend across window
#         trace[start_idx:end_idx+1, :] = (1 - weights) * before_val + weights * after_val
    
#     return trace


# def smooth_phase_boundaries_average(trace, phase_boundaries=None, window_size=3):
#     """
#     Smooth boundaries by averaging neighbors (simpler approach).
    
#     At each boundary point, replace with average of previous and next timesteps.
    
#     Args:
#         trace: (T, M) array
#         phase_boundaries: list of boundary timesteps
#         window_size: how many neighbors to average (default=3 means ±1)
    
#     Returns:
#         Smoothed trace
#     """
#     if phase_boundaries is None:
#         phase_boundaries = [96, 180, 300, 420, 600]
    
#     trace = trace.copy()
#     T, M = trace.shape
    
#     half_window = window_size // 2
    
#     for boundary in phase_boundaries:
#         if boundary < half_window or boundary >= T - half_window:
#             continue
        
#         # Average across window centered on boundary
#         window_start = boundary - half_window
#         window_end = boundary + half_window + 1
        
#         trace[boundary, :] = trace[window_start:window_end, :].mean(axis=0)
    
#     return trace


# def smooth_multiple_traces(traces, phase_boundaries=None, window_size=5, method='linear'):
#     """
#     Apply boundary smoothing to multiple traces.
    
#     Args:
#         traces: (N, T, M) array - batch of traces
#         phase_boundaries: list of boundary timesteps
#         window_size: smoothing window size
#         method: 'linear' or 'average'
    
#     Returns:
#         Smoothed traces (N, T, M)
#     """
#     N, T, M = traces.shape
#     smoothed = np.zeros_like(traces)
    
#     smooth_fn = smooth_phase_boundaries if method == 'linear' else smooth_phase_boundaries_average
    
#     for i in range(N):
#         smoothed[i] = smooth_fn(traces[i], phase_boundaries, window_size)
    
#     return smoothed


# def visualize_smoothing_effect(trace_before, trace_after, metric_idx=0, 
#                                phase_boundaries=None, save_path=None):
#     """
#     Plot before/after smoothing for visual inspection.
    
#     Args:
#         trace_before: (T, M) original trace
#         trace_after: (T, M) smoothed trace
#         metric_idx: which metric to plot
#         phase_boundaries: list of boundaries
#         save_path: optional path to save figure
#     """
#     import matplotlib.pyplot as plt
    
#     if phase_boundaries is None:
#         phase_boundaries = [96, 180, 300, 420, 600]
    
#     T = trace_before.shape[0]
#     time_min = np.arange(T) * 5 / 60.0  # 5-second intervals to minutes
    
#     fig, ax = plt.subplots(figsize=(12, 4))
    
#     ax.plot(time_min, trace_before[:, metric_idx], 'b-', alpha=0.7, 
#             label='Before smoothing', lw=1.5)
#     ax.plot(time_min, trace_after[:, metric_idx], 'r-', alpha=0.7, 
#             label='After smoothing', lw=1.5)
    
#     # Mark boundaries
#     for b in phase_boundaries:
#         b_min = b * 5 / 60.0
#         ax.axvline(b_min, color='gray', linestyle='--', alpha=0.4, lw=1)
    
#     ax.set_xlabel('Time (minutes)')
#     ax.set_ylabel(f'Metric {metric_idx}')
#     ax.set_title('Boundary Smoothing Effect')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
    
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()


# # Example usage
# if __name__ == "__main__":
#     # Test with random data
#     np.random.seed(42)
    
#     # Create trace with artificial discontinuities at boundaries
#     T, M = 715, 10
#     trace = np.random.randn(T, M).cumsum(axis=0) * 0.1 + 0.5
    
#     # Add spikes at boundaries
#     boundaries = [96, 180, 300, 420, 600]
#     for b in boundaries:
#         if b < T:
#             trace[b, :] += np.random.randn(M) * 0.5  # Random spike
    
#     # Smooth
#     trace_smooth_linear = smooth_phase_boundaries(trace, window_size=5)
#     trace_smooth_avg = smooth_phase_boundaries_average(trace, window_size=3)
    
#     print("Boundary Smoothing Test")
#     print("=" * 50)
#     print(f"Original trace shape: {trace.shape}")
#     print(f"Smoothed trace shape: {trace_smooth_linear.shape}")
#     print("\nDiscontinuity reduction:")
    
#     for b in boundaries:
#         if b > 0 and b < T - 1:
#             before_disc = abs(trace[b, 0] - trace[b-1, 0])
#             after_disc = abs(trace_smooth_linear[b, 0] - trace_smooth_linear[b-1, 0])
#             print(f"  Boundary {b:3d}: {before_disc:.4f} -> {after_disc:.4f} "
#                   f"({(1 - after_disc/before_disc)*100:.1f}% reduction)")
    
#     # Visualize
#     visualize_smoothing_effect(trace, trace_smooth_linear, metric_idx=0,
#                                save_path='boundary_smoothing_demo.png')
#     print("\nDemo plot saved: boundary_smoothing_demo.png")



"""
Boundary Smoothing Utility
===========================
Post-processes generated traces to eliminate phase boundary discontinuities.

Strategy: Linear blending across boundary windows

Usage:
    from boundary_smoothing import smooth_phase_boundaries
    
    # After generating trace
    smooth_trace = smooth_phase_boundaries(raw_trace, window_size=5)
"""

import numpy as np


def smooth_phase_boundaries(trace, phase_boundaries=None, window_size=5):
    """
    Smooth discontinuities at phase boundaries using linear blending.
    
    Args:
        trace: (T, M) array - time series trace
        phase_boundaries: list of boundary timesteps (default: [96, 180, 300, 420, 600])
        window_size: number of timesteps to blend on each side of boundary
    
    Returns:
        Smoothed trace (T, M)
    
    Example:
        >>> trace = np.random.rand(715, 10)
        >>> smooth_trace = smooth_phase_boundaries(trace, window_size=5)
    """
    if phase_boundaries is None:
        # Default S27/S29 boundaries at 5-second intervals
        phase_boundaries = [96, 180, 300, 420, 600]
    
    trace = trace.copy()  # Don't modify original
    T, M = trace.shape
    
    for boundary in phase_boundaries:
        if boundary <= window_size or boundary >= T - window_size:
            continue  # Skip boundaries too close to edges
        
        # Define blending window
        start_idx = boundary - window_size
        end_idx = boundary + window_size
        
        # Get values before and after boundary
        before_val = trace[start_idx, :]  # (M,)
        after_val = trace[end_idx, :]     # (M,)
        
        # Create linear interpolation weights
        blend_length = end_idx - start_idx + 1
        weights = np.linspace(0, 1, blend_length).reshape(-1, 1)  # (blend_length, 1)
        
        # Blend across window
        trace[start_idx:end_idx+1, :] = (1 - weights) * before_val + weights * after_val
    
    return trace


def smooth_phase_boundaries_average(trace, phase_boundaries=None, window_size=3):
    """
    Smooth boundaries by averaging neighbors (simpler approach).
    
    At each boundary point, replace with average of previous and next timesteps.
    
    Args:
        trace: (T, M) array
        phase_boundaries: list of boundary timesteps
        window_size: how many neighbors to average (default=3 means ±1)
    
    Returns:
        Smoothed trace
    """
    if phase_boundaries is None:
        phase_boundaries = [96, 180, 300, 420, 600]
    
    trace = trace.copy()
    T, M = trace.shape
    
    half_window = window_size // 2
    
    for boundary in phase_boundaries:
        if boundary < half_window or boundary >= T - half_window:
            continue
        
        # Average across window centered on boundary
        window_start = boundary - half_window
        window_end = boundary + half_window + 1
        
        trace[boundary, :] = trace[window_start:window_end, :].mean(axis=0)
    
    return trace


def smooth_multiple_traces(traces, phase_boundaries=None, window_size=5, method='linear'):
    """
    Apply boundary smoothing to multiple traces.
    
    Args:
        traces: (N, T, M) array - batch of traces
        phase_boundaries: list of boundary timesteps
        window_size: smoothing window size
        method: 'linear' or 'average'
    
    Returns:
        Smoothed traces (N, T, M)
    """
    N, T, M = traces.shape
    smoothed = np.zeros_like(traces)
    
    smooth_fn = smooth_phase_boundaries if method == 'linear' else smooth_phase_boundaries_average
    
    for i in range(N):
        smoothed[i] = smooth_fn(traces[i], phase_boundaries, window_size)
    
    return smoothed


def visualize_smoothing_effect(trace_before, trace_after, metric_idx=0, 
                               phase_boundaries=None, save_path=None):
    """
    Plot before/after smoothing for visual inspection.
    
    Args:
        trace_before: (T, M) original trace
        trace_after: (T, M) smoothed trace
        metric_idx: which metric to plot
        phase_boundaries: list of boundaries
        save_path: optional path to save figure
    """
    import matplotlib.pyplot as plt
    
    if phase_boundaries is None:
        phase_boundaries = [96, 180, 300, 420, 600]
    
    T = trace_before.shape[0]
    time_min = np.arange(T) * 5 / 60.0  # 5-second intervals to minutes
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.plot(time_min, trace_before[:, metric_idx], 'b-', alpha=0.7, 
            label='Before smoothing', lw=1.5)
    ax.plot(time_min, trace_after[:, metric_idx], 'r-', alpha=0.7, 
            label='After smoothing', lw=1.5)
    
    # Mark boundaries
    for b in phase_boundaries:
        b_min = b * 5 / 60.0
        ax.axvline(b_min, color='gray', linestyle='--', alpha=0.4, lw=1)
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel(f'Metric {metric_idx}')
    ax.set_title('Boundary Smoothing Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    # Test with random data
    np.random.seed(42)
    
    # Create trace with artificial discontinuities at boundaries
    T, M = 715, 10
    trace = np.random.randn(T, M).cumsum(axis=0) * 0.1 + 0.5
    
    # Add spikes at boundaries
    boundaries = [96, 180, 300, 420, 600]
    for b in boundaries:
        if b < T:
            trace[b, :] += np.random.randn(M) * 0.5  # Random spike
    
    # Smooth
    trace_smooth_linear = smooth_phase_boundaries(trace, window_size=5)
    trace_smooth_avg = smooth_phase_boundaries_average(trace, window_size=3)
    
    print("Boundary Smoothing Test")
    print("=" * 50)
    print(f"Original trace shape: {trace.shape}")
    print(f"Smoothed trace shape: {trace_smooth_linear.shape}")
    print("\nDiscontinuity reduction:")
    
    for b in boundaries:
        if b > 0 and b < T - 1:
            before_disc = abs(trace[b, 0] - trace[b-1, 0])
            after_disc = abs(trace_smooth_linear[b, 0] - trace_smooth_linear[b-1, 0])
            print(f"  Boundary {b:3d}: {before_disc:.4f} -> {after_disc:.4f} "
                  f"({(1 - after_disc/before_disc)*100:.1f}% reduction)")
    
    # Visualize
    visualize_smoothing_effect(trace, trace_smooth_linear, metric_idx=0,
                               save_path='boundary_smoothing_demo.png')
    print("\nDemo plot saved: boundary_smoothing_demo.png")