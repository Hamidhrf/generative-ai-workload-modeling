#!/usr/bin/env python3
"""
Validation Test for Phase 1 Data Loader

Tests that the data loader correctly handles Phase 1 data structure.
to verify everything works before proceeding to TimeVAE training.

"""

import sys
import numpy as np
from pathlib import Path

# Import the loader
try:
    from phase1_data_loader import load_phase1_data
    print(" Successfully imported phase1_data_loader")
except ImportError as e:
    print(f" Failed to import loader: {e}")
    sys.exit(1)


def test_basic_loading():
    """Test 1: Basic data loading works."""
    print("\n" + "="*80)
    print("TEST 1: Basic Loading")
    print("="*80)
    
    try:
        pod_traces = load_phase1_data(verbose=False)
        print(f" Loaded {len(pod_traces)} pod traces")
        return pod_traces
    except Exception as e:
        print(f" Loading failed: {e}")
        return None


def test_expected_counts(pod_traces):
    """Test 2: Expected number of traces."""
    print("\n" + "="*80)
    print("TEST 2: Expected Counts")
    print("="*80)
    
    # Expected total
    expected_total = 60
    actual_total = len(pod_traces)
    
    if actual_total == expected_total:
        print(f" Total traces: {actual_total} (expected {expected_total})")
    else:
        print(f" Total traces: {actual_total} (expected {expected_total})")
    
    # Count by workload
    workload_counts = {}
    for _, meta in pod_traces:
        wl = meta['workload']
        workload_counts[wl] = workload_counts.get(wl, 0) + 1
    
    expected_workloads = {
        'distilbert': 19,  # 1+2+6+10
        'resnet50': 22,    # 1+2+3+6+10
        'whisper': 19      # 1+2+3+5+8
    }
    
    print("\nWorkload counts:")
    all_correct = True
    for wl, expected in expected_workloads.items():
        actual = workload_counts.get(wl, 0)
        status = "✓" if actual == expected else "✗"
        print(f"  {status} {wl:15s}: {actual:3d} (expected {expected:3d})")
        if actual != expected:
            all_correct = False
    
    return all_correct


def test_trace_shapes(pod_traces):
    """Test 3: All traces have correct shape."""
    print("\n" + "="*80)
    print("TEST 3: Trace Shapes")
    print("="*80)
    
    expected_timesteps = 715
    expected_metrics = 15
    
    shape_errors = []
    
    for i, (trace, meta) in enumerate(pod_traces):
        # Check 2D array
        if trace.ndim != 2:
            shape_errors.append(f"Pod {i} ({meta['experiment_dir']}): Wrong dimensions {trace.ndim}D")
            continue
        
        # Check number of metrics
        if trace.shape[1] != expected_metrics:
            shape_errors.append(f"Pod {i} ({meta['experiment_dir']}): {trace.shape[1]} metrics (expected {expected_metrics})")
        
        # Check timesteps (should be around 715, allow some variation)
        if abs(trace.shape[0] - expected_timesteps) > 10:
            shape_errors.append(f"Pod {i} ({meta['experiment_dir']}): {trace.shape[0]} timesteps (expected ~{expected_timesteps})")
    
    if shape_errors:
        print(" Shape errors found:")
        for error in shape_errors[:5]:  # Show first 5
            print(f"  {error}")
        if len(shape_errors) > 5:
            print(f"  ... and {len(shape_errors)-5} more")
        return False
    else:
        print(f" All {len(pod_traces)} traces have correct shape (~{expected_timesteps}, {expected_metrics})")
        return True


def test_no_nan_values(pod_traces):
    """Test 4: No NaN values in traces."""
    print("\n" + "="*80)
    print("TEST 4: NaN Check")
    print("="*80)
    
    nan_found = []
    
    for i, (trace, meta) in enumerate(pod_traces):
        if np.isnan(trace).any():
            nan_count = np.isnan(trace).sum()
            nan_found.append((meta['experiment_dir'], meta['pod_id'][:30], nan_count))
    
    if nan_found:
        print(" NaN values found:")
        for exp, pod, count in nan_found[:5]:
            print(f"  {exp} - {pod}: {count} NaN values")
        if len(nan_found) > 5:
            print(f"  ... and {len(nan_found)-5} more")
        return False
    else:
        print(f" No NaN values in any of {len(pod_traces)} traces")
        return True


def test_metadata_structure(pod_traces):
    """Test 5: Metadata has required fields."""
    print("\n" + "="*80)
    print("TEST 5: Metadata Structure")
    print("="*80)
    
    required_fields = ['workload', 'replica_count', 'pod_id', 'experiment_dir']
    
    metadata_errors = []
    
    for i, (trace, meta) in enumerate(pod_traces):
        for field in required_fields:
            if field not in meta:
                metadata_errors.append(f"Pod {i}: Missing field '{field}'")
    
    if metadata_errors:
        print(" Metadata errors:")
        for error in metadata_errors[:5]:
            print(f"  {error}")
        return False
    else:
        print(f" All {len(pod_traces)} traces have complete metadata")
        
        # Show example
        _, example_meta = pod_traces[0]
        print("\nExample metadata:")
        for key, value in example_meta.items():
            if key == 'pod_id':
                value = value[:40] + "..."
            print(f"  {key:20s}: {value}")
        
        return True


def test_replica_count_consistency(pod_traces):
    """Test 6: Replica counts match experiment directory names."""
    print("\n" + "="*80)
    print("TEST 6: Replica Count Consistency")
    print("="*80)
    
    inconsistencies = []
    
    for i, (trace, meta) in enumerate(pod_traces):
        exp_dir = meta['experiment_dir']
        r_from_meta = meta['replica_count']
        
        # Extract r from directory name (e.g., "resnet50_r3" -> 3)
        import re
        match = re.search(r'_r(\d+)$', exp_dir)
        if match:
            r_from_dir = int(match.group(1))
            if r_from_dir != r_from_meta:
                inconsistencies.append(f"{exp_dir}: meta says r={r_from_meta}, dir says r={r_from_dir}")
    
    if inconsistencies:
        print(" Replica count inconsistencies:")
        for error in inconsistencies:
            print(f"  {error}")
        return False
    else:
        print(f" Replica counts consistent across {len(pod_traces)} traces")
        return True


def test_value_ranges(pod_traces):
    """Test 7: Metric values are in reasonable ranges."""
    print("\n" + "="*80)
    print("TEST 7: Value Range Checks")
    print("="*80)
    
    # Collect all traces
    all_traces = np.array([trace for trace, _ in pod_traces])
    
    # Compute statistics per metric
    print("\nMetric statistics:")
    print(f"{'Metric':<5} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print("-" * 60)
    
    for i in range(15):
        values = all_traces[:, :, i].flatten()
        values = values[~np.isnan(values)]  # Remove NaN for stats
        
        if len(values) > 0:
            print(f"{i+1:2d}    {values.min():12.4f} {values.max():12.4f} {values.mean():12.4f} {values.std():12.4f}")
    
    print("\n Value ranges computed successfully")
    return True


def run_all_tests():
    """Run all validation tests."""
    print("="*80)
    print("PHASE 1 DATA LOADER - VALIDATION TESTS")
    print("="*80)
    
    # Test 1: Load data
    pod_traces = test_basic_loading()
    if pod_traces is None:
        print("\n Loading failed - cannot continue tests")
        return False
    
    # Run all tests
    tests = [
        ("Expected Counts", test_expected_counts),
        ("Trace Shapes", test_trace_shapes),
        ("NaN Check", test_no_nan_values),
        ("Metadata Structure", test_metadata_structure),
        ("Replica Count Consistency", test_replica_count_consistency),
        ("Value Ranges", test_value_ranges)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func(pod_traces)
        except Exception as e:
            print(f"\n Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = " PASS" if result else " FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n ALL TESTS PASSED Data loader is working correctly")
        return True
    else:
        print(f"\n {total-passed} test(s) failed. ")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)