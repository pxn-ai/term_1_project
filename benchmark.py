#!/usr/bin/env python3
"""
Benchmark script to compare old vs new implementation
Shows actual speedup achieved on your Raspberry Pi 4
"""

import time
import sys
from Human_Identifier import HumanInOutCounter

def benchmark(video_path, num_runs=3):
    """Run benchmark comparing different optimization levels"""
    
    print("="*80)
    print("PERFORMANCE BENCHMARK - Raspberry Pi 4 Optimization")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"Runs per test: {num_runs}")
    print("="*80)
    print()
    
    counter = HumanInOutCounter(model_size='n')
    
    results = {}
    
    # Test 1: Simulated OLD method (single core, no batch, all frames)
    print("🐌 Test 1: OLD METHOD (single core, all frames)")
    print("   - Processing every frame")
    print("   - Single core")
    print("   - No batching")
    times = []
    for run in range(num_runs):
        start = time.time()
        # This simulates the old slow method
        net_count = counter.get_net_entered_count(video_path, count_line_pos=320)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   Run {run+1}: {elapsed:.2f}s (Result: {net_count:+d})")
    avg_old = sum(times) / len(times)
    results['old'] = {'times': times, 'avg': avg_old}
    print(f"   Average: {avg_old:.2f}s\n")
    
    # Test 2: NEW method with 2 workers
    print("🚀 Test 2: NEW METHOD (2 cores)")
    print("   - Frame skipping (every 3rd)")
    print("   - Batch processing")
    print("   - 2 parallel workers")
    times = []
    for run in range(num_runs):
        start = time.time()
        net_count = counter.get_net_entered_count_multicore(
            video_path, 
            count_line_pos=320,
            num_workers=2
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   Run {run+1}: {elapsed:.2f}s (Result: {net_count:+d})")
    avg_2core = sum(times) / len(times)
    results['2core'] = {'times': times, 'avg': avg_2core}
    print(f"   Average: {avg_2core:.2f}s\n")
    
    # Test 3: NEW method with 3 workers (RECOMMENDED)
    print("⚡ Test 3: NEW METHOD (3 cores) - RECOMMENDED")
    print("   - Frame skipping (every 3rd)")
    print("   - Batch processing")
    print("   - 3 parallel workers")
    times = []
    for run in range(num_runs):
        start = time.time()
        net_count = counter.get_net_entered_count_multicore(
            video_path, 
            count_line_pos=320,
            num_workers=3
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   Run {run+1}: {elapsed:.2f}s (Result: {net_count:+d})")
    avg_3core = sum(times) / len(times)
    results['3core'] = {'times': times, 'avg': avg_3core}
    print(f"   Average: {avg_3core:.2f}s\n")
    
    # Results Summary
    print("="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    speedup_2core = avg_old / avg_2core
    speedup_3core = avg_old / avg_3core
    
    print(f"\n📊 Performance Comparison:")
    print(f"   OLD METHOD:        {avg_old:7.2f}s  (baseline)")
    print(f"   NEW (2 cores):     {avg_2core:7.2f}s  ({speedup_2core:.1f}x faster)")
    print(f"   NEW (3 cores):     {avg_3core:7.2f}s  ({speedup_3core:.1f}x faster) ⭐")
    
    print(f"\n⚡ SPEEDUP ACHIEVED:")
    print(f"   2-core optimization: {speedup_2core:.1f}x faster")
    print(f"   3-core optimization: {speedup_3core:.1f}x faster")
    
    # Time saved per video
    time_saved = avg_old - avg_3core
    print(f"\n💾 TIME SAVED PER VIDEO:")
    print(f"   {time_saved:.1f} seconds saved")
    print(f"   ({(time_saved/avg_old*100):.1f}% reduction)")
    
    # Projected savings
    videos_per_day = 10
    daily_savings = time_saved * videos_per_day
    yearly_savings = daily_savings * 365
    print(f"\n📈 PROJECTED SAVINGS:")
    print(f"   If processing {videos_per_day} videos/day:")
    print(f"   Daily time saved:   {daily_savings/60:.1f} minutes")
    print(f"   Monthly time saved: {daily_savings*30/3600:.1f} hours")
    print(f"   Yearly time saved:  {yearly_savings/3600:.1f} hours")
    
    print("\n" + "="*80)
    print("✅ Benchmark Complete!")
    print("="*80)
    print()
    
    # Recommendation
    if speedup_3core >= 10:
        print("🎉 EXCELLENT! Optimization is working perfectly!")
    elif speedup_3core >= 5:
        print("👍 GOOD! Solid speedup achieved.")
    else:
        print("⚠️  Speedup lower than expected. Check:")
        print("   - Are all 4 cores available?")
        print("   - Is system under load?")
        print("   - Run: python3 -c 'import multiprocessing; print(multiprocessing.cpu_count())'")
    
    print()
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 benchmark.py <video_file>")
        print("Example: python3 benchmark.py test_video.mp4")
        sys.exit(1)
    
    video_file = sys.argv[1]
    
    try:
        benchmark(video_file, num_runs=3)
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\n\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()