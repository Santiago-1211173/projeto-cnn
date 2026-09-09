"""
Benchmark script to compare matrix multiplication throughput between CPU and NVIDIA GPU.
Measures execution times of large matrix multiplications, forcing synchronization to avoid asynchronous evaluation bias.
"""

import sys
import os
import time
import logging
import tensorflow as tf

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_benchmark(device_name: str, matrix_size: int = 8192, iterations: int = 10) -> float:
    """
    Executes matrix multiplications on the specified device.
    
    Args:
        device_name: Device identifier (e.g. '/CPU:0' or '/GPU:0').
        matrix_size: N dimension of the N x N matrix (default: 8192x8192, requires ~700MB RAM per matrix).
        iterations: Number of iterations to average the measurement.
        
    Returns:
        float: Average execution time per iteration in seconds.
    """
    logger.info(f"\n--- Starting test on {device_name} ---")
    
    try:
        with tf.device(device_name):
            # 1. Allocate tensors (pure float32)
            # An 8192x8192 matrix in float32 occupies approx. 268 MB.
            shape = [matrix_size, matrix_size]
            matrix_a = tf.random.normal(shape)
            matrix_b = tf.random.normal(shape)
            
            # 2. Warmup (crucial for GPU memory allocation and JIT/PTX compilation overhead)
            logger.info("Executing Warmup...")
            warmup_result = tf.matmul(matrix_a, matrix_b)
            # In eager mode, tf sends commands to GPU asynchronously.
            # Calling .numpy() blocks CPU until the GPU finishes.
            _ = warmup_result.numpy() 
            
            # 3. Real Benchmark
            logger.info(f"Executing {iterations} iterations of {matrix_size}x{matrix_size} multiplication...")
            start_time = time.time()
            
            for _ in range(iterations):
                result = tf.matmul(matrix_a, matrix_b)
                
            # Block until execution completes
            _ = result.numpy()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / iterations
            logger.info(f"-> Average time per iteration: {avg_time:.4f} seconds")
            
            return avg_time

    except Exception as e:
        logger.error(f"Failed to execute benchmark on {device_name}: {e}")
        raise


def main() -> None:
    # Check GPU availability and enable memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Detected hardware: {tf.config.list_physical_devices()}")

    # Run on CPU
    time_cpu = run_benchmark('/CPU:0', matrix_size=8192, iterations=5)
    
    # Run on GPU
    gpu_devices = tf.config.list_logical_devices('GPU')
    if not gpu_devices:
        logger.info("\nNo GPU detected. Skipping GPU benchmark.")
        return
        
    time_gpu = run_benchmark('/GPU:0', matrix_size=8192, iterations=20)
    
    # Calculate Speedup
    if time_gpu > 0:
        speedup = time_cpu / time_gpu
        logger.info("\n==========================================")
        logger.info(f"FINAL RESULT: GPU was {speedup:.2f}x faster than CPU!")
        logger.info("==========================================")


if __name__ == "__main__":
    main()
