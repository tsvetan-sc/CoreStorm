import argparse
import numpy as np
import itertools
from numba import cuda
import time
from pathlib import Path

DEFAULT_BATCH_SIZE = 1000  # Default number of records to write at once
MAX_GPU_MEMORY_CHUNK = 10000000  # Maximum combinations to process at once on GPU

def write_batch(file, batch):
    """Write a batch of combinations to file"""
    file.write('\n'.join(batch) + '\n')

def generate_combinations_cpu(length, output_file, batch_size):
    """Generate all combinations using CPU (itertools) with batched writing"""
    total = 10**length
    print(f"Generating all {total} combinations (0-9) of length {length}...")
    start_time = time.time()
    
    batch = []
    with open(output_file, 'w') as f:
        for combo in itertools.product('0123456789', repeat=length):
            batch.append(''.join(combo))
            if len(batch) >= batch_size:
                write_batch(f, batch)
                batch = []
        
        if batch:  # Write remaining combinations
            write_batch(f, batch)
    
    duration = time.time() - start_time
    print(f"Done! Generated {total} combinations in {duration:.2f} seconds.")
    print(f"Output saved to {output_file}")

@cuda.jit
def generate_combinations_gpu_kernel(result, start_idx, length):
    """CUDA kernel to generate combinations on GPU"""
    idx = cuda.grid(1)
    global_idx = start_idx + idx
    if global_idx < (start_idx + result.shape[0]):
        num = global_idx
        for i in range(length):
            digit = num % 10
            result[idx, length - 1 - i] = digit
            num = num // 10

def generate_combinations_gpu(length, output_file, batch_size):
    """Generate combinations using GPU with memory-safe chunking"""
    total = 10**length
    print(f"Generating all {total} combinations (0-9) of length {length} on GPU...")
    start_time = time.time()
    
    # Calculate chunk size based on available memory
    chunk_size = min(total, MAX_GPU_MEMORY_CHUNK)
    chunks = (total + chunk_size - 1) // chunk_size
    
    with open(output_file, 'w') as f:
        for chunk_idx in range(chunks):
            start = chunk_idx * chunk_size
            current_chunk = min(chunk_size, total - start)
            
            # Allocate GPU memory for this chunk
            d_result = cuda.device_array((current_chunk, length), dtype=np.int32)
            
            # Configure and launch kernel
            threads_per_block = 256
            blocks_per_grid = (current_chunk + threads_per_block - 1) // threads_per_block
            
            generate_combinations_gpu_kernel[blocks_per_grid, threads_per_block](
                d_result, start, length)
            cuda.synchronize()
            
            # Process results in batches
            h_result = d_result.copy_to_host()
            batch = []
            for i in range(current_chunk):
                combo = ''.join(map(str, h_result[i]))
                batch.append(combo)
                if len(batch) >= batch_size:
                    write_batch(f, batch)
                    batch = []
            
            if batch:  # Write remaining combinations
                write_batch(f, batch)
            
            # Free GPU memory
            del d_result
            
            # Progress reporting
            if chunks > 1:
                print(f"Processed chunk {chunk_idx+1}/{chunks} ({start+current_chunk}/{total} combinations)")
    
    duration = time.time() - start_time
    print(f"Done! Generated {total} combinations in {duration:.2f} seconds.")
    print(f"Output saved to {output_file}")

def estimate_gpu_memory_usage(length, num_combinations):
    """Estimate memory needed for combinations (in bytes)"""
    # 4 bytes per digit (np.int32) * length * num_combinations
    return 4 * length * num_combinations

def main():
    parser = argparse.ArgumentParser(description='Generate all possible number combinations (0-9) of specified length.')
    parser.add_argument('length', type=int, help='Length of the number combinations')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--output', type=str, default='combinations.txt', help='Output file path')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, 
                       help=f'Number of records per batch (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--max-gpu-chunk', type=int, default=MAX_GPU_MEMORY_CHUNK,
                       help=f'Max combinations per GPU chunk (default: {MAX_GPU_MEMORY_CHUNK})')
    
    args = parser.parse_args()
    
    if args.length <= 0:
        print("Error: Length must be a positive integer.")
        return
    
    total_combinations = 10**args.length
    if args.length > 5:
        print(f"Warning: Generating {total_combinations:,} combinations. This may require significant resources.")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() != 'y':
            return
    
    output_path = Path(args.output)
    if output_path.exists():
        print(f"Warning: {args.output} already exists.")
        confirm = input("Overwrite? (y/n): ")
        if confirm.lower() != 'y':
            return
    
    try:
        if args.gpu:
            try:
                # Estimate memory requirements
                mem_needed = estimate_gpu_memory_usage(args.length, min(total_combinations, args.max_gpu_chunk))
                print(f"Estimated GPU memory needed: {mem_needed/1024/1024:.2f} MB per chunk")
                
                generate_combinations_gpu(args.length, args.output, args.batch_size)
            except Exception as e:
                print(f"GPU acceleration failed: {e}")
                print("Falling back to CPU implementation...")
                generate_combinations_cpu(args.length, args.output, args.batch_size)
        else:
            generate_combinations_cpu(args.length, args.output, args.batch_size)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        if output_path.exists():
            output_path.unlink()
            print("Partial output file deleted.")
    except Exception as e:
        print(f"An error occurred: {e}")
        if output_path.exists():
            output_path.unlink()
            print("Partial output file deleted.")

if __name__ == '__main__':
    main()