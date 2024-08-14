import math
import numpy as np
from numba import cuda, float32, njit, prange
import timeit

@cuda.jit
def mersenne_twister_kernel(seed, n, minimum, maximum, result):
    idx = cuda.grid(1)
    if idx < n:
        mt = cuda.local.array(624, dtype=np.uint32)
        mt[0] = seed
        for i in range(1, 624):
            mt[i] = 0xFFFFFFFF & (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i)

        index = 624
        for _ in range(idx + 1):
            if index >= 624:
                for i in range(624):
                    y = (mt[i] & 0x80000000) + (mt[(i + 1) % 624] & 0x7FFFFFFF)
                    mt[i] = mt[(i + 397) % 624] ^ (y >> 1)
                    if y % 2 != 0:
                        mt[i] ^= 0x9908B0DF
                index = 0

            y = mt[index]
            y ^= (y >> 11)
            y ^= (y << 7) & 0x9D2C5680
            y ^= (y << 15) & 0xEFC60000
            y ^= (y >> 18)
            index += 1

        result[idx] = minimum + (y % (maximum - minimum + 1))

def random_array(n, minimum, maximum):
    seed = 5489
    result = np.zeros(n, dtype=np.float32)
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block
    mersenne_twister_kernel[blocks_per_grid, threads_per_block](seed, n, minimum, maximum, result)
    return result

def writefile(name, n, minimum, maximum):
    arr = np.array(random_array(n, minimum, maximum),dtype=np.float32)
    arr.tofile(name)

def readfile(name):
    arr_from_file = np.fromfile(name, dtype=np.float32)
    return arr_from_file

@cuda.jit
def quicksort_kernel(arr, left, right):
    stack = cuda.local.array(1024, dtype=np.int32)
    top = -1

    top += 1
    stack[top] = left
    top += 1
    stack[top] = right

    while top >= 0:
        right = stack[top]
        top -= 1
        left = stack[top]
        top -= 1

        i = left - 1
        pivot = arr[right]

        for j in range(left, right):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]

        arr[i + 1], arr[right] = arr[right], arr[i + 1]
        p = i + 1

        if p - 1 > left:
            top += 1
            stack[top] = left
            top += 1
            stack[top] = p - 1

        if p + 1 < right:
            top += 1
            stack[top] = p + 1
            top += 1
            stack[top] = right

def quicksort(arr):
    n = len(arr)
    d_arr = cuda.to_device(arr)
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    quicksort_kernel[blocks_per_grid, threads_per_block](arr, 0, n-1)
    d_arr.copy_to_host(arr)

def Alpha_Beta_Pruning_Sequential(values, max_depth):
    stack = np.zeros((1000, 5), dtype=np.float32)
    stack_size = 0

    # Initialize stack
    stack[stack_size, 0] = 0  # depth
    stack[stack_size, 1] = 0  # index
    stack[stack_size, 2] = 1  # maximizingPlayer (True as 1)
    stack[stack_size, 3] = float('-inf')  # alpha
    stack[stack_size, 4] = float('inf')  # beta
    stack_size += 1

    while stack_size > 0:
        stack_size -= 1
        depth = stack[stack_size, 0]
        index = stack[stack_size, 1]
        maximizingPlayer = stack[stack_size, 2]
        alpha = stack[stack_size, 3]
        beta = stack[stack_size, 4]

        if depth == max_depth:
            return values[int(index)]
        else:
            if maximizingPlayer:
                optimum = float('-inf')
                for i in range(1, -1, -1):  # Push right child first
                    stack[stack_size, 0] = depth + 1
                    stack[stack_size, 1] = index * 2 + i
                    stack[stack_size, 2] = 0  # False as 0
                    stack[stack_size, 3] = alpha
                    stack[stack_size, 4] = beta
                    stack_size += 1

                while stack_size > 0 and stack[stack_size - 1, 0] == depth + 1:
                    stack_size -= 1
                    idx = stack[stack_size, 1]
                    val = values[int(idx)]
                    optimum = max(optimum, val)
                    alpha = max(alpha, optimum)
                    if beta <= alpha:
                        break
                return optimum
            else:
                optimum = float('inf')
                for i in range(1, -1, -1):  # Push right child first
                    stack[stack_size, 0] = depth + 1
                    stack[stack_size, 1] = index * 2 + i
                    stack[stack_size, 2] = 1  # True as 1
                    stack[stack_size, 3] = alpha
                    stack[stack_size, 4] = beta
                    stack_size += 1

                while stack_size > 0 and stack[stack_size - 1, 0] == depth + 1:
                    stack_size -= 1
                    idx = stack[stack_size, 1]
                    val = values[int(idx)]
                    optimum = min(optimum, val)
                    beta = min(beta, optimum)
                    if beta <= alpha:
                        break
                return optimum

@njit(fastmath=True, cache=True, parallel=True)
def Alpha_Beta_Pruning_CPU(values, max_depth):
    stack = np.zeros((1000, 5), dtype=np.float32)
    stack_size = 0

    # Initialize stack
    stack[stack_size, 0] = 0  # depth
    stack[stack_size, 1] = 0  # index
    stack[stack_size, 2] = 1  # maximizingPlayer (True as 1)
    stack[stack_size, 3] = float('-inf')  # alpha
    stack[stack_size, 4] = float('inf')  # beta
    stack_size += 1

    while stack_size > 0:
        stack_size -= 1
        depth = stack[stack_size, 0]
        index = stack[stack_size, 1]
        maximizingPlayer = stack[stack_size, 2]
        alpha = stack[stack_size, 3]
        beta = stack[stack_size, 4]

        if depth == max_depth:
            result = values[int(index)]
        else:
            if maximizingPlayer:
                optimum = float('-inf')
                for i in range(1, -1, -1):  # Push right child first
                    stack[stack_size, 0] = depth + 1
                    stack[stack_size, 1] = index * 2 + i
                    stack[stack_size, 2] = 0  # False as 0
                    stack[stack_size, 3] = alpha
                    stack[stack_size, 4] = beta
                    stack_size += 1

                while stack_size > 0 and stack[stack_size - 1, 0] == depth + 1:
                    stack_size -= 1
                    idx = stack[stack_size, 1]
                    val = values[int(idx)]
                    optimum = max(optimum, val)
                    alpha = max(alpha, optimum)
                    if beta <= alpha:
                        break
                result = optimum
            else:
                optimum = float('inf')
                for i in range(1, -1, -1):  # Push right child first
                    stack[stack_size, 0] = depth + 1
                    stack[stack_size, 1] = index * 2 + i
                    stack[stack_size, 2] = 1  # True as 1
                    stack[stack_size, 3] = alpha
                    stack[stack_size, 4] = beta
                    stack_size += 1
                while stack_size > 0 and stack[stack_size - 1, 0] == depth + 1:
                    stack_size -= 1
                    idx = stack[stack_size, 1]
                    val = values[int(idx)]
                    optimum = min(optimum, val)
                    beta = min(beta, optimum)
                    if beta <= alpha:
                        break
                result = optimum
    return result

@cuda.jit
def Alpha_Beta_Pruning_CUDA(values, max_depth, result):
    stack = cuda.local.array((1000, 5), dtype=float32)
    stack_size = 0

    # Initialize stack
    stack[stack_size, 0] = 0  # depth
    stack[stack_size, 1] = 0  # index
    stack[stack_size, 2] = 1  # maximizingPlayer (True as 1)
    stack[stack_size, 3] = float('-inf')  # alpha
    stack[stack_size, 4] = float('inf')  # beta
    stack_size += 1

    while stack_size > 0:
        stack_size -= 1
        depth = stack[stack_size, 0]
        index = stack[stack_size, 1]
        maximizingPlayer = stack[stack_size, 2]
        alpha = stack[stack_size, 3]
        beta = stack[stack_size, 4]

        if depth == max_depth:
            result[0] = values[int(index)]
        else:
            if maximizingPlayer:
                optimum = float('-inf')
                for i in range(1, -1, -1):  # Push right child first
                    stack[stack_size, 0] = depth + 1
                    stack[stack_size, 1] = index * 2 + i
                    stack[stack_size, 2] = 0  # False as 0
                    stack[stack_size, 3] = alpha
                    stack[stack_size, 4] = beta
                    stack_size += 1

                while stack_size > 0 and stack[stack_size - 1, 0] == depth + 1:
                    stack_size -= 1
                    idx = stack[stack_size, 1]
                    val = values[int(idx)]
                    optimum = max(optimum, val)
                    alpha = max(alpha, optimum)
                    if beta <= alpha:
                        break
                result[0] = optimum
            else:
                optimum = float('inf')
                for i in range(1, -1, -1):  # Push right child first
                    stack[stack_size, 0] = depth + 1
                    stack[stack_size, 1] = index * 2 + i
                    stack[stack_size, 2] = 1  # True as 1
                    stack[stack_size, 3] = alpha
                    stack[stack_size, 4] = beta
                    stack_size += 1

                while stack_size > 0 and stack[stack_size - 1, 0] == depth + 1:
                    stack_size -= 1
                    idx = stack[stack_size, 1]
                    val = values[int(idx)]
                    optimum = min(optimum, val)
                    beta = min(beta, optimum)
                    if beta <= alpha:
                        break
                result[0] = optimum


#Data
writefile('number_data.bin',100,0,10000)
values = readfile('number_data.bin')

#values = np.array([3, 5, 6, 9, 1, 2, 0, -1], dtype=np.float32)
max_depth = int(math.log(len(values), 2))
result = np.zeros(1, dtype=np.float32)

#Chạy tuần tự
print("Chạy tuần tự:")
start = timeit.default_timer()
result = Alpha_Beta_Pruning_Sequential(values, max_depth)
stop = timeit.default_timer()
print("Trước khi sắp xếp dữ liệu:")
print("Result:", result)
print('Time: ', stop - start)
v1=values
quicksort(v1)
start = timeit.default_timer()
result = Alpha_Beta_Pruning_Sequential(values, max_depth)
stop = timeit.default_timer()
print("Sau khi sắp xếp dữ liệu:")
print("Result:", result)
print('Time: ', stop - start)

#Chạy song song trên CPU
print("\nChạy song song trên CPU:")
start = timeit.default_timer()
result_CPU = Alpha_Beta_Pruning_CPU(values, max_depth)
stop = timeit.default_timer()
print("Trước khi sắp xếp dữ liệu:")
print("Result:", result_CPU)
print('Time: ', stop - start)
v2=values
quicksort(v2)
start = timeit.default_timer()
result_CPU = Alpha_Beta_Pruning_CPU(values, max_depth)
stop = timeit.default_timer()
print("Sau khi sắp xếp dữ liệu:")
print("Result:", result_CPU)
print('Time: ', stop - start)

#Chạy song song trên GPU
result_GPU = np.zeros(1, dtype=np.float32)
v3=values
quicksort(v3)
# Launch kernel with more threads and blocks
threads_per_block = 256
blocks_per_grid = (values.size + (threads_per_block - 1)) // threads_per_block
start = timeit.default_timer()
Alpha_Beta_Pruning_CUDA[blocks_per_grid, threads_per_block](values, max_depth, result_GPU)
stop = timeit.default_timer()
print("\nChạy song song trên GPU CUDA:")
print("Trước khi sắp xếp dữ liệu:")
print("Result:", result_GPU[0])
print('Time: ', stop - start)

result_GPU_2 = np.zeros(1, dtype=np.float32)
threads_per_block = 256
blocks_per_grid = (values.size + (threads_per_block - 1)) // threads_per_block
start = timeit.default_timer()
Alpha_Beta_Pruning_CUDA[blocks_per_grid, threads_per_block](v3, max_depth, result_GPU_2)
stop = timeit.default_timer()
print("Sau khi sắp xếp dữ liệu:")
print("Result:", result_GPU_2[0])
print('Time: ', stop - start)
