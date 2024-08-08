import math
from numba import njit, prange
import timeit

def Alpha_Beta_Pruning_Recursion(depth, index, maximizingPlayer, values, alpha, beta):
    # Terminating condition
    if depth == int(math.log(len(values),2)):
        return values[index]
    if maximizingPlayer:
        optimum = float('-inf')
        # Recursion for left and right children
        for i in range(0, int(math.log(len(values),2))-1):
            val = Alpha_Beta_Pruning_Recursion(depth + 1, index * 2 + i,False, values, alpha, beta)
            optimum = max(optimum, val)
            alpha = max(alpha, optimum)
            # Alpha Beta Pruning condition
            if beta <= alpha:
                break
        return optimum
    else:
        optimum = float('+inf')
        # Recursion for left and right children
        for i in range(0, int(math.log(len(values),2))-1):
            val = Alpha_Beta_Pruning_Recursion(depth + 1, index * 2 + i,True, values, alpha, beta)
            optimum = min(optimum, val)
            beta = min(beta, optimum)
            # Alpha Beta Pruning
            if beta <= alpha:
                break
        return optimum

@njit(fastmath=True, cache=True)
def Alpha_Beta_Pruning_Without_Recursion(values, max_depth):
    stack = [(0, 0, True, float('-inf'), float('+inf'))]
    result = None
    while stack:
        depth, index, maximizingPlayer, alpha, beta = stack.pop()
        if depth == max_depth:
            result = values[index]
        else:
            if maximizingPlayer:
                optimum = float('-inf')
                for i in prange(1, -1, -1):  # Push right child first
                    stack.append((depth + 1, index * 2 + i, False, alpha, beta))
                while stack and stack[-1][0] == depth + 1:
                    _, idx, _, a, b = stack.pop()
                    val = values[idx]
                    optimum = max(optimum, val)
                    alpha = max(alpha, optimum)
                    if beta <= alpha:
                        break
                result = optimum
            else:
                optimum = float('+inf')
                for i in prange(1, -1, -1):  # Push right child first
                    stack.append((depth + 1, index * 2 + i, True, alpha, beta))
                while stack and stack[-1][0] == depth + 1:
                    _, idx, _, a, b = stack.pop()
                    val = values[idx]
                    optimum = min(optimum, val)
                    beta = min(beta, optimum)
                    if beta <= alpha:
                        break
                result = optimum
    return result

# Main Code

if __name__ == "__main__":
    array = [5,6,7,4,5,3,6,6,9,7,5,9,8,6]
    max_depth = int(math.log(len(array), 2))
    start = timeit.default_timer()
#    result = Alpha_Beta_Pruning_Recursion(0, 0, True, array, float('-inf'), float('+inf'))
    result = Alpha_Beta_Pruning_Without_Recursion(array, max_depth)
    stop = timeit.default_timer()
    print("The result is:", result, "\n")
    print("The time taken is:", stop - start, " (second)")
