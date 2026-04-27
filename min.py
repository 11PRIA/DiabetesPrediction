import math

def minimax(depth, node_index, is_max, scores, height):
    # If leaf node is reached
    if depth == height:
        return scores[node_index]

    # If Maximizer's turn
    if is_max:
        return max(
            minimax(depth + 1, node_index * 2, False, scores, height),
            minimax(depth + 1, node_index * 2 + 1, False, scores, height)
        )
    # If Minimizer's turn
    else:
        return min(
            minimax(depth + 1, node_index * 2, True, scores, height),
            minimax(depth + 1, node_index * 2 + 1, True, scores, height)
        )

# Driver code
if __name__ == "__main__":
    # Leaf node values
    scores = [3, 5, 2, 9, 12, 5, 23, 23]

    # Height of the game tree
    height = int(math.log2(len(scores)))

    optimal_value = minimax(0, 0, True, scores, height)

    print("The optimal value is:", optimal_value)
