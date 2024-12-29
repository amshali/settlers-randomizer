import numpy as np
import gym
from gym import spaces
import pickle


def pairwise_distances(array):
    n = len(array)
    if n < 2:
        raise ValueError("Array must contain at least two elements.")

    total_distance = 0
    max_distance = float("-inf")

    # Compute all pairwise distances
    for i in range(n):
        for j in range(i + 1, n):  # Only compute for pairs (i, j) with i < j
            distance = abs(array[i] - array[j])
            total_distance += distance
            max_distance = max(max_distance, distance)

    # Calculate average pairwise distance
    num_pairs = n * (n - 1) // 2  # Number of unique pairs
    avg_distance = total_distance / num_pairs

    return avg_distance, max_distance


class CatanBoardEnv(gym.Env):
    def __init__(self, reward_function):
        super(CatanBoardEnv, self).__init__()
        # Define the board layout
        self.tiles = 19  # Total tiles (including Desert)
        self.reward_function = reward_function
        self.good_assignments = {}
        self.dice_numbers = [
            2,
            3,
            3,
            4,
            4,
            5,
            5,
            6,
            6,
            7,
            8,
            8,
            9,
            9,
            10,
            10,
            11,
            11,
            12,
        ]
        self.probability_points = {
            7: 0,
            2: 1,
            12: 1,
            3: 2,
            11: 2,
            4: 3,
            10: 3,
            5: 4,
            9: 4,
            6: 5,
            8: 5,
        }

        # Action space: Swap dice numbers between any two tiles
        self.action_space = spaces.Discrete(self.tiles * (self.tiles - 1) // 2)

        # Observation space: Current dice numbers on tiles
        self.observation_space = spaces.Box(
            low=2, high=12, shape=(self.tiles,), dtype=np.int32
        )

        self.reset()

    def reset(self):
        # Randomly shuffle the dice numbers at the start
        self.current_assignment = np.random.permutation(self.dice_numbers)
        return self.current_assignment

    def step(self, action):
        # Decode the action (index to tile pairs)
        tile_a, tile_b = self.decode_action(action)

        # Swap the dice numbers
        self.current_assignment[tile_a], self.current_assignment[tile_b] = (
            self.current_assignment[tile_b],
            self.current_assignment[tile_a],
        )

        # Calculate reward (negative variance of cumulative probabilities)
        mean, variance = self.calculate_cumulative_probability_mean_variance()
        reward = self.reward_function(mean, variance)

        # Check if we want to terminate (fixed number of steps or optimal variance)
        done = False
        if reward > 0:
            done = True
            self.good_assignments[tuple(self.current_assignment)] = (mean, variance)
            print(f"Good assignments: {len(self.good_assignments)}, {(mean, variance)}")

        return (
            self.current_assignment,
            reward,
            done,
            {"mean": mean, "variance": variance},
        )

    def decode_action(self, action):
        # Map action index to pair of tiles
        tile_a = action // (self.tiles - 1)
        tile_b = action % (self.tiles - 1)
        if tile_b >= tile_a:
            tile_b += 1
        return tile_a, tile_b

    def calculate_cumulative_probability(self):
        # Calculate the cumulative probabilities at all intersection points
        # (For simplicity, assume intersections are pre-defined)
        cumulative_probs = []
        for point in self.get_intersection_points():
            cumulative_probs.append(
                sum(self.probability_points[self.current_assignment[t]] for t in point)
            )
        return cumulative_probs

    def calculate_cumulative_probability_mean_variance(self):
        # Calculate the cumulative probabilities at all intersection points
        # (For simplicity, assume intersections are pre-defined)
        cumulative_probs = self.calculate_cumulative_probability()

        # Compute variance of cumulative probabilities
        mean = np.mean(cumulative_probs)
        variance = np.mean((np.array(cumulative_probs) - mean) ** 2)
        return mean, variance

    def get_intersection_points(self):
        # Define intersection points as groups of tile indices
        # This will depend on the specific board layout
        return [
            [0, 3, 4],
            [0, 1, 4],
            [1, 4, 5],
            [1, 2, 5],
            [2, 5, 6],
            [3, 7, 8],
            [3, 4, 8],
            [4, 8, 9],
            [4, 5, 9],
            [5, 9, 10],
            [5, 6, 10],
            [6, 10, 11],
            [7, 8, 12],
            [8, 12, 13],
            [8, 9, 13],
            [9, 13, 14],
            [9, 10, 14],
            [10, 14, 15],
            [10, 11, 15],
            [12, 13, 16],
            [13, 16, 17],
            [13, 14, 17],
            [14, 17, 18],
            [14, 15, 18],
        ]

    def render_board(self):
        print("\nTile Assignments (Tile Index -> Dice Number):")
        for i, number in enumerate(self.current_assignment):
            print(f"Tile {i}: {number}")

        cumulative_probs = []
        for point in self.get_intersection_points():
            prob = sum(
                self.probability_points[self.current_assignment[t]] for t in point
            )
            cumulative_probs.append(prob)

        print("\nIntersection Cumulative Probabilities:")
        for i, prob in enumerate(cumulative_probs):
            print(f"Intersection {i}: {prob}")

        variance = np.var(cumulative_probs)
        print(f"\nCumulative Probability Variance: {variance}")
        return variance

    def save_good_assignments(self, filename):
        print(f"Good assignments: {len(self.good_assignments)}")
        with open(filename, mode="wb") as f:
            pickle.dump(self.good_assignments, f)
