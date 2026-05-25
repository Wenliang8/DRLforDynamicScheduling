import heapq

class AStar:
    def __init__(self, grid_size=(10, 10)):
        self.grid_size = grid_size

    # -----------------------------------
    # Heuristic (Manhattan distance)
    # -----------------------------------
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # -----------------------------------
    # Get valid neighbors
    # -----------------------------------
    def get_neighbors(self, node):
        x, y = node
        neighbors = []

        # 4-direction movement
        directions = [(1,0), (-1,0), (0,1), (0,-1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check boundaries
            if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                neighbors.append((nx, ny))

        return neighbors

    # -----------------------------------
    # A* Search
    # -----------------------------------
    def search(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            # Goal reached
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current

        return []  # No path found

    # -----------------------------------
    # Reconstruct path
    # -----------------------------------
    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
