import numpy as np
from src.simulator.a_star import AStar

class AGVEnv:
    def __init__(self, num_agvs, layout_size=(10, 10)):
        self.num_agvs = num_agvs
        self.layout_size = layout_size

        # A* router
        self.router = AStar(layout_size)

        # AGV states
        self.positions = np.zeros((num_agvs, 2))  # (x, y)
        self.busy = np.zeros(num_agvs)            # 0 = free, 1 = busy
        self.targets = [None] * num_agvs          # target positions
        self.paths = [None] * num_agvs            # A* path for each AGV
        # Job transport attributes
        self.carrying_job = [None] * num_agvs
        self.pickup_location = [None] * num_agvs
        self.dropoff_location = [None] * num_agvs
        self.state = ["idle"] * num_agvs   # idle, moving_to_pickup, moving_to_dropoff
        self.job_to_transport = [None] * num_agvs

        # Episode tracking
        self.time = 0
        self.max_time = 200

    # -----------------------------
    # Reset environment
    # -----------------------------
    def reset(self):
        self.positions = np.random.randint(0, self.layout_size[0], (self.num_agvs, 2))
        self.busy = np.zeros(self.num_agvs)
        self.targets = [None] * self.num_agvs
        self.paths = [None] * self.num_agvs
        self.time = 0

        return self.get_state()

    # -----------------------------
    # Build state vector
    # -----------------------------
    def get_state(self):
        state = np.concatenate([
            self.positions.flatten(),
            self.busy
        ])
        return state

    # -----------------------------
    # Step function (AGV movement)
    # -----------------------------
    def step(self, action):
        reward = 0

        # If AGV is free → assign a random target + compute A* path
        if self.busy[action] == 0:
            target = np.random.randint(0, self.layout_size[0], 2)
            self.targets[action] = target

            start = tuple(self.positions[action])
            goal = tuple(target)

            path = self.router.search(start, goal)

            if len(path) > 1:
                self.paths[action] = path[1:]  # skip current position
                self.busy[action] = 1
            else:
                reward = -5  # no path found

        # If AGV is busy → follow A* path
        if self.busy[action] == 1:
            if self.paths[action] and len(self.paths[action]) > 0:
                next_step = self.paths[action].pop(0)
                self.positions[action] = np.array(next_step)

                x, y = map(float, self.positions[action])
                print(f"[AGV] AGV {action} moved to ({x}, {y})")


                reward = -1  # travel penalty

                # Check pickup
                if self.state[action] == "moving_to_pickup":
                    if tuple(self.positions[action]) == self.pickup_location[action]:
                        self.pickup(action)
                        # Compute path to dropoff
                        path = self.router.search(self.pickup_location[action], self.dropoff_location[action])
                        self.paths[action] = path[1:]
                        self.state[action] = "moving_to_dropoff"

                # Check dropoff
                elif self.state[action] == "moving_to_dropoff":
                    if tuple(self.positions[action]) == self.dropoff_location[action]:
                        job = self.dropoff(action)
                        self.state[action] = "idle"
                        self.busy[action] = 0
                        return self.get_state(), 10, False, job

            else:
                # Task completed
                self.busy[action] = 0
                self.targets[action] = None
                reward = 10  # success reward

        # Time penalty
        self.time += 1
        done = self.time >= self.max_time

        next_state = self.get_state()
        return next_state, reward, done

    def pickup(self, agv_id):
        job = self.job_to_transport[agv_id]
        self.carrying_job[agv_id] = job
        print(f"[AGV] AGV {agv_id} picked up Job {job.j_idx}")

    def dropoff(self, agv_id):
        job = self.carrying_job[agv_id]
        self.carrying_job[agv_id] = None
        print(f"[AGV] AGV {agv_id} dropped off Job {job.j_idx}")
        return job

    def assign_transport(self, agv_id, job, pickup, dropoff):
        # Store job and locations
        self.job_to_transport[agv_id] = job
        self.pickup_location[agv_id] = tuple(pickup)
        self.dropoff_location[agv_id] = tuple(dropoff)
        self.state[agv_id] = "moving_to_pickup"

        # Compute A* path to pickup
        start = tuple(self.positions[agv_id])
        #path = self.router.search(start, self.pickup_location[agv_id])
        path = [(start), self.pickup_location[agv_id]]

        if len(path) > 1:
            self.paths[agv_id] = path[1:]
            self.busy[agv_id] = 1
        else:
            print("[AGV] No path to pickup")
