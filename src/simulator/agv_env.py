import numpy as np
from src.simulator.a_star import AStar

class AGVEnv:
    def __init__(self, num_agvs, layout_size=(10, 10), max_time=200):
        self.num_agvs = num_agvs
        self.layout_size = layout_size
        self.max_time = max_time

        # A* router
        self.router = AStar(layout_size)

        # AGV states
        self.positions = np.zeros((num_agvs, 2))      # (x, y)
        self.busy = np.zeros(num_agvs)                # 0 = free, 1 = busy
        self.carrying_job = [None] * num_agvs
        self.state = ["idle"] * num_agvs              # idle, moving_to_pickup, moving_to_dropoff
        self.paths = [None] * num_agvs

        # Machine coordinates (REQUIRED for pickup/dropoff)
        self.machine_positions = {
            0: (0, 0),
            1: (3, 0),
            2: (6, 0),
            3: (9, 0)
        }

        # Episode tracking
        self.time = 0

    # -----------------------------
    # Reset environment
    # -----------------------------
    def reset(self):
        self.positions = np.random.randint(0, self.layout_size[0], (self.num_agvs, 2))
        self.busy = np.zeros(self.num_agvs)
        self.carrying_job = [None] * self.num_agvs
        self.state = ["idle"] * self.num_agvs
        self.paths = [None] * self.num_agvs
        self.time = 0

        return self.get_state()

    # -----------------------------
    # Build state vector
    # -----------------------------
    def get_state(self):
        # AGV part: positions + busy flags
        agv_state = np.concatenate([
            self.positions.flatten(),
            self.busy
        ])

        # Time (normalized)
        time_state = np.array([self.time / self.max_time])

        return np.concatenate([agv_state, time_state])

    # -----------------------------
    # Assign transport for one AGV + one job
    # -----------------------------
    def assign_transport(self, agv_id, job, pickup, dropoff):
        self.carrying_job[agv_id] = job
        self.busy[agv_id] = 1

        px, py = self.machine_positions[pickup]
        dx, dy = self.machine_positions[dropoff]

        job.pickup = (px, py)
        job.dropoff = (dx, dy)

        start = tuple(self.positions[agv_id])
        path_to_pickup = self.router.search(start, job.pickup)

        if not path_to_pickup or len(path_to_pickup) < 2:
            # no path → leave AGV idle
            self.busy[agv_id] = 0
            self.carrying_job[agv_id] = None
            self.state[agv_id] = "idle"
            self.paths[agv_id] = None
        else:
            self.paths[agv_id] = path_to_pickup[1:]  # skip current
            self.state[agv_id] = "moving_to_pickup"
            # In agv_env.py
    def assign_transport(self, agv_id, job, pickup, dropoff):
        self.carrying_job[agv_id] = job
        self.busy[agv_id] = 1

        px, py = self.machine_positions[pickup]
        dx, dy = self.machine_positions[dropoff]

        job.pickup = (px, py)
        job.dropoff = (dx, dy)

        start = tuple(self.positions[agv_id])
    
        # Path to pickup
        path_to_pickup = self.router.search(start, job.pickup)
        # Path from pickup to dropoff
        path_to_dropoff = self.router.search(job.pickup, job.dropoff)

        total_path = []
        if path_to_pickup and len(path_to_pickup) > 1:
            total_path.extend(path_to_pickup[1:])
        if path_to_dropoff and len(path_to_dropoff) > 1:
            total_path.extend(path_to_dropoff[1:])

        if not total_path:
            self.busy[agv_id] = 0
            self.carrying_job[agv_id] = None
            self.state[agv_id] = "idle"
            self.paths[agv_id] = None
            return 0  # 0 travel steps

        self.paths[agv_id] = total_path
        self.state[agv_id] = "moving_to_pickup"
    
        # Return path length / travel time to caller so discrete-event simulation delays arrival
        return len(total_path)
    def assign_transport(self, agv_id, job, pickup, dropoff):
        self.carrying_job[agv_id] = job
        self.busy[agv_id] = 1

        px, py = self.machine_positions[pickup]
        dx, dy = self.machine_positions[dropoff]

        job.pickup = (px, py)
        job.dropoff = (dx, dy)

        start = tuple(self.positions[agv_id])

        # 1. Path from AGV current position to pickup machine
        path_to_pickup = self.router.search(start, job.pickup)
        # 2. Path from pickup machine to dropoff machine
        path_to_dropoff = self.router.search(job.pickup, job.dropoff)

        total_path = []
    
        # Add pickup steps (excluding starting position)
        if path_to_pickup and len(path_to_pickup) > 1:
            total_path.extend(path_to_pickup[1:])
        
        # Add dropoff steps (excluding starting pickup location)
        if path_to_dropoff and len(path_to_dropoff) > 1:
            total_path.extend(path_to_dropoff[1:])

        if not total_path:
            # AGV is already at target dropoff or no route exists
            self.busy[agv_id] = 0
            self.carrying_job[agv_id] = None
            self.state[agv_id] = "idle"
            self.paths[agv_id] = None
            return 0  # 0 travel steps

        self.paths[agv_id] = total_path
    
        # Set initial transport state
        if path_to_pickup and len(path_to_pickup) > 1:
            self.state[agv_id] = "moving_to_pickup"
        else:
            self.state[agv_id] = "moving_to_dropoff"
        return len(total_path)

    # -----------------------------
    # Step function
    # action = (agv_id, job_id) or (agv_id, -1) for wait
    # -----------------------------
    def step(self, action):
        reward = 0
        done = False

        agv_id, job_id = action

        # "wait" action
        if job_id == -1:
            reward -= 0.5
        else:
            # Move AGV if busy
            if self.busy[agv_id] == 1:
                reward += self._move_agv(agv_id)

        # Time update
        self.time += 1
        if self.time >= self.max_time:
            done = True

        next_state = self.get_state()
        return next_state, reward, done

    # -----------------------------
    # Internal: move AGV along path
    # -----------------------------
    def _move_agv(self, agv_id):
        reward = 0
        job = self.carrying_job[agv_id]

        if self.paths[agv_id] and len(self.paths[agv_id]) > 0:
            next_step = self.paths[agv_id].pop(0)
            self.positions[agv_id] = np.array(next_step)
            reward -= 0.1  # small travel penalty

            # Check pickup
            if self.state[agv_id] == "moving_to_pickup":
                if tuple(self.positions[agv_id]) == job.pickup:
                    path_to_dropoff = self.router.search(job.pickup, job.dropoff)
                    if path_to_dropoff and len(path_to_dropoff) > 1:
                        self.paths[agv_id] = path_to_dropoff[1:]
                        self.state[agv_id] = "moving_to_dropoff"
                    else:
                        reward -= 5

            # Check dropoff
            elif self.state[agv_id] == "moving_to_dropoff":
                if tuple(self.positions[agv_id]) == job.dropoff:
                    self.busy[agv_id] = 0
                    self.state[agv_id] = "idle"
                    self.carrying_job[agv_id] = None
                    self.paths[agv_id] = None
                    reward += 10  # success
        else:
            # no path left → idle
            self.busy[agv_id] = 0
            self.state[agv_id] = "idle"
            self.paths[agv_id] = None

        return reward
