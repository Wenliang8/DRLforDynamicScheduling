import matplotlib.pyplot as plt
import time
from src.scheduler.agv_scheduler import AGVScheduler, TransportManager

# Fake machine class for testing
class FakeMachine:
    def __init__(self, m_idx):
        self.m_idx = m_idx

    def job_arrival(self, job):
        print(f"[Machine {self.m_idx}] Job {job.j_idx} arrived")

# Fake job
class FakeJob:
    def __init__(self, j_idx):
        self.j_idx = j_idx

# Initialize environment
scheduler = AGVScheduler(num_agvs=3, layout_size=(10, 10))
machines = [FakeMachine(i) for i in range(4)]
tm = TransportManager(scheduler, machines)

jobs = [FakeJob(i) for i in range(3)]

# Dispatch jobs
for i, job in enumerate(jobs):
    tm.request_transport(job, current_machine=i, next_machine=(i + 1) % len(machines))

# Visualization setup
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Add this:
machine_positions = [(0, 0), (3, 0), (6, 0), (9, 0)]
for mx, my in machine_positions:
    ax.plot(mx, my, "s", color="black", markersize=8)

ax.set_title("AGV Movement Visualization")
ax.set_xlabel("X")
ax.set_ylabel("Y")

colors = ["red", "blue", "green"]
points = [ax.plot([], [], "o", color=colors[i])[0] for i in range(3)]

# Add this:
labels = [ax.text(0, 0, f"AGV{i}", color=colors[i]) for i in range(3)]
paths = [[] for _ in range(scheduler.env.num_agvs)]

# Animate AGV movement
for t in range(50):
    for i in range(scheduler.env.num_agvs):
        scheduler.env.step(i)
        x, y = scheduler.env.positions[i]

        # update path
        paths[i].append((x, y))
        xs, ys = zip(*paths[i])
        points[i].set_data(xs, ys)

        # update label
        labels[i].set_position((x + 0.2, y + 0.2))

    plt.pause(0.2)

plt.ioff()
plt.show()