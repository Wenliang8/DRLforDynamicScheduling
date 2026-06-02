import matplotlib.pyplot as plt
import time

from src.simulator.simulator import Shopfloor
from src.scheduler.sequencing_rule import SequencingMethod


# -----------------------------
# 1. Full simulator config
# -----------------------------
config = {
    "m_no": 4,
    "j_no": 3,

    "pt_range": [3, 8],
    "pt_cv": 0,
    "due_tightness": 1.5,
    "processing_time_variability": False,

    "E_utliz": 0.85,
    "span": 200,

    "sqc_method": SequencingMethod.FIFO,

    "machine_breakdown": False,
    "random_MTBF": False,
    "random_MTTR": False,
    "MTBF": 99999,
    "MTTR": 0,

    "stream": True,
    "seed": 12345,
    "draw_gantt": False,
}


# -----------------------------
# 2. Create real shopfloor
# -----------------------------
shop = Shopfloor(**config)
env = shop.env
machines = shop.m_list

# Narrator already created:
#   - AGVScheduler
#   - TransportManager
#   - machine.transport_manager = tm
tm = shop.narrator.transport_manager
agv_sched = shop.narrator.agv_scheduler


# -----------------------------
# 3. Visualization setup
# -----------------------------
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")

ax.set_title("AGV Movement (Real Job-Shop Integration)")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Manual machine layout
machine_positions = [(0, 0), (3, 0), (6, 0), (9, 0)]
for i, (mx, my) in enumerate(machine_positions):
    ax.plot(mx, my, "s", color="black", markersize=8)
    ax.text(mx + 0.2, my + 0.2, f"M{i}", color="black")

# AGV visuals
colors = ["red", "blue", "green"]
points = [ax.plot([], [], "o", color=colors[i])[0] for i in range(agv_sched.env.num_agvs)]
labels = [ax.text(0, 0, f"AGV{i}", color=colors[i]) for i in range(agv_sched.env.num_agvs)]
paths = [[] for _ in range(agv_sched.env.num_agvs)]


# -----------------------------
# 4. Run simulation + animate
# -----------------------------
MAX_STEPS = 500

for t in range(MAX_STEPS):

    # Step the SimPy environment (this moves machines, jobs, AND AGVs)
    env.step()
    
    # CLEAR previous AGV drawings
    for p in points:
        p.set_data([], [])

    # Update AGV visuals
    for i in range(agv_sched.env.num_agvs):
        x, y = agv_sched.env.positions[i]
        # Show only the current AGV position
        points[i].set_data([x], [y])
        labels[i].set_position((x + 0.2, y + 0.2))

        
        #paths[i].append((x, y))
        #xs, ys = zip(*paths[i])
        #points[i].set_data(xs, ys)
        #labels[i].set_position((x + 0.2, y + 0.2))

    plt.pause(0.1)

plt.ioff()
plt.show()
