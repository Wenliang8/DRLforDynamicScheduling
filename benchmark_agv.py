import pandas as pd
import numpy as np
import random
import logging
import src.simulator.simulator as sim_module

# Bypass utilities.py logger bug
def dummy_setup_logger(stream=False, keep=10):
    logger = logging.getLogger("sim_logger")
    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    logger.setLevel(logging.ERROR)  # Keep terminal clean for benchmark table
    return logger

sim_module.setup_logger = dummy_setup_logger

from src.simulator.simulator import Shopfloor
from src.scheduler.sequencing_rule import SequencingMethod

seeds = [42, 100, 2024]
loads = [0.75, 0.90]
modes = ["random", "nearest", "dqn"]

results = []  # <--- MUST BE OUTSIDE ALL LOOPS

for load in loads:
    for mode in modes:
        for seed in seeds:
            # 1. Reset seed
            np.random.seed(seed)
            random.seed(seed)

            # 2. Setup config & shopfloor
            config = {
                'seed': seed,
                'span': 200,
                'm_no': 4,
                'pt_range': [3, 8],
                'E_utliz': load,
                'machine_breakdown': False,
                'processing_time_variability': False,
                'pt_cv': 0,
                'due_tightness': 1.5,
                'stream': False,
                'show': False,
                'sqc_method': SequencingMethod.FIFO,
            }

            spf = Shopfloor(**config)

            # 3. Inject dispatch_mode & load weights
            tm = getattr(spf, 'transport_manager', None)
            if tm is None and len(spf.m_list) > 0:
                tm = getattr(spf.m_list[0], 'transport_manager', None)

            if tm is not None:
                tm.dispatch_mode = mode
                for m in spf.m_list:
                    if hasattr(m, 'transport_manager') and m.transport_manager is not None:
                        m.transport_manager.dispatch_mode = mode

                if mode == "dqn":
                    try:
                        if hasattr(tm.agent, 'q_net'):
                            tm.agent.q_net.load_state_dict(torch.load("agv_dqn_weights.pth"))
                        elif hasattr(tm.agent, 'model'):
                            tm.agent.model.load_state_dict(torch.load("agv_dqn_weights.pth"))
                    except Exception:
                        pass
                    if hasattr(tm.agent, 'epsilon'):
                        tm.agent.epsilon = 0.0

            # 4. Run simulation
            spf.env.run(until=config['span'])

            # 5. Extract metrics
            rec = spf.recorder
            n_jobs = len(rec.j_flowtime_dict)
            makespan = rec.last_job_comp_T if n_jobs > 0 else 0
            mean_flow = sum(rec.j_flowtime_dict.values()) / n_jobs if n_jobs > 0 else 0
            mean_tard = sum(rec.j_tardiness_dict.values()) / n_jobs if n_jobs > 0 else 0

            total_steps = getattr(tm.agv_env, 'total_travel_steps', 0) if tm else 0
            wait_steps = getattr(tm.agv_env, 'wait_steps', 0) if tm else 0
            busy_ticks = getattr(tm.agv_env, 'agv_busy_ticks', np.zeros(1)) if tm else np.zeros(1)
            agv_util = float(np.mean(np.minimum(busy_ticks, config['span'])) / config['span']) * 100

            results.append({
                "Load": f"{int(load*100)}%",
                "Policy": mode.upper(),
                "Seed": seed,
                "Completed": n_jobs,
                "Makespan": round(makespan, 1),
                "Mean_Flowtime": round(mean_flow, 2),
                "Mean_Tardiness": round(mean_tard, 2),
                "AGV_Steps": int(total_steps),
                "Wait_Steps": int(wait_steps),
                "AGV_Util(%)": round(agv_util, 1)
            })

# Print Full Table (All Seeds)
df = pd.DataFrame(results)
print("\n==================== ALL SEED RUNS ====================")
print(df.to_string(index=False))

# Print Grouped Summary (Mean across seeds for Chapter 4)
summary_df = df.groupby(["Load", "Policy"]).agg({
    "Completed": "mean",
    "Makespan": "mean",
    "Mean_Flowtime": "mean",
    "Mean_Tardiness": "mean",
    "AGV_Steps": "mean",
    "Wait_Steps": "mean",
    "AGV_Util(%)": "mean"
}).round(2).reset_index()

print("\n==================== AGGREGATED BENCHMARK SUMMARY (MEAN) ====================")
print(summary_df.to_string(index=False))