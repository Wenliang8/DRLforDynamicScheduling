import torch
import numpy as np
import logging
import src.simulator.simulator as sim_module

# Silence logger during training
def dummy_setup_logger(stream=False, keep=10):
    logger = logging.getLogger("sim_logger")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger

sim_module.setup_logger = dummy_setup_logger

from src.simulator.simulator import Shopfloor
from src.scheduler.sequencing_rule import SequencingMethod

num_episodes = 60
base_span = 200

print("=== Starting DQN Offline Training for Multi-AGV Dispatching ===")

saved_agent = None

for ep in range(1, num_episodes + 1):
    seed = 1000 + ep
    np.random.seed(seed)

    config = {
        'seed': seed,
        'span': base_span,
        'm_no': 4,
        'pt_range': [3, 8],
        'E_utliz': 0.85,
        'machine_breakdown': False,
        'processing_time_variability': False,
        'pt_cv': 0,
        'due_tightness': 1.5,
        'stream': False,
        'show': False,
        'sqc_method': SequencingMethod.FIFO,
    }

    spf = Shopfloor(**config)

    # Locate TransportManager
    tm = getattr(spf, 'transport_manager', None)
    if tm is None and len(spf.m_list) > 0:
        tm = getattr(spf.m_list[0], 'transport_manager', None)

    if saved_agent is not None and tm is not None:
        # Carry over learned weights and replay buffer across episodes
        tm.agent = saved_agent

    if tm is not None:
        tm.dispatch_mode = "dqn"
        # Linear epsilon decay from 1.0 down to 0.05
        tm.agent.epsilon = max(0.05, 1.0 - (ep / (num_episodes * 0.8)))

    spf.env.run(until=config['span'])

    if tm is not None:
        saved_agent = tm.agent
        if ep % 10 == 0 or ep == num_episodes:
            rec = spf.recorder
            completed = len(rec.j_flowtime_dict)
            steps = getattr(tm.agv_env, 'total_travel_steps', 0)
            print(f"Episode {ep:02d}/{num_episodes} | Epsilon: {tm.agent.epsilon:.2f} | Completed: {completed} | Total AGV Steps: {steps}")

# Save the trained model weights
if saved_agent is not None and hasattr(saved_agent, 'q_net'):
    torch.save(saved_agent.q_net.state_dict(), "agv_dqn_weights.pth")
    print("\n[SUCCESS] Trained model weights saved to 'agv_dqn_weights.pth'!")
elif saved_agent is not None and hasattr(saved_agent, 'model'):
    torch.save(saved_agent.model.state_dict(), "agv_dqn_weights.pth")
    print("\n[SUCCESS] Trained model weights saved to 'agv_dqn_weights.pth'!")