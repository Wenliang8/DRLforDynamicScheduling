import numpy as np
from src.simulator.agv_env import AGVEnv
from .drl_agent import DRLAgent

class AGVScheduler:
    def __init__(self, num_agvs=3, layout_size=(10, 10)):
        # Environment + DRL agent
        self.env = AGVEnv(num_agvs, layout_size)
        state_dim = num_agvs * 2 + num_agvs  # positions + busy flags
        action_dim = num_agvs                # choose which AGV to dispatch
        self.agent = DRLAgent(state_dim, action_dim)

        # Training parameters
        self.episodes = 50
        self.max_steps = 200

    # -----------------------------
    # Run one training episode
    # -----------------------------
    def run_episode(self):
        state = self.env.reset()
        total_reward = 0

        for step in range(self.max_steps):
            # Choose AGV using DRL
            action = self.agent.choose_action(state)

            # Environment step
            next_state, reward, done = self.env.step(action)

            # Store experience and learn
            self.agent.store(state, action, reward, next_state)
            self.agent.learn()

            state = next_state
            total_reward += reward

            if done:
                break

        return total_reward

    # -----------------------------
    # Train over multiple episodes
    # -----------------------------
    def train(self):
        for ep in range(self.episodes):
            ep_reward = self.run_episode()
            print(f"Episode {ep+1}/{self.episodes} | Total Reward: {ep_reward:.2f}")

        print("Training completed.")

    # -----------------------------
    # Evaluate without learning
    # -----------------------------
    def evaluate(self, episodes=5):
        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0

            for step in range(self.max_steps):
                action = self.agent.choose_action(state)
                next_state, reward, done = self.env.step(action)
                state = next_state
                total_reward += reward
                if done:
                    break

            print(f"Evaluation Episode {ep+1}: Reward = {total_reward:.2f}")

class TransportManager:
    def __init__(self, agv_scheduler, machine_list):
        self.agv_scheduler = agv_scheduler
        self.machine_list = machine_list

        # Step 5: Map machine index → grid coordinates
        # Simple layout: machines placed in a row (x = m_idx, y = 0)
        self.machine_positions = {
            m.m_idx: (m.m_idx, 0)
            for m in machine_list
        }
    def request_transport(self, job, current_machine, next_machine):
        # For now: immediate teleport (placeholder)
        # Later: replace with AGV DRL + A* movement
        print(f"[DEBUG] TransportManager called: Job {job.j_idx} from M{current_machine}")
        #self.machine_list[next_machine].job_arrival(job)
        #agv_id = 0  # later DRL chooses
        
        #Round‑robin assignment
        #agv_id = current_machine % self.agv_scheduler.env.num_agvs
        import random
        agv_id = random.randint(0, self.agv_scheduler.env.num_agvs - 1)

        pickup = self.machine_positions[current_machine]
        dropoff = self.machine_positions[next_machine]

        # Assign job to AGV
        self.agv_scheduler.env.assign_transport(agv_id, job, pickup, dropoff)

        # ⭐ Move AGV until job is delivered
        while True:
            result = self.agv_scheduler.env.step(agv_id)

            # step() may return 3 or 4 values depending on your version
            if len(result) == 4:
                next_state, reward, done, delivered_job = result
            else:
                next_state, reward, done = result
                delivered_job = None

            if delivered_job is not None:
                # AGV dropped off the job
                self.machine_list[next_machine].job_arrival(delivered_job)
                break
