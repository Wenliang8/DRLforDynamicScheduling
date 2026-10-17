import numpy as np

class TransportManager:
    def __init__(self, agv_env, agent, decode_action, num_agvs, env=None):
        self.agv_env = agv_env
        self.agent = agent
        self.decode_action = decode_action
        self.num_agvs = num_agvs
        self.env = env  # SimPy or discrete-event simulation environment instance

    def select_agv(self, current_machine):
        # Read mode directly from instance or fallback to 'dqn'
        mode = getattr(self, 'dispatch_mode', 'dqn')
        pickup_pos = np.array(self.agv_env.machine_positions[current_machine])
        
        # Find idle AGVs first, otherwise use all
        idle_agvs = [i for i in range(self.num_agvs) if self.agv_env.busy[i] == 0]
        candidate_agvs = idle_agvs if len(idle_agvs) > 0 else list(range(self.num_agvs))

        if mode == "random":
            chosen_agv = int(np.random.choice(candidate_agvs))
            return chosen_agv, chosen_agv

        elif mode == "nearest":
            dists = [np.linalg.norm(self.agv_env.positions[i] - pickup_pos, ord=1) for i in candidate_agvs]
            chosen_agv = int(candidate_agvs[int(np.argmin(dists))])
            return chosen_agv, chosen_agv

        else:  # "dqn"
            state = self.agv_env.get_state()
            action_int = self.agent.select_action(state)
            agv_id, _ = self.decode_action(action_int, self.num_agvs)
            if agv_id not in candidate_agvs:
                agv_id = candidate_agvs[0]
                action_int = agv_id
            return int(agv_id), int(action_int)

    def request_transport(self, job, current_machine, next_machine, env=None):
        sim_env = env if env is not None else self.env

        job.pickup = self.agv_env.machine_positions[current_machine]
        job.dropoff = self.agv_env.machine_positions[next_machine]

        # Select AGV based on active policy
        agv_id, action_int = self.select_agv(current_machine)
        job_id = job.j_idx
        state = self.agv_env.get_state()

        # Get A* travel time
        travel_time = self.agv_env.assign_transport(
            agv_id, job, pickup=current_machine, dropoff=next_machine
        )
        if travel_time is None or travel_time <= 0:
            travel_time = 2

        # Step through SimPy time ticks smoothly
        if sim_env is not None and hasattr(sim_env, 'timeout'):
            for _ in range(travel_time):
                yield sim_env.timeout(1)
                next_state, reward, done = self.agv_env.step((agv_id, job_id))
                
                if getattr(self, 'dispatch_mode', 'dqn') == 'dqn':
                    self.agent.store(state, action_int, reward, next_state, done)
                    self.agent.train()
                state = next_state
        else:
            for _ in range(travel_time):
                next_state, reward, done = self.agv_env.step((agv_id, job_id))
                state = next_state

        # Deliver job to destination machine queue
        job.machine_list[next_machine].job_arrival(job)