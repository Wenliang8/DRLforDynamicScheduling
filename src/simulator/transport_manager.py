class TransportManager:
    def __init__(self, agv_env, agent, decode_action, num_agvs, env=None):
        self.agv_env = agv_env
        self.agent = agent
        self.decode_action = decode_action
        self.num_agvs = num_agvs
        self.env = env  # SimPy or discrete-event simulation environment instance

    def request_transport(self, job, current_machine, next_machine, env=None):
        # Use passed-in environment or fall back to class instance
        sim_env = env if env is not None else self.env

        # Assign pickup/dropoff coordinates
        job.pickup = self.agv_env.machine_positions[current_machine]
        job.dropoff = self.agv_env.machine_positions[next_machine]

        # Build AGVEnv state
        state = self.agv_env.get_state()

        # DQN chooses action
        action_int = self.agent.select_action(state)

        # Decode action → (agv_id, job_id)
        agv_id, job_id = self.decode_action(action_int, self.num_agvs)

        # Force job_id to match actual job
        job_id = job.j_idx

        print(f"[TM] Job {job.j_idx} requesting transport from M{current_machine} → M{next_machine}")
        print(f"[TM] DQN selected action {action_int} → AGV {agv_id}")

        # Assign job to AGVEnv and get total travel steps
        travel_time = self.agv_env.assign_transport(
            agv_id,
            job,
            pickup=current_machine,
            dropoff=next_machine
        )

        # Step AGV movement through time ticks
        if sim_env is not None and hasattr(sim_env, 'timeout') and travel_time > 0:
            for _ in range(travel_time):
                yield sim_env.timeout(1)  # Advance simulation time tick
                next_state, reward, done = self.agv_env.step((agv_id, job_id))

                self.agent.store(state, action_int, reward, next_state, done)
                self.agent.train()

                state = next_state
        else:
            # Fallback if running outside a SimPy event loop
            for _ in range(travel_time):
                next_state, reward, done = self.agv_env.step((agv_id, job_id))
                self.agent.store(state, action_int, reward, next_state, done)
                self.agent.train()
                state = next_state

        print(f"[AGVEnv] Job {job.j_idx} delivered to machine {next_machine}")

        # Deliver job to machine queue
        job.machine_list[next_machine].job_arrival(job)