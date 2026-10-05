class TransportManager:
    def __init__(self, agv_env, agent, decode_action, num_agvs):
        self.agv_env = agv_env
        self.agent = agent
        self.decode_action = decode_action
        self.num_agvs = num_agvs

    def request_transport(self, job, current_machine, next_machine):

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

        # Assign job to AGVEnv
        self.agv_env.assign_transport(
            agv_id,
            job,
            pickup=current_machine,
            dropoff=next_machine
        )

        # Simulate AGV movement until job delivered
        done = False
        while not done:
            next_state, reward, done = self.agv_env.step((agv_id, job_id))

            self.agent.store(state, action_int, reward, next_state, done)
            self.agent.train()

            state = next_state

        print(f"[AGVEnv] Job {job.j_idx} delivered to machine {next_machine}")

        job.machine_list[next_machine].job_arrival(job)
