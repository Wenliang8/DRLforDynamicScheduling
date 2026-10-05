env = AGVEnv(num_agvs=2, jobs=job_list)
state_dim = len(env.get_state())
action_dim = env.num_agvs * len(env.jobs) + 1

agent = DQNAgent(state_dim, action_dim)

for episode in range(500):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action_int = agent.select_action(state)
        action = decode_action(action_int, env.num_agvs, len(env.jobs))

        next_state, reward, done = env.step(action)
        agent.store(state, action_int, reward, next_state, done)
        agent.train()

        state = next_state
        total_reward += reward

    agent.update_target()
    print(f"Episode {episode}, reward = {total_reward}")
