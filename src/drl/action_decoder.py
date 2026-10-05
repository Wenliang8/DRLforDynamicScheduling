def decode_action(action_int, num_agvs):
    # AGV actions: 0..num_agvs-1
    if action_int < num_agvs:
        return action_int, 0   # job_id always 0 (TransportManager overrides)
    # Wait action
    return 0, -1
