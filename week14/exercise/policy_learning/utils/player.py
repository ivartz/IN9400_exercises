class Player():
    def __init__(self, model, modelParam, config, saveRestorer, env):
        self.model = model
        self.modelParam   = modelParam
        self.config       = config
        self.saveRestorer = saveRestorer
        self.env          = env

        #initialize network
        model.policyNet.eval()
        return

    def play_episode(self):
        ep_log_probs = []
        ep_rewards = []
        state, ep_total_reward = self.env.reset(), 0
        for t in range(1, self.modelParam['max_episode_len']):  # Don't infinite loop while learning
            print(t)
            action, log_prob = self.model.select_action(state)
            state, reward, done, _ = self.env.step(action)
            if self.modelParam['render']:
                self.env.render()
            ep_rewards.append(reward)
            ep_log_probs.append(log_prob)
            ep_total_reward += reward
            if done:
                break
        return
