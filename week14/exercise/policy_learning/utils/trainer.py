from tqdm import tqdm
from tqdm import tqdm_notebook
from utils.plotter import Plotter
import sys
import torch
import numpy as np

#######################################################################################################################
class Trainer():
    def __init__(self, model, modelParam, config, saveRestorer, env):
        self.model        = model
        self.modelParam   = modelParam
        self.config       = config
        self.saveRestorer = saveRestorer
        self.env = env
        self.plotter      = Plotter(self.modelParam, self.config)
        return

    def train(self):
        running_reward = 10
        given_range = range(self.model.update_counter, self.modelParam['numb_of_updates'])
        if self.modelParam['inNotebook']:
            tt = tqdm_notebook(given_range, desc='', leave=True, mininterval=0.01, file=sys.stdout)
        else:
            tt = tqdm(given_range, desc='', leave=True, mininterval=0.01, file=sys.stdout)
        for update_counter in tt:
            if self.modelParam['is_train']:
                self.model.policyNet.train()
            else:
                self.model.policyNet.eval()
            loss, reward = self.run_update()

            reward = reward/self.modelParam["episode_batch"]
            running_reward = 0.2 * reward + (1 - 0.2) * running_reward

            desc = f'Update_counter={update_counter} | reward={reward:.4f} | | running_reward={running_reward:.4f}'
            tt.set_description(desc)
            tt.update()

            if update_counter % self.modelParam['storeModelFreq']==0:
                self.plotter.update(update_counter, running_reward)
                self.saveRestorer.save(update_counter, running_reward, self.model)
        return

    def run_update(self):
        episodes_summary = {'episodes_log_probs': [],
                            'episodes_rewards': [],
                            'episodes_total_reward': [],
                            'episodes_return': [],
                            }

        for episode_ind in range(self.modelParam['episode_batch']):
            #play episode
            ep_log_probs, ep_rewards, ep_total_reward, ep_returns = self.play_episode()
            episodes_summary['episodes_log_probs'].append(ep_log_probs)
            episodes_summary['episodes_rewards'].append(ep_rewards)
            episodes_summary['episodes_total_reward'].append(ep_total_reward)
            episodes_summary['episodes_return'].append(ep_returns)

        loss   = self.gradient_update(episodes_summary)
        reward = sum(episodes_summary['episodes_total_reward'])
        return loss, reward

    def play_episode(self):
        ep_log_probs = []
        ep_rewards = []
        state, ep_total_reward = self.env.reset(), 0
        for t in range(1, self.modelParam['max_episode_len']):  # Don't infinite loop while learning
            action, log_prob = self.model.select_action(state)
            state, reward, done, _ = self.env.step(action)
            if self.modelParam['render']:
                self.env.render()
            ep_rewards.append(reward)
            ep_log_probs.append(log_prob)
            ep_total_reward += reward
            if done:
                break
        #calculate return
        ep_returns = []
        G_t = 0
        for r in ep_rewards[::-1]:
            G_t = r + self.config['gamma'] * G_t
            ep_returns.insert(0, G_t)
        return ep_log_probs, ep_rewards, ep_total_reward, ep_returns

    def gradient_update(self, episodes_summary):
        policy_loss = []
        #flatten the list of lists into a single list
        episodes_return    = [item for sublist in episodes_summary['episodes_return'] for item in sublist]
        episodes_log_probs = [item for sublist in episodes_summary['episodes_log_probs'] for item in sublist]

        #normalize the return to zero mean ans unit variance
        eps = np.finfo(np.float32).eps.item()
        episodes_return = torch.tensor(episodes_return, device=self.model.device)
        episodes_return = (episodes_return - episodes_return.mean()) / (episodes_return.std() + eps)
        for log_prob, R in zip(episodes_log_probs, episodes_return):
            policy_loss.append(-log_prob * R) #we multiply with -1 to get gradient ascent
        self.model.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.model.optimizer.step()

        return policy_loss.detach().cpu().item()

