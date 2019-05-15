from torch import optim
from torch.distributions import Categorical
import importlib

class Model():
    def __init__(self, config, modelParam, env):
        self.update_counter = 0

        if modelParam['cuda']['use_cuda']:
            self.device = f"cuda:{modelParam['cuda']['device_idx']}"
        else:
            self.device = "cpu"

        self.config         = config
        self.modelParam     = modelParam

        self.policyNet = self.selectPolicyNet(config, env.size_of_state_space, env.size_of_action_space)
        self.policyNet.to(self.device)

        self.optimizer = self.selectOptimizer(config)

        return

    def selectPolicyNet(self, config, size_of_state_space, size_of_action_space):
        #Importing the network class based on the config[network] key
        module = importlib.import_module("networks." + config['network'])
        net    = getattr(module, config['network'])(size_of_state_space, size_of_action_space)
        return net

    def selectOptimizer(self, config):
        if config['optimizer'] == 'adam':
            optimizer = optim.Adam(self.policyNet.parameters(), lr=config['learningRate']['lr'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.policyNet.parameters(), lr=config['learningRate']['lr'],weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'RMSprop':
            optimizer = optim.RMSprop(self.policyNet.parameters(), lr=config['learningRate']['lr'],weight_decay=config['weight_decay'])
        else:
            raise Exception('invalid optimizer')
        return optimizer

    def select_action(self, state):
        state = state.to(self.device)
        probs = self.policyNet(state)
        m = Categorical(probs)
        action = m.sample()
        log_probs = m.log_prob(action)
        return action.item(), log_probs
