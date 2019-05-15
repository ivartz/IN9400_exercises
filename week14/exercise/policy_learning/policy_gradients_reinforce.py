from utils.saverRestorer import SaverRestorer
from utils.model import Model
from utils.trainer import Trainer
from utils.player import Player
from utils.environment import EnvironmentWrapper, EnvironmentWrapper_image

def main(config, modelParam):
    if config['network'] == 'CartPole_v1_image':
        env = EnvironmentWrapper_image(modelParam)
    else:
        env = EnvironmentWrapper(modelParam)

    # create an instance of the model you want
    model = Model(config, modelParam, env)

    # create an instacne of the saver and resoterer class
    saveRestorer = SaverRestorer(config, modelParam)
    model        = saveRestorer.restore(model)

    # here you train your model
    if modelParam['play'] == False:
        trainer = Trainer(model, modelParam, config, saveRestorer, env)
        trainer.train()

    #play
    if modelParam['play'] == True:
        player = Player(model, modelParam, config, saveRestorer, env)
        player.play_episode()

    return


########################################################################################################################
if __name__ == '__main__':
    modelParam = {
        'episode_batch': 32,           # Training batch size, number of games before parameter update
        'numb_of_updates': 100,       # Number of gradient descent updates
        'max_episode_len': 500,        # Max number of steps before the game is terminated
        'cuda': {'use_cuda': True,     # Use_cuda=True: use GPU
                 'device_idx': 0},     # Select gpu index: 0,1,2,3
        'environment': 'CartPole-v1',  # Game selected
        'modelsDir': 'storedModels/',
        'restoreModelLast': 0,
        'restoreModelBest': 0,
        'storeModelFreq': 5,
        'render': True,
        'is_train': True,
        'inNotebook': False,  # If running script in jupyter notebook
        'play': False         # False=train the model | True=restore pretrained model and play an episode
    }

    config = {
        'optimizer': 'adam',           # 'SGD' | 'adam' | 'RMSprop'
        'learningRate': {'lr': 0.01},  # learning rate to the optimizer
        'weight_decay': 0,             # weight_decay value
        'gamma': 0.99,                 # discount factor
        'seed': 543,
        'network': 'CartPole_v1'    # 'CartPole_v1' | 'CartPole_v1_image'
    }

    if modelParam['play'] == True:
        modelParam['restoreModelLast'] = 1
        modelParam['render'] = True

    main(config, modelParam)
    aa = 1
