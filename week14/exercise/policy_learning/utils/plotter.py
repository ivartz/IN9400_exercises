import matplotlib.pyplot as plt
from time import sleep
from IPython import display

class Plotter():
    def __init__(self, modelParam, config):
        self.modelParam = modelParam
        self.config     = config
        self.fig, self.ax = plt.subplots()
        if not self.modelParam['inNotebook']:
            self.fig.show()
        plt.ylabel('Reward')
        plt.xlabel('number of updates [#]')
        self.ax.set_axisbelow(True)
        self.ax.grid()
        return


    def update(self, current_epoch, loss):
        color = 'b'
        if self.modelParam['inNotebook']:
            if loss > plt.ylim()[1]:
                top = loss
            self.ax.scatter(current_epoch, loss, c=color)
            self.ax.set_ylim(bottom=0, top=top)
            # display.display(plt.gcf())
            # display.clear_output(wait=True)
            self.fig.canvas.draw()
            sleep(0.1)
        else:
            plt.scatter(current_epoch, loss, c=color)
            if loss > plt.ylim()[1]:
                plt.ylim(0.0, loss)
            plt.draw()
            plt.pause(0.1)
        self.save()
        return


    def save(self):
        # path = self._getPath()
        path = 'loss_images/'+self.config['network']
        plt.savefig(path+'.png')
        return

    def _getPath(self):
        keys = self.config.keys()
        path = 'loss_images/'
        first=1
        for key in keys:
            if first!=1:
                path += '_'
            else:
                first=0
            element = self.config[key]
            if isinstance(element, str):
                path += element
            elif isinstance(element, int):
                path += key+str(element)
            elif isinstance(element, float):
                path += key+str(element)
            elif isinstance(element, list):
                path += ''
                for elm in element:
                    path += str(elm)
            elif isinstance(element, dict):
                path += ''
                for elKey, elVal in element.items():
                    path += str(elKey) + str(elVal).replace('.', '_')
            else:
                raise Exception('Unknown element in config')
        return path
