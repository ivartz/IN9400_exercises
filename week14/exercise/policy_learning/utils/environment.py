import gym
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

class EnvironmentWrapper_image():
    def __init__(self, modelParam):
        self.resize = T.Compose([T.ToPILImage(),
                      T.Resize(40, interpolation=Image.CUBIC),
                      T.ToTensor()])

        self.last_screen    = None
        self.current_screen = None
        self.state          = None

        self.gym_env = gym.make(modelParam['environment']).env
        self.size_of_action_space = self.gym_env.action_space.n
        self.size_of_state_space  = self.get_size_of_state_space()


        return

    def step(self, action):
        state, reward, done, _ = self.gym_env.step(action)
        self.last_screen = self.current_screen
        self.current_screen = self.get_screen()
        self.state = self.current_screen - self.last_screen
        return self.state, reward, done, _

    def reset(self):
        self.gym_env.reset()
        self.last_screen = self.get_screen()
        self.current_screen = self.get_screen()
        self.state = self.current_screen - self.last_screen
        return self.state

    def render(self):
        self.gym_env.render()
        return

    def get_cart_location(self, screen_width):
        world_width = self.gym_env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.gym_env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.gym_env.render(mode='rgb_array').transpose((2, 0, 1))
        # x = self.gym_env.env.ale.getScreenRGB()
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0)


    def get_size_of_state_space(self):
        self.gym_env.reset()
        init_screen = self.get_screen()
        _, screen_channels, screen_height, screen_width = init_screen.shape
        return [screen_channels, screen_height, screen_width]


#######################################################################################################################
class EnvironmentWrapper():
    def __init__(self, modelParam):
        self.gym_env = gym.make(modelParam['environment']).env
        self.size_of_action_space = self.gym_env.action_space.n
        self.size_of_state_space  = self.gym_env.observation_space.shape[0]
        return

    def step(self, action):
        state, reward, done, info = self.gym_env.step(action)
        state = torch.from_numpy(state).float().unsqueeze(0)
        return [state, reward, done, info]

    def reset(self):
        state = self.gym_env.reset()
        return torch.from_numpy(state).float().unsqueeze(0)

    def render(self):
        self.gym_env.render()
        return