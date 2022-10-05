from gym import Wrapper


class RandomizeWrapper(Wrapper):
    def __init__(self, env, randomize_config_path):
        super().__init__(env)
        self.default_randomize_config_path = randomize_config_path

    def turn_on_randomization(self):
        self.unwrapped.world.randomize_config_path = self.default_randomize_config_path

    def turn_off_randomization(self):
        self.unwrapped.world.randomize_config_path = None
