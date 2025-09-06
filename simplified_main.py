# simplified_main.py — stable old-Gym run (channels-first, no tuple errors)

import warnings
import numpy as np
from gym import spaces

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

# Classic Gym wrappers
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

from agent import Agent

# Quiet legacy chatter
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Gym has been unmaintained since 2022")
warnings.filterwarnings("ignore", message="No render modes was declared")

ENV_NAME = "SuperMarioBros-1-1-v0"   # matches old-Gym stack
SHOULD_TRAIN = True
DISPLAY = True                       # no .render() call -> no render warning
NUM_OF_EPISODES = 50_000


class StepToFiveTuple:
    """Convert old 4-tuple (obs, reward, done, info) -> 5-tuple (obs, reward, terminated, truncated, info)."""
    def __init__(self, env):
        self.env = env
        for k in ("observation_space", "action_space", "metadata", "spec"):
            if hasattr(env, k):
                setattr(self, k, getattr(env, k))

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        # Normalize to (obs, info)
        if isinstance(out, tuple) and len(out) >= 2 and isinstance(out[1], dict):
            return out[0], out[1]
        return out, {}

    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            return out
        obs, reward, done, info = out
        return obs, reward, bool(done), False, info

    def render(self, *a, **k): return self.env.render(*a, **k)
    def close(self): return self.env.close()
    def __getattr__(self, n): return getattr(self.env, n)


class ToChannelsFirst(spaces.Space):
    """
    Observation wrapper: make obs shape (C,H,W) = (4,84,84).
    - Input after FrameStack will be (H,W,4) or sometimes (H,W,1,4).
    - We squeeze any singleton and transpose to (4,H,W).
    """
    def __init__(self, env):
        self.env = env
        # Target space
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
        )
        self.action_space = env.action_space
        for k in ("metadata", "spec"):
            if hasattr(env, k):
                setattr(self, k, getattr(env, k))

    def _fix(self, obs):
        arr = np.array(obs)
        # squeeze grayscale singleton if present
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)          # (H,W,4)
        if arr.ndim == 3 and arr.shape[-1] == 4:     # (H,W,4) -> (4,H,W)
            arr = np.transpose(arr, (2, 0, 1))
        elif arr.ndim == 3 and arr.shape[0] == 4:    # already (4,H,W)
            pass
        else:
            # As a safe fallback, try to move the stack axis to front
            axes = list(range(arr.ndim))
            # heuristics: pick the axis with size 4 as channel
            ch_axis = int(np.argwhere(np.array(arr.shape) == 4).flatten()[0])
            axes = [ch_axis] + [i for i in axes if i != ch_axis]
            arr = np.transpose(arr, axes)
        return arr.astype(np.uint8, copy=False)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs, info = out
        else:
            obs, info = out, {}
        return self._fix(obs), info

    def step(self, action):
        out = self.env.step(action)  # expect 5-tuple after StepToFiveTuple
        obs, reward, terminated, truncated, info = out
        return self._fix(obs), reward, terminated, truncated, info

    def render(self, *a, **k): return self.env.render(*a, **k)
    def close(self): return self.env.close()
    def __getattr__(self, n): return getattr(self.env, n)


# --- Build env with classic API + wrappers ---
env = gym_super_mario_bros.make(ENV_NAME)
env = JoypadSpace(env, RIGHT_ONLY)

# Preprocess: grayscale (drop singleton), resize, framestack
env = GrayScaleObservation(env, keep_dim=False)   # -> (H,W)
env = ResizeObservation(env, (84, 84))           # -> (84,84)
env = FrameStack(env, num_stack=4, lz4_compress=False)  # -> (84,84,4)

# Normalize step to 5-tuple for modern loop
env = StepToFiveTuple(env)

# Make observations channels-first for PyTorch Conv2d: (4,84,84)
env = ToChannelsFirst(env)

# Agent now gets (C,H,W) = (4,84,84)
agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

for episode in range(NUM_OF_EPISODES):
    terminated = truncated = False
    state, _ = env.reset()

    # Optional: enable to see the window (accept render warning)
    if DISPLAY:
        env.render()

    while not (terminated or truncated):
        action = agent.choose_action(state)
        new_state, reward, terminated, truncated, info = env.step(action)

        if SHOULD_TRAIN:
            agent.store_in_memory(state, action, reward, new_state, bool(terminated or truncated))
            agent.learn()

        state = new_state

        if DISPLAY:
            env.render()

env.close()
