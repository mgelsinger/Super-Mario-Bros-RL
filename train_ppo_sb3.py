"""
train_ppo_sb3.py — Super Mario Bros PPO (classic Gym stack, Windows-safe)
Recording + robust load/resume + shape hardening + render-mode compat.

- Classic gym + gym_super_mario_bros + nes-py
- Preprocessing: grayscale -> resize 84x84 -> framestack(4)
- Ensure base env outputs HWC=(84,84,4) and exposes render_mode='rgb_array'
- Vectorized envs (SubprocVecEnv spawn on Windows, fallback DummyVecEnv)
- Stable-Baselines3 1.8.0 PPO + TensorBoard
- Recording via SB3 VecVideoRecorder
"""

import argparse
import os
import warnings
import numpy as np
import imageio
import threading
import time

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecMonitor,
    VecTransposeImage,
    VecEnvWrapper,
    VecVideoRecorder,
)
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

# --- Suppress noisy deprecation warnings from legacy Gym ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=r".*No render modes.*")

ENV_ID = "SuperMarioBros-1-1-v0"

# Defaults for env wrappers (can be overridden via CLI)
SKIP_FRAMES = 4
NOOP_MAX = 0
CLIP_REWARD = False
ACTION_SET = "RIGHT_ONLY"
STICKY_PROB = 0.0
REWARD_FWD = 0.0
DEATH_PENALTY = 0.0


class VecSqueezeLastChannel(VecEnvWrapper):
    """If vec obs come with a trailing singleton channel, squeeze it away."""
    def __init__(self, venv):
        super().__init__(venv)
        shp = venv.observation_space.shape
        if len(shp) == 3 and shp[-1] == 1:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=shp[:-1], dtype=venv.observation_space.dtype
            )
        else:
            self.observation_space = venv.observation_space

    def reset(self):
        obs = self.venv.reset()
        return self._fix(obs)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        return self._fix(obs), rews, dones, infos

    def _fix(self, obs):
        arr = np.asarray(obs)
        if arr.ndim >= 1 and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
        return arr


class EnsureHWCLast(gym.ObservationWrapper):
    """
    Force observations to HWC=(84,84,4):
      - squeeze singletons
      - channels-first (4,H,W) -> (H,W,4)
      - if 4D, move the '4' axis to last, then squeeze
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)

    def observation(self, obs):
        arr = np.asarray(obs)
        arr = np.squeeze(arr)

        if arr.ndim == 3 and arr.shape == (84, 84, 4):
            return arr.astype(np.uint8, copy=False)

        if arr.ndim == 3 and arr.shape[0] == 4:
            # (4,84,84) -> (84,84,4)
            return np.transpose(arr, (1, 2, 0)).astype(np.uint8, copy=False)

        if arr.ndim == 4:
            ch_axes = np.where(np.array(arr.shape) == 4)[0]
            if ch_axes.size:
                arr = np.moveaxis(arr, int(ch_axes[0]), -1)
            arr = np.squeeze(arr)
            if arr.ndim == 3 and arr.shape[-1] == 4:
                return arr.astype(np.uint8, copy=False)

        # Last resort: pick the axis with size 4 and move it to last
        if arr.ndim == 3:
            ch_axes = np.where(np.array(arr.shape) == 4)[0]
            if ch_axes.size:
                arr = np.moveaxis(arr, int(ch_axes[0]), -1)
                return arr.astype(np.uint8, copy=False)

        raise ValueError(f"Unexpected observation shape {arr.shape}, expected something coercible to (84,84,4)")


class SafeClose(gym.Wrapper):
    """Make env.close() idempotent and swallow ValueError from NES-Py double-close."""
    def __init__(self, env):
        super().__init__(env)
        self._closed = False

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            return self.env.close()
        except Exception:
            # NES-Py may raise ValueError('env has already been closed.') on double close
            return None


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame and max-pool over the last two.

    Standard for retro/Atari-like environments to reduce flicker and speed up.
    """
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self._skip = max(1, int(skip))
        self._obs_buffer = [None, None]

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        obs = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i >= self._skip - 2:
                self._obs_buffer[i - (self._skip - 2)] = obs
            total_reward += float(reward)
            if done:
                break
        if self._obs_buffer[0] is None:
            frame = obs
        else:
            try:
                frame = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
            except Exception:
                frame = obs
        return frame, total_reward, done, info

    def reset(self, **kwargs):
        self._obs_buffer = [None, None]
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):
    """On reset, take up to `noop_max` no-op actions to randomize starting state."""
    def __init__(self, env, noop_max: int = 0, noop_action: int = 0):
        super().__init__(env)
        self.noop_max = max(0, int(noop_max))
        self.noop_action = int(noop_action)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self.noop_max <= 0:
            return obs
        n_noop = np.random.randint(1, self.noop_max + 1)
        done = False
        for _ in range(n_noop):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1}."""
    def reward(self, reward):
        return float(np.sign(reward))


class StickyActionEnv(gym.Wrapper):
    """Repeat the previous action with probability p to add stochasticity."""
    def __init__(self, env, sticky_prob: float = 0.25, noop_action: int = 0):
        super().__init__(env)
        self.p = float(max(0.0, min(1.0, sticky_prob)))
        self.noop = int(noop_action)
        self._last_action = self.noop

    def reset(self, **kwargs):
        self._last_action = self.noop
        return self.env.reset(**kwargs)

    def step(self, action):
        if np.random.rand() < self.p:
            action = self._last_action
        else:
            self._last_action = int(action)
        return self.env.step(action)


class RewardForwardProgress(gym.Wrapper):
    """Add shaping for forward x-position progress and an optional death penalty."""
    def __init__(self, env, forward_coeff: float = 1.0, death_penalty: float = 0.0):
        super().__init__(env)
        self.c = float(forward_coeff)
        self.death = float(death_penalty)
        self._last_x = 0.0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_x = 0.0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        x = float(info.get('x_pos', 0.0)) if isinstance(info, dict) else 0.0
        dx = max(0.0, x - self._last_x)
        self._last_x = x
        shaped = reward + self.c * dx
        if done and isinstance(info, dict):
            if not info.get('flag_get', False) and self.death != 0.0:
                shaped += self.death
        return obs, shaped, done, info


class RenderModeCompat(gym.Wrapper):
    """
    Old Gym envs (NES-Py) may not expose env.render_mode / render_modes.
    This wrapper *advertises* 'rgb_array' and forwards render calls to return RGB frames.
    """
    def __init__(self, env):
        super().__init__(env)
        base_md = getattr(self.env, "metadata", {})
        md = dict(base_md) if isinstance(base_md, dict) else {}
        rmods = set(md.get("render_modes") or [])
        rmods.update(md.get("render.modes") or [])
        rmods.add("rgb_array")
        md["render_modes"] = sorted(rmods)
        md["render.modes"] = sorted(rmods)
        # Ensure fps hints for video recorder
        md.setdefault("render_fps", 60)
        md.setdefault("video.frames_per_second", 60)
        self.metadata = md

    # Read-only exposure so code can do `env.render_mode`
    def __getattr__(self, name):
        if name == "render_mode":
            return "rgb_array"
        return getattr(self.env, name)

    def render(self, *args, **kwargs):
        try:
            return self.env.render("rgb_array")
        except TypeError:
            return self.env.render(mode="rgb_array")


class VecRenderModeCompat(VecEnvWrapper):
    """
    Give the *vectorized* env a visible render_mode and a working .render() so
    Gym's VideoRecorder (via SB3's VecVideoRecorder) is happy.
    Implements reset/step_wait to satisfy VecEnv ABC on some SB3 versions.
    """
    def __init__(self, venv):
        super().__init__(venv)

    @property
    def render_mode(self):
        return "rgb_array"

    def _ensure_metadata_rgb_array(self, md):
        try:
            md = dict(md) if md is not None else {}
        except Exception:
            md = {}
        rmods = set(md.get("render_modes") or [])
        rmods.update(md.get("render.modes") or [])
        rmods.add("rgb_array")
        md["render_modes"] = sorted(rmods)
        md["render.modes"] = sorted(rmods)
        md.setdefault("render_fps", 60)
        md.setdefault("video.frames_per_second", 60)
        return md

    @property
    def metadata(self):
        # Backing store so SB3 can set env.metadata
        if not hasattr(self, "_metadata_store"):
            base = getattr(self.venv, "metadata", None)
            self._metadata_store = self._ensure_metadata_rgb_array(base)
        return self._metadata_store

    @metadata.setter
    def metadata(self, value):
        self._metadata_store = self._ensure_metadata_rgb_array(value)

    def render(self, mode="rgb_array"):
        # Ask each underlying env to render an rgb_array; return the first frame (what recorder expects)
        try:
            frames = self.venv.env_method("render", "rgb_array")
        except TypeError:
            frames = self.venv.env_method("render", mode="rgb_array")
        return frames[0] if isinstance(frames, (list, tuple)) else frames

    # Pass-throughs to satisfy abstract methods on some versions
    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    # Ensure SB3's BaseVecEnv getattr recursion can find these on the wrapper
    def getattr_recursive(self, name):
        if name == "render_mode":
            return self.render_mode
        if name == "metadata":
            return self.metadata
        # Delegate otherwise
        return self.venv.getattr_recursive(name)


def make_env(seed: int = 0):
    """Factory for a single env instance (classic Gym API)."""
    def _thunk():
        env = gym_super_mario_bros.make(ENV_ID)
        # Select action set
        movements = RIGHT_ONLY if ACTION_SET.upper() == "RIGHT_ONLY" else SIMPLE_MOVEMENT
        env = JoypadSpace(env, movements)
        # Optional random no-ops at reset for better exploration
        if NOOP_MAX > 0:
            env = NoopResetEnv(env, noop_max=NOOP_MAX, noop_action=0)
        # Frame skipping + max-pooling before preprocessing
        if SKIP_FRAMES > 1:
            env = MaxAndSkipEnv(env, skip=SKIP_FRAMES)
        env = GrayScaleObservation(env, keep_dim=False)      # (H,W)
        env = ResizeObservation(env, (84, 84))               # (84,84)
        env = FrameStack(env, num_stack=4, lz4_compress=False)  # typically (84,84,4)
        env = EnsureHWCLast(env)                             # harden to (84,84,4)
        env = RenderModeCompat(env)                          # advertise rgb_array render mode
        env = SafeClose(env)                                 # idempotent close
        if STICKY_PROB > 0.0:
            env = StickyActionEnv(env, sticky_prob=STICKY_PROB, noop_action=0)
        if CLIP_REWARD:
            env = ClipRewardEnv(env)
        if REWARD_FWD != 0.0 or DEATH_PENALTY != 0.0:
            env = RewardForwardProgress(env, forward_coeff=REWARD_FWD, death_penalty=DEATH_PENALTY)
        env.seed(seed)
        return env
    return _thunk


def build_vec_env(n_envs: int, start_seed: int = 0, force_dummy: bool = False):
    thunks = [make_env(seed=start_seed + i) for i in range(n_envs)]
    if not force_dummy and n_envs > 1:
        try:
            vec = SubprocVecEnv(thunks, start_method="spawn")
        except Exception:
            vec = DummyVecEnv(thunks)
    else:
        vec = DummyVecEnv(thunks)

    vec = VecMonitor(vec)
    vec = VecSqueezeLastChannel(vec)  # belt & suspenders

    # Transpose only if channel-last (H,W,C)
    shp = vec.observation_space.shape
    if len(shp) == 3:
        channels_first = shp[0] in (1, 3, 4) and shp[-1] not in (1, 3, 4)
        channels_last  = shp[-1] in (1, 3, 4) and shp[0] not in (1, 3, 4)
        if channels_last:
            vec = VecTransposeImage(vec)  # (H,W,C) -> (C,H,W)

    # NEW: make the vectorized env advertise render_mode + render()
    vec = VecRenderModeCompat(vec)
    return vec


def maybe_wrap_video(venv, enable: bool, record_dir: str, record_freq: int, video_length: int):
    """Wrap vec env with VecVideoRecorder if recording is enabled."""
    if not enable:
        return venv
    os.makedirs(record_dir, exist_ok=True)
    def trigger(step_count):
        return (step_count // max(1, record_freq)) != ((step_count - 1) // max(1, record_freq))
    return VecVideoRecorder(
        venv,
        record_dir,
        record_video_trigger=trigger,
        video_length=video_length,
        name_prefix="ppo_mario",
    )


def load_model_with_env(path: str, vec_env, print_system_info: bool = False):
    """
    Robust SB3 1.8.0 load:
    - attach env at load time (allows different n_envs)
    - pass live spaces via custom_objects to bypass pickled-space deserialization issues
    """
    custom = {
        "observation_space": vec_env.observation_space,
        "action_space": vec_env.action_space,
    }
    model = PPO.load(
        path,
        env=vec_env,
        device="cuda",
        custom_objects=custom,
        print_system_info=print_system_info,
    )
    return model


def eval_and_record(model_path: str, out_dir: str, episodes: int = 1, force_dummy: bool = False, video_length: int = 2000, fps: int = 60):
    """Load a saved model and record a single MP4 of `video_length` frames.

    Uses a manual imageio writer to avoid SB3 VecVideoRecorder's per-episode
    segmentation and metadata side files. Continues across resets.
    """
    os.makedirs(out_dir, exist_ok=True)
    vec = build_vec_env(n_envs=1, force_dummy=True if force_dummy else True)  # force Dummy for eval
    model = load_model_with_env(model_path, vec, print_system_info=False)

    # Prepare output
    fname = os.path.join(out_dir, "eval_mario.mp4")
    writer = imageio.get_writer(fname, fps=int(max(1, fps)))
    try:
        obs = vec.reset()
        total = 0
        while total < int(video_length):
            # Render frame BEFORE or AFTER step; use after to reflect action result
            frame = vec.render("rgb_array")
            if frame is not None:
                writer.append_data(frame)
                total += 1
                if total >= int(video_length):
                    break
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = vec.step(action)
            if bool(dones[0]) if np.ndim(dones) > 0 else bool(dones):
                obs = vec.reset()
    finally:
        try:
            writer.close()
        except Exception:
            pass
        vec.close()


class LiveViewer(threading.Thread):
    """Watches model files and previews them periodically in a window.

    follow: 'best' -> runs best_model.zip
            'latest' -> runs most recent checkpoint ppo_mario_*_steps.zip
            'both' -> whichever (best or latest) has the newest mtime
    """
    def __init__(self, logdir: str, follow: str = "best", fps: int = 60, steps: int = 2000, interval: int = 30, seed: int = 0):
        super().__init__(daemon=True)
        self.logdir = logdir
        self.follow = follow
        self.fps = max(1, int(fps))
        self.steps = max(1, int(steps))
        self.interval = max(1, int(interval))
        self.seed = int(seed)
        self._stop = threading.Event()
        self._last_mtime = 0.0
        self._last_path = None

    def stop(self):
        self._stop.set()

    def _make_obs_vec(self):
        # One env, dummy vec, same wrappers as training
        return build_vec_env(n_envs=1, start_seed=self.seed, force_dummy=True)

    def _make_vis_env(self):
        # Raw visual env for pretty rendering (no preprocessing wrappers)
        env = gym_super_mario_bros.make(ENV_ID)
        movements = RIGHT_ONLY if ACTION_SET.upper() == "RIGHT_ONLY" else SIMPLE_MOVEMENT
        env = JoypadSpace(env, movements)
        return env

    def run(self):
        last_preview = 0.0
        while not self._stop.is_set():
            try:
                # Determine which model file to preview
                candidates = []
                best_path = os.path.join(self.logdir, 'best_model.zip')
                if self.follow in ("best", "both") and os.path.exists(best_path):
                    candidates.append(best_path)
                if self.follow in ("latest", "both"):
                    try:
                        files = [
                            os.path.join(self.logdir, f)
                            for f in os.listdir(self.logdir)
                            if f.startswith('ppo_mario_') and f.endswith('_steps.zip')
                        ]
                        if files:
                            latest = max(files, key=lambda p: os.path.getmtime(p))
                            candidates.append(latest)
                    except Exception:
                        pass

                if not candidates:
                    time.sleep(1.0)
                    continue

                # Choose the newest by mtime
                path = max(candidates, key=lambda p: os.path.getmtime(p))
                mtime = os.path.getmtime(path)
                now = time.time()
                changed = (self._last_path != path) or (mtime > self._last_mtime)
                should_preview = changed or (now - last_preview >= self.interval)
                if not should_preview:
                    time.sleep(0.5)
                    continue

                self._last_mtime = mtime
                self._last_path = path
                last_preview = now

                obs_vec = self._make_obs_vec()
                model = load_model_with_env(path, obs_vec, print_system_info=False)
                vis_env = self._make_vis_env()

                # Try OpenCV for display; fallback to env.render('human') if unavailable
                try:
                    import cv2  # type: ignore
                    use_cv2 = True
                except Exception:
                    use_cv2 = False

                try:
                    obs = obs_vec.reset()
                    vis_env.reset()
                    t_per_frame = 1.0 / float(self.fps)
                    for _ in range(self.steps):
                        start = time.perf_counter()
                        action, _ = model.predict(obs, deterministic=True)
                        obs, _, dones, _ = obs_vec.step(action)
                        a = int(action[0]) if np.ndim(action) > 0 else int(action)
                        obs_r, _, done_r, _ = vis_env.step(a)
                        if use_cv2:
                            frame = vis_env.render(mode='rgb_array')
                            if frame is not None:
                                cv2.imshow('Live Mario (best model)', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                        else:
                            # Let NES-Py draw its own window
                            vis_env.render()

                        if (bool(dones[0]) if np.ndim(dones) > 0 else bool(dones)) or bool(done_r):
                            obs = obs_vec.reset()
                            vis_env.reset()

                        # Throttle to desired FPS
                        elapsed = time.perf_counter() - start
                        remaining = t_per_frame - elapsed
                        if remaining > 0:
                            time.sleep(remaining)
                finally:
                    try:
                        obs_vec.close()
                    except Exception:
                        pass
                    try:
                        vis_env.close()
                    except Exception:
                        pass
                    if 'cv2' in locals() and use_cv2:
                        try:
                            cv2.destroyWindow('Live Mario (best model)')
                        except Exception:
                            pass
            except Exception:
                # Don't crash training due to viewer errors; back off briefly
                time.sleep(2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--lr-schedule", type=str, choices=["constant", "linear"], default="linear")
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--clip-schedule", type=str, choices=["constant", "linear"], default="linear")
    parser.add_argument("--ent-coef", type=float, default=1e-3)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--logdir", type=str, default="runs/mario_ppo")
    parser.add_argument("--checkpoint-every", type=int, default=250_000)
    parser.add_argument("--eval-every", type=int, default=250_000, help="Evaluate every N timesteps.")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--force-dummy", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    # Env control
    parser.add_argument("--skip-frames", type=int, default=4, help="Frames to skip per action (max-pool last two).")
    parser.add_argument("--noop-max", type=int, default=0, help="Random no-ops at reset (0 to disable).")
    parser.add_argument("--clip-reward", action="store_true", help="Clip rewards to {-1,0,+1} like Atari.")
    parser.add_argument("--action-set", type=str, choices=["RIGHT_ONLY","SIMPLE"], default="SIMPLE", help="Action set to use.")
    parser.add_argument("--sticky-prob", type=float, default=0.0, help="Probability to repeat last action (sticky).")
    parser.add_argument("--reward-forward", type=float, default=0.0, help="Forward progress shaping coefficient.")
    parser.add_argument("--death-penalty", type=float, default=0.0, help="Penalty added on death when shaping is enabled.")

    # Recording controls (during training)
    parser.add_argument("--record", action="store_true", help="Record clips during training.")
    parser.add_argument("--record-dir", type=str, default=None, help="Directory for training videos.")
    parser.add_argument("--record-freq", type=int, default=250_000, help="Record every N steps.")
    parser.add_argument("--video-length", type=int, default=2000, help="Max steps per training video.")
    parser.add_argument("--fps", type=int, default=60, help="FPS for manual eval recording.")
    # Live viewer of best model
    parser.add_argument("--live-view", action="store_true", help="Show periodic live preview of best model.")
    parser.add_argument("--live-steps", type=int, default=2000, help="Frames per live preview run.")
    parser.add_argument("--live-fps", type=int, default=60, help="FPS for live preview.")
    parser.add_argument("--live-interval", type=int, default=30, help="Min seconds between previews when no new best.")
    parser.add_argument("--live-follow", type=str, choices=["best","latest","both"], default="best", help="Which model to follow for live preview.")

    # Evaluation-only recording
    parser.add_argument("--record-only", action="store_true", help="Only record evaluation episodes from --resume.")
    parser.add_argument("--eval-episodes", type=int, default=1, help="Episodes to record in record-only mode.")
    args = parser.parse_args()

    # Make configurable env knobs visible in this scope before any assignment
    global SKIP_FRAMES, NOOP_MAX, CLIP_REWARD

    # Record-only path
    if args.record_only:
        if not args.resume:
            raise SystemExit("--record-only requires --resume <path_to_model.zip>")
        # Configure env knobs for eval
        SKIP_FRAMES = max(1, int(getattr(args, "skip_frames", 4)))
        NOOP_MAX = max(0, int(getattr(args, "noop_max", 0)))
        CLIP_REWARD = bool(getattr(args, "clip_reward", False))
        ACTION_SET = "RIGHT_ONLY" if getattr(args, "action_set", "SIMPLE").upper().startswith("RIGHT") else "SIMPLE"
        STICKY_PROB = float(getattr(args, "sticky_prob", 0.0))
        REWARD_FWD = float(getattr(args, "reward_forward", 0.0))
        DEATH_PENALTY = float(getattr(args, "death_penalty", 0.0))
        out_dir = args.record_dir or os.path.join(args.logdir, "videos_eval")
        eval_and_record(
            args.resume,
            out_dir,
            episodes=args.eval_episodes,
            force_dummy=True,
            video_length=args.video_length,
            fps=args.fps,
        )
        return

    os.makedirs(args.logdir, exist_ok=True)

    # Global knobs for env factory
    SKIP_FRAMES = max(1, int(args.skip_frames))
    NOOP_MAX = max(0, int(args.noop_max))
    CLIP_REWARD = bool(args.clip_reward)
    ACTION_SET = "RIGHT_ONLY" if args.action_set.upper().startswith("RIGHT") else "SIMPLE"
    STICKY_PROB = float(args.sticky_prob)
    REWARD_FWD = float(args.reward_forward)
    DEATH_PENALTY = float(args.death_penalty)

    vec_env = build_vec_env(args.n_envs, start_seed=args.seed, force_dummy=args.force_dummy)

    # Optional recording during training
    if args.record or args.record_dir:
        record_dir = args.record_dir or os.path.join(args.logdir, "videos")
        vec_env = maybe_wrap_video(vec_env, True, record_dir, args.record_freq, args.video_length)

    # Schedules
    def linear_schedule(v):
        def f(progress_remaining):
            # progress_remaining goes from 1 (start) to 0 (end)
            return progress_remaining * v
        return f

    lr = args.lr if args.lr_schedule == "constant" else linear_schedule(args.lr)
    clip_range = args.clip if args.clip_schedule == "constant" else linear_schedule(args.clip)

    # Use default SB3 behavior for image inputs: normalize to [0,1]
    policy_kwargs = dict(normalize_images=True)
    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=lr,
        clip_range=clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        gae_lambda=args.gae_lambda,
        n_epochs=args.n_epochs,
        tensorboard_log=args.logdir,
        policy_kwargs=policy_kwargs,
        device="cuda",
    )

    # Resume training if requested (robust load)
    if args.resume:
        model = load_model_with_env(args.resume, vec_env)

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.checkpoint_every // (args.n_envs * args.n_steps)),
        save_path=args.logdir,
        name_prefix="ppo_mario",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    # Build a separate evaluation vec env (single env, dummy, unclipped reward)
    prev_clip = CLIP_REWARD
    prev_fwd, prev_death = REWARD_FWD, DEATH_PENALTY
    try:
        CLIP_REWARD = False
        REWARD_FWD, DEATH_PENALTY = 0.0, 0.0
        eval_env = build_vec_env(n_envs=1, start_seed=args.seed + 12345, force_dummy=True)
    finally:
        CLIP_REWARD = prev_clip
        REWARD_FWD, DEATH_PENALTY = prev_fwd, prev_death

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.logdir,
        log_path=args.logdir,
        eval_freq=max(1, args.eval_every // (args.n_envs * args.n_steps)),
        n_eval_episodes=max(1, args.eval_episodes),
        deterministic=True,
        render=False,
    )

    callbacks = CallbackList([ckpt_cb, eval_cb])

    # Optionally start live viewer thread to preview best model
    viewer = None
    if args.live_view:
        viewer = LiveViewer(
            args.logdir,
            follow=args.live_follow,
            fps=args.live_fps,
            steps=args.live_steps,
            interval=args.live_interval,
            seed=args.seed + 54321,
        )
        viewer.start()

    model.learn(total_timesteps=args.total_steps, callback=callbacks, tb_log_name="PPO")
    model.save(os.path.join(args.logdir, "ppo_mario_final"))
    vec_env.close()
    if viewer is not None:
        try:
            viewer.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
