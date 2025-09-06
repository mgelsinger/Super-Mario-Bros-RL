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

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
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
from stable_baselines3.common.callbacks import CheckpointCallback

# --- Suppress noisy deprecation warnings from legacy Gym ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=r".*No render modes.*")

ENV_ID = "SuperMarioBros-1-1-v0"


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
        env = JoypadSpace(env, RIGHT_ONLY)
        env = GrayScaleObservation(env, keep_dim=False)      # (H,W)
        env = ResizeObservation(env, (84, 84))               # (84,84)
        env = FrameStack(env, num_stack=4, lz4_compress=False)  # typically (84,84,4)
        env = EnsureHWCLast(env)                             # harden to (84,84,4)
        env = RenderModeCompat(env)                          # advertise rgb_array render mode
        env = SafeClose(env)                                 # idempotent close
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


def eval_and_record(model_path: str, out_dir: str, episodes: int = 1, force_dummy: bool = False, video_length: int = 2000):
    """Load a saved model and record evaluation to a single MP4 of desired length.

    - Records a single clip up to `video_length` steps, continuing across resets
      so short episodes don't produce tiny videos.
    """
    os.makedirs(out_dir, exist_ok=True)
    vec = build_vec_env(n_envs=1, force_dummy=True if force_dummy else True)  # force Dummy for eval
    # Trigger only once to keep a single file even across resets
    _fired = {"v": False}
    def _trigger(step):
        if not _fired["v"]:
            _fired["v"] = True
            return True
        return False

    vec = VecVideoRecorder(
        vec,
        out_dir,
        record_video_trigger=_trigger,  # start once at first reset
        video_length=int(video_length),
        name_prefix="eval_mario",
    )
    model = load_model_with_env(model_path, vec, print_system_info=False)

    # Record up to video_length steps, continue across episode boundaries
    obs = vec.reset()
    total = 0
    while total < int(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec.step(action)
        if bool(dones[0]) if np.ndim(dones) > 0 else bool(dones):
            obs = vec.reset()
        total += 1
    vec.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--clip", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--logdir", type=str, default="runs/mario_ppo")
    parser.add_argument("--checkpoint-every", type=int, default=250_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--force-dummy", action="store_true")

    # Recording controls (during training)
    parser.add_argument("--record", action="store_true", help="Record clips during training.")
    parser.add_argument("--record-dir", type=str, default=None, help="Directory for training videos.")
    parser.add_argument("--record-freq", type=int, default=250_000, help="Record every N steps.")
    parser.add_argument("--video-length", type=int, default=2000, help="Max steps per training video.")

    # Evaluation-only recording
    parser.add_argument("--record-only", action="store_true", help="Only record evaluation episodes from --resume.")
    parser.add_argument("--eval-episodes", type=int, default=1, help="Episodes to record in record-only mode.")
    args = parser.parse_args()

    # Record-only path
    if args.record_only:
        if not args.resume:
            raise SystemExit("--record-only requires --resume <path_to_model.zip>")
        out_dir = args.record_dir or os.path.join(args.logdir, "videos_eval")
        eval_and_record(
            args.resume,
            out_dir,
            episodes=args.eval_episodes,
            force_dummy=True,
            video_length=args.video_length,
        )
        return

    os.makedirs(args.logdir, exist_ok=True)
    vec_env = build_vec_env(args.n_envs, force_dummy=args.force_dummy)

    # Optional recording during training
    if args.record or args.record_dir:
        record_dir = args.record_dir or os.path.join(args.logdir, "videos")
        vec_env = maybe_wrap_video(vec_env, True, record_dir, args.record_freq, args.video_length)

    policy_kwargs = dict(normalize_images=False)
    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.lr,
        clip_range=args.clip,
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

    model.learn(total_timesteps=args.total_steps, callback=ckpt_cb, tb_log_name="PPO")
    model.save(os.path.join(args.logdir, "ppo_mario_final"))
    vec_env.close()


if __name__ == "__main__":
    main()
