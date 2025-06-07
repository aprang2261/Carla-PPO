import os
import atexit
import traceback
import numpy as np
import gymnasium as gym
import carla
import torch
import torch.nn as nn

from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ─────────────────────────────────────────────────────────────────────────────
# 설정 상수 (Config Section)
CARLA_HOST = "localhost"
CARLA_PORT = 2000
TM_PORT = 8000
MAP_NAME = "Town05"

FIXED_DELTA_SECONDS = 0.1
MAX_TICKS_PER_EPISODE = 1000          # 에피소드 당 최대 틱 (1000 * 0.1초 = 100초)
TICKS_PER_AGENT_STEP = 5               # 에이전트의 step() 호출 당 world.tick() 횟수
MIN_DURATION = 50                      # 신호등 상태 최소 유지 틱 (50틱 * 0.1초 = 5초)

NUM_VEHICLES = 150
MAX_ACTIVE_VEHICLES = 200              # 에피소드 중 최대 차량 수
RESPAWN_INTERVAL = 20                  # ticks마다 차량 부족 시 추가 스폰
CHECKPOINT_DIR = "info/model_checkpoints"
TENSORBOARD_LOG_DIR = "info/tensorboard_logs/"

# 관측 공간 관련 상수
QUEUE_COUNT_OBS_MAX = 80.0             # 대기 차량 수 최대치
ELAPSED_TIME_OBS_MAX = 200.0           # 신호 지속 시간 최대치

# 보상 함수 가중치
ALPHA_QUEUE_REWARD = 1.0               # 대기열 보상 가중치
BETA_CHANGE_PENALTY = 0.1              # 신호 변경 패널티 가중치
GAMMA_IMBALANCE_PENALTY = 0.01         # 대기열 불균형 패널티 가중치

TOTAL_TIMESTEPS = 2_000_000            # 학습 총 스텝 수

# 디렉토리 생성
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)


def cleanup_on_exit(env_instance=None):
    if env_instance is not None:
        try:
            env_instance.close()
        except Exception:
            pass


class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.last_step = 0

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        current_step = self.num_timesteps
        self.pbar.update(current_step - self.last_step)
        self.last_step = current_step
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


class InfoLoggingCallback(BaseCallback):
    def __init__(self, log_interval_steps=100, verbose=0):
        super().__init__(verbose)
        self.log_interval_steps = log_interval_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval_steps != 0:
            return True

        infos = self.locals.get("infos", [])
        if not infos:
            return True

        reward_queue_sum = 0.0
        reward_change_penalty_sum = 0.0
        reward_imbalance_sum = 0.0
        count = 0

        for info in infos:
            if "reward_queue" in info:
                reward_queue_sum += info["reward_queue"]
                reward_change_penalty_sum += info["reward_change_penalty"]
                reward_imbalance_sum += info["reward_imbalance"]
                count += 1

        if count > 0:
            self.logger.record("custom/reward_queue", reward_queue_sum / count)
            self.logger.record("custom/reward_change_penalty", reward_change_penalty_sum / count)
            self.logger.record("custom/reward_imbalance", reward_imbalance_sum / count)

        return True


class TLFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, tl_emb_dim: int = 16):
        # observation_space.shape = (num_tls * 4, )
        super().__init__(observation_space, features_dim=1)  # 임시값
        obs_dim = observation_space.shape[0]
        assert obs_dim % 4 == 0, "Observation dimension must be divisible by 4"
        self.num_tls = obs_dim // 4
        self.tl_emb_dim = tl_emb_dim

        # 공유 MLP: 4차원 입력 → 32 → tl_emb_dim 임베딩
        self.shared_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, tl_emb_dim),
            nn.ReLU()
        )

        # 최종 피처 차원 = tl_emb_dim (평균 풀링)
        self._features_dim = tl_emb_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        # (batch_size, num_tls * 4) → (batch_size, num_tls, 4)
        x = observations.view(batch_size, self.num_tls, 4)
        # (batch_size * num_tls, 4)
        x = x.reshape(batch_size * self.num_tls, 4)
        # 공유 네트워크 통과 → (batch_size * num_tls, tl_emb_dim)
        h = self.shared_net(x)
        # (batch_size, num_tls, tl_emb_dim)
        h = h.view(batch_size, self.num_tls, self.tl_emb_dim)
        # TL별 임베딩을 평균(pooling) → (batch_size, tl_emb_dim)
        pooled = h.mean(dim=1)
        return pooled


class MultiTLCarlaEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super().__init__()

        # CARLA 서버 연결
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(10.0)
        self.world = None
        self.tm = None

        # 신호등, 차량, 스폰 포인트
        self.tls = []
        self.num_tls = 0
        self.blueprint_lib = None
        self.spawn_points = []
        self.vehicles = []

        # 상태 추적
        self.last_change_tick = None
        self.action_carry = None
        self.current_episode_tick_count = 0

        # 환경 설정
        self._setup_carla_world()
        self._setup_tls_and_spawnpoints()

        # action/observation 공간 정의
        self._define_spaces()

        # 프로세스 종료 시 정리 보장
        atexit.register(cleanup_on_exit, self)

    def _setup_carla_world(self):
        try:
            self.client.load_world(MAP_NAME)
            self.world = self.client.get_world()
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
            self.world.apply_settings(settings)

            self.tm = self.client.get_trafficmanager(TM_PORT)
            self.tm.set_synchronous_mode(True)
            self.tm.global_percentage_speed_difference(0)

        except Exception as e:
            raise RuntimeError(f"CARLA 월드 설정 중 오류 발생: {e}")

    def _setup_tls_and_spawnpoints(self):
        all_actors = self.world.get_actors()
        self.tls = [actor for actor in all_actors if actor.type_id == "traffic.traffic_light"]
        if not self.tls:
            raise RuntimeError("월드에 신호등이 하나도 없습니다.")
        self.num_tls = len(self.tls)
        print(f"총 {self.num_tls}개의 신호등을 제어합니다.")

        self.blueprint_lib = self.world.get_blueprint_library().filter("vehicle.*")
        self.spawn_points = self.world.get_map().get_spawn_points()
        if not self.spawn_points:
            raise RuntimeError(f"{MAP_NAME} 맵에 차량 스폰 포인트가 없습니다.")

        self.last_change_tick = np.zeros(self.num_tls, dtype=np.int32)
        self.action_carry = np.zeros(self.num_tls, dtype=np.int32)

    def _define_spaces(self):
        num_obs_features_per_tl = 4
        obs_space_size = self.num_tls * num_obs_features_per_tl

        obs_low = np.zeros(obs_space_size, dtype=np.float32)
        obs_high = np.zeros(obs_space_size, dtype=np.float32)

        for i in range(self.num_tls):
            base_idx = i * num_obs_features_per_tl
            # NS, EW 큐 카운트 범위
            obs_low[base_idx: base_idx + 2] = 0.0
            obs_high[base_idx: base_idx + 2] = QUEUE_COUNT_OBS_MAX
            # current_state (0 or 1)
            obs_low[base_idx + 2] = 0.0
            obs_high[base_idx + 2] = 1.0
            # elapsed_time
            obs_low[base_idx + 3] = 0.0
            obs_high[base_idx + 3] = ELAPSED_TIME_OBS_MAX

        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([2] * self.num_tls)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._cleanup_actors()
        self._spawn_vehicles(initial=True)

        for tl in self.tls:
            tl.set_green_time(99999.0)
            tl.set_red_time(99999.0)
            tl.set_yellow_time(0.0)
            tl.set_state(carla.TrafficLightState.Green)

        self.current_episode_tick_count = 0
        self.last_change_tick.fill(0)
        self.action_carry.fill(0)

        if FIXED_DELTA_SECONDS is not None:
            self.world.tick()
        obs = self._get_observation()
        return obs, {}

    def _spawn_vehicles(self, initial=False):
        if initial:
            cps = self.spawn_points.copy()
            np.random.shuffle(cps)
            spawned = 0
            for spawn_point in cps:
                if spawned >= NUM_VEHICLES:
                    break
                blueprint = np.random.choice(self.blueprint_lib)
                vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
                if vehicle:
                    vehicle.set_autopilot(True, self.tm.get_port())
                    self.tm.ignore_lights_percentage(vehicle, 0)
                    self.vehicles.append(vehicle)
                    spawned += 1
        else:
            # 일정 틱 간격마다 차량 수가 부족하면 추가 스폰
            active = [v for v in self.vehicles if v.is_alive]
            if len(active) < NUM_VEHICLES and self.current_episode_tick_count % RESPAWN_INTERVAL == 0:
                cps = self.spawn_points.copy()
                np.random.shuffle(cps)
                for spawn_point in cps:
                    if len(active) >= min(MAX_ACTIVE_VEHICLES, NUM_VEHICLES):
                        break
                    blueprint = np.random.choice(self.blueprint_lib)
                    vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
                    if vehicle:
                        vehicle.set_autopilot(True, self.tm.get_port())
                        self.tm.ignore_lights_percentage(vehicle, 0)
                        self.vehicles.append(vehicle)
                        active.append(vehicle)

    def _get_observation(self):
        """
        각 신호등 주변 차량 수 집계 및 상태 관측 반환
        """
        obs_list = []
        vehicle_actors = self.world.get_actors().filter("vehicle.*")

        for i, tl in enumerate(self.tls):
            ns_count = 0
            ew_count = 0
            tl_loc = tl.get_transform().location

            for veh in vehicle_actors:
                if veh.get_location().distance(tl_loc) > 50.0:
                    continue
                loc = veh.get_transform().location
                dx = loc.x - tl_loc.x
                dy = loc.y - tl_loc.y
                if abs(dx) < 7.5 and abs(dy) < 30.0:
                    ns_count += 1
                elif abs(dy) < 7.5 and abs(dx) < 30.0:
                    ew_count += 1

            obs_list.append(np.clip(ns_count, 0, QUEUE_COUNT_OBS_MAX))
            obs_list.append(np.clip(ew_count, 0, QUEUE_COUNT_OBS_MAX))
            obs_list.append(float(self.action_carry[i]))
            elapsed = float(self.current_episode_tick_count - self.last_change_tick[i])
            obs_list.append(np.clip(elapsed, 0, ELAPSED_TIME_OBS_MAX))

        return np.array(obs_list, dtype=np.float32)

    def step(self, actions):
        previous_action_carry = np.copy(self.action_carry)

        # 여러 틱 동안 액션을 유지하며 시뮬레이션 진행
        for _ in range(TICKS_PER_AGENT_STEP):
            if self.current_episode_tick_count >= MAX_TICKS_PER_EPISODE:
                break

            applied_actions = np.zeros(self.num_tls, dtype=np.int32)
            for i in range(self.num_tls):
                if (self.current_episode_tick_count - self.last_change_tick[i]) < MIN_DURATION:
                    applied_actions[i] = self.action_carry[i]
                else:
                    new_act = int(actions[i])
                    applied_actions[i] = new_act
                    if new_act != self.action_carry[i]:
                        self.last_change_tick[i] = self.current_episode_tick_count
                    self.action_carry[i] = new_act

            for i, tl in enumerate(self.tls):
                if applied_actions[i] == 0:
                    tl.set_state(carla.TrafficLightState.Green)
                else:
                    tl.set_state(carla.TrafficLightState.Red)

            self.world.tick()
            self.current_episode_tick_count += 1

            # 부족 시 추가 스폰
            self._spawn_vehicles(initial=False)

        obs = self._get_observation()

        # --- 보상 계산 ---
        queue_sum = 0.0
        imbalance_sum = 0.0
        for i in range(self.num_tls):
            ns_q = obs[i * 4 + 0]
            ew_q = obs[i * 4 + 1]
            queue_sum += (ns_q + ew_q)
            imbalance_sum += abs(ns_q - ew_q)

        reward_queue = -ALPHA_QUEUE_REWARD * queue_sum
        num_changes = np.sum(self.action_carry != previous_action_carry)
        reward_change = -BETA_CHANGE_PENALTY * num_changes
        reward_imbalance = -GAMMA_IMBALANCE_PENALTY * imbalance_sum

        total_reward = reward_queue + reward_change + reward_imbalance

        done = (self.current_episode_tick_count >= MAX_TICKS_PER_EPISODE)
        info = {
            "reward_queue": reward_queue,
            "reward_change_penalty": reward_change,
            "reward_imbalance": reward_imbalance
        }
        return obs, float(total_reward), done, False, info

    def _cleanup_actors(self):
        """
        기존 스폰된 차량 제거
        """
        for v in self.vehicles:
            try:
                if v and v.is_alive:
                    v.destroy()
            except Exception:
                pass
        self.vehicles.clear()

    def close(self):
        print("Closing Carla environment...")
        self._cleanup_actors()
        if self.world:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            except Exception:
                pass
        if self.tm:
            try:
                self.tm.set_synchronous_mode(False)
            except Exception:
                pass
        print("Carla environment closed.")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_env():
    env = MultiTLCarlaEnv()
    env = Monitor(env)  # 에피소드별 리턴/길이 자동 기록
    return env


def train():
    num_cpu = 1
    env_fns = [make_env for _ in range(num_cpu)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=ELAPSED_TIME_OBS_MAX)

    policy_kwargs = dict(
        features_extractor_class=TLFeatureExtractor,
        features_extractor_kwargs=dict(tl_emb_dim=16),
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // num_cpu, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_carla_tls_improved"
    )
    progress_callback = TqdmCallback(total_timesteps=TOTAL_TIMESTEPS)
    info_callback = InfoLoggingCallback(log_interval_steps=100)

    model = PPO(
        "MlpPolicy",
        vec_env,
        device="cpu",
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        policy_kwargs=policy_kwargs
    )

    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, progress_callback, info_callback],
        log_interval=10
    )
    print("Training finished.")

    model_path = os.path.join(CHECKPOINT_DIR, "ppo_carla_tls_improved_final_model")
    vecnormalize_path = os.path.join(CHECKPOINT_DIR, "ppo_carla_tls_improved_vecnormalize.pkl")
    model.save(model_path)
    vec_env.save(vecnormalize_path)
    print(f"Saved model to {model_path}")
    print(f"Saved VecNormalize stats to {vecnormalize_path}")

    vec_env.close()


def test(model_path, vecnormalize_path, num_episodes=5):
    env = MultiTLCarlaEnv()
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load(vecnormalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)

    for ep in range(num_episodes):
        obs, _ = vec_env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = vec_env.step(action)
            total_reward += reward
            step_count += 1

        print(f"Episode {ep+1}/{num_episodes} - Total Reward: {total_reward:.2f}, Steps: {step_count}")

    vec_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or Test PPO agent on CARLA Traffic Lights Control")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="실행 모드: train (학습) 또는 test (테스트)")
    parser.add_argument("--model-path", type=str,
                        default=os.path.join(CHECKPOINT_DIR, "ppo_carla_tls_improved_final_model.zip"),
                        help="테스트 모드에서 사용할 PPO 모델 경로 (.zip).")
    parser.add_argument("--vec-path", type=str,
                        default=os.path.join(CHECKPOINT_DIR, "ppo_carla_tls_improved_vecnormalize.pkl"),
                        help="테스트 모드에서 사용할 VecNormalize 통계 경로 (.pkl).")
    parser.add_argument("--episodes", type=int, default=5,
                        help="테스트 모드에서 실행할 에피소드 수")
    args = parser.parse_args()

    try:
        if args.mode == "train":
            train()
        elif args.mode == "test":
            test(args.model_path, args.vec_path, num_episodes=args.episodes)
    except KeyboardInterrupt:
        print("사용자 중단 (KeyboardInterrupt). 환경을 정리합니다...")
    except Exception as e:
        print(f"오류 발생: {e}")
        traceback.print_exc()
    finally:
        pass