import os
import carla
import gym
import numpy as np
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# ─────────────────────────────────────────────────────────────────────────────
# 설정 상수
CARLA_HOST = "localhost"
CARLA_PORT = 2000
TM_PORT = 8000
MAP_NAME = "Town05"

FIXED_DELTA_SECONDS = 0.1
MAX_TICKS_PER_EPISODE = 1000  # 에피소드 당 최대 틱 (1000틱 * 0.1초/틱 = 100초 시뮬레이션)
TICKS_PER_AGENT_STEP = 5  # 에이전트의 step() 호출 당 내부 시뮬레이션 world.tick() 횟수 (5틱 * 0.1초/틱 = 0.5초마다 에이전트 결정)
MIN_DURATION = 50  # 신호등 상태 최소 유지 틱 수 (50틱 * 0.1초/틱 = 5초)

NUM_VEHICLES = 150
CHECKPOINT_DIR = "../model_checkpoints"
TENSORBOARD_LOG_DIR = "../tensorboard_logs/"

# 관측 공간 관련 상수
QUEUE_COUNT_OBS_MAX = 50.0  # 대기 차량 수 관측값 최대치
ELAPSED_TIME_OBS_MAX = 200.0  # 신호 지속 시간 관측값 최대치 (MIN_DURATION * 5 또는 적절한 값)

# 보상 함수 가중치 (튜닝 필요)
ALPHA_QUEUE_REWARD = 1.0  # 대기열 보상 가중치
BETA_CHANGE_PENALTY = 0.1  # 신호 변경 패널티 가중치
GAMMA_IMBALANCE_PENALTY = 0.01  # 대기열 불균형 패널티 가중치

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.last_step = 0  # 이전 timesteps 저장

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        current_step = self.num_timesteps
        self.pbar.update(current_step - self.last_step)
        self.last_step = current_step
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()

class InfoLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        reward_queue_sum = 0
        reward_change_penalty_sum = 0
        reward_imbalance_sum = 0
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

class MultiTLCarlaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MultiTLCarlaEnv, self).__init__()

        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(10.0)
        self.client.load_world(MAP_NAME)
        self.world = self.client.get_world()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)

        self.tm = self.client.get_trafficmanager(TM_PORT)
        self.tm.set_synchronous_mode(True)
        self.tm.global_percentage_speed_difference(0)

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
        self.action_carry = np.zeros(self.num_tls, dtype=np.int32)  # 현재 신호등의 실제 상태 (0:Green, 1:Red)
        self.vehicles = []
        self.current_episode_tick_count = 0

        # 관측 공간 정의: 각 TL마다 [ns_queue, ew_queue, current_state, elapsed_time]
        # current_state: 0 for Green, 1 for Red
        # elapsed_time: 현재 상태가 지속된 틱 수
        num_obs_features_per_tl = 4
        obs_space_size = self.num_tls * num_obs_features_per_tl

        obs_low = np.zeros(obs_space_size, dtype=np.float32)
        obs_high = np.zeros(obs_space_size, dtype=np.float32)

        for i in range(self.num_tls):
            base_idx = i * num_obs_features_per_tl
            obs_low[base_idx: base_idx + 2] = 0.0  # ns_queue, ew_queue low
            obs_high[base_idx: base_idx + 2] = QUEUE_COUNT_OBS_MAX  # ns_queue, ew_queue high

            obs_low[base_idx + 2] = 0.0  # current_state low (0)
            obs_high[base_idx + 2] = 1.0  # current_state high (1)

            obs_low[base_idx + 3] = 0.0  # elapsed_time low
            obs_high[base_idx + 3] = ELAPSED_TIME_OBS_MAX  # elapsed_time high

        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([2] * self.num_tls)

    def reset(self):
        self._cleanup_actors()
        self._spawn_vehicles()

        for tl in self.tls:
            tl.set_green_time(99999.0)
            tl.set_red_time(99999.0)
            tl.set_yellow_time(0.0)
            tl.set_state(carla.TrafficLightState.Green)

        self.current_episode_tick_count = 0
        self.last_change_tick.fill(0)
        self.action_carry.fill(0)  # 모든 신호등 초기 상태 Green(0)

        if FIXED_DELTA_SECONDS is not None:
            self.world.tick()  # 차량들이 자리잡도록 한 틱 진행
        return self._get_observation()

    def _spawn_vehicles(self):
        np.random.shuffle(self.spawn_points)
        spawned_count = 0
        for i in range(min(NUM_VEHICLES, len(self.spawn_points))):
            if spawned_count >= NUM_VEHICLES: break
            blueprint = np.random.choice(self.blueprint_lib)
            spawn_point = self.spawn_points[i % len(self.spawn_points)]
            vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True, self.tm.get_port())
                self.tm.ignore_lights_percentage(vehicle, 0)
                self.vehicles.append(vehicle)
                spawned_count += 1

    def _get_observation(self):
        obs_list = []
        vehicle_actors = self.world.get_actors().filter("vehicle.*")

        for i in range(self.num_tls):
            tl = self.tls[i]
            ns_count = 0
            ew_count = 0
            tl_loc = tl.get_transform().location

            for veh_actor in vehicle_actors:
                if veh_actor.get_location().distance(tl_loc) > 50.0: continue
                loc = veh_actor.get_transform().location
                dx = loc.x - tl_loc.x
                dy = loc.y - tl_loc.y
                if abs(dx) < 7.5 and abs(dy) < 30.0:
                    ns_count += 1
                elif abs(dy) < 7.5 and abs(dx) < 30.0:
                    ew_count += 1

            obs_list.append(np.clip(ns_count, 0, QUEUE_COUNT_OBS_MAX))
            obs_list.append(np.clip(ew_count, 0, QUEUE_COUNT_OBS_MAX))

            current_tl_state = float(self.action_carry[i])  # 0 for Green, 1 for Red
            obs_list.append(current_tl_state)

            elapsed_time_ticks = float(self.current_episode_tick_count - self.last_change_tick[i])
            obs_list.append(np.clip(elapsed_time_ticks, 0, ELAPSED_TIME_OBS_MAX))

        return np.array(obs_list, dtype=np.float32)

    def step(self, actions):
        num_signal_changes_this_agent_step = 0

        # TICKS_PER_AGENT_STEP 만큼 루프 돌기 전의 신호 상태 (변경 감지용)
        previous_action_carry_for_change_detection = np.copy(self.action_carry)

        for _ in range(TICKS_PER_AGENT_STEP):
            if self.current_episode_tick_count >= MAX_TICKS_PER_EPISODE:
                break

            applied_actions_this_tick = np.zeros(self.num_tls, dtype=np.int32)
            for i in range(self.num_tls):
                # MIN_DURATION 제약 확인
                if self.current_episode_tick_count - self.last_change_tick[i] < MIN_DURATION:
                    applied_actions_this_tick[i] = self.action_carry[i]
                else:
                    new_action_for_tl = int(actions[i])
                    applied_actions_this_tick[i] = new_action_for_tl
                    # 실제 신호 상태가 변경되면 last_change_tick 업데이트
                    if new_action_for_tl != self.action_carry[i]:
                        self.last_change_tick[i] = self.current_episode_tick_count
                    self.action_carry[i] = new_action_for_tl  # 현재 적용된 (또는 유지된) 액션 저장

            for i, tl in enumerate(self.tls):
                if applied_actions_this_tick[i] == 0:
                    tl.set_state(carla.TrafficLightState.Green)
                else:
                    tl.set_state(carla.TrafficLightState.Red)

            self.world.tick()
            self.current_episode_tick_count += 1

        # TICKS_PER_AGENT_STEP 이후의 관측
        obs = self._get_observation()

        # --- 보상 계산 ---
        # 1. 대기열 보상 (음수)
        queue_lengths_sum = 0
        imbalance_penalty_sum = 0
        for i in range(self.num_tls):
            ns_q = obs[i * 4 + 0]
            ew_q = obs[i * 4 + 1]
            queue_lengths_sum += (ns_q + ew_q)
            imbalance_penalty_sum += (ns_q - ew_q) ** 2  # 차이의 제곱으로 불균형 측정

        reward_queue = -ALPHA_QUEUE_REWARD * queue_lengths_sum

        # 2. 신호 변경 패널티 (음수)
        # TICKS_PER_AGENT_STEP 동안 실제 신호 상태가 변경된 횟수 카운트
        # self.action_carry는 루프 후의 최종 상태, previous_action_carry는 루프 전 상태
        num_actual_changes_in_step = np.sum(self.action_carry != previous_action_carry_for_change_detection)
        reward_change_penalty = -BETA_CHANGE_PENALTY * num_actual_changes_in_step

        # 3. 대기열 불균형 패널티 (음수)
        reward_imbalance = -GAMMA_IMBALANCE_PENALTY * imbalance_penalty_sum

        total_reward = reward_queue + reward_change_penalty + reward_imbalance

        done = (self.current_episode_tick_count >= MAX_TICKS_PER_EPISODE)
        info = {
            "reward_queue": reward_queue,
            "reward_change_penalty": reward_change_penalty,
            "reward_imbalance": reward_imbalance
        }
        return obs, total_reward, done, info

    def _cleanup_actors(self):
        for v in self.vehicles:
            if v and v.is_alive:
                v.destroy()
        self.vehicles.clear()

    def close(self):
        print("Closing Carla environment...")
        self._cleanup_actors()
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            if self.tm: self.tm.set_synchronous_mode(False)
        print("Carla environment closed.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        def make_env():
            env = MultiTLCarlaEnv()
            return env


        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                               clip_obs=ELAPSED_TIME_OBS_MAX)  # clip_obs는 관측값 중 가장 큰 범위로

        model = PPO(
            "MlpPolicy",
            vec_env,
            device="cpu",
            verbose=1,
            learning_rate=3e-4,  # 조금 높여봄 (일반적인 값)
            n_steps=2048,  # 롤아웃 버퍼 크기
            batch_size=64,  # 미니배치 크기
            n_epochs=10,  # 업데이트 당 에폭 수
            gamma=0.99,  # 할인 계수
            gae_lambda=0.95,  # GAE 람다
            clip_range=0.2,  # PPO 클리핑 범위
            ent_coef=0.005,  # 엔트로피 계수 (탐험 장려)
            tensorboard_log=TENSORBOARD_LOG_DIR
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(50000 // vec_env.num_envs, 1),  # 저장 빈도 증가
            save_path=CHECKPOINT_DIR,
            name_prefix="ppo_carla_tls_fi"  # fi: fully_improved
        )

        TOTAL_TIMESTEPS = 2_000_000  # 학습 시간 증가 고려
        print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")

        progress_callback = TqdmCallback(total_timesteps=TOTAL_TIMESTEPS)
        info_callback = InfoLoggingCallback()

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, progress_callback, info_callback],
            log_interval=1 # 기본값은 100 (에피소드 단위). 1로 하면 스텝마다 로깅 (너무 많을 수 있음)
        )
        print("Training finished.")

        model.save(os.path.join(CHECKPOINT_DIR, "ppo_carla_tls_fi_final_model"))
        vec_env.save(os.path.join(CHECKPOINT_DIR, "ppo_carla_tls_fi_vecnormalize.pkl"))
        print(f"Final model and VecNormalize stats saved to {CHECKPOINT_DIR}")

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if 'vec_env' in locals():
            vec_env.close()
        print("Main script finished.")

    # 테스트 부분은 이전 코드와 유사하게 별도로 실행하거나 주석 해제하여 사용
    # VecNormalize 로드, model 로드, deterministic=True로 predict 등