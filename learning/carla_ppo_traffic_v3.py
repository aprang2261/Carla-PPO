import os
import atexit
import traceback
import time
import numpy as np
import gymnasium as gym
import carla
import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 설정 상수 (Config Section)
# CARLA 설정
CARLA_HOST = "localhost"
CARLA_PORT = 2000
TM_PORT = 8000
MAP_NAME = "Town05"

# 시뮬레이션 설정
FIXED_DELTA_SECONDS = 0.1
MAX_TICKS_PER_EPISODE = 2000  # 에피소드 당 최대 틱 (2000 * 0.1초 = 200초)
TICKS_PER_AGENT_STEP = 10  # 에이전트의 step() 호출 당 world.tick() 횟수
MIN_PHASE_DURATION = 50  # 신호 페이즈 최소 유지 틱 (50틱 * 0.1초 = 5초)
YELLOW_LIGHT_DURATION_TICKS = 30  # 황색 신호등 지속 시간 (30틱 * 0.1초 = 3초)

# 차량 설정
NUM_VEHICLES = 150  # 커리큘럼 학습을 위해 조절 (예: 50 -> 100 -> 150)
MAX_ACTIVE_VEHICLES = 200
RESPAWN_INTERVAL = 20

# 저장 경로 설정
CHECKPOINT_DIR = "info/model_checkpoints_gnn_advanced"
TENSORBOARD_LOG_DIR = "info/tensorboard_logs_gnn_advanced/"

# GNN/관측 공간 관련 상수
GNN_ADJACENCY_THRESHOLD = 150.0  # 신호등을 인접하다고 판단할 최대 거리 (미터)
QUEUE_COUNT_OBS_MAX = 80.0
MAX_WAIT_TIME_OBS_MAX = 300.0  # 최대 대기 시간 관측 최대치 (300틱 = 30초)
ELAPSED_TIME_OBS_MAX = 200.0

# 보상 함수 가중치
ALPHA_QUEUE_REWARD = 0.8  # 대기열 보상 가중치
BETA_CHANGE_PENALTY = 0.1  # 신호 변경 패널티 가중치
GAMMA_IMBALANCE_PENALTY = 0.01  # 대기열 불균형 패널티 가중치
DELTA_THROUGHPUT_REWARD = 0.5  # 처리량 보상 가중치
EPSILON_DELAY_PENALTY = 0.005  # 딜레이 페널티 가중치

# 학습 설정
TOTAL_TIMESTEPS = 2_000_000

# 디렉토리 생성
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)


# 콜백 함수들 (Callbacks)
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

    def _on_training_start(self): self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self):
        self.pbar.update(self.num_timesteps - self.last_step)
        self.last_step = self.num_timesteps
        return True

    def _on_training_end(self):
        if self.pbar: self.pbar.close()


class InfoLoggingCallback(BaseCallback):
    def __init__(self, log_interval_steps=100, verbose=0):
        super().__init__(verbose)
        self.log_interval_steps = log_interval_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval_steps != 0: return True
        infos = self.locals.get("infos", [])
        if not infos: return True

        # 여러 보상 컴포넌트들을 로깅
        reward_components = {k: [] for k in infos[0] if k.startswith('reward_')}
        for info in infos:
            for key in reward_components:
                if key in info: reward_components[key].append(info[key])

        for key, values in reward_components.items():
            if values: self.logger.record(f"custom/{key}", np.mean(values))
        return True


# GAT 특징 추출기 (GATFeatureExtractor)
class GATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, adjacency_matrix: np.ndarray, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        # 6 features per TL: ns_q, ew_q, ns_wait, ew_wait, phase, elapsed
        self.node_feature_dim = 6
        assert obs_dim % self.node_feature_dim == 0, "Observation dimension is not divisible by node feature count"
        self.num_tls = obs_dim // self.node_feature_dim

        # GAT 레이어 정의
        self.conv1 = gnn.GATConv(self.node_feature_dim, 32, heads=4, concat=True)
        self.conv2 = gnn.GATConv(32 * 4, features_dim, heads=1, concat=False)
        self.relu = nn.ReLU()

        # 인접 행렬을 torch_geometric의 edge_index 형식으로 변환하여 버퍼에 등록
        sparse_matrix = coo_matrix(adjacency_matrix)
        edge_index, _ = from_scipy_sparse_matrix(sparse_matrix)
        self.register_buffer("edge_index", edge_index)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        device = observations.device

        # (B, N * 6) -> (B * N, 6)
        x = observations.view(batch_size * self.num_tls, self.node_feature_dim)

        # GAT 레이어 통과
        x = self.relu(self.conv1(x, self.edge_index.repeat(1, batch_size)))
        x = self.relu(self.conv2(x, self.edge_index.repeat(1, batch_size)))

        # 배치 벡터 생성 (예: [0,0,0, 1,1,1, ...])
        batch_vector = torch.arange(batch_size, device=device).repeat_interleave(self.num_tls)

        # Global Mean Pooling
        return gnn.global_mean_pool(x, batch_vector)


def create_adjacency_matrix(tls: list, threshold: float) -> np.ndarray:
    num_tls = len(tls)
    adj_matrix = np.eye(num_tls, dtype=int)  # 자기 자신과의 연결
    locations = [tl.get_location() for tl in tls]

    for i in range(num_tls):
        for j in range(i + 1, num_tls):
            dist = locations[i].distance(locations[j])
            if dist <= threshold:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
    print("인접 행렬 생성 완료 (1: 연결됨):")
    print(adj_matrix)
    return adj_matrix


# 개선된 CARLA 환경 (MultiTLCarlaEnvAdvanced)
class MultiTLCarlaEnvAdvanced(gym.Env):
    # 신호 페이즈 정의: (NS: North-South, EW: East-West)
    PHASE_NS_GREEN = 0
    PHASE_NS_YELLOW = 1
    PHASE_EW_GREEN = 2
    PHASE_EW_YELLOW = 3

    def __init__(self):
        super().__init__()
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(10.0)
        self._setup_carla_world()
        self._setup_tls_and_spawnpoints()
        self._define_spaces()
        atexit.register(cleanup_on_exit, self)

    def _setup_carla_world(self):
        self.world = self.client.get_world()
        if self.world.get_map().name.split('/')[-1] != MAP_NAME:
            self.world = self.client.load_world(MAP_NAME)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)

        self.tm = self.client.get_trafficmanager(TM_PORT)
        self.tm.set_synchronous_mode(True)
        self.tm.global_percentage_speed_difference(0)

    def _setup_tls_and_spawnpoints(self):
        all_actors = self.world.get_actors()
        self.tls = sorted([a for a in all_actors if a.type_id == "traffic.traffic_light"], key=lambda x: x.id)
        if not self.tls: raise RuntimeError("월드에 신호등이 없습니다.")

        self.num_tls = len(self.tls)
        print(f"총 {self.num_tls}개의 신호등을 제어합니다.")

        self.adjacency_matrix = create_adjacency_matrix(self.tls, GNN_ADJACENCY_THRESHOLD)
        self.blueprint_lib = self.world.get_blueprint_library().filter("vehicle.*")
        self.spawn_points = self.world.get_map().get_spawn_points()

        # 상태 추적 변수 초기화
        self.vehicles = []
        self.vehicle_wait_times = {}  # {vehicle_id: wait_ticks}
        self.tl_phases = np.zeros(self.num_tls, dtype=int)
        self.last_phase_change_tick = np.zeros(self.num_tls, dtype=int)
        self.current_episode_tick_count = 0

    def _define_spaces(self):
        # 관측: [NS큐, EW큐, NS최대대기, EW최대대기, 현재페이즈, 페이즈지속시간] * num_tls
        num_features = 6
        obs_space_size = self.num_tls * num_features
        obs_low = np.zeros(obs_space_size, dtype=np.float32)
        obs_high = np.full(obs_space_size, 1.0, dtype=np.float32)  # 정규화된 값 사용
        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)

        # 행동: [유지, 전환] * num_tls
        self.action_space = gym.spaces.MultiDiscrete([2] * self.num_tls)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._cleanup_actors()

        self.vehicle_wait_times.clear()
        self.tl_phases.fill(self.PHASE_NS_GREEN)
        self.last_phase_change_tick.fill(0)
        self.current_episode_tick_count = 0

        for tl in self.tls:
            tl.set_state(carla.TrafficLightState.Red)  # 초기에는 모두 Red로 설정
        self._apply_phase_to_lights()  # NS Green으로 변경

        self._spawn_vehicles(initial=True)
        self.world.tick()

        obs = self._get_observation()
        return obs, {}

    def step(self, actions):
        # 1. 처리량 계산 및 대기 시간 업데이트
        passed_vehicles = self._update_and_get_passed_vehicles()

        # 2. 에이전트 행동 및 상태 머신 적용 (N틱 동안)
        num_changes = 0
        for _ in range(TICKS_PER_AGENT_STEP):
            if self.current_episode_tick_count >= MAX_TICKS_PER_EPISODE: break

            # 각 신호등에 대해 페이즈 업데이트
            for i in range(self.num_tls):
                is_yellow = (self.tl_phases[i] == self.PHASE_NS_YELLOW or self.tl_phases[i] == self.PHASE_EW_YELLOW)
                time_in_phase = self.current_episode_tick_count - self.last_phase_change_tick[i]

                if is_yellow and time_in_phase >= YELLOW_LIGHT_DURATION_TICKS:
                    # 황색 신호 시간이 끝나면 자동으로 다음 녹색 신호로 전환
                    self.tl_phases[i] = (self.tl_phases[i] + 1) % 4
                    self.last_phase_change_tick[i] = self.current_episode_tick_count

                elif not is_yellow and actions[i] == 1 and time_in_phase >= MIN_PHASE_DURATION:
                    # 에이전트가 '전환'을 선택하고 최소 지속 시간이 지났으면 다음(황색) 페이즈로
                    self.tl_phases[i] = (self.tl_phases[i] + 1) % 4
                    self.last_phase_change_tick[i] = self.current_episode_tick_count
                    num_changes += 1

            self._apply_phase_to_lights()
            self.world.tick()
            self._update_vehicle_wait_times()
            self._spawn_vehicles()
            self.current_episode_tick_count += 1

        # 3. 관측 계산
        obs = self._get_observation()

        # 4. 보상 계산
        total_delay = sum(self.vehicle_wait_times.values())

        queue_sum, imbalance_sum = 0.0, 0.0
        for i in range(self.num_tls):
            ns_q, ew_q = obs[i * 6], obs[i * 6 + 1]
            queue_sum += (ns_q * QUEUE_COUNT_OBS_MAX)  # 역정규화
            imbalance_sum += abs(ns_q - ew_q) * QUEUE_COUNT_OBS_MAX

        reward_queue = -ALPHA_QUEUE_REWARD * queue_sum
        reward_change = -BETA_CHANGE_PENALTY * num_changes
        reward_imbalance = -GAMMA_IMBALANCE_PENALTY * imbalance_sum
        reward_throughput = DELTA_THROUGHPUT_REWARD * len(passed_vehicles)
        reward_delay = -EPSILON_DELAY_PENALTY * total_delay

        total_reward = reward_queue + reward_change + reward_imbalance + reward_throughput + reward_delay

        # 5. 종료 조건 및 정보 반환
        done = self.current_episode_tick_count >= MAX_TICKS_PER_EPISODE
        info = {
            "reward_queue": reward_queue, "reward_change_penalty": reward_change,
            "reward_imbalance": reward_imbalance, "reward_throughput": reward_throughput,
            "reward_delay": reward_delay
        }
        return obs, float(total_reward), done, False, info

    def _apply_phase_to_lights(self):
        """현재 페이즈 상태를 실제 CARLA 신호등에 적용"""
        for i, tl in enumerate(self.tls):
            phase = self.tl_phases[i]
            if phase == self.PHASE_NS_GREEN:
                tl.set_state(carla.TrafficLightState.Green)
            elif phase == self.PHASE_NS_YELLOW:
                tl.set_state(carla.TrafficLightState.Yellow)
            elif phase == self.PHASE_EW_GREEN:
                # GroupTrafficLights API를 사용하는 대신, 모든 등에 직접 상태 설정
                # 이 방식은 복잡한 교차로에서 오작동할 수 있으나, Town05에서는 동작함
                affected_tls = tl.get_group_traffic_lights()
                for affected_tl in affected_tls:
                    if tl.id == affected_tl.id:
                        affected_tl.set_state(carla.TrafficLightState.Red)
                    else:
                        affected_tl.set_state(carla.TrafficLightState.Green)
            elif phase == self.PHASE_EW_YELLOW:
                affected_tls = tl.get_group_traffic_lights()
                for affected_tl in affected_tls:
                    if tl.id == affected_tl.id:
                        affected_tl.set_state(carla.TrafficLightState.Red)
                    else:
                        affected_tl.set_state(carla.TrafficLightState.Yellow)

    def _get_observation(self):
        """모든 신호등에 대한 정규화된 관측 벡터 생성"""
        obs_list = []
        all_vehicles = self.world.get_actors().filter("vehicle.*")

        for i, tl in enumerate(self.tls):
            ns_q, ew_q, ns_max_wait, ew_max_wait = 0, 0, 0, 0
            tl_loc = tl.get_location()

            for v in all_vehicles:
                if v.get_location().distance(tl_loc) > 50.0: continue

                loc = v.get_location()
                dx, dy = loc.x - tl_loc.x, loc.y - tl_loc.y
                wait_time = self.vehicle_wait_times.get(v.id, 0)

                if abs(dx) < 7.5 and abs(dy) < 30.0:  # NS 방향
                    ns_q += 1
                    if wait_time > ns_max_wait: ns_max_wait = wait_time
                elif abs(dy) < 7.5 and abs(dx) < 30.0:  # EW 방향
                    ew_q += 1
                    if wait_time > ew_max_wait: ew_max_wait = wait_time

            # 정규화
            obs_list.append(min(ns_q / QUEUE_COUNT_OBS_MAX, 1.0))
            obs_list.append(min(ew_q / QUEUE_COUNT_OBS_MAX, 1.0))
            obs_list.append(min(ns_max_wait / MAX_WAIT_TIME_OBS_MAX, 1.0))
            obs_list.append(min(ew_max_wait / MAX_WAIT_TIME_OBS_MAX, 1.0))
            obs_list.append(self.tl_phases[i] / 3.0)  # 0,1,2,3 -> [0, 1]
            elapsed = self.current_episode_tick_count - self.last_phase_change_tick[i]
            obs_list.append(min(elapsed / ELAPSED_TIME_OBS_MAX, 1.0))

        return np.array(obs_list, dtype=np.float32)

    def _update_vehicle_wait_times(self):
        for v in self.world.get_actors().filter("vehicle.*"):
            if v.get_velocity().length() < 0.1:  # 정지 상태로 간주
                self.vehicle_wait_times[v.id] = self.vehicle_wait_times.get(v.id, 0) + 1
            else:  # 움직이면 리셋
                if v.id in self.vehicle_wait_times:
                    del self.vehicle_wait_times[v.id]

    def _update_and_get_passed_vehicles(self):
        passed_vehicles = set()
        current_vehicle_ids = {v.id for v in self.world.get_actors().filter("vehicle.*")}

        # 이전에 대기 중이었으나 지금은 없는 차량 (파괴되었거나 맵을 떠남)
        for vid in list(self.vehicle_wait_times.keys()):
            if vid not in current_vehicle_ids:
                passed_vehicles.add(vid)
                del self.vehicle_wait_times[vid]
        return passed_vehicles

    def _spawn_vehicles(self, initial=False):
        active_count = len(self.world.get_actors().filter("vehicle.*"))
        target_count = NUM_VEHICLES if initial else min(MAX_ACTIVE_VEHICLES, NUM_VEHICLES)

        if initial or (active_count < target_count and self.current_episode_tick_count % RESPAWN_INTERVAL == 0):
            spawn_points = self.spawn_points.copy()
            np.random.shuffle(spawn_points)

            for sp in spawn_points:
                if active_count >= target_count: break
                blueprint = np.random.choice(self.blueprint_lib)
                vehicle = self.world.try_spawn_actor(blueprint, sp)
                if vehicle:
                    vehicle.set_autopilot(True, self.tm.get_port())
                    self.tm.ignore_lights_percentage(vehicle, 0)
                    self.vehicles.append(vehicle)
                    active_count += 1

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

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_env():
    env = MultiTLCarlaEnvAdvanced()
    env = Monitor(env)
    return env


# 학습 및 테스트 함수 (Train & Test Functions)
def train():
    print("GAT 모델을 위한 인접 행렬 생성 중...")
    temp_env = make_env()
    # .env를 추가하여 Monitor 래퍼 내부의 원본 환경에 접근
    adj_matrix = temp_env.env.adjacency_matrix
    temp_env.close()

    vec_env = DummyVecEnv([make_env])
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    policy_kwargs = dict(
        features_extractor_class=GATFeatureExtractor,
        features_extractor_kwargs=dict(adjacency_matrix=adj_matrix, features_dim=128),
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )

    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=CHECKPOINT_DIR, name_prefix="ppo_carla_gat")
    progress_callback = TqdmCallback(total_timesteps=TOTAL_TIMESTEPS)
    info_callback = InfoLoggingCallback(log_interval_steps=100)

    model = PPO(
        "MlpPolicy", vec_env, device="cuda", verbose=1,
        learning_rate=3e-4, n_steps=4096, batch_size=128, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005,
        tensorboard_log=TENSORBOARD_LOG_DIR, policy_kwargs=policy_kwargs
    )

    print(f"GAT 모델로 고급 환경에서 학습을 시작합니다. 총 Timesteps: {TOTAL_TIMESTEPS}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, progress_callback, info_callback],
        log_interval=10
    )
    print("학습 완료.")

    model.save(os.path.join(CHECKPOINT_DIR, "final_model"))
    vec_env.save(os.path.join(CHECKPOINT_DIR, "vecnormalize.pkl"))
    print("최종 모델 및 VecNormalize 통계 저장 완료.")
    vec_env.close()


def test(model_path, vecnormalize_path, num_episodes=5):
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(vecnormalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)

    for ep in range(num_episodes):
        obs = vec_env.reset()
        done = False
        pbar = tqdm(desc=f"Episode {ep + 1}/{num_episodes}", total=MAX_TICKS_PER_EPISODE // TICKS_PER_AGENT_STEP)
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            pbar.update(1)
            if done:
                pbar.close()
                print(
                    f"\nEpisode {ep + 1} 종료. Total Reward: {info[0]['episode']['r']:.2f}, Length: {info[0]['episode']['l']}")
                # 상세 보상 출력
                reward_details = {k: v for k, v in info[0].items() if k.startswith('reward_')}
                print(f"  - 상세 보상: {reward_details}")

    vec_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or Test Advanced GAT-PPO agent for CARLA TL Control")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--model-path", type=str, default=os.path.join(CHECKPOINT_DIR, "final_model.zip"))
    parser.add_argument("--vec-path", type=str, default=os.path.join(CHECKPOINT_DIR, "vecnormalize.pkl"))
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    try:
        if args.mode == "train":
            train()
        elif args.mode == "test":
            if not os.path.exists(args.model_path) or not os.path.exists(args.vec_path):
                print(f"에러: 테스트 모델/통계 파일을 찾을 수 없습니다.\n  - 모델: {args.model_path}\n  - 통계: {args.vec_path}")
            else:
                test(args.model_path, args.vec_path, num_episodes=args.episodes)
    except KeyboardInterrupt:
        print("\n사용자 중단. 환경 정리 중...")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        traceback.print_exc()
    finally:
        # 모든 CARLA 관련 객체가 확실히 정리되도록 함
        time.sleep(2)
        os.system('killall -9 carla-simulator')  # CARLA 서버 강제 종료 (필요 시 사용)