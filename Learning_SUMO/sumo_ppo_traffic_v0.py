import os
import sys
import atexit
import traceback
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import sumolib

# traci 라이브러리 임포트 (SUMO_HOME 환경 변수 필요)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("환경 변수 'SUMO_HOME'을 설정해주세요.")
import traci

from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# --- 설정 상수 (Config Section) ---

# SUMO 설정
SUMO_NET_FILE = "sumo_files/Town05.with_tls.net.xml"
SUMO_ROUTES_FILE = "sumo_files/Town05.rou.xml"
SUMO_VTYPES_FILE = "sumo_files/carlavtypes.rou.xml"
SUMO_CONFIG_FILE = "sumo_files/Town05.sumocfg"

USE_GUI = False  # 학습 시에는 False, 테스트 시에는 True로 바꿔서 확인 가능
SUMO_STEP_LENGTH = 1.0

# 시뮬레이션 설정
MAX_STEPS_PER_EPISODE = 5400
STEPS_PER_AGENT_ACTION = 10
MIN_PHASE_DURATION_STEPS = 5
YELLOW_LIGHT_DURATION_STEPS = 3

# 저장 경로 설정
CHECKPOINT_DIR = "info/model_checkpoints"
TENSORBOARD_LOG_DIR = "info/tensorboard_logs"

# GNN/관측 공간 관련 상수
GNN_ADJACENCY_THRESHOLD = 200.0
QUEUE_COUNT_OBS_MAX = 50.0
MAX_WAIT_TIME_OBS_MAX = 300.0
ELAPSED_TIME_OBS_MAX = 200.0

# 보상 함수 가중치
ALPHA_QUEUE_REWARD = 0.8
BETA_CHANGE_PENALTY = 0.1
GAMMA_IMBALANCE_PENALTY = 0.01
DELTA_THROUGHPUT_REWARD = 0.5
EPSILON_DELAY_PENALTY = 0.005

# 학습 설정
TOTAL_TIMESTEPS = 500_000

# 디렉토리 생성
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)


# --- SUMO 파일 자동 생성 ---
def generate_sumo_config():
    """제공된 파일들을 바탕으로 SUMO 설정 파일을 생성합니다."""
    # os.path.basename을 사용하여 전체 경로에서 파일 이름만 추출합니다.
    net_filename = os.path.basename(SUMO_NET_FILE)
    vtypes_filename = os.path.basename(SUMO_VTYPES_FILE)
    routes_filename = os.path.basename(SUMO_ROUTES_FILE)

    # 경로 파일들을 콤마로 연결
    routes_files_str = f'"{vtypes_filename},{routes_filename}"'

    sumocfg_content = f"""<configuration>
        <input>
            <net-file value="{net_filename}"/>
            <route-files value={routes_files_str}/>
        </input>
        <time>
            <begin value="0"/>
            <step-length value="{SUMO_STEP_LENGTH}"/>
        </time>
        <report>
            <no-step-log value="true"/>
        </report>
    </configuration>
    """
    # 설정 파일을 올바른 위치에 생성합니다.
    # SUMO_CONFIG_FILE에 'sumo_files' 디렉터리가 포함되어 있으므로, 먼저 디렉터리를 만듭니다.
    os.makedirs(os.path.dirname(SUMO_CONFIG_FILE), exist_ok=True)
    with open(SUMO_CONFIG_FILE, "w") as f:
        f.write(sumocfg_content)
    print(f"SUMO 설정 파일 '{SUMO_CONFIG_FILE}' 생성 완료.")


# --- 콜백 함수들 (Callbacks) ---
class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.last_step = 0

    def _on_training_start(self): self.pbar = tqdm(total=self.total_timesteps, desc="학습 진행도")

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
        reward_components = {k: [] for k in infos[0] if k.startswith('reward_')}
        for info in infos:
            for key in reward_components:
                if key in info: reward_components[key].append(info[key])
        for key, values in reward_components.items():
            if values: self.logger.record(f"custom/{key}", np.mean(values))
        return True


# --- GAT 특징 추출기 (GATFeatureExtractor) ---
class GATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, adjacency_matrix: np.ndarray, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        self.node_feature_dim = 6
        assert obs_dim % self.node_feature_dim == 0, "관측 공간 차원이 노드 특징 수로 나누어 떨어지지 않습니다."
        self.num_tls = obs_dim // self.node_feature_dim
        self.conv1 = gnn.GATConv(self.node_feature_dim, 32, heads=4, concat=True)
        self.conv2 = gnn.GATConv(32 * 4, features_dim, heads=1, concat=False)
        self.relu = nn.ReLU()
        sparse_matrix = coo_matrix(adjacency_matrix)
        edge_index, _ = from_scipy_sparse_matrix(sparse_matrix)
        self.register_buffer("edge_index", edge_index)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        device = observations.device
        x = observations.view(batch_size * self.num_tls, self.node_feature_dim)

        edge_index_batch = self.edge_index.to(device)
        edge_indices = [edge_index_batch + i * self.num_tls for i in range(batch_size)]
        edge_index_batched = torch.cat(edge_indices, dim=1)

        x = self.relu(self.conv1(x, edge_index_batched))
        x = self.relu(self.conv2(x, edge_index_batched))
        batch_vector = torch.arange(batch_size, device=device).repeat_interleave(self.num_tls)
        return gnn.global_mean_pool(x, batch_vector)


def create_adjacency_matrix_from_sumo(net_file: str, threshold: float) -> (np.ndarray, list):
    """SUMO .net.xml 파일에서 신호등 위치를 읽어 인접 행렬과 ID 목록을 생성합니다."""
    net_file_abs = os.path.abspath(net_file)
    if not os.path.exists(net_file_abs):
        raise FileNotFoundError(f"네트워크 파일을 찾을 수 없습니다: {net_file_abs}")

    net = sumolib.net.readNet(net_file_abs)

    # --- 수정된 부분 ---
    # getType()으로 노드를 필터링하는 대신, getTrafficLights()를 사용하여 신호등이 있는 모든 교차로를 직접 가져옵니다.
    # 이 방식이 훨씬 안정적입니다.
    tls_nodes = net.getTrafficLights()
    tls_ids = sorted([tl.getID() for tl in tls_nodes])
    # --- 수정 끝 ---

    num_tls = len(tls_ids)

    if num_tls == 0:
        # 이 경고가 계속 표시된다면, .net.xml 파일 자체에 <tlLogic> 태그가 없는 것입니다.
        print("경고: 해당 네트워크 파일에서 <tlLogic> 정의를 찾을 수 없습니다.")
        return np.array([]), []

    adj_matrix = np.eye(num_tls, dtype=int)

    locations = {tl_id: net.getNode(tl_id).getCoord() for tl_id in tls_ids}

    for i in range(num_tls):
        for j in range(i + 1, num_tls):
            loc_i = locations[tls_ids[i]]
            loc_j = locations[tls_ids[j]]
            dist = np.sqrt((loc_i[0] - loc_j[0]) ** 2 + (loc_i[1] - loc_j[1]) ** 2)
            if dist <= threshold:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    print("SUMO 네트워크 기반 인접 행렬 생성 완료 (1: 연결됨):")
    print(adj_matrix)
    print(f"찾은 신호등 수: {num_tls}")
    return adj_matrix, tls_ids


# --- SUMO 강화학습 환경 (SumoMultiTLGym) ---
class SumoMultiTLGym(gym.Env):
    PHASE_NS_GREEN = 0
    PHASE_NS_YELLOW = 1
    PHASE_EW_GREEN = 2
    PHASE_EW_YELLOW = 3

    def __init__(self, tls_ids, use_gui=False):
        super().__init__()
        self.use_gui = use_gui
        self.sumo_binary = sumolib.checkBinary('sumo-gui' if self.use_gui else 'sumo')
        self.tls_ids = tls_ids
        self.num_tls = len(self.tls_ids)

        if self.num_tls == 0:
            raise RuntimeError("제어할 신호등이 없습니다.")
        print(f"총 {self.num_tls}개의 신호등을 제어합니다: {self.tls_ids}")

        self._start_sumo()
        self._setup_lanes_and_phases()
        self._define_spaces()

        atexit.register(self.close)

    def _start_sumo(self):
        traci.start([self.sumo_binary, "-c", SUMO_CONFIG_FILE, "--no-warnings", "true"])

    def _setup_lanes_and_phases(self):
        """ 각 신호등에 연결된 차선과 페이즈별 신호 상태를 자동으로 정의합니다. """
        self.incoming_lanes = {}
        self.phase_ryg_states = {}

        for tl_id in self.tls_ids:
            self.incoming_lanes[tl_id] = {'ns': [], 'ew': []}
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)

            # 차선 ID를 분석하여 방향을 추론합니다.
            # 이 로직은 사용자의 네트워크 명명 규칙에 따라 조정될 수 있습니다.
            for lane_id in controlled_lanes:
                # 차선 ID의 마지막 부분(보통 방향을 나타냄)을 분석합니다.
                direction_part = lane_id.split('_')[-2].upper()
                if 'N' in direction_part or 'S' in direction_part:
                    self.incoming_lanes[tl_id]['ns'].append(lane_id)
                elif 'E' in direction_part or 'W' in direction_part:
                    self.incoming_lanes[tl_id]['ew'].append(lane_id)

            # 페이즈별 RYG 상태 문자열 자동 생성
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
            num_signals = len(logic.phases[0].state)

            # NS Green, EW Red
            ns_green_state = list('r' * num_signals)
            # EW Green, NS Red
            ew_green_state = list('r' * num_signals)

            for i in range(num_signals):
                is_ns_lane = False
                links = traci.trafficlight.getControlledLinks(tl_id)
                if i < len(links):
                    lane_id = links[i][0][0]
                    if lane_id in self.incoming_lanes[tl_id]['ns']:
                        is_ns_lane = True

                # 우회전은 녹색으로 유지 (g)
                if 'g' in logic.phases[0].state[i].lower():
                    if is_ns_lane:
                        ns_green_state[i] = 'G'
                    else:
                        ew_green_state[i] = 'G'

            ns_yellow_state = list("".join(ns_green_state).replace('G', 'y'))
            ew_yellow_state = list("".join(ew_green_state).replace('G', 'y'))

            self.phase_ryg_states[tl_id] = [
                "".join(ns_green_state),
                "".join(ns_yellow_state),
                "".join(ew_green_state),
                "".join(ew_yellow_state)
            ]

        self.tl_phases = np.zeros(self.num_tls, dtype=int)
        self.last_phase_change_step = np.zeros(self.num_tls, dtype=int)
        self.current_episode_step = 0

    def _define_spaces(self):
        num_features = 6
        obs_space_size = self.num_tls * num_features
        obs_low = np.zeros(obs_space_size, dtype=np.float32)
        obs_high = np.full(obs_space_size, 1.0, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([2] * self.num_tls)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if traci.isLoaded():
            traci.close()
        self._start_sumo()

        self.tl_phases.fill(self.PHASE_NS_GREEN)
        self.last_phase_change_step.fill(0)
        self.current_episode_step = 0

        self._apply_phase_to_lights()
        for _ in range(50): traci.simulationStep()

        obs = self._get_observation()
        return obs, {}

    def step(self, actions):
        start_arrived_vehicles = traci.simulation.getArrivedNumber()
        num_changes = 0

        for _ in range(STEPS_PER_AGENT_ACTION):
            if self.current_episode_step >= MAX_STEPS_PER_EPISODE: break

            for i in range(self.num_tls):
                time_in_phase_agent_steps = (self.current_episode_step - self.last_phase_change_step[
                    i]) // STEPS_PER_AGENT_ACTION
                is_yellow = (self.tl_phases[i] == self.PHASE_NS_YELLOW or self.tl_phases[i] == self.PHASE_EW_YELLOW)

                if is_yellow and time_in_phase_agent_steps >= YELLOW_LIGHT_DURATION_STEPS:
                    self.tl_phases[i] = (self.tl_phases[i] + 1) % 4
                    self.last_phase_change_step[i] = self.current_episode_step
                elif not is_yellow and actions[i] == 1 and time_in_phase_agent_steps >= MIN_PHASE_DURATION_STEPS:
                    self.tl_phases[i] = (self.tl_phases[i] + 1) % 4
                    self.last_phase_change_step[i] = self.current_episode_step
                    num_changes += 1

            self._apply_phase_to_lights()
            traci.simulationStep()
            self.current_episode_step += 1

        obs = self._get_observation()
        end_arrived_vehicles = traci.simulation.getArrivedNumber()

        total_queue, total_wait_time, total_imbalance = 0.0, 0.0, 0.0
        for i in range(self.num_tls):
            ns_q, ew_q, ns_wait, ew_wait = self._get_queue_and_wait_for_tl(self.tls_ids[i])
            total_queue += (ns_q + ew_q)
            total_wait_time += (ns_wait + ew_wait)
            total_imbalance += abs(ns_q - ew_q)

        reward_queue = -ALPHA_QUEUE_REWARD * total_queue
        reward_change = -BETA_CHANGE_PENALTY * num_changes
        reward_imbalance = -GAMMA_IMBALANCE_PENALTY * total_imbalance
        reward_throughput = DELTA_THROUGHPUT_REWARD * (end_arrived_vehicles - start_arrived_vehicles)
        reward_delay = -EPSILON_DELAY_PENALTY * total_wait_time
        total_reward = reward_queue + reward_change + reward_imbalance + reward_throughput + reward_delay

        done = self.current_episode_step >= MAX_STEPS_PER_EPISODE
        info = {
            "reward_queue": reward_queue, "reward_change_penalty": reward_change,
            "reward_imbalance": reward_imbalance, "reward_throughput": reward_throughput,
            "reward_delay": reward_delay
        }
        return obs, float(total_reward), done, False, info

    def _apply_phase_to_lights(self):
        for i, tl_id in enumerate(self.tls_ids):
            phase = self.tl_phases[i]
            ryg_state = self.phase_ryg_states[tl_id][phase]
            traci.trafficlight.setRedYellowGreenState(tl_id, ryg_state)

    def _get_queue_and_wait_for_tl(self, tl_id):
        ns_q, ew_q, ns_wait, ew_wait = 0, 0, 0.0, 0.0
        for lane in self.incoming_lanes[tl_id]['ns']:
            if lane in traci.lane.getIDList():
                ns_q += traci.lane.getLastStepHaltingNumber(lane)
                ns_wait += traci.lane.getWaitingTime(lane)
        for lane in self.incoming_lanes[tl_id]['ew']:
            if lane in traci.lane.getIDList():
                ew_q += traci.lane.getLastStepHaltingNumber(lane)
                ew_wait += traci.lane.getWaitingTime(lane)
        return ns_q, ew_q, ns_wait, ew_wait

    def _get_observation(self):
        obs_list = []
        for i, tl_id in enumerate(self.tls_ids):
            ns_q, ew_q, ns_wait, ew_wait = self._get_queue_and_wait_for_tl(tl_id)
            obs_list.append(min(ns_q / QUEUE_COUNT_OBS_MAX, 1.0))
            obs_list.append(min(ew_q / QUEUE_COUNT_OBS_MAX, 1.0))
            obs_list.append(min(ns_wait / MAX_WAIT_TIME_OBS_MAX, 1.0))
            obs_list.append(min(ew_wait / MAX_WAIT_TIME_OBS_MAX, 1.0))
            obs_list.append(self.tl_phases[i] / 3.0)
            elapsed = (self.current_episode_step - self.last_phase_change_step[i])
            obs_list.append(min(elapsed / ELAPSED_TIME_OBS_MAX, 1.0))

        return np.array(obs_list, dtype=np.float32)

    def close(self):
        if traci.isLoaded():
            traci.close()

    def __del__(self):
        self.close()


def make_env(tls_ids, use_gui=False):
    env = SumoMultiTLGym(tls_ids, use_gui=use_gui)
    env = Monitor(env)
    return env


# --- 학습 및 테스트 함수 ---
def train():
    generate_sumo_config()
    print("GAT 모델을 위한 인접 행렬 생성 중...")
    adj_matrix, tls_ids = create_adjacency_matrix_from_sumo(SUMO_NET_FILE, GNN_ADJACENCY_THRESHOLD)

    vec_env = DummyVecEnv([lambda: make_env(tls_ids, use_gui=USE_GUI)])
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    policy_kwargs = dict(
        features_extractor_class=GATFeatureExtractor,
        features_extractor_kwargs=dict(adjacency_matrix=adj_matrix, features_dim=128),
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )

    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=CHECKPOINT_DIR,
                                             name_prefix="ppo_sumo_gat_town05")
    progress_callback = TqdmCallback(total_timesteps=TOTAL_TIMESTEPS)
    info_callback = InfoLoggingCallback(log_interval_steps=100)

    model = PPO(
        "MlpPolicy", vec_env, device="cuda", verbose=1,
        learning_rate=3e-4, n_steps=4096, batch_size=128, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005,
        tensorboard_log=TENSORBOARD_LOG_DIR, policy_kwargs=policy_kwargs
    )

    print(f"GAT 모델로 SUMO(Town05) 환경에서 학습을 시작합니다. 총 Timesteps: {TOTAL_TIMESTEPS}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, progress_callback, info_callback],
        log_interval=10
    )
    print("학습 완료.")

    model.save(os.path.join(CHECKPOINT_DIR, "final_model_sumo_town05"))
    vec_env.save(os.path.join(CHECKPOINT_DIR, "vecnormalize_sumo_town05.pkl"))
    print("최종 모델 및 VecNormalize 통계 저장 완료.")
    vec_env.close()


def test(model_path, vecnormalize_path, num_episodes=5):
    if not os.path.exists(SUMO_CONFIG_FILE): generate_sumo_config()
    _, tls_ids = create_adjacency_matrix_from_sumo(SUMO_NET_FILE, GNN_ADJACENCY_THRESHOLD)

    vec_env = DummyVecEnv([lambda: make_env(tls_ids, use_gui=True)])  # 테스트 시에는 GUI 켜기
    vec_env = VecNormalize.load(vecnormalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)

    for ep in range(num_episodes):
        obs = vec_env.reset()
        done = False
        pbar = tqdm(desc=f"에피소드 {ep + 1}/{num_episodes}", total=MAX_STEPS_PER_EPISODE // STEPS_PER_AGENT_ACTION)
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            pbar.update(1)
            if done:
                pbar.close()
                print(f"\n에피소드 {ep + 1} 종료. 총 보상: {total_reward:.2f}")
    vec_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or Test GAT-PPO agent for SUMO TL Control")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="실행 모드: 'train' 또는 'test'")
    parser.add_argument("--model-path", type=str, default=os.path.join(CHECKPOINT_DIR, "final_model_sumo_town05.zip"),
                        help="테스트할 모델 파일 경로")
    parser.add_argument("--vec-path", type=str, default=os.path.join(CHECKPOINT_DIR, "vecnormalize_sumo_town05.pkl"),
                        help="VecNormalize 통계 파일 경로")
    parser.add_argument("--episodes", type=int, default=5, help="테스트할 에피소드 수")
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
        if traci.isLoaded():
            traci.close()