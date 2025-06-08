import os
import sys
import atexit
import traceback
import subprocess
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import sumolib
import warnings
import multiprocessing as mp

mp.set_start_method('spawn', force=True)

# --- 경고 메시지 무시 ---
warnings.filterwarnings("ignore")

# --- traci 라이브러리 임포트 ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("환경 변수 'SUMO_HOME'을 설정해주세요.")
import traci

# --- 나머지 라이브러리 임포트 ---
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from collections import defaultdict

# --- 설정 상수 (Config Section) ---

# 파일 이름 정의
ORIGINAL_NET_FILE = "sumo_files/Town05.net.xml"
TLS_NET_FILE = "sumo_files/Town05.with_tls.net.xml"
RANDOM_ROUTES_FILE = "sumo_files/Town05.random.rou.xml"
VTYPES_FILE = "sumo_files/carlavtypes.rou.xml"
SUMO_CONFIG_FILE = "sumo_files/Town05.sumocfg"

# 시뮬레이션 설정
USE_GUI = False
MAX_STEPS_PER_EPISODE = 3000
AGENT_STEP_INTERVAL = 10
# 로우 레벨 제어에서는 황색/적색 단계를 에이전트가 직접 학습하거나, 환경이 강제해야 합니다.
# 여기서는 단순성을 위해 직접 적용하지 않습니다. (즉시 신호 변경)
YELLOW_PHASE_DURATION = 0  # 즉시 변경
ALL_RED_PHASE_DURATION = 0  # 즉시 변경
TRAFFIC_PERIOD = 10.0

# 저장 경로 설정
CHECKPOINT_DIR = "info/model_checkpoints"  # 경로 변경
TENSORBOARD_LOG_DIR = "info/tensorboard_logs/"  # 경로 변경

# 관측 공간 관련 상수 (로우 레벨 제어에 맞게 변경)
# [북큐, 남큐, 동큐, 서큐, 북대기, 남대기, 동대기, 서대기, 북신호, 남신호, 동신호, 서신호]
NODE_FEATURE_DIM = 12
GNN_ADJACENCY_THRESHOLD = 200.0
QUEUE_COUNT_OBS_MAX = 50.0
MAX_WAIT_TIME_OBS_MAX = 300.0
ELAPSED_TIME_OBS_MAX = 200.0  # 이 값은 이제 사용되지 않을 가능성이 높음 (신호 변경 즉시 적용)

# 보상 함수 가중치
REWARD_THROUGHPUT = 4.0
PENALTY_QUEUE = -0.05
PENALTY_WAITING_TIME = -0.0025

# 학습 설정
TOTAL_TIMESTEPS = 1_000_000

# 제어할 교차로 ID 목록 (이전과 동일)
REAL_TLS_IDS = ['1050', '1070', '1148', '1175', '1260', '139', '421', '509', '53', '599', '751', '829', '905', '943',
                '965']

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)


class EnvFactory:
    def __init__(self, tls_ids, use_gui, port):
        self.tls_ids = tls_ids
        self.use_gui = use_gui
        self.port = port

    def __call__(self):
        # 반드시 Monitor 래퍼까지 포함해서 반환
        env = SumoLowLevelTlsEnv(tls_ids=self.tls_ids, use_gui=self.use_gui, port=self.port)
        return Monitor(env)


# --- 파일 생성 함수들 (변경 없음) ---
def check_and_generate_net_file():
    if os.path.exists(TLS_NET_FILE): return
    print(f"'{TLS_NET_FILE}' 생성 중...")
    netconvert = sumolib.checkBinary('netconvert')
    subprocess.run(
        [netconvert, "--sumo-net-file", os.path.abspath(ORIGINAL_NET_FILE), "--tls.set", ",".join(REAL_TLS_IDS), "-o",
         os.path.abspath(TLS_NET_FILE), "--tls.guess.joining", "--junctions.join", "--tls.default-type", "actuated"],
        check=True, capture_output=True, text=True)


def generate_random_traffic():
    random_trips = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
    print(f"'{RANDOM_ROUTES_FILE}' 생성 중...")
    subprocess.run(
        [sys.executable, random_trips, "-n", os.path.abspath(TLS_NET_FILE), "-r", os.path.abspath(RANDOM_ROUTES_FILE),
         "-e", str(MAX_STEPS_PER_EPISODE), "-p", str(TRAFFIC_PERIOD), "10.0", "--validate"], check=True,
        capture_output=True, text=True)


def generate_sumo_config():
    routes_str = f'"{os.path.basename(VTYPES_FILE)},{os.path.basename(RANDOM_ROUTES_FILE)}"'
    with open(SUMO_CONFIG_FILE, "w") as f:
        f.write(
            f'<configuration><input><net-file value="{os.path.basename(TLS_NET_FILE)}"/><route-files value={routes_str}/></input><time><begin value="0"/></time><report><no-step-log value="true"/></report></configuration>')


# --- 콜백, GAT, 인접 행렬 생성 함수 (변경 없음) ---
class SaveVecNormalizeCheckpoint(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.last_save_timestep = 0

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_save_timestep) >= self.save_freq:
            self.last_save_timestep = self.num_timesteps
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"\n모델 체크포인트 저장: {model_path}")
            if isinstance(self.training_env, VecNormalize):
                stats_path = os.path.join(self.save_path, f"vecnormalize_{self.num_timesteps}_steps.pkl")
                self.training_env.save(stats_path)
                if self.verbose > 0:
                    print(f"VecNormalize 상태 저장: {stats_path}")
        return True


class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.last_step = 0

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="학습 진행도")

    def _on_step(self) -> bool:
        self.pbar.update(self.num_timesteps - self.last_step)
        self.last_step = self.num_timesteps
        return True

    def _on_training_end(self):
        if self.pbar: self.pbar.close()


class InfoLoggingCallback(BaseCallback):
    """
    info 딕셔너리의 커스텀 보상 값들을 Tensorboard에 로깅하고 콘솔에 출력하는 콜백
    """

    def __init__(self, log_interval_steps: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval_steps = log_interval_steps
        self.last_log_timestep = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_log_timestep) >= self.log_interval_steps:
            self.last_log_timestep = self.num_timesteps
            aggregated_rewards = defaultdict(list)
            for info in self.locals.get("infos", []):
                if not info or 'reward_throughput' not in info:
                    continue
                for key, value in info.items():
                    if key.startswith("reward_"):
                        aggregated_rewards[key].append(value)

            if aggregated_rewards:
                mean_rewards = {key: np.mean(values) for key, values in aggregated_rewards.items()}
                for key, mean_value in mean_rewards.items():
                    self.logger.record(f"custom_mean/{key}", mean_value)
                log_str = f"Timestep: {self.num_timesteps:<8}"
                for key, mean_value in mean_rewards.items():
                    clean_key = key.replace('reward_', '')
                    log_str += f" | {clean_key}: {mean_value:8.2f}"
                print(log_str)
        return True


class GATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, adjacency_matrix: np.ndarray, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        self.node_feature_dim = NODE_FEATURE_DIM  # NODE_FEATURE_DIM 사용
        assert obs_dim % self.node_feature_dim == 0
        self.num_tls = obs_dim // self.node_feature_dim
        self.conv1 = gnn.GATConv(self.node_feature_dim, 32, heads=4, concat=True)
        self.conv2 = gnn.GATConv(32 * 4, features_dim, heads=1, concat=False)
        self.relu = nn.ReLU()
        sparse_matrix = coo_matrix(adjacency_matrix)
        edge_index, _ = from_scipy_sparse_matrix(sparse_matrix)
        self.register_buffer("edge_index", edge_index)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, device = observations.shape[0], observations.device
        x = observations.view(batch_size * self.num_tls, self.node_feature_dim)
        edge_index_batch = self.edge_index.to(device)
        edge_indices = [edge_index_batch + i * self.num_tls for i in range(batch_size)]
        edge_index_batched = torch.cat(edge_indices, dim=1)
        x = self.relu(self.conv1(x, edge_index_batched))
        x = self.relu(self.conv2(x, edge_index_batched))
        batch_vector = torch.arange(batch_size, device=device).repeat_interleave(self.num_tls)
        return gnn.global_mean_pool(x, batch_vector)


def create_adjacency_matrix_from_sumo(net_file: str, tls_ids: list, threshold: float) -> (np.ndarray, list):
    net = sumolib.net.readNet(os.path.abspath(net_file))
    adj_matrix = np.eye(len(tls_ids), dtype=int)
    locations = {tl_id: net.getNode(tl_id).getCoord() for tl_id in tls_ids}
    for i in range(len(tls_ids)):
        for j in range(i + 1, len(tls_ids)):
            loc_i, loc_j = locations[tls_ids[i]], locations[tls_ids[j]]
            dist = np.sqrt((loc_i[0] - loc_j[0]) ** 2 + (loc_i[1] - loc_j[1]) ** 2)
            if dist <= threshold: adj_matrix[i, j] = adj_matrix[j, i] = 1
    print(f"인접 행렬 생성 완료 ({len(tls_ids)}x{len(tls_ids)})")
    return adj_matrix, tls_ids


# --- SUMO 강화학습 환경 (로우 레벨 제어) ---
class SumoLowLevelTlsEnv(gym.Env):
    # 액션: [북향 신호 (0/1), 남향 신호 (0/1), 동향 신호 (0/1), 서향 신호 (0/1)]
    # 0: Red, 1: Green
    APPROACH_NORTH, APPROACH_SOUTH, APPROACH_EAST, APPROACH_WEST = 0, 1, 2, 3
    APPROACH_MAP = {
        'N': APPROACH_NORTH,
        'S': APPROACH_SOUTH,
        'E': APPROACH_EAST,
        'W': APPROACH_WEST
    }
    APPROACH_NAMES = ['N', 'S', 'E', 'W']  # 인덱스에 매핑

    def __init__(self, tls_ids, use_gui=False, port=8813):
        super().__init__()
        self.use_gui, self.port, self.tls_ids = use_gui, port, tls_ids
        self.sumo_binary = sumolib.checkBinary('sumo-gui' if self.use_gui else 'sumo')
        self.num_tls = len(tls_ids)
        self.net = sumolib.net.readNet(os.path.abspath(TLS_NET_FILE))
        self.prev_arrived = 0

        # 각 tl_id별 진입로 차선 그룹 (로우 레벨 제어를 위함)
        self.approach_lanes = {tl_id: {
            'N': [], 'S': [], 'E': [], 'W': []
        } for tl_id in self.tls_ids}

        # 현재 각 진입로의 신호 상태 (0:Red, 1:Green)
        self.current_approach_signals = {
            tl_id: {app: 0 for app in self.APPROACH_NAMES} for tl_id in self.tls_ids
        }

        self.current_episode_step = 0

        if self.port == 8813: print(f"총 {self.num_tls}개의 신호등 제어 시작 (로우 레벨)")

        self._start_sumo()
        self._setup_approach_lanes()  # 로우 레벨에 맞게 함수 이름 변경 및 로직 수정
        self._define_spaces()
        atexit.register(self.close)

    def _start_sumo(self):
        traci.start([self.sumo_binary, "-c", SUMO_CONFIG_FILE, "--no-warnings", "true", "--time-to-teleport", "-1"],
                    port=self.port)

    def _setup_approach_lanes(self):
        """각 신호등의 진입로 차선들을 식별합니다 (로우 레벨 제어를 위함)."""
        for tl_id in self.tls_ids:
            # SUMO의 getControlledLinks는 신호등이 제어하는 모든 링크 (차선)의 정보를 반환합니다.
            # (in_lane, out_lane, link_index)
            links = traci.trafficlight.getControlledLinks(tl_id)

            # 각 링크의 진입 방향을 파악하여 분류
            for link_idx, link_info in enumerate(links):
                if not link_info or not link_info[0]: continue
                in_lane, _, _ = link_info[0]

                try:
                    in_edge = self.net.getLane(in_lane).getEdge()
                    # 진입 엣지의 방향 (yaw)을 기준으로 진입로 분류
                    # SUMO edge의 좌표는 (from_node.x, from_node.y) -> (to_node.x, to_node.y)
                    # 이를 통해 엣지의 방향 벡터를 얻고, 각도를 계산합니다.
                    v_edge = np.array(in_edge.getToNode().getCoord()) - np.array(in_edge.getFromNode().getCoord())
                    angle = np.arctan2(v_edge[1], v_edge[0]) * 180 / np.pi

                    # 각도 범위에 따라 진입로 분류
                    # 북향: 45~135, 동향: -45~45, 남향: -135~-45, 서향: 135~225(or -225~-135)
                    if 45 <= angle < 135:  # 북향 (y 증가)
                        approach = 'N'
                    elif -45 <= angle < 45:  # 동향 (x 증가)
                        approach = 'E'
                    elif -135 <= angle < -45:  # 남향 (y 감소)
                        approach = 'S'
                    else:  # 서향 (x 감소, 135~180 or -180~-135)
                        approach = 'W'

                    self.approach_lanes[tl_id][approach].append(in_lane)

                except (KeyError, IndexError):
                    continue

        # 초기 신호 상태 설정 (모두 Red)
        for tl_id in self.tls_ids:
            for app_name in self.APPROACH_NAMES:
                self.current_approach_signals[tl_id][app_name] = 0  # 0: Red

    def _define_spaces(self):
        obs_space_size = self.num_tls * NODE_FEATURE_DIM
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_space_size,), dtype=np.float32)

        # 액션 스페이스: 각 교차로(num_tls)마다 4개의 진입로(N, S, E, W)를 0(Red) 또는 1(Green)로 제어
        self.action_space = gym.spaces.MultiBinary(4 * self.num_tls)  # MultiBinary는 [0,1] 액션만 가짐

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if traci.isLoaded():
            traci.close()

        self._start_sumo()
        # setup 함수를 다시 호출하여 traci 재연결 후 상태를 다시 읽어옵니다.
        self._setup_approach_lanes()  # 로우 레벨 제어에 맞게 호출

        # 초기 신호 상태 설정 (모두 Red)
        for tl_id in self.tls_ids:
            for app_name in self.APPROACH_NAMES:
                self.current_approach_signals[tl_id][app_name] = 0  # 0: Red
            # 각 tl_id별로 직접 신호 상태 적용 (로우 레벨)
            self._set_tls_state_low_level(tl_id, self.current_approach_signals[tl_id])

        self.current_episode_step = 0

        # 시뮬레이션 안정화
        for _ in range(50):
            traci.simulationStep()

        self.prev_arrived = traci.simulation.getArrivedNumber()
        return self._get_observation(), {}

    def step(self, actions):
        # actions는 (num_tls * 4) 크기의 MultiBinary 배열
        # 각 교차로별 액션 분리
        action_idx = 0
        for i, tl_id in enumerate(self.tls_ids):
            # 현재 교차로의 4개 진입로에 대한 액션 추출
            tl_actions = actions[action_idx: action_idx + 4]  # [N, S, E, W] 에 대한 0/1
            action_idx += 4

            new_signals = {}
            for j, app_name in enumerate(self.APPROACH_NAMES):
                new_signals[app_name] = tl_actions[j]  # 0 또는 1 (Red/Green)

            # 로우 레벨 신호등 제어 적용 (즉시 변경, Yellow/All-Red 없음)
            self._set_tls_state_low_level(tl_id, new_signals)

            # 현재 신호 상태 업데이트
            self.current_approach_signals[tl_id] = new_signals

        # AGENT_STEP_INTERVAL 만큼 시뮬레이션 진행
        for _ in range(AGENT_STEP_INTERVAL):
            traci.simulationStep()
            self.current_episode_step += 1

        obs = self._get_observation()
        total_q, total_wait = 0.0, 0.0
        for tl_id in self.tls_ids:
            queues, waits = self._get_queue_and_wait_for_tl(tl_id)  # 4개 진입로별 큐/대기
            total_q += sum(queues.values())
            total_wait += sum(waits.values())

        reward_queue = PENALTY_QUEUE * total_q
        reward_wait = PENALTY_WAITING_TIME * total_wait

        done = self.current_episode_step >= MAX_STEPS_PER_EPISODE

        current_arrived = traci.simulation.getArrivedNumber()
        delta_arrived = current_arrived - self.prev_arrived
        if delta_arrived < 0:
            delta_arrived = 0
        self.prev_arrived = current_arrived

        reward_throughput = REWARD_THROUGHPUT * delta_arrived

        total_reward = reward_throughput + reward_queue + reward_wait

        info = {
            "reward_throughput": reward_throughput,
            "reward_queue": reward_queue,
            "reward_wait": reward_wait,
            "raw_queue": total_q,
            "raw_wait": total_wait,
            "current_signals": {tl_id: self.current_approach_signals[tl_id].copy() for tl_id in self.tls_ids}
        }
        return obs, total_reward, done, False, info

    def _set_tls_state_low_level(self, tl_id, approach_signals: dict):
        """
        주어진 진입로별 신호 상태(0:Red, 1:Green)에 따라 SUMO 신호등을 직접 제어합니다.
        Yellow, All-Red 단계는 건너뜁니다.
        """
        # 먼저 해당 교차로의 모든 링크를 Red로 초기화
        # traci.trafficlight.getControlledLinks(tl_id)는 (inLane, outLane, linkIdx) 튜플의 리스트를 반환
        links = traci.trafficlight.getControlledLinks(tl_id)
        current_ryg_state = ['r'] * len(links)  # 모든 링크를 빨간불로 초기화

        # 각 진입로의 신호 상태에 따라 RYG 문자열 구성
        for app_name, signal_state in approach_signals.items():
            if signal_state == 1:  # Green
                for in_lane in self.approach_lanes[tl_id][app_name]:
                    # 해당 in_lane이 링크에 포함된 모든 link_idx를 찾아 'G'로 변경
                    for link_idx, link_info in enumerate(links):
                        if link_info and link_info[0] and link_info[0][0] == in_lane:
                            current_ryg_state[link_idx] = 'G'
            # else (signal_state == 0): Red (이미 'r'로 초기화 되어 있음)

        # 최종 RYG 문자열을 SUMO에 적용
        traci.trafficlight.setRedYellowGreenState(tl_id, "".join(current_ryg_state))

    def _get_queue_and_wait_for_tl(self, tl_id):
        """각 진입로(N, S, E, W)별 큐 및 대기 시간을 반환합니다."""
        queues = {app: 0 for app in self.APPROACH_NAMES}
        waits = {app: 0 for app in self.APPROACH_NAMES}

        for app_name, lanes in self.approach_lanes[tl_id].items():
            for lane in lanes:
                try:
                    queues[app_name] += traci.lane.getLastStepHaltingNumber(lane)
                    waits[app_name] += traci.lane.getWaitingTime(lane)
                except traci.TraCIException:
                    continue  # 유효하지 않은 차선일 경우 무시
        return queues, waits

    def _get_observation(self):
        obs = np.zeros((self.num_tls, NODE_FEATURE_DIM), dtype=np.float32)
        for i, tl_id in enumerate(self.tls_ids):
            queues, waits = self._get_queue_and_wait_for_tl(tl_id)

            # 큐 (0-3)
            obs[i, 0] = min(queues['N'] / QUEUE_COUNT_OBS_MAX, 1.0)
            obs[i, 1] = min(queues['S'] / QUEUE_COUNT_OBS_MAX, 1.0)
            obs[i, 2] = min(queues['E'] / QUEUE_COUNT_OBS_MAX, 1.0)
            obs[i, 3] = min(queues['W'] / QUEUE_COUNT_OBS_MAX, 1.0)

            # 대기 시간 (4-7)
            obs[i, 4] = min(waits['N'] / MAX_WAIT_TIME_OBS_MAX, 1.0)
            obs[i, 5] = min(waits['S'] / MAX_WAIT_TIME_OBS_MAX, 1.0)
            obs[i, 6] = min(waits['E'] / MAX_WAIT_TIME_OBS_MAX, 1.0)
            obs[i, 7] = min(waits['W'] / MAX_WAIT_TIME_OBS_MAX, 1.0)

            # 현재 신호 상태 (8-11) - 0:Red, 1:Green
            obs[i, 8] = self.current_approach_signals[tl_id]['N']
            obs[i, 9] = self.current_approach_signals[tl_id]['S']
            obs[i, 10] = self.current_approach_signals[tl_id]['E']
            obs[i, 11] = self.current_approach_signals[tl_id]['W']

        return obs.flatten()

    def close(self):
        if traci.isLoaded(): traci.close()


# --- 환경 생성 및 학습/테스트 함수 ---
def make_single_env(tls_ids, rank, use_gui=False):
    def _init(): return Monitor(SumoLowLevelTlsEnv(tls_ids=tls_ids, use_gui=use_gui, port=8813 + rank))

    return _init


def train():
    check_and_generate_net_file()
    generate_random_traffic()
    generate_sumo_config()

    adj_matrix, tls_ids = create_adjacency_matrix_from_sumo(TLS_NET_FILE, REAL_TLS_IDS, GNN_ADJACENCY_THRESHOLD)
    if not tls_ids: print("오류: 신호등을 찾을 수 없습니다."); return

    num_cpu = os.cpu_count() - 2 if os.cpu_count() > 2 else 1
    print(f"CPU {num_cpu}개를 사용하여 병렬 학습을 시작합니다.")

    # 환경 팩토리도 SumoLowLevelTlsEnv를 사용하도록 변경
    env_fns = [EnvFactory(REAL_TLS_IDS, USE_GUI, 8813 + i) for i in range(num_cpu)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, filename=os.path.join(CHECKPOINT_DIR, "monitor.csv"))
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    policy_kwargs = dict(features_extractor_class=GATFeatureExtractor,
                         features_extractor_kwargs=dict(adjacency_matrix=adj_matrix, features_dim=128),
                         net_arch=dict(pi=[256, 256], vf=[256, 256]))

    checkpoint_callback = SaveVecNormalizeCheckpoint(save_freq=50_000, save_path=CHECKPOINT_DIR,
                                                     name_prefix="ppo_sumo_low_level", verbose=1)  # 이름 변경
    progress_callback = TqdmCallback(total_timesteps=TOTAL_TIMESTEPS)
    info_callback = InfoLoggingCallback(log_interval_steps=2000)

    model = PPO("MlpPolicy", vec_env, device="cuda", verbose=2, learning_rate=3e-4, n_steps=4096, batch_size=256,
                n_epochs=10, gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                tensorboard_log=TENSORBOARD_LOG_DIR, policy_kwargs=policy_kwargs)

    print(f"GAT 모델로 SUMO(Town05) 로우 레벨 신호 제어 환경에서 학습을 시작합니다. 총 Timesteps: {TOTAL_TIMESTEPS}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, progress_callback, info_callback],
        progress_bar=False
    )

    model.save(os.path.join(CHECKPOINT_DIR, "final_model_low_level.zip"))  # 이름 변경
    vec_env.save(os.path.join(CHECKPOINT_DIR, "vecnormalize_low_level.pkl"))  # 이름 변경


def test(model_path, vecnormalize_path, num_episodes=5):
    _, tls_ids = create_adjacency_matrix_from_sumo(TLS_NET_FILE, REAL_TLS_IDS, GNN_ADJACENCY_THRESHOLD)
    # 테스트 환경도 SumoLowLevelTlsEnv로 변경
    env = SumoLowLevelTlsEnv(tls_ids, use_gui=True)
    vec_env = DummyVecEnv([lambda: Monitor(env)])
    if os.path.exists(vecnormalize_path):
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    model = PPO.load(model_path, env=vec_env)
    for ep in range(num_episodes):
        obs = vec_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = vec_env.step(action)
    vec_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or Test Low-Level GAT-PPO agent")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    # 모델 및 통계 파일 기본 경로 변경 (로우 레벨용)
    parser.add_argument("--model-path", type=str, default=os.path.join(CHECKPOINT_DIR, "final_model_low_level.zip"))
    parser.add_argument("--vec-path", type=str, default=os.path.join(CHECKPOINT_DIR, "vecnormalize_low_level.pkl"))
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    try:
        if args.mode == 'train':
            train()
        else:
            if not os.path.exists(args.model_path):
                print(f"테스트 모델 파일 없음: {args.model_path}")
            else:
                test(args.model_path, args.vec_path, args.episodes)
    except KeyboardInterrupt:
        print("\n사용자 중단.")
    except Exception as e:
        print(f"\n오류 발생: {e}");
        traceback.print_exc()
    finally:
        if 'traci' in sys.modules and traci.isLoaded():
            try:
                while traci.isLoaded(): traci.close()
            except Exception:
                pass