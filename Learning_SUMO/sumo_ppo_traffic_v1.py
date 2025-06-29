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
USE_GUI = True
MAX_STEPS_PER_EPISODE = 3000
AGENT_STEP_INTERVAL = 10
YELLOW_PHASE_DURATION = 3
ALL_RED_PHASE_DURATION = 2
TRAFFIC_PERIOD = 10.0

# 저장 경로 설정
CHECKPOINT_DIR = "info/model_checkpoints"
TENSORBOARD_LOG_DIR = "info/tensorboard_logs/"

# 관측 공간 관련 상수
NODE_FEATURE_DIM = 10  # [NS직진큐,NS좌회전큐,EW직진큐,EW좌회전큐, NS직진대기,NS좌회전대기,EW직진대기,EW좌회전대기, 현재페이즈, 경과시간]
GNN_ADJACENCY_THRESHOLD = 200.0
QUEUE_COUNT_OBS_MAX = 50.0
MAX_WAIT_TIME_OBS_MAX = 300.0
ELAPSED_TIME_OBS_MAX = 200.0

# 보상 함수 가중치
REWARD_THROUGHPUT = 4.0
PENALTY_QUEUE = -0.05
PENALTY_WAITING_TIME = -0.0025

# 학습 설정
TOTAL_TIMESTEPS = 1_000_000

# 제어할 교차로 ID 목록
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
        env = SumoMultiPhaseEnv(tls_ids=self.tls_ids, use_gui=self.use_gui, port=self.port)
        return Monitor(env)

# --- 파일 생성 함수들 ---
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
         "-e", str(MAX_STEPS_PER_EPISODE), "-p", str(TRAFFIC_PERIOD), "10.0", "--validate"], check=True, capture_output=True, text=True)


def generate_sumo_config():
    routes_str = f'"{os.path.basename(VTYPES_FILE)},{os.path.basename(RANDOM_ROUTES_FILE)}"'
    with open(SUMO_CONFIG_FILE, "w") as f:
        f.write(
            f'<configuration><input><net-file value="{os.path.basename(TLS_NET_FILE)}"/><route-files value={routes_str}/></input><time><begin value="0"/></time><report><no-step-log value="true"/></report></configuration>')

# --- 콜백, GAT, 인접 행렬 생성 함수 ---
class SaveVecNormalizeCheckpoint(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        # 마지막으로 저장한 타임스텝을 기록할 변수 추가
        self.last_save_timestep = 0

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # self.n_calls 대신 self.num_timesteps를 사용하여 조건 확인
        if (self.num_timesteps - self.last_save_timestep) >= self.save_freq:
            # 마지막 저장 시점 업데이트
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
        # 이전에 수정한 것과 같이 self.num_timesteps를 기준으로 간격 확인
        if (self.num_timesteps - self.last_log_timestep) >= self.log_interval_steps:
            self.last_log_timestep = self.num_timesteps

            # 모든 병렬 환경의 보상 값을 수집하기 위한 딕셔너리
            aggregated_rewards = defaultdict(list)

            # self.locals['infos']는 각 병렬 환경의 info 딕셔너리를 담고 있는 튜플입니다.
            for info in self.locals.get("infos", []):
                if not info or 'reward_throughput' not in info:  # 에피소드가 끝나지 않은 환경의 info는 스킵
                    continue
                # 'reward_'로 시작하는 모든 키의 값을 수집합니다.
                for key, value in info.items():
                    if key.startswith("reward_"):
                        aggregated_rewards[key].append(value)

            # 수집된 값이 있을 경우에만 로깅 및 출력
            if aggregated_rewards:
                # 각 보상 항목의 평균을 계산합니다.
                mean_rewards = {key: np.mean(values) for key, values in aggregated_rewards.items()}

                # TensorBoard에 평균값을 로깅합니다.
                for key, mean_value in mean_rewards.items():
                    self.logger.record(f"custom_mean/{key}", mean_value)

                # --- 콘솔 출력을 위한 부분 ---
                # 출력할 문자열을 보기 좋게 포맷팅합니다.
                log_str = f"Timestep: {self.num_timesteps:<8}"
                for key, mean_value in mean_rewards.items():
                    # 'reward_' 접두사를 제거하여 깔끔하게 만듭니다.
                    clean_key = key.replace('reward_', '')
                    log_str += f" | {clean_key}: {mean_value:8.2f}"

                # 콘솔에 최종 문자열을 출력합니다.
                print(log_str)

        return True


class GATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, adjacency_matrix: np.ndarray, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        self.node_feature_dim = NODE_FEATURE_DIM
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


# --- SUMO 강화학습 환경 (4-페이즈) ---
class SumoMultiPhaseEnv(gym.Env):
    ACTION_EW_S, ACTION_EW_L, ACTION_NS_S, ACTION_NS_L = 0, 1, 2, 3

    def __init__(self, tls_ids, use_gui=False, port=8813):
        super().__init__()
        self.use_gui, self.port, self.tls_ids = use_gui, port, tls_ids
        self.sumo_binary = sumolib.checkBinary('sumo-gui' if self.use_gui else 'sumo')
        self.num_tls = len(tls_ids)
        self.net = sumolib.net.readNet(os.path.abspath(TLS_NET_FILE))
        self.prev_arrived = 0

        # --- 수정된 부분: 변수들을 __init__에서 미리 초기화 ---
        self.lanes = {tl_id: {'ns_s': [], 'ns_l': [], 'ew_s': [], 'ew_l': []} for tl_id in self.tls_ids}
        self.phase_map = {tl_id: {} for tl_id in self.tls_ids}
        self.yellow_map = {tl_id: {} for tl_id in self.tls_ids}
        self.all_red_state = {}  # all_red_state도 여기서 초기화
        self.current_phase_action = {tl_id: self.ACTION_NS_S for tl_id in self.tls_ids}
        self.time_since_last_action = {tl_id: 0 for tl_id in self.tls_ids}
        # --- 수정 끝 ---

        if self.port == 8813: print(f"총 {self.num_tls}개의 신호등 제어 시작")

        self._start_sumo()
        self._setup_lanes_and_phases()  # 실제 값들은 여기서 채워짐
        self._define_spaces()
        atexit.register(self.close)

    def _start_sumo(self):
        traci.start([self.sumo_binary, "-c", SUMO_CONFIG_FILE, "--no-warnings", "true", "--time-to-teleport", "-1"],
                    port=self.port)

    def _setup_lanes_and_phases(self):
        """직진/좌회전 차선을 식별하고, 4개의 제어 가능한 페이즈를 수동으로 생성합니다."""
        self.lanes = {tl_id: {'ns_s': [], 'ns_l': [], 'ew_s': [], 'ew_l': []} for tl_id in self.tls_ids}
        self.phase_ryg_map = {tl_id: {} for tl_id in self.tls_ids}
        self.all_red_state = {}  # 각 tl_id별 “모두 적색” 상태 저장

        for tl_id in self.tls_ids:
            links = traci.trafficlight.getControlledLinks(tl_id)

            self.all_red_state[tl_id] = 'r' * len(links)

            # --- 수정된 부분: link_to_move 딕셔너리 초기화 ---
            link_to_move = {}  # 각 신호 링크(인덱스)가 어떤 움직임에 해당하는지 저장
            # --- 수정 끝 ---

            for i, link in enumerate(links):
                if not link or not link[0]: continue
                in_lane, out_lane, _ = link[0]

                # 이전에 lane_info를 사용했던 부분을 link_to_move로 통합하여 더 명확하게 만듭니다.
                try:
                    in_edge = self.net.getLane(in_lane).getEdge()
                    out_edge = self.net.getLane(out_lane).getEdge()

                    v_in = np.array(in_edge.getToNode().getCoord()) - np.array(in_edge.getFromNode().getCoord())
                    v_out = np.array(out_edge.getToNode().getCoord()) - np.array(out_edge.getFromNode().getCoord())

                    angle_in = np.arctan2(v_in[1], v_in[0]) * 180 / np.pi
                    angle_out = np.arctan2(v_out[1], v_out[0]) * 180 / np.pi

                    angle_diff = abs(angle_in - angle_out)
                    if angle_diff > 180: angle_diff = 360 - angle_diff

                    is_ns = (45 < abs(angle_in) < 135)
                    turn = 'l' if angle_diff > 60 else 's'
                    move = f"{'ns' if is_ns else 'ew'}_{turn}"

                    self.lanes[tl_id][move].append(in_lane)
                    link_to_move[i] = move  # 계산된 움직임을 링크 인덱스에 매핑
                except (KeyError, IndexError):
                    continue

            # 4개의 주요 페이즈에 대한 RYG 문자열 직접 생성
            for action in [self.ACTION_EW_S, self.ACTION_EW_L, self.ACTION_NS_S, self.ACTION_NS_L]:
                ryg_state = ['r'] * len(links)
                target_move = ""
                if action == self.ACTION_EW_S:
                    target_move = "ew_s"
                elif action == self.ACTION_EW_L:
                    target_move = "ew_l"
                elif action == self.ACTION_NS_S:
                    target_move = "ns_s"
                elif action == self.ACTION_NS_L:
                    target_move = "ns_l"

                for i in range(len(links)):
                    if link_to_move.get(i) == target_move:
                        ryg_state[i] = 'G'
                self.phase_ryg_map[tl_id][action] = "".join(ryg_state)

        self.current_phase_action = {tl_id: self.ACTION_NS_S for tl_id in self.tls_ids}
        self.time_since_last_action = {tl_id: 0 for tl_id in self.tls_ids}

    def _define_spaces(self):
        obs_space_size = self.num_tls * NODE_FEATURE_DIM
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_space_size,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([4] * self.num_tls)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if traci.isLoaded():
            traci.close()

        self._start_sumo()
        # setup 함수를 다시 호출하여 traci 재연결 후 상태를 다시 읽어옵니다.
        self._setup_lanes_and_phases()

        # 초기 페이즈를 NS_S(남북 직진)으로 설정
        self.current_phase_action = {tl_id: self.ACTION_NS_S for tl_id in self.tls_ids}
        for tl_id in self.tls_ids:
            # reset 시에는 황색/적색 전환 과정 없이 즉시 신호 설정
            self._set_phase(tl_id, self.current_phase_action[tl_id], transition=False)

        self.time_since_last_action = {tl_id: 0 for tl_id in self.tls_ids}
        self.current_episode_step = 0

        # 시뮬레이션 안정화
        for _ in range(50):
            traci.simulationStep()

        self.prev_arrived = traci.simulation.getArrivedNumber()
        return self._get_observation(), {}

    def step(self, actions):
        for i, tl_id in enumerate(self.tls_ids):
            self._set_phase(tl_id, actions[i])

        sim_steps = AGENT_STEP_INTERVAL
        if self.phase_transitioned: sim_steps -= (YELLOW_PHASE_DURATION + ALL_RED_PHASE_DURATION)

        for _ in range(max(1, sim_steps)):
            traci.simulationStep()
            self.current_episode_step += 1

        for tl_id in self.tls_ids: self.time_since_last_action[tl_id] += AGENT_STEP_INTERVAL

        obs = self._get_observation()
        total_q, total_wait = 0.0, 0.0
        for tl_id in self.tls_ids:
            queues, waits = self._get_queue_and_wait_for_tl(tl_id)
            total_q += sum(queues.values())
            total_wait += sum(waits.values())

        reward_queue = PENALTY_QUEUE * total_q
        reward_wait = PENALTY_WAITING_TIME * total_wait

        done = self.current_episode_step >= MAX_STEPS_PER_EPISODE

        current_arrived = traci.simulation.getArrivedNumber()
        # 2) 순증가분만 delta_arrived에 담고, 음수면 0으로 보정
        delta_arrived = current_arrived - self.prev_arrived
        if delta_arrived < 0:
            delta_arrived = 0
        # 3) 다음 스텝을 위해 갱신
        self.prev_arrived = current_arrived

        # throughput 보상 계산
        reward_throughput = REWARD_THROUGHPUT * delta_arrived

        # … queue, wait 보상 계산 …
        total_reward = reward_throughput + reward_queue + reward_wait
        # … done, info 반환 …
        info = {
            "reward_throughput": reward_throughput,
            "reward_queue": reward_queue,
            "reward_wait": reward_wait,
            "raw_queue": total_q,
            "raw_wait": total_wait,
        }
        return obs, total_reward, done, False, info

    def _set_phase(self, tl_id, action, transition=True):
        self.phase_transitioned = False
        if self.current_phase_action.get(tl_id) == action: return

        self.phase_transitioned = True

        if transition:
            current_ryg = self.phase_ryg_map[tl_id].get(self.current_phase_action[tl_id],
                                                        'r' * len(self.all_red_state[tl_id]))
            yellow_state = "".join([c.replace('G', 'y') for c in current_ryg])
            traci.trafficlight.setRedYellowGreenState(tl_id, yellow_state)
            for _ in range(YELLOW_PHASE_DURATION): traci.simulationStep(); self.current_episode_step += 1

            traci.trafficlight.setRedYellowGreenState(tl_id, 'r' * len(current_ryg))
            for _ in range(ALL_RED_PHASE_DURATION): traci.simulationStep(); self.current_episode_step += 1

        target_ryg = self.phase_ryg_map[tl_id].get(action)
        if target_ryg is not None:
            traci.trafficlight.setRedYellowGreenState(tl_id, target_ryg)
            self.current_phase_action[tl_id] = action

    def _get_queue_and_wait_for_tl(self, tl_id):
        queues = {'ns_s': 0, 'ns_l': 0, 'ew_s': 0, 'ew_l': 0}
        waits = {'ns_s': 0, 'ns_l': 0, 'ew_s': 0, 'ew_l': 0}
        for move, lanes in self.lanes[tl_id].items():
            for lane in lanes:
                try:
                    queues[move] += traci.lane.getLastStepHaltingNumber(lane)
                    waits[move] += traci.lane.getWaitingTime(lane)
                except traci.TraCIException:
                    continue
        return queues, waits

    def _get_observation(self):
        obs = np.zeros((self.num_tls, NODE_FEATURE_DIM), dtype=np.float32)
        for i, tl_id in enumerate(self.tls_ids):
            queues, waits = self._get_queue_and_wait_for_tl(tl_id)
            obs[i, 0] = min(queues['ns_s'] / QUEUE_COUNT_OBS_MAX, 1.0)
            obs[i, 1] = min(queues['ns_l'] / QUEUE_COUNT_OBS_MAX, 1.0)
            obs[i, 2] = min(queues['ew_s'] / QUEUE_COUNT_OBS_MAX, 1.0)
            obs[i, 3] = min(queues['ew_l'] / QUEUE_COUNT_OBS_MAX, 1.0)
            obs[i, 4] = min(waits['ns_s'] / MAX_WAIT_TIME_OBS_MAX, 1.0)
            obs[i, 5] = min(waits['ns_l'] / MAX_WAIT_TIME_OBS_MAX, 1.0)
            obs[i, 6] = min(waits['ew_s'] / MAX_WAIT_TIME_OBS_MAX, 1.0)
            obs[i, 7] = min(waits['ew_l'] / MAX_WAIT_TIME_OBS_MAX, 1.0)
            obs[i, 8] = self.current_phase_action[tl_id] / 3.0
            obs[i, 9] = min(self.time_since_last_action[tl_id] / ELAPSED_TIME_OBS_MAX, 1.0)
        return obs.flatten()

    def close(self):
        if traci.isLoaded(): traci.close()


# --- 환경 생성 및 학습/테스트 함수 ---
def make_single_env(tls_ids, rank, use_gui=False):
    def _init(): return Monitor(SumoMultiPhaseEnv(tls_ids=tls_ids, use_gui=use_gui, port=8813 + rank))

    return _init


def train():
    check_and_generate_net_file()
    generate_random_traffic()
    generate_sumo_config()

    adj_matrix, tls_ids = create_adjacency_matrix_from_sumo(TLS_NET_FILE, REAL_TLS_IDS, GNN_ADJACENCY_THRESHOLD)
    if not tls_ids: print("오류: 신호등을 찾을 수 없습니다."); return

    num_cpu = os.cpu_count() - 2 if os.cpu_count() > 2 else 1
    print(f"CPU {num_cpu}개를 사용하여 병렬 학습을 시작합니다.")

    env_fns = [EnvFactory(REAL_TLS_IDS, USE_GUI, 8813 + i) for i in range(num_cpu)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, filename=os.path.join(CHECKPOINT_DIR, "monitor.csv"))
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    policy_kwargs = dict(features_extractor_class=GATFeatureExtractor,
                         features_extractor_kwargs=dict(adjacency_matrix=adj_matrix, features_dim=128),
                         net_arch=dict(pi=[256, 256], vf=[256, 256]))

    checkpoint_callback = SaveVecNormalizeCheckpoint(save_freq=50_000, save_path=CHECKPOINT_DIR,
                                                     name_prefix="ppo_sumo_4phase", verbose=1)
    progress_callback = TqdmCallback(total_timesteps=TOTAL_TIMESTEPS)  # 이 줄 추가
    info_callback = InfoLoggingCallback(log_interval_steps=2000)

    model = PPO("MlpPolicy", vec_env, device="cuda", verbose=2, learning_rate=3e-4, n_steps=4096, batch_size=256,
                n_epochs=10, gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                tensorboard_log=TENSORBOARD_LOG_DIR, policy_kwargs=policy_kwargs)

    print(f"GAT 모델로 SUMO(Town05) 4-페이즈 환경에서 학습을 시작합니다. 총 Timesteps: {TOTAL_TIMESTEPS}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, progress_callback, info_callback],
        progress_bar=False
    )

    model.save(os.path.join(CHECKPOINT_DIR, "final_model.zip"))
    vec_env.save(os.path.join(CHECKPOINT_DIR, "vecnormalize.pkl"))


def test(model_path, vecnormalize_path, num_episodes=5):
    _, tls_ids = create_adjacency_matrix_from_sumo(TLS_NET_FILE, REAL_TLS_IDS, GNN_ADJACENCY_THRESHOLD)
    env = SumoMultiPhaseEnv(tls_ids, use_gui=True)
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

    parser = argparse.ArgumentParser(description="Train or Test 4-Phase GAT-PPO agent")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--model-path", type=str, default=os.path.join(CHECKPOINT_DIR, "final_model.zip"))
    parser.add_argument("--vec-path", type=str, default=os.path.join(CHECKPOINT_DIR, "vecnormalize.pkl"))
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    try:
        if args.mode == 'train':
            train()
        else:
            if not os.path.exists(args.model_path):
                print("테스트 모델 파일 없음")
            else:
                test(args.model_path, args.vec_path, args.episodes)
    except KeyboardInterrupt:
        print("\n사용자 중단.")
    except Exception as e:
        print(f"\n오류 발생: {e}"); traceback.print_exc()
    finally:
        if 'traci' in sys.modules and traci.isLoaded():
            try:
                while traci.isLoaded(): traci.close()
            except Exception:
                pass