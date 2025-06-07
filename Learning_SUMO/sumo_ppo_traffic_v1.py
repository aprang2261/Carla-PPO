import os
import sys
import atexit
import traceback
import subprocess
from pathlib import Path
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import sumolib
import warnings

warnings.filterwarnings("ignore", message=".*Call to deprecated function getAllProgramLogics.*")

# traci 라이브러리 임포트
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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# --- 설정 상수 (Config Section) ---

# 원본 및 생성될 파일 이름 정의
ORIGINAL_NET_FILE = "sumo_files/Town05.net.xml"
TLS_NET_FILE = "sumo_files/Town05.with_tls.net.xml"
RANDOM_TRIPS_FILE = "sumo_files/Town05.trips.xml"
RANDOM_ROUTES_FILE = "sumo_files/Town05.random.rou.xml"
VTYPES_FILE = "sumo_files/carlavtypes.rou.xml"
SUMO_CONFIG_FILE = "sumo_files/Town05.sumocfg"

USE_GUI = False  # 학습 시에는 False, 테스트 시에는 True로 바꿔서 확인 가능
SUMO_STEP_LENGTH = 1.0

# 시뮬레이션 설정
MAX_STEPS_PER_EPISODE = 3600
STEPS_PER_AGENT_ACTION = 10
MIN_PHASE_DURATION_STEPS = 5
YELLOW_LIGHT_DURATION_STEPS = 3

# 저장 경로 설정
CHECKPOINT_DIR = "info/model_checkpoints"
TENSORBOARD_LOG_DIR = "info/tensorboard_logs/"

# GNN/관측 공간 관련 상수
GNN_ADJACENCY_THRESHOLD = 200.0
QUEUE_COUNT_OBS_MAX = 50.0
MAX_WAIT_TIME_OBS_MAX = 300.0
ELAPSED_TIME_OBS_MAX = 200.0

# 보상 함수 가중치 (최적화된 값)
ALPHA_QUEUE_REWARD = 0.05
BETA_CHANGE_PENALTY = 0.01
GAMMA_IMBALANCE_PENALTY = 0.005
DELTA_THROUGHPUT_REWARD = 1.5
EPSILON_DELAY_PENALTY = 0.001

# 학습 설정
TOTAL_TIMESTEPS = 500_000

# 디렉토리 생성
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)


# --- 모든 파일 생성 자동화 ---

def check_and_generate_net_file():
    """TLS 네트워크 파일이 없으면 자동으로 생성합니다."""
    if os.path.exists(TLS_NET_FILE):
        print(f"'{TLS_NET_FILE}' 파일이 이미 존재합니다. 생성을 건너뜁니다.")
        return

    print(f"'{TLS_NET_FILE}' 파일이 없어 새로 생성합니다...")
    netconvert_path = sumolib.checkBinary('netconvert')

    original_net_abs = os.path.abspath(ORIGINAL_NET_FILE)
    tls_net_abs = os.path.abspath(TLS_NET_FILE)

    tls_ids_str = "1050,1070,1148,1162,1175,1260,139,224,245,334,421,509,53,599,685,751,829,905,924,943,965"

    command = [
        netconvert_path,
        "--sumo-net-file", original_net_abs,
        "--tls.set", tls_ids_str,
        "-o", tls_net_abs,
        "--tls.guess.joining",
        "--junctions.join"
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print("신호등 로직이 포함된 네트워크 파일 생성 성공!")
    except subprocess.CalledProcessError as e:
        print("\n--- ERROR: netconvert 실행 실패 ---")
        print("STDERR:", e.stderr)
        raise e


def generate_random_traffic():
    """randomTrips.py를 사용하여 랜덤 트래픽을 생성합니다."""
    random_trips_script = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')

    net_file_abs = os.path.abspath(TLS_NET_FILE)
    trips_file_abs = os.path.abspath(RANDOM_TRIPS_FILE)
    route_file_abs = os.path.abspath(RANDOM_ROUTES_FILE)

    command = [
        sys.executable, random_trips_script,
        "-n", net_file_abs,
        "-r", route_file_abs,
        "-e", str(MAX_STEPS_PER_EPISODE),
        "-p", "1.0",
        "--validate"
    ]

    print(f"'{RANDOM_ROUTES_FILE}' 파일 생성을 시작합니다...")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')

        if os.path.exists(trips_file_abs):
            os.remove(trips_file_abs)
            print(f"임시 파일 '{RANDOM_TRIPS_FILE}'을 삭제했습니다.")

    except subprocess.CalledProcessError as e:
        print("\n--- ERROR: randomTrips.py 스크립트 실행 실패 ---")
        print("STDERR:", e.stderr)
        raise e
    print("랜덤 트래픽 생성 완료.")


def generate_sumo_config():
    """SUMO 설정 파일을 생성합니다."""
    net_filename = os.path.basename(TLS_NET_FILE)
    vtypes_filename = os.path.basename(VTYPES_FILE)
    routes_filename = os.path.basename(RANDOM_ROUTES_FILE)

    route_files_str = f'"{vtypes_filename},{routes_filename}"'

    sumocfg_content = f"""<configuration>
    <input>
        <net-file value="{net_filename}"/>
        <route-files value={route_files_str}/>
    </input>
    <time><begin value="0"/><step-length value="{SUMO_STEP_LENGTH}"/></time>
    <report><no-step-log value="true"/></report>
</configuration>"""
    with open(SUMO_CONFIG_FILE, "w") as f:
        f.write(sumocfg_content)
    print(f"SUMO 설정 파일 '{SUMO_CONFIG_FILE}' 생성 완료.")


# --- 콜백, 특징 추출기, 인접 행렬 생성 함수 ---
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


class SaveVecNormalizeCheckpoint(BaseCallback):
    """ VecNormalize 통계를 함께 저장하는 커스텀 체크포인트 콜백 """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
            self.model.save(model_path)
            if self.verbose > 0: print(f"모델 체크포인트 저장: {model_path}")
            if isinstance(self.training_env, VecNormalize):
                stats_path = os.path.join(self.save_path, f"vecnormalize_{self.num_timesteps}_steps.pkl")
                self.training_env.save(stats_path)
                if self.verbose > 0: print(f"VecNormalize 통계 저장: {stats_path}")
        return True


class GATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, adjacency_matrix: np.ndarray, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        self.node_feature_dim = 6
        assert obs_dim % self.node_feature_dim == 0
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
    net_file_abs = os.path.abspath(net_file)
    if not os.path.exists(net_file_abs):
        raise FileNotFoundError(f"네트워크 파일을 찾을 수 없습니다: {net_file_abs}")
    net = sumolib.net.readNet(net_file_abs)
    tls_nodes = net.getTrafficLights()
    tls_ids = sorted([tl.getID() for tl in tls_nodes])
    num_tls = len(tls_ids)
    if num_tls == 0:
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
    print("SUMO 네트워크 기반 인접 행렬 생성 완료.")
    print(f"찾은 신호등 수: {num_tls}")
    return adj_matrix, tls_ids


# --- SUMO 강화학습 환경 (SumoMultiTLGym) ---
class SumoMultiTLGym(gym.Env):
    PHASE_NS_GREEN, PHASE_NS_YELLOW, PHASE_EW_GREEN, PHASE_EW_YELLOW = 0, 1, 2, 3

    def __init__(self, tls_ids, use_gui=False, port=8813):
        super().__init__()
        self.use_gui = use_gui
        self.port = port
        self.sumo_binary = sumolib.checkBinary('sumo-gui' if self.use_gui else 'sumo')
        self.tls_ids = tls_ids
        self.num_tls = len(self.tls_ids)
        if self.num_tls == 0: raise RuntimeError("제어할 신호등이 없습니다.")
        if self.port % (os.cpu_count() - 2 if os.cpu_count() > 2 else 1) == 0:
            print(f"총 {self.num_tls}개의 신호등 제어 시작 (PID: {os.getpid()})")
        self._start_sumo()
        self._setup_lanes_and_phases()
        self._define_spaces()
        atexit.register(self.close)

    def _start_sumo(self):
        command = [self.sumo_binary, "-c", SUMO_CONFIG_FILE, "--no-warnings", "true"]
        traci.start(command, port=self.port)

    def _setup_lanes_and_phases(self):
        self.phase_ryg_states = {}
        for tl_id in self.tls_ids:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
            if len(logic.phases) >= 4:
                self.phase_ryg_states[tl_id] = [p.state for p in logic.phases[:4]]
            else:
                print(f"경고: {tl_id}의 페이즈가 4개 미만({len(logic.phases)}개)입니다. 첫 페이즈를 반복 사용합니다.")
                self.phase_ryg_states[tl_id] = [logic.phases[0].state] * 4
        self.tl_phases = np.zeros(self.num_tls, dtype=int)
        self.last_phase_change_step, self.current_episode_step = np.zeros(self.num_tls, dtype=int), 0

    def _define_spaces(self):
        num_features = 6
        obs_space_size = self.num_tls * num_features
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_space_size,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([2] * self.num_tls)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if traci.isLoaded(): traci.close()
        self._start_sumo()
        self._setup_lanes_and_phases()
        self.tl_phases.fill(self.PHASE_NS_GREEN)
        self.last_phase_change_step.fill(0)
        self.current_episode_step = 0
        self._apply_phase_to_lights()
        for _ in range(50): traci.simulationStep()
        obs = self._get_observation()
        return obs, {}

    def step(self, actions):
        start_arrived = traci.simulation.getArrivedNumber()
        num_changes = 0
        for _ in range(STEPS_PER_AGENT_ACTION):
            if self.current_episode_step >= MAX_STEPS_PER_EPISODE: break
            for i in range(self.num_tls):
                time_in_phase = (self.current_episode_step - self.last_phase_change_step[i]) // STEPS_PER_AGENT_ACTION
                is_yellow = self.tl_phases[i] in [self.PHASE_NS_YELLOW, self.PHASE_EW_YELLOW]
                if is_yellow and time_in_phase >= YELLOW_LIGHT_DURATION_STEPS:
                    self.tl_phases[i] = (self.tl_phases[i] + 1) % 4
                    self.last_phase_change_step[i] = self.current_episode_step
                elif not is_yellow and actions[i] == 1 and time_in_phase >= MIN_PHASE_DURATION_STEPS:
                    self.tl_phases[i] = (self.tl_phases[i] + 1) % 4
                    self.last_phase_change_step[i] = self.current_episode_step
                    num_changes += 1
            self._apply_phase_to_lights()
            traci.simulationStep()
            self.current_episode_step += 1
        obs = self._get_observation()
        q, wait, imb = 0.0, 0.0, 0.0
        for i in range(self.num_tls):
            ns_q, ew_q, ns_w, ew_w = self._get_queue_and_wait_for_tl(self.tls_ids[i])
            q += (ns_q + ew_q);
            wait += (ns_w + ew_w);
            imb += abs(ns_q - ew_q)
        reward_queue = -ALPHA_QUEUE_REWARD * q;
        reward_change = -BETA_CHANGE_PENALTY * num_changes;
        reward_imbalance = -GAMMA_IMBALANCE_PENALTY * imb
        reward_throughput = DELTA_THROUGHPUT_REWARD * (traci.simulation.getArrivedNumber() - start_arrived);
        reward_delay = -EPSILON_DELAY_PENALTY * wait
        total_reward = reward_queue + reward_change + reward_imbalance + reward_throughput + reward_delay
        done = self.current_episode_step >= MAX_STEPS_PER_EPISODE
        info = {"reward_queue": reward_queue, "reward_change_penalty": reward_change,
                "reward_imbalance": reward_imbalance, "reward_throughput": reward_throughput,
                "reward_delay": reward_delay}
        return obs, total_reward, done, False, info

    def _apply_phase_to_lights(self):
        for i, tl_id in enumerate(self.tls_ids):
            phase = self.tl_phases[i]
            ryg_state = self.phase_ryg_states[tl_id][phase]
            traci.trafficlight.setRedYellowGreenState(tl_id, ryg_state)

    def _get_queue_and_wait_for_tl(self, tl_id):
        ns_q, ew_q, ns_w, ew_w = 0, 0, 0.0, 0.0
        # 이 함수는 환경 내부에서만 호출되므로 traci.lane.getIDList() 호출은 생략 가능 (성능 향상)
        incoming_lanes = traci.trafficlight.getControlledLanes(tl_id)
        for lane in incoming_lanes:
            # 대기열 계산을 위해 방향 구분 필요 (여기서는 단순 합산)
            # 이 부분을 더 정교하게 만들려면, _setup_lanes_and_phases에서처럼 방향을 알아내야 함
            q_val = traci.lane.getLastStepHaltingNumber(lane)
            w_val = traci.lane.getWaitingTime(lane)
            # 임시로 ns_q에 모두 더함 (관측값은 합산되므로 결과는 동일)
            ns_q += q_val
            ns_w += w_val
        return ns_q, ew_q, ns_w, ew_w

    def _get_observation(self):
        obs_list = []
        for i, tl_id in enumerate(self.tls_ids):
            ns_q, ew_q, ns_w, ew_w = self._get_queue_and_wait_for_tl(tl_id)
            obs_list.extend([min(ns_q / QUEUE_COUNT_OBS_MAX, 1.0), min(ew_q / QUEUE_COUNT_OBS_MAX, 1.0),
                             min(ns_w / MAX_WAIT_TIME_OBS_MAX, 1.0), min(ew_w / MAX_WAIT_TIME_OBS_MAX, 1.0),
                             self.tl_phases[i] / 3.0,
                             min((self.current_episode_step - self.last_phase_change_step[i]) / ELAPSED_TIME_OBS_MAX,
                                 1.0)])
        return np.array(obs_list, dtype=np.float32)

    def close(self):
        if traci.isLoaded(): traci.close()


# --- 환경 생성 및 학습/테스트 함수 ---
def make_single_env(tls_ids, rank, use_gui=False):
    """단일 환경을 생성하는 헬퍼 함수"""

    def _init():
        port = 8813 + rank
        env = SumoMultiTLGym(tls_ids=tls_ids, use_gui=use_gui, port=port)
        return env

    return _init


def train():
    print("학습에 필요한 SUMO 파일들을 준비합니다...")
    check_and_generate_net_file()
    generate_random_traffic()
    generate_sumo_config()
    print("파일 준비 완료.")

    print("GAT 모델을 위한 인접 행렬 생성 중...")
    adj_matrix, tls_ids = create_adjacency_matrix_from_sumo(TLS_NET_FILE, GNN_ADJACENCY_THRESHOLD)
    if not tls_ids:
        print("오류: 생성된 네트워크 파일에 신호등이 없습니다.")
        return

    num_cpu = os.cpu_count() - 2 if os.cpu_count() > 2 else 1
    print(f"CPU {num_cpu}개를 사용하여 병렬 학습을 시작합니다.")

    env_fns = [lambda rank=i: make_single_env(tls_ids=tls_ids, rank=rank, use_gui=USE_GUI)() for i in range(num_cpu)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    policy_kwargs = dict(
        features_extractor_class=GATFeatureExtractor,
        features_extractor_kwargs=dict(adjacency_matrix=adj_matrix, features_dim=128),
        # net_arch를 리스트가 아닌 딕셔너리로 직접 전달
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    # 체크포인트 콜백
    checkpoint_callback = SaveVecNormalizeCheckpoint(save_freq=50_000, save_path=CHECKPOINT_DIR,
                                                     name_prefix="ppo_sumo_gat_town05", verbose=1)
    progress_callback = TqdmCallback(total_timesteps=TOTAL_TIMESTEPS)
    info_callback = InfoLoggingCallback(log_interval_steps=1000)

    model = PPO("MlpPolicy", vec_env, device="cuda", verbose=1, learning_rate=3e-4, n_steps=4096, batch_size=128,
                n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005,
                tensorboard_log=TENSORBOARD_LOG_DIR, policy_kwargs=policy_kwargs)

    print(f"GAT 모델로 SUMO(Town05) 환경에서 학습을 시작합니다. 총 Timesteps: {TOTAL_TIMESTEPS}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, progress_callback, info_callback],
                log_interval=10)

    print("학습 완료.")
    model.save(os.path.join(CHECKPOINT_DIR, "final_model_sumo_town05"))
    vec_env.save(os.path.join(CHECKPOINT_DIR, "vecnormalize_sumo_town05.pkl"))
    print("최종 모델 및 VecNormalize 통계 저장 완료.")
    vec_env.close()


def test(model_path, vecnormalize_path, num_episodes=5):
    check_and_generate_net_file()
    if not os.path.exists(RANDOM_ROUTES_FILE): generate_random_traffic()
    generate_sumo_config()
    _, tls_ids = create_adjacency_matrix_from_sumo(TLS_NET_FILE, GNN_ADJACENCY_THRESHOLD)

    env = SumoMultiTLGym(tls_ids=tls_ids, use_gui=True)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])

    if os.path.exists(vecnormalize_path):
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print(f"경고: VecNormalize 통계 파일 '{vecnormalize_path}'을 찾을 수 없습니다. 정규화 없이 테스트를 진행합니다.")

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
            if not os.path.exists(args.model_path):
                print(f"에러: 테스트 모델/통계 파일을 찾을 수 없습니다.\n  - 모델: {args.model_path}")
            else:
                # 테스트 시에는 vec-path가 없어도 경고만 하고 진행하도록 수정
                test(args.model_path, args.vec_path, num_episodes=args.episodes)
    except KeyboardInterrupt:
        print("\n사용자 중단. 환경 정리 중...")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        traceback.print_exc()
    finally:
        if 'traci' in sys.modules and traci.isLoaded():
            traci.close()