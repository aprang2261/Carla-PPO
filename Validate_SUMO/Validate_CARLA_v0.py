import os
import sys
import random
import time
import numpy as np
import gymnasium as gym
import torch
import torch_geometric.nn as gnn
import torch.nn as nn
import carla
import traceback
import sumolib

from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# --- 설정 상수 ---
# CARLA 서버 설정
CARLA_HOST = "localhost"
CARLA_PORT = 2000
MAP_NAME = "Town05"

# 시뮬레이션 설정
FIXED_DELTA_SECONDS = 0.1
MAX_STEPS_PER_EPISODE = 3600  # 3600 * 0.1초 = 6분
TICKS_PER_AGENT_STEP = 10  # 에이전트 행동 1번 당 시뮬레이션 10틱 (1초)
MIN_PHASE_DURATION_STEPS = 5  # <-- 이 줄 추가
YELLOW_LIGHT_DURATION_STEPS = 3
NUM_VEHICLES = 150

# 모델 및 통계 파일 경로
MODEL_PATH = "../Learning_SUMO/info/town05_model_checkpoints_final/final_model_sumo_town05.zip"
STATS_PATH = "../Learning_SUMO/info/town05_model_checkpoints_final/vecnormalize_sumo_town05.pkl"

# GNN/관측 공간 관련 상수 (SUMO 학습 때와 동일하게 유지)
GNN_ADJACENCY_THRESHOLD = 200.0
QUEUE_COUNT_OBS_MAX = 50.0
MAX_WAIT_TIME_OBS_MAX = 300.0  # 300틱 = 30초
ELAPSED_TIME_OBS_MAX = 200.0  # 200 에이전트 스텝


# --- GAT 특징 추출기 (SUMO 학습 때와 완전히 동일) ---
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


# --- CARLA 환경 어댑터 ---
class CarlaValidationEnv(gym.Env):
    PHASE_NS_GREEN, PHASE_NS_YELLOW, PHASE_EW_GREEN, PHASE_EW_YELLOW = 0, 1, 2, 3

    def __init__(self):
        super().__init__()
        # CARLA 클라이언트 연결
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        if self.world.get_map().name.split('/')[-1] != MAP_NAME:
            self.world = self.client.load_world(MAP_NAME)

        # 동기 모드 설정
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)

        self.tm = self.client.get_trafficmanager()
        self.tm.set_synchronous_mode(True)

        self._cleanup_actors()
        self.vehicles = []
        self.vehicle_wait_times = {}

        # --- 수정된 부분: node_feature_dim 변수 추가 ---
        self.node_feature_dim = 6
        # --- 수정 끝 ---

        # 신호등 그룹화 및 공간 정의
        self._setup_junctions()
        self._define_spaces()

    def _cleanup_actors(self):
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')
        for vehicle in vehicle_list:
            vehicle.destroy()
        self.vehicles = []

    def _setup_junctions(self):
        """SUMO의 21개 교차로 기준에 맞춰 CARLA의 신호등 액터를 강제로 그룹화합니다. (최종 수정)"""

        sumo_tls_ids = ['1050', '1070', '1148', '1162', '1175', '1260', '139', '224', '245', '334', '421', '509', '53',
                        '599', '685', '751', '829', '905', '924', '943', '965']

        sumo_net_path = "../Town05.with_tls.net.xml"
        if not os.path.exists(sumo_net_path):
            raise FileNotFoundError(f"SUMO 네트워크 파일 '{sumo_net_path}'를 찾을 수 없습니다. 학습 스크립트를 실행하여 생성해주세요.")

        sumo_net = sumolib.net.readNet(sumo_net_path)
        offsetX, offsetY = sumo_net.getLocationOffset()
        print(f"SUMO netOffset: x={offsetX}, y={offsetY}")

        sumo_junction_locations = {jid: sumo_net.getNode(jid).getCoord() for jid in sumo_tls_ids}

        all_tls_actors = list(self.world.get_actors().filter("traffic.traffic_light"))

        # --- 수정된 부분: 그룹화 로직 강화 ---
        # 1. 모든 CARLA 신호등이 어느 그룹에 속하는지 먼저 매핑합니다.
        actor_to_junction_map = {}
        for carla_tl in all_tls_actors:
            carla_loc = carla_tl.get_location()

            min_dist = float('inf')
            closest_junction_id = None
            for jid, jloc in sumo_junction_locations.items():
                carla_equiv_x = jloc[0] - offsetX
                carla_equiv_y = -(jloc[1] - offsetY)
                dist = carla_loc.distance(carla.Location(x=carla_equiv_x, y=carla_equiv_y, z=carla_loc.z))
                if dist < min_dist:
                    min_dist = dist
                    closest_junction_id = jid

            # 모든 액터는 가장 가까운 그룹에 반드시 할당됩니다.
            if closest_junction_id:
                actor_to_junction_map[carla_tl.id] = closest_junction_id

        # 2. 매핑된 결과를 바탕으로 그룹을 재구성합니다.
        junction_groups = {jid: [] for jid in sumo_tls_ids}
        for actor_id, junction_id in actor_to_junction_map.items():
            actor = self.world.get_actor(actor_id)
            if actor:
                junction_groups[junction_id].append(actor)

        # 3. 최종 self.junctions 리스트를 생성합니다.
        #    이제 멤버가 없는 그룹도 빈 리스트로 포함하여 항상 21개를 유지합니다.
        self.junctions = [sorted(junction_groups[jid], key=lambda x: x.id) for jid in sorted(junction_groups.keys())]
        # --- 수정 끝 ---

        self.num_junctions = len(self.junctions)

        print(f"총 {len(all_tls_actors)}개의 신호등 액터를 SUMO 기준 {self.num_junctions}개의 교차로로 강제 그룹화했습니다.")

        # 비어있는 그룹이 있는지 확인하고 출력
        empty_groups = [sorted(junction_groups.keys())[i] for i, group in enumerate(self.junctions) if not group]
        if empty_groups:
            print(f"경고: 다음 {len(empty_groups)}개 교차로 그룹이 비어있습니다: {empty_groups}")
            print("이 교차로들은 관측/행동 계산에서 제외되지만, 모델 구조(21개)는 유지됩니다.")

        self.tl_phases = np.zeros(self.num_junctions, dtype=int)
        self.last_phase_change_step = np.zeros(self.num_junctions, dtype=int)
        self.current_episode_step = 0

    def _define_spaces(self):
        """SUMO 환경과 동일한 관측/행동 공간을 정의합니다."""
        num_features = 6
        obs_space_size = self.num_junctions * num_features
        obs_low = np.zeros(obs_space_size, dtype=np.float32)
        obs_high = np.full(obs_space_size, 1.0, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([2] * self.num_junctions)

    def reset(self, *, seed=None, options=None):
        self._cleanup_actors()
        self.vehicle_wait_times.clear()

        # 차량 스폰
        blueprint_library = self.world.get_blueprint_library().filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()
        for _ in range(NUM_VEHICLES):
            try:
                blueprint = random.choice(blueprint_library)
                spawn_point = random.choice(spawn_points)
                vehicle = self.world.spawn_actor(blueprint, spawn_point)
                vehicle.set_autopilot(True)
                self.vehicles.append(vehicle)
            except:
                continue  # 스폰 위치 충돌 시 무시

        # 초기 상태 설정
        self.tl_phases.fill(self.PHASE_NS_GREEN)
        self.last_phase_change_step.fill(0)
        self.current_episode_step = 0
        self._apply_phase_to_lights()

        # 시뮬레이션 안정화
        for _ in range(50):
            self.world.tick()

        return self._get_observation(), {}

    def step(self, actions):
        """에이전트의 행동(21개)을 받아 CARLA 환경에 적용합니다."""
        for _ in range(TICKS_PER_AGENT_STEP):
            # 상태 머신 로직 (SUMO 환경과 동일)
            for i in range(self.num_junctions):
                time_in_phase = (self.current_episode_step - self.last_phase_change_step[i])
                is_yellow = self.tl_phases[i] in [self.PHASE_NS_YELLOW, self.PHASE_EW_YELLOW]

                if is_yellow and time_in_phase >= (YELLOW_LIGHT_DURATION_STEPS * TICKS_PER_AGENT_STEP):
                    self.tl_phases[i] = (self.tl_phases[i] + 1) % 4
                    self.last_phase_change_step[i] = self.current_episode_step
                elif not is_yellow and actions[i] == 1 and time_in_phase >= (
                        MIN_PHASE_DURATION_STEPS * TICKS_PER_AGENT_STEP):
                    self.tl_phases[i] = (self.tl_phases[i] + 1) % 4
                    self.last_phase_change_step[i] = self.current_episode_step

            self._apply_phase_to_lights()
            self.world.tick()
            self.current_episode_step += 1

        obs = self._get_observation()
        done = self.current_episode_step >= MAX_STEPS_PER_EPISODE

        # 검증에서는 보상 계산이 중요하지 않으므로 0으로 처리
        return obs, 0.0, done, False, {}

    def _apply_phase_to_lights(self):
        """21개 논리 페이즈를 54개 물리적 신호등에 적용합니다."""
        for i, junction_group in enumerate(self.junctions):
            phase = self.tl_phases[i]
            if phase == self.PHASE_NS_GREEN:
                state = carla.TrafficLightState.Green
            elif phase == self.PHASE_NS_YELLOW:
                state = carla.TrafficLightState.Yellow
            else:  # EW Green or Yellow
                state = carla.TrafficLightState.Red  # EW 주행은 그룹의 다른 신호등이 처리한다고 가정

            # 그룹 내 모든 신호등에 상태 적용 (단순화된 로직)
            for tl in junction_group:
                tl.set_state(state)

    def _get_observation(self):
        """CARLA 세계에서 21개 교차로에 대한 관측을 생성합니다."""
        obs_list = []
        all_vehicles = self.world.get_actors().filter('vehicle.*')

        for i, junction_group in enumerate(self.junctions):
            # --- 수정된 부분: 비어있는 그룹에 대한 예외 처리 ---
            if not junction_group:
                # 그룹이 비어있으면, 해당 교차로의 모든 관측값은 0으로 처리하고 다음으로 넘어갑니다.
                # (큐 0, 대기시간 0, 페이즈 0, 경과시간 0) -> 6개 특징
                obs_list.extend([0.0] * self.node_feature_dim)
                continue
            # --- 수정 끝 ---

            center_loc = self._get_junction_center(junction_group)

            ns_q, ew_q, ns_max_wait, ew_max_wait = 0, 0, 0, 0

            for v in all_vehicles:
                if v.get_location().distance(center_loc) > 50.0:
                    continue

                if v.get_velocity().length() < 0.1:
                    self.vehicle_wait_times[v.id] = self.vehicle_wait_times.get(v.id, 0) + 1
                elif v.id in self.vehicle_wait_times:
                    del self.vehicle_wait_times[v.id]

                wait_time = self.vehicle_wait_times.get(v.id, 0)

                v_loc = v.get_location()
                dx = v_loc.x - center_loc.x
                dy = v_loc.y - center_loc.y

                if abs(dx) < 7.5:
                    ns_q += 1
                    ns_max_wait = max(ns_max_wait, wait_time)
                elif abs(dy) < 7.5:
                    ew_q += 1
                    ew_max_wait = max(ew_max_wait, wait_time)

            # 정규화
            obs_list.append(min(ns_q / QUEUE_COUNT_OBS_MAX, 1.0))
            obs_list.append(min(ew_q / QUEUE_COUNT_OBS_MAX, 1.0))
            obs_list.append(min(ns_max_wait / MAX_WAIT_TIME_OBS_MAX, 1.0))
            obs_list.append(min(ew_max_wait / MAX_WAIT_TIME_OBS_MAX, 1.0))
            obs_list.append(self.tl_phases[i] / 3.0)
            elapsed = (self.current_episode_step - self.last_phase_change_step[i]) / TICKS_PER_AGENT_STEP
            obs_list.append(min(elapsed / ELAPSED_TIME_OBS_MAX, 1.0))

        return np.array(obs_list, dtype=np.float32)

    def _get_junction_center(self, junction_group):
        """교차로 그룹의 중심 위치를 계산합니다."""
        x = sum(tl.get_location().x for tl in junction_group) / len(junction_group)
        y = sum(tl.get_location().y for tl in junction_group) / len(junction_group)
        z = sum(tl.get_location().z for tl in junction_group) / len(junction_group)
        return carla.Location(x, y, z)

    def close(self):
        self._cleanup_actors()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        print("CARLA environment closed.")


def create_carla_adjacency_matrix(net_file: str, tls_ids: list, threshold: float) -> np.ndarray:
    """
    SUMO 네트워크 파일의 좌표를 직접 사용하여 인접 행렬을 생성합니다.
    이 방식은 CARLA에 해당 신호등 액터가 없어도 안정적으로 동작합니다.
    """
    net_path = os.path.abspath(net_file)
    if not os.path.exists(net_path):
        raise FileNotFoundError(f"SUMO 네트워크 파일 '{net_path}'를 찾을 수 없습니다.")

    net = sumolib.net.readNet(net_path)
    offsetX, offsetY = net.getLocationOffset()

    num_junctions = len(tls_ids)
    adj_matrix = np.eye(num_junctions, dtype=int)

    # SUMO 파일에서 직접 교차로 좌표를 읽어옵니다.
    locations = {}
    for jid in tls_ids:
        try:
            node = net.getNode(jid)
            jloc = node.getCoord()
            # CARLA 좌표계에 맞게 변환
            carla_x = jloc[0] - offsetX
            carla_y = -(jloc[1] - offsetY)
            locations[jid] = (carla_x, carla_y)
        except KeyError:
            print(f"경고: {net_file} 파일에서 노드 '{jid}'를 찾을 수 없습니다.")
            locations[jid] = (0, 0)  # 임시 좌표

    sorted_ids = sorted(locations.keys())
    for i in range(num_junctions):
        for j in range(i + 1, num_junctions):
            loc_i = locations[sorted_ids[i]]
            loc_j = locations[sorted_ids[j]]
            dist = np.sqrt((loc_i[0] - loc_j[0]) ** 2 + (loc_i[1] - loc_j[1]) ** 2)
            if dist <= threshold:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    print(f"SUMO 좌표 기반 인접 행렬 생성 완료 ({num_junctions}x{num_junctions})")
    return adj_matrix


def main():
    """메인 검증 함수"""
    vec_env = None
    try:
        # 1. CARLA 환경 생성
        raw_env = CarlaValidationEnv()

        # 2. 인접 행렬 생성
        sumo_tls_ids = ['1050', '1070', '1148', '1162', '1175', '1260', '139', '224', '245', '334', '421', '509', '53',
                        '599', '685', '751', '829', '905', '924', '943', '965']
        sumo_net_path = "../Town05.with_tls.net.xml"
        adj_matrix = create_carla_adjacency_matrix(sumo_net_path, sumo_tls_ids, GNN_ADJACENCY_THRESHOLD)

        # 3. 모델 로드에 필요한 policy_kwargs 정의
        policy_kwargs = dict(
            features_extractor_class=GATFeatureExtractor,
            features_extractor_kwargs=dict(adjacency_matrix=adj_matrix, features_dim=128),
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        )

        # --- 수정된 부분: 환경을 먼저 만들고 load 시점에 전달 ---
        # 4. Monitor 및 VecNormalize 래핑 (1개 환경)
        env = Monitor(raw_env)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(STATS_PATH, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

        # 5. 모델을 로드할 때, 준비된 vec_env를 env 인자로 직접 전달
        model = PPO.load(MODEL_PATH, env=vec_env, custom_objects={'policy_kwargs': policy_kwargs})
        # model.set_env(vec_env) 라인은 이제 필요 없으므로 삭제합니다.
        # --- 수정 끝 ---

        # 6. 검증 루프
        print("\n--- CARLA 환경에서 모델 검증 시작 ---")
        for i in range(3):
            obs = vec_env.reset()
            done = False
            print(f"\n--- 에피소드 {i + 1} 시작 ---")
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, info = vec_env.step(action)
            print(f"--- 에피소드 {i + 1} 종료 ---")

    except Exception as e:
        print(f"\n오류 발생: {e}")
        traceback.print_exc()
    finally:
        if vec_env:
            vec_env.close()


if __name__ == "__main__":
    main()