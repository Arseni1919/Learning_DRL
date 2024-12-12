from abc import abstractmethod

import numpy as np

from globals import *


class MetaMultiAgentEnv(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def reset(self) -> Tuple[Any, Dict]:
        pass

    @abstractmethod
    def sample_action(self, agent_name: str) -> Any:
        pass

    @abstractmethod
    def sample_actions(self) -> Any:
        pass

    @abstractmethod
    def step(self, actions: Dict[str, Any]) -> Tuple[Any, Any, Any, Dict]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class MetaSingleAgentEnv(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def reset(self) -> Tuple[Any, Dict]:
        pass

    @abstractmethod
    def sample_action(self) -> Any:
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, Any, bool, bool, Dict]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass



def choose_unoccupied_loc(field: np.ndarray) -> List[int]:
    width = field.shape[0]
    height = field.shape[1]
    while True:
        target_x = random.randint(0, height - 1)
        target_y = random.randint(0, width - 1)
        if field[target_x, target_y] == 0:
            break
    return [target_x, target_y]


def create_agents_loc(field: np.ndarray, num_agents: int) -> dict:
    agents_loc = {}
    for i in range(num_agents):
        agents_loc[f'agent_{i}'] = choose_unoccupied_loc(field)
    return agents_loc


def create_rand_field(width: int, height: int, obstacle_ratio: float = 0.1) -> np.ndarray:
    field = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            if random.random() < obstacle_ratio:
                field[i, j] = 1
    return field


def build_agent_obs(
        radius: int,
        agent_name: str,
        next_loc: List[int],
        field: np.ndarray,
        width: int,
        height: int,
        agents_loc: Dict[str, List[int]],
        target: List[int]
) -> np.ndarray:

    side = radius * 2 + 1
    agent_obs = np.zeros((3, side, side))  # map, agents, target
    curr_x, curr_y = next_loc[0], next_loc[1]

    for i_x in range(-radius, radius + 1):
        for i_y in range(-radius, radius + 1):
            i_loc_x = curr_x + i_x
            i_loc_y = curr_y + i_y

            # map
            if i_loc_x < 0 or i_loc_x >= width:
                agent_obs[0, i_x + radius, i_y + radius] = 1
            elif i_loc_y < 0 or i_loc_y >= height:
                agent_obs[0, i_x + radius, i_y + radius] = 1
            else:
                agent_obs[0, i_x + radius, i_y + radius] = field[i_loc_x, i_loc_y]

            # agents
            for a_name, (a_x, a_y) in agents_loc.items():
                if a_name == agent_name:
                    continue
                if i_loc_x == a_x and i_loc_y == a_y:
                    agent_obs[1, i_x + radius, i_y + radius] = 1

            # target
            if [i_loc_x, i_loc_y] == target:
                agent_obs[2, i_x + radius, i_y + radius] = 1

    return agent_obs

