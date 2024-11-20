from abc import abstractmethod
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
    def step(self, actions: Dict[str, Any]) -> Tuple[Any, Any, bool, bool, Dict]:
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
