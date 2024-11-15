from globals import *


class MetaEnv:

    def __init__(self):
        pass

    def reset(self) -> Tuple[Any, Dict]:
        pass

    def sample_action(self) -> Any:
        pass

    def step(self, action) -> Tuple[Any, Any, bool, bool, Dict]:
        pass

    def close(self) -> None:
        pass
