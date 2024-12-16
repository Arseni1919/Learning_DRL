
from globals import *

class AlgMetaClass(ABC):

    @abstractmethod
    def select_actions(self, obs_dict: dict, **kwargs):
        pass