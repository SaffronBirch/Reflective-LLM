###################### Imports ######################
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass    

###################### Abstract Class ######################
class Provider(ABC):

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]]) -> List[str]:
        pass
    
    def generate_one(self, messages: List[Dict[str, str]]) -> str:
        return self.generate(messages)[0]
    
    def cleanup(self):
        pass