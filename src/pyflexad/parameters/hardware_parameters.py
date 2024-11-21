from abc import ABCMeta
from dataclasses import dataclass


@dataclass
class HardwareParameters(metaclass=ABCMeta):
    """
    name:str
        Name of the hardware model
    """
    name: str
