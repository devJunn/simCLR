from dataclasses import dataclass


@dataclass
class Cfg:
    batch_size: int=3
    epochs: int=1000
    temperature: float=0.1
    lr: float=0.001
    log_interval: int=10
    
