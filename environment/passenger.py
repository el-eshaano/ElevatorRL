from dataclasses import dataclass
import random

from environment.enums import Direction

@dataclass
class Passenger(object):
    source_floor: int
    destination_floor: int
    direction: int
    total_wait_time: int = 0
    
    def __init__(self, current_floor: int, max_floor: int):
        self.source_floor = current_floor
        self.destination_floor = random.choice(
            [i for i in range(current_floor)] + 
            [i for i in range(current_floor+1, max_floor)]
        )
        self.direction = Direction.UP if self.destination_floor > self.source_floor else Direction.DOWN
        
    def __str__(self):
        return f"Passenger({self.source_floor} -> {self.destination_floor})"
