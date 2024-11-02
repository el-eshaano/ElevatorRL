from typing import List

from environment.passenger import Passenger
from environment.enums import Direction, Action

class Elevator(object):
    
    def __init__(self, capacity: int, num_floors: int, idx: int):
        self.capacity = capacity
        self.num_floors = num_floors
        self.idx = idx
        self.current_floor = 0
        self.passengers = []
        self.direction = Direction.UP
        
        self.passengers_delivered_this_step = 0
        
        self.total_passengers_delivered = 0 # For reward calculation
        
    def move_up(self):
        if self.current_floor < self.num_floors-1:
            self.current_floor += 1
        self.direction = Direction.UP
            
    def move_down(self):
        if self.current_floor > 0:
            self.current_floor -= 1
        self.direction = Direction.DOWN
        
    def reset(self):
        self.current_floor = 0
        self.passengers = []
        self.total_passengers_delivered = 0
        
    

    def get_passenger_info(self):
        return [p.destination_floor for p in self.passengers]

    def get_state(self):
        passenger_destinations = self.get_passenger_info()
        return passenger_destinations if passenger_destinations else [-1]
    
    def passenger_count(self):
        return len(self.passengers)
