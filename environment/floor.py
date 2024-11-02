class Floor(object):
    def __init__(self, capacity: int, idx: int):
        self.capacity = capacity
        self.passengers = []
        self.idx = idx
        
    def reset(self):
        self.passengers = []
        
    def passenger_count(self):
        return len(self.passengers)
    
    def get_state(self):
        # Convert list of passengers into [current_floor, destination] pairs
        return [[self.idx, p.destination_floor] for p in self.passengers] or [[-1, -1]]
