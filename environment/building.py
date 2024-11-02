import random
from typing import List
import pygame

from environment.elevator import Elevator
from environment.floor import Floor
from environment.passenger import Passenger
from environment.enums import Action, Direction

class Building(object):
    
    def __init__(
        self, 
        num_floors: int, 
        num_elevators: int, 
        floor_capacity: int, 
        elevator_capacity: int,
        spawn_prob: float = 0.1,
        max_group_size: int = 5,
        total_wait_penalty: float = 1.0,
        total_delivered_reward: float = 10.0
    ):
        
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.floor_capacity = floor_capacity
        self.elevator_capacity = elevator_capacity
        self.spawn_prob = spawn_prob
        self.max_group_size = max_group_size

        self.elevators = [Elevator(elevator_capacity, num_floors, idx=i) for i in range(num_elevators)]
        self.floors = [Floor(floor_capacity, idx=i) for i in range(num_floors)]
        
        self.total_wait_penalty = total_wait_penalty
        self.total_delivered_reward = total_delivered_reward
        
        self.total_passengers_delivered = 0
        self.total_passengers_remaining = 0
        
        # generate passengers to start with
        self.generate_passengers(spawn_prob=1, max_group_size=self.max_group_size)
        
    def reset(self):
        for elevator in self.elevators:
            elevator.reset()
        self.total_passengers_delivered = 0
        self.total_passengers_remaining = 0
        
        return self.get_state()
        
    def get_total_passengers_delivered(self):
        self.total_passengers_delivered = sum(e.total_passengers_delivered for e in self.elevators)
        return self.total_passengers_delivered
    
    def get_total_passengers_remaining(self):
        return sum(f.passenger_count() for f in self.floors)
    
    def generate_passengers(self, spawn_prob: float, max_group_size: int):
        for floor in range(self.num_floors):
            if random.random() < spawn_prob:
                group_size = random.randint(1, max_group_size)
                while group_size > 0 and self.floors[floor].passenger_count() < self.floor_capacity:
                    self.floors[floor].passengers.append(Passenger(floor, self.num_floors))
                    group_size -= 1
    
    def get_state(self):
        floor_state = [(f.idx, f.get_state()) for f in self.floors]
        elevator_state = [(e.idx, e.get_state(), e.current_floor, e.direction) for e in self.elevators]
        
        return floor_state, elevator_state
    
    def load_passengers(elevator: Elevator, floor: Floor):
        for passenger in floor.passengers:
            if len(elevator.passengers) < elevator.capacity and passenger.direction == elevator.direction:
                elevator.passengers.append(passenger)
                floor.passengers.remove(passenger)
                
    def unload_passengers(elevator: Elevator):
        for passenger in elevator.passengers:
            if passenger.destination_floor == elevator.current_floor:
                elevator.passengers.remove(passenger)
                elevator.total_passengers_delivered += 1
                elevator.passengers_delivered_this_step += 1
    
    def step(self, action: List[int]):
        for idx, elevator in enumerate(self.elevators):
            action = action[idx]
            elevator.passengers_delivered_this_step = 0
            if action == Action.UP:
                elevator.move_up()
            elif action == Action.DOWN:
                elevator.move_down()
            elif action == Action.LOAD_AND_UP:
                Building.unload_passengers(elevator)
                elevator.direction = Direction.UP
                Building.load_passengers(elevator, self.floors[elevator.current_floor])
            elif action == Action.LOAD_AND_DOWN:
                Building.unload_passengers(elevator)
                elevator.direction = Direction.DOWN
                Building.load_passengers(elevator, self.floors[elevator.current_floor])
            
        for floor in self.floors:
            for passenger in floor.passengers:
                passenger.total_wait_time += 1
                
        for elevator in self.elevators:
            for passenger in elevator.passengers:
                passenger.total_wait_time += 1
                
        done = self.get_total_passengers_remaining() == 0
        self.done = done
        return self.get_state(), self.get_reward(), done, {}
                
    def get_reward(self):
        cumulative_wait_time = sum(p.total_wait_time for f in self.floors for p in f.passengers)
        cumulative_wait_time += sum(p.total_wait_time for e in self.elevators for p in e.passengers)
        
        total_delivered_this_step = sum(e.passengers_delivered_this_step for e in self.elevators)
        
        reward = total_delivered_this_step * self.total_delivered_reward - cumulative_wait_time * self.total_wait_penalty
        return reward

    def print_building(self, step: int):
        separator = "=" * 55
        for idx in reversed(range(1, self.num_floors)):
            print(separator)
            print(f"= Floor #{idx:02d} =", end=' ')
            for e in self.elevators:
                if e.current_floor == idx:
                    print(f"  Lift #{e.idx}  ", end=' ')
                else:
                    print("           ", end=' ')
            print()
            
            print("=  Waiting  =", end=' ')
            for e in self.elevators:
                if e.current_floor == idx:
                    waiting = len(e.passengers)
                    print(f"    {waiting:02d}     ", end=' ')
                else:
                    print("           ", end=' ')
            print()
            
            waiting_passengers = self.floors[idx].passenger_count()
            print(f"=    {waiting_passengers:03d}    =")
        print(separator)
        
        # Ground Floor
        print(f"= Floor #00 =", end=' ')
        for e in self.elevators:
            if e.current_floor == 0:
                print(f"  Lift #{e.idx}  ", end=' ')
            else:
                print("           ", end=' ')
        print()
        
        print("=  Arrived  =", end=' ')
        for e in self.elevators:
            if e.current_floor == 0:
                arrived = len(e.passengers)
                print(f"    {arrived:02d}     ", end=' ')
            else:
                print("           ", end=' ')
        print()
        
        arrived_passengers = self.floors[0].passenger_count()
        print(f"=    {arrived_passengers:03d}    =")
        print(separator)
        print()
        
        people_to_move = self.get_total_passengers_remaining() - arrived_passengers
        total_people = self.get_total_passengers_remaining()
        print(f"People to move: {people_to_move} ")
        print(f"Total # of people: {total_people}")
        print(f"Step: {step}")
        print('State:', self.get_state())
        print('Now reward:', self.get_reward())
        
        # Go back to the top of the building to print again
        print("\033[F" * (self.num_floors + 2))
        
    def draw_building(self, screen, step: int):
        # Constants for visualization
        FLOOR_HEIGHT = 100
        BUILDING_WIDTH = 800
        ELEVATOR_WIDTH = 60
        ELEVATOR_HEIGHT = 80
        PASSENGER_SIZE = 10
        
        # Colors
        BLACK = (0, 0, 0)
        GRAY = (200, 200, 200)
        BLUE = (0, 0, 255)
        RED = (255, 0, 0)
        WHITE = (255, 255, 255)
        
        # Clear screen
        screen.fill(WHITE)
        
        # Draw floors
        for floor in range(self.num_floors):
            y = FLOOR_HEIGHT * (self.num_floors - floor - 1)
            # Draw floor line
            pygame.draw.line(screen, BLACK, (0, y), (BUILDING_WIDTH, y), 2)
            
            # Draw floor number
            font = pygame.font.Font(None, 36)
            floor_text = font.render(f"Floor {floor}", True, BLACK)
            screen.blit(floor_text, (10, y + 10))
            
            # Draw waiting passengers on floor
            waiting_passengers = self.floors[floor].passenger_count()
            waiting_text = font.render(f"Waiting: {waiting_passengers}", True, BLACK)
            screen.blit(waiting_text, (BUILDING_WIDTH - 150, y + 10))
            
            # Draw passengers as circles
            passenger_x = 200
            for _ in range(min(waiting_passengers, 10)):  # Show max 10 passengers
                pygame.draw.circle(screen, RED, (passenger_x, y + 50), PASSENGER_SIZE)
                passenger_x += PASSENGER_SIZE * 2
        
        # Draw elevators
        for elevator in self.elevators:
            x = 300 + elevator.idx * (ELEVATOR_WIDTH + 50)
            y = FLOOR_HEIGHT * (self.num_floors - elevator.current_floor - 1)
            
            # Draw elevator shaft
            pygame.draw.line(screen, GRAY, 
                            (x + ELEVATOR_WIDTH//2, 0), 
                            (x + ELEVATOR_WIDTH//2, FLOOR_HEIGHT * self.num_floors), 
                            2)
            
            # Draw elevator
            pygame.draw.rect(screen, BLUE, 
                            (x, y + 10, ELEVATOR_WIDTH, ELEVATOR_HEIGHT))
            
            # Draw elevator number and passengers
            elevator_text = font.render(f"#{elevator.idx}", True, WHITE)
            screen.blit(elevator_text, (x + 5, y + 15))
            
            passengers_text = font.render(f"{len(elevator.passengers)}", True, WHITE)
            screen.blit(passengers_text, (x + 5, y + 45))
        
        # Draw status information
        info_y = FLOOR_HEIGHT * self.num_floors + 10
        status_font = pygame.font.Font(None, 30)
        
        people_to_move = self.get_total_passengers_remaining()
        total_people = self.get_total_passengers_remaining()
        reward = self.get_reward()
        
        texts = [
            f"Step: {step}",
            f"People to move: {people_to_move}",
            f"Total people: {total_people}",
            f"Reward: {reward:.2f}"
        ]
        
        for i, text in enumerate(texts):
            text_surface = status_font.render(text, True, BLACK)
            screen.blit(text_surface, (10, info_y + i * 30))
        
        # Update display
        pygame.display.flip()
