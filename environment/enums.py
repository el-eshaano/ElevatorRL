from enum import IntEnum

class Direction(IntEnum):
    UP = 1
    STILL = 0
    DOWN = -1

class Action(IntEnum):
    LOAD_AND_UP = 2
    UP = 1
    DOWN = -1
    LOAD_AND_DOWN = -2
