"""
game_state.py

This file contains a class representing an Crystal Rover game state. You should make use of this class in your solver.

COMP3702 Assignment 2 "CrystalRover" Support Code

Last updated by vp 09/09/2026
"""


class GameState:
    """
    Instance of an Crystal Rover game state. row and col represent the current player position. crystal_status is 1 for
    each collected crystal, and 0 for each remaining crystal.
    """

    def __init__(self, row, col, crystal_status):
        self.row = row
        self.col = col
        assert isinstance(crystal_status, tuple), '!!! crystal_status should be a tuple !!!'
        self.crystal_status = crystal_status

    def __eq__(self, other):
        if not isinstance(other, GameState):
            return False
        return self.row == other.row and self.col == other.col and self.crystal_status == other.crystal_status

    def __hash__(self):
        return hash((self.row, self.col, *self.crystal_status))

    def __repr__(self):
        return f'row: {self.row},\t\t col: {self.col},\t\t crystal status: {self.crystal_status}'

    def deepcopy(self):
        return GameState(self.row, self.col, self.crystal_status)

