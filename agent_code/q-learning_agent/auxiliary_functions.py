import numpy as np

def bomb_map(self, game_state: dict)->np.ndarray:
    
    """
    Generates a bomb map based on the given game state, and that behind a wall the agent is protected from the bomb.
    Parameters:
    - game_state (dict): The current game state containing the field and bomb information.
    Returns:
    - bomb_map: A 2D array representing the bomb map, where each element represents the timer value of a bomb.
    The bomb map is generated by iterating over the bombs in the game state and updating the timer values of the affected tiles.
    The size of the bomb map is adjusted to match the size of the field in the game state.
    The initial timer value for each tile is set to 5, and explodes when it reaches 1.
    The timer value of a tile is updated if a bomb with a shorter timer is found.
    The bomb map is returned as a 2D array.
    """
    field = game_state['field'].copy()

    #adjust size of the bomb map to the field size
    bomb_map = np.full((self.rows,self.cols), -1)
    bombs = game_state['bombs'].copy()
    
    for bomb in bombs:
        bomb_center_x, bomb_center_y = bomb[0][0], bomb[0][1]
        bomb_timer = bomb[1] + 1
        
        #affected tiles classic scenario
        #affected_tiles_up = [(bomb_center_x, bomb_center_y-1), (bomb_center_x, bomb_center_y-2), (bomb_center_x, bomb_center_y-3)]
        #affected_tiles_down = [(bomb_center_x, bomb_center_y-1), (bomb_center_x, bomb_center_y-2), (bomb_center_x, bomb_center_y-3)]
        #affected_tiles_left = [(bomb_center_x-1, bomb_center_y), (bomb_center_x-2, bomb_center_y), (bomb_center_x-3, bomb_center_y)]
        #affected_tiles_right = [(bomb_center_x+1, bomb_center_y), (bomb_center_x+2, bomb_center_y), (bomb_center_x+3, bomb_center_y)]
        
        #affected tiles 1-goal scenario
        #We consider in affected_tiles_up the bomb center and in the rest not.
        affected_tiles_up = [(bomb_center_x, bomb_center_y),(bomb_center_x, bomb_center_y-1)]
        affected_tiles_down = [(bomb_center_x, bomb_center_y+1)]
        affected_tiles_left = [(bomb_center_x-1, bomb_center_y)]
        affected_tiles_right = [(bomb_center_x+1, bomb_center_y)]

        for tile in affected_tiles_up:
            if field[tile] == -1:
                break
            bomb_map[tile] = bomb_timer 
            # bomb_map[tile] = bomb_timer if (timer_tile > bomb_timer) else timer_tile

        for tile in affected_tiles_down:
            if field[tile] == -1:
                break
            timer_tile = bomb_map[tile]
            bomb_map[tile] = bomb_timer #if (timer_tile > bomb_timer) else timer_tile

        for tile in affected_tiles_left:
            if field[tile] == -1:
                break
            timer_tile = bomb_map[tile]
            bomb_map[tile] = bomb_timer #if (timer_tile > bomb_timer) else timer_tile

        for tile in affected_tiles_right:
            if field[tile] == -1:
                break
            timer_tile = bomb_map[tile]
            bomb_map[tile] = bomb_timer #if (timer_tile > bomb_timer) else timer_tile

    explosion_map = game_state['explosion_map'].copy()

    bomb_map[explosion_map == 1] = 1
    return bomb_map


def build_field(self, game_state: dict):
    """
    Builds the game field based on the given game state.
    Parameters:
    - self: The instance of the class.
    - game_state (dict): The dictionary containing the game state.
    Returns:
    - field (numpy.ndarray): The built game field.
    The function builds the game field by adjusting the scale of crates to 1, coins to 3, 
    bombs to 2, and free spaces to 0.
    The function returns the built game field.
    """
    field = game_state['field'].copy()
    
    for (coords, value) in game_state['bombs'].copy():
        field[coords] = 2  

    coin_map = game_state['coins'].copy()
    for coin in coin_map:
        field[coin] = 3

    oppponents = game_state['others'].copy()
    for opp in oppponents:
        field[opp[3]] = 4
    
    return field


def agent_vision(self, field: np.ndarray, player_pos: tuple, radius: int = 3):
    """
    Generates the agent's vision based on the field and player position.
    Parameters:
    - self: The instance of the class.
    - field (numpy.ndarray): The game field.
    - player_pos (tuple): The player position.
    - radius (int): The vision radius around the player (default is 3).
    
    Returns:
    - vision (numpy.ndarray): The agent's vision.
    
    The function generates the agent's vision by creating a (2*radius+1)x(2*radius+1) grid centered 
    around the player position. If the player is near the boundary, vision is clipped accordingly.
    """
    vision = np.zeros((2*radius+1, 2*radius+1))    

    left = player_pos[0] - radius
    right = player_pos[0] + radius 
    up = player_pos[1] - radius
    down = player_pos[1] + radius

    for i in range(left, right+1):
        for j in range(up, down+1):
            if i < 0 or i >= field.shape[0] or j < 0 or j >= field.shape[1]:
                vision[i-left, j-up] = -1
            else:
                vision[i-left, j-up] = field[i, j]

    return vision
