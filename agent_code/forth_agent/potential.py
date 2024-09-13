
def is_between_placing_and_collection(gamestate, got_killed: bool):
    # check if a bomb or coin is in a 4 radius of the player,
    # to stretch the reward from collecting the coin to placing the bomb:
    if got_killed:
        return True
    is_between = False
    xp, yp = gamestate["self"][3]
    for xy, timer in gamestate["bombs"]:
        x, y = xy
        if x in range(xp - 4, xp + 5):
            if y in range(yp - 4, yp + 5):
                return True
    for x, y in gamestate["coins"]:
        if x in range(xp - 4, xp + 5):
            if y in range(yp - 4, yp + 5):
                return True
    return False


def is_safe(game_state, got_killed: bool):
    """
    inputs:
    player_cords: tuple(int,int) being the x and y coords
    bobs_ticking: the list of bombs
    field: the field with only walls, crates and air

    returns:
    bool: True if player is currently not in any blast radius.
    Else false
    """
    if got_killed:
        return True
    player_cords = game_state["self"][3]
    bombs_ticking = game_state["bombs"]
    field = game_state["field"]
    xp, yp = player_cords
    for xy, _timer in bombs_ticking:
        xb, yb = xy
        if xp == xb:
            if yp in range(yb - 3, yb + 4):
                if (yb - yp) == 2:
                    if field[(xp, yb - 1)] != 0:
                        continue
                elif (yp - yb) == 2:
                    if field[(xp, yb + 1)] != 0:
                        continue
                return False
        if yp == yb:
            if xp in range(xb - 3, xb + 3):
                if (xb - xp) == 2:
                    if field[(xp + 1, yb)] != 0:
                        continue
                elif (xp - xb) == 2:
                    if field[(xp - 1, yb)] != 0:
                        continue
                return False
    return True


def distance(xyp, bombs):
    xp, yp = xyp
    distance = 14 + 14  # the distance between the two adjacent corners i think
    for xyb, timer in bombs:
        xb, yb = xyb
        distance = min(distance, abs(xb - xp) + abs(yb - yp))
    return distance


def distance_coins(xyp, coins):
    xp, yp = xyp
    distance = 14 + 14  # the distance between the two adjacent corners i think
    for (
        x,
        y,
    ) in coins:
        distance = min(distance, abs(x - xp) + abs(y - yp))
    return distance


def bomb_place_incentive(state, got_killed):
    if got_killed:
        return 4
    xp, yp = state["self"][3]
    if state["self"][2] == True:  # a bomb can be placed
        # check if a coin in near:
        dist_coin = distance_coins(state["self"][3], state["coins"])
        if dist_coin <= 5:
            return 5 - dist_coin

        else:
            # check if crate is next to the agent:
            # yes -> punish for not placing bomb,
            # no -> dont, encourage walking
            if (
                state["field"][(xp + 1, yp)] == 1
                or state["field"][(xp, yp + 1)] == 1
                or state["field"][(xp - 1, yp)] == 1
                or state["field"][(xp, yp - 1)] == 1
            ):
                # yes, a crate is next to
                return -2
            else:
                return 0
    # a bomb is currently ticking
    save = is_safe(state, got_killed)
    if save:
        return 4
    # else:
    dist = distance(state["self"][3], state["bombs"])
    return min(dist, 4)  # 4 should also be the largest distance possible, because we just placed a bomb


def potential_function(state, got_killed):
    # got killed is implicitly part of the state, but gets passed
    # anyway, so we do not need to compute it
    reward = 0
    reward = bomb_place_incentive(state, got_killed)
    return reward

