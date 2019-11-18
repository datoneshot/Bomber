from collections import deque
from datetime import datetime
from random import sample

import logging
import sys
import numpy as np
import copy
import socketio
from src.board import Board
from src.bomb import Bomb
from src.node import Node, Position
import time
import threading


class ItemType(object):
    EMPTY = 0
    STONE = 1
    WOOD = 2


class CellType(object):
    EMPTY = 0
    STONE = 1
    WOOD = 2
    BOMB = 666
    ENEMY = 555


class Commands(object):
    BOMB = "b"
    LEFT = "1"
    RIGHT = "2"
    UP = "3"
    DOWN = "4"


log_format = '%(asctime)s (%(levelname)s) : [%(name)s],%(filename)s:%(lineno)d %(message)s'
logging.root.handlers = []
logging.basicConfig(level='INFO', format=log_format, datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

sio = socketio.Client()
URL, GAME_ID, MY_PLAYER_ID, ENEMY_PLAYER_ID = '', '', '', ''

ROW_NUM = [-1, 0, 0, 1]
COL_NUM = [0, -1, 1, 0]
COMMANDS = [Commands.UP, Commands.LEFT, Commands.RIGHT, Commands.DOWN]

TIME_START = datetime.utcnow()
BOARD = None

IS_WAIT_BOMB_EXPLOSIVE = False
IS_RUN_TO_AVOID_BOMB = False

IS_RUN_TO_SAFE_POS = False
SAFE_RUN_PATH = None
MY_PLAYER_POS_WHEN_RUNNING = None



def get_delay_time():
    global TIME_START
    delta = datetime.utcnow() - TIME_START
    miniseconds = delta.microseconds
    seconds = delta.seconds
    time_consume = seconds * 1.0 + (miniseconds * 1.0) / 1000000
    logger.info("================> TIME CONSUME: %s" % time_consume)
    return max(0.0, 5.0 - time_consume)


def waiting_bomb_explosive():
    global IS_WAIT_BOMB_EXPLOSIVE
    global BOARD
    global IS_RUN_TO_AVOID_BOMB
    logger.info("================> start waiting_bomb_explosive")
    # waiting for x seconds
    time.sleep(3.0)
    IS_WAIT_BOMB_EXPLOSIVE = False
    IS_RUN_TO_AVOID_BOMB = False
    logger.info("================> end waiting_bomb_explosive")
    handle_command(BOARD)


def start_waiting_bomb_explosive():
    thread = threading.Thread(target=waiting_bomb_explosive)
    thread.start()


def run_character_watch_dog():
    global TIME_START
    global BOARD
    global IS_RUN_TO_SAFE_POS
    while True:
        delta = datetime.utcnow() - TIME_START
        seconds = delta.seconds
        if seconds >= 10:
            TIME_START = datetime.utcnow()
            IS_RUN_TO_SAFE_POS = False
            logger.info("================> start run character watch dog")
            handle_command(BOARD)
        time.sleep(1.0)


def start_run_character_watch_dog():
    logger.info("================> start run character watch done")
    thread = threading.Thread(target=run_character_watch_dog)
    thread.start()


def run_character(paths):
    global IS_RUN_TO_SAFE_POS
    global MY_PLAYER_POS_WHEN_RUNNING
    global TIME_START
    global BOARD

    TIME_START = datetime.utcnow()

    if paths and len(paths) > 0:
        logger.info("====> Running path: %s" % paths)
        idx = 1
        my_post = Position(MY_PLAYER_POS_WHEN_RUNNING.row, MY_PLAYER_POS_WHEN_RUNNING.col)

        for next_position in paths:

            action = find_action(next_position, my_post)
            send_command(action)
            logger.info("=====> Step %s: my position: %s, next position: %s, Action: %s" % (
                idx, my_post, next_position, action))

            my_post = Position(next_position.row, next_position.col)

            if idx == len(paths):
                delay_time = get_delay_time()

                logger.info("==========> DELAY: %s <==============" % delay_time)
                time.sleep(delay_time)

                IS_RUN_TO_SAFE_POS = False
                MY_PLAYER_POS_WHEN_RUNNING = None
                handle_command(BOARD)

            idx += 1

            time.sleep(0.5)

    else:
        time.sleep(get_delay_time())
        IS_RUN_TO_SAFE_POS = False
        MY_PLAYER_POS_WHEN_RUNNING = None

        handle_command(BOARD)


def running(paths):
    run_character(paths)


def begin_join_game():
    game_con_info = {"game_id": GAME_ID,
                     "player_id": MY_PLAYER_ID}
    sio.emit("join game", game_con_info)


def get_bomb_radius(board, bomb):
    """
    Calculate bomb radius
    :param board: game board
    :param bomb: bomb info
    :return: bomb radius
    """
    my_player = board.get_player(MY_PLAYER_ID)
    enemy_player = board.get_player(ENEMY_PLAYER_ID)

    if bomb.player_id == MY_PLAYER_ID:
        return 1 + my_player.power_stone
    else:
        return 1 + enemy_player.power_stone


def check_safe(board, pos):
    for bomb in board.bombs or []:
        radius = get_bomb_radius(board, bomb)
        up = bomb.row - radius
        down = bomb.row + radius
        left = bomb.col - radius
        right = bomb.col + radius
        if (bomb.col == pos.col and up <= pos.row <= down) or \
                (bomb.row == pos.row and left <= pos.col <= right):
            return False

    return True


def is_valid(row, col, rows, cols):
    """
    Check row, col in board
    :param row: row to check
    :param col: col to check
    :param rows: board's number of rows
    :param cols: board's number of cols
    :return: True if in board
    """
    return 0 <= row < rows and 0 <= col < cols


def bfs(board, src, dest):
    """
    Find shortest path to src to dest
    :param board: game board
    :param src: source position
    :param dest: destination position
    :return: shortest path if has
    """
    visited = np.full((board.rows, board.cols), False, dtype=bool)
    visited[src.row][src.col] = True
    queue = deque()
    queue.append(Node(row=src.row, col=src.col))

    matrix = board.map
    while queue:
        node = queue.popleft()
        if node.row == dest.row and node.col == dest.col:
            directions, paths = node.get_path()
            return node.dist, directions, paths
        for i in range(4):
            row = node.row + ROW_NUM[i]
            col = node.col + COL_NUM[i]
            command = COMMANDS[i]
            if is_valid(row, col, board.rows, board.cols) and matrix[row][col] not in [1, 2] and not visited[row][col]:
                visited[row][col] = True
                queue.append(Node(row=row, col=col, dist=node.dist + 1, parent=node, command=command))

    return -1, None, None


def find_pos(matrix, types, is_in=True):
    """
    Find position of elements

    :param matrix: Array numpy
    :param types: Ex: [0, 3]
    :param is_in: bool
    :return:
    """
    res = np.in1d(matrix, types).reshape(matrix.shape)
    pos = np.where(res == is_in)

    return list(zip(pos[0], pos[1]))


def near_by_pos_wood(board, woods_pos):
    """
    Find empty cell near a wood cell
    :param board: game board
    :param woods_pos: wood position
    :return: list empty cell
    """
    relative_woods_pos = []

    for wood_pos in woods_pos or []:
        for i in range(4):
            row = wood_pos[0] + ROW_NUM[i]
            col = wood_pos[1] + COL_NUM[i]

            row = 0 if row < 0 else row
            row = board.rows - 1 if row >= board.rows else row

            col = 0 if col < 0 else col
            col = board.cols - 1 if col >= board.cols else col

            if not is_in_danger_area(board, col, row) and board.map[row][col] not in [ItemType.WOOD, ItemType.STONE]:
                relative_woods_pos.append((row, col))

    return relative_woods_pos


def find_positions(board, items_type, not_condition=True):
    """
    Find nearest safe position
    :return: paths if has
    """
    limit = max(board.cols, board.rows)
    my_player = board.get_player(MY_PLAYER_ID)
    enemy_player = board.get_player(ENEMY_PLAYER_ID)
    my_pos = Position(my_player.row, my_player.col)
    enemy_pos = Position(enemy_player.row, enemy_player.col)

    for d in range(1, limit):
        r1 = max(0, my_pos.row - d)
        r2 = min(my_pos.row + d, board.rows)
        c1 = max(0, my_pos.col - d)
        c2 = min(my_pos.col + d, board.cols)

        sub_map = board.map[r1:r2 + 1, c1:c2 + 1]

        item_cells_pos_sub = find_pos(sub_map, items_type, not_condition)
        item_cells_pos = [(r1 + e[0], c1 + e[1]) for e in item_cells_pos_sub]
        # Filter pos of my player and enemy player
        item_cells_pos = [item for item in item_cells_pos if item[0] != my_pos.row and item[1] != my_pos.col and
                          item[0] != enemy_pos.row and item[1] != enemy_pos.col]

        if len(item_cells_pos_sub) == 0:
            continue

        min_dist = sys.maxsize
        paths = None
        directions = None
        dest_pos = None

        if ItemType.WOOD in items_type and not_condition:
            item_cells_pos = near_by_pos_wood(board, item_cells_pos)

        for cor in item_cells_pos:
            if not is_in_danger_area(board=board, x=cor[1], y=cor[0]):
                des_pos = Position(cor[0], cor[1])
                distance, drcs, pth = bfs(board, my_pos, des_pos)
                if distance != -1 and distance < min_dist:
                    min_dist = distance
                    paths = pth
                    directions = drcs
                    dest_pos = des_pos

    return directions, dest_pos, paths


def find_action(dest_pos, my_pos):
    """
    Calculate direction for moving
    :param dest_pos: next position
    :param my_pos: current position
    :return: direction code
    """
    if my_pos.row - dest_pos.row < 0:
        return "4"  # down
    elif my_pos.row - dest_pos.row > 0:
        return "3"  # up
    elif my_pos.col - dest_pos.col < 0:
        return "2"  # right
    elif my_pos.col - dest_pos.col > 0:
        return "1"  # left

    return ""


def near_by_item(board, pos, item_code):
    """
    Check around pos whether an item with code is available or not
    :param board: game board
    :param pos: position to check
    :param item_code: code of item to check
    :return: (Bool, Position)
    """
    matrix = board.map
    up = max(0, pos.row - 1)
    down = min(board.rows, pos.row + 1)
    left = max(0, pos.col - 1)
    right = min(board.cols, pos.col + 1)

    if matrix[up][pos.col] == item_code:
        return True, Position(up, pos.col)

    if matrix[pos.row][left] == item_code:
        return True, Position(pos.row, left)

    if matrix[down][pos.col] == item_code:
        return True, Position(down, pos.col)

    if matrix[pos.row][right] == item_code:
        return True, Position(pos.row, right)

    return False, None


def is_near_enemy(board):
    """
    Check my player whether it's near enemy or not
    :param board: game board
    :return: True if near
    """
    my_player = board.get_player(MY_PLAYER_ID)
    enemy_player = board.get_player(ENEMY_PLAYER_ID)
    my_pos = Position(my_player.row, my_player.col)
    enemy_pos = Position(enemy_player.row, enemy_player.col)

    if (my_pos.row == enemy_pos.row) and ((my_pos.col - 1 == enemy_pos.col) or
                                          (my_pos.col + 1 == enemy_pos.col)):
        return True
    if ((my_pos.col == enemy_pos.col) and ((my_pos.row - 1 == enemy_pos.row) or
                                           my_pos.row + 1 == enemy_pos.row)):
        return True
    return False


def bom_setup(board):
    """
    Try setup bomb and run to safe position
    :param board: game board
    :return: None
    """
    global MY_PLAYER_POS_WHEN_RUNNING, IS_RUN_TO_SAFE_POS, IS_WAIT_BOMB_EXPLOSIVE
    global IS_RUN_TO_AVOID_BOMB
    my_player = board.get_player(MY_PLAYER_ID)
    my_pos = Position(my_player.row, my_player.col)

    bomb = Bomb({
        'row': my_pos.row,
        'col': my_pos.col,
        'playerId': MY_PLAYER_ID,
        'remainTime': 2000
    })

    # Copy map and bomb to process

    board.bombs.append(bomb)
    paths, dest_pos, _ = find_positions(
        board=board,
        items_type=[ItemType.STONE, ItemType.WOOD],
        not_condition=False
    )
    if paths and len(paths) > 1:
        paths.insert(0, "b")
        cmds = "".join(paths)
        logger.info("====> BOMBS COMMAND: %s" % cmds)
        IS_RUN_TO_SAFE_POS = True
        MY_PLAYER_POS_WHEN_RUNNING = dest_pos
        send_command(cmds)
        IS_RUN_TO_AVOID_BOMB = True
        # IS_WAIT_BOMB_EXPLOSIVE = True
        # start_waiting_bomb_explosive()

    board.bombs.remove(bomb)

    logger.info("=====> Setup bomb at %s" % my_pos)


def bom_enemy(board):
    """
    Try setup bomb at enemy position and run to safe position
    :param board: game board
    :return: None
    """
    global MY_PLAYER_POS_WHEN_RUNNING, IS_RUN_TO_SAFE_POS, IS_WAIT_BOMB_EXPLOSIVE
    global IS_RUN_TO_AVOID_BOMB
    my_player = board.get_player(MY_PLAYER_ID)
    my_pos = Position(my_player.row, my_player.col)
    enemy_player = board.get_player(ENEMY_PLAYER_ID)
    enemy_pos = Position(enemy_player.row, enemy_player.col)

    bomb = Bomb({
        'row': my_pos.row,
        'col': my_pos.col,
        'playerId': MY_PLAYER_ID,
        'remainTime': 2000
    })

    # Copy map and bomb to process
    copy_board = copy.deepcopy(board)
    copy_board.bombs.append(bomb)
    copy_board.map[enemy_pos.row][enemy_pos.col] = ItemType.STONE

    paths, dest_pos, _ = find_positions(
        board=copy_board,
        items_type=[ItemType.STONE, ItemType.WOOD],
        not_condition=False
    )
    if paths and len(paths) > 1:
        paths.insert(0, "b")
        cmds = "".join(paths)
        logger.info("====> BOMBS COMMAND: %s" % cmds)
        IS_RUN_TO_SAFE_POS = True
        MY_PLAYER_POS_WHEN_RUNNING = dest_pos
        send_command(cmds)
        IS_RUN_TO_AVOID_BOMB = True
        # IS_WAIT_BOMB_EXPLOSIVE = True
        # start_waiting_bomb_explosive()

    logger.info("=====> Setup bomb enemy at %s" % my_pos)


def find_nearest_spoils(board, my_pos):
    """
    Find nearest spoils
    :param board: game board
    :param my_pos: current position of my player
    :return: shortest path from my player's position to one of the spoils
    """
    if len(board.spoils) <= 0:
        return None, None, None

    min_distance = sys.maxsize
    min_directions = None
    min_paths = None
    dest_pos = None
    for spoil in board.spoils:
        spoil_pos = Position(spoil.row, spoil.col)
        if not is_in_danger_area(board=board, x=spoil.col, y=spoil.row):
            distance, directions, paths = bfs(board, my_pos, spoil_pos)
            if distance > 0 and paths and len(paths) > 1:
                if distance < min_distance:
                    min_distance = distance
                    min_paths = paths
                    min_directions = directions
                    dest_pos = spoil_pos
    logger.info("=====> Spoil paths at %s is: %s" % (dest_pos, min_paths))
    return min_directions, dest_pos, min_paths


def shortest_path_to_enemy(board):
    """
    Find shortest path from current my position to enemy position
    :param board: board game
    :return: shortest path if has
    """
    my_player = board.get_player(MY_PLAYER_ID)
    enemy_player = board.get_player(ENEMY_PLAYER_ID)
    my_pos = Position(my_player.row, my_player.col)
    enemy_pos = Position(enemy_player.row, enemy_player.col)

    up = max(0, enemy_pos.row - 1)
    down = min(board.rows, enemy_pos.row + 1)
    left = max(0, enemy_pos.col - 1)
    right = min(board.cols, enemy_pos.col + 1)

    up_pos = Position(up, enemy_pos.col)
    down_pos = Position(down, enemy_pos.col)
    left_pos = Position(enemy_pos.row, left)
    right_pos = Position(enemy_pos.row, right)

    directions = None
    position = None
    paths = None
    min_dist = -1

    for p in [up_pos, down_pos, left_pos, right_pos]:
        dist, drcs, pths = bfs(board, my_pos, p)
        if dist < min_dist:
            directions = drcs
            paths = paths
            position = p

    return directions, position, paths


def board_is_valid():
    global BOARD
    if not BOARD.cols or not BOARD.rows:
        return False
    return True


def handle_command(board):
    """
    Find next move
    :param board: game board
    :return: None
    """
    global IS_RUN_TO_SAFE_POS
    global SAFE_RUN_PATH
    global MY_PLAYER_POS_WHEN_RUNNING
    global TIME_START
    global IS_WAIT_BOMB_EXPLOSIVE
    global IS_RUN_TO_AVOID_BOMB

    my_player = board.get_player(MY_PLAYER_ID)
    my_pos = Position(my_player.row, my_player.col)

    logger.info("=====> HANDLE COMMAND")
    logger.info("=====> IS_RUN_TO_SAFE_POS: %s" % IS_RUN_TO_SAFE_POS)
    if IS_RUN_TO_SAFE_POS:
        if my_pos.row != MY_PLAYER_POS_WHEN_RUNNING.row or MY_PLAYER_POS_WHEN_RUNNING.col != my_pos.col:
            logger.info("=====> RUNNING TO SAFE POSITION")
            return
        else:
            IS_RUN_TO_SAFE_POS = False
            logger.info("=====> STAND TO SAFE POSITION")
            if IS_RUN_TO_AVOID_BOMB:
                IS_WAIT_BOMB_EXPLOSIVE = True
                start_waiting_bomb_explosive()
                return

    logger.info("=====> IS_WAIT_BOMB_EXPLOSIVE: %s" % IS_WAIT_BOMB_EXPLOSIVE)
    if IS_WAIT_BOMB_EXPLOSIVE:
        return

    logger.info("=====> START HANDLE COMMAND")

    # If current my position not safe then I must find nearest safe position
    # and run to there as fast as possible
    # I must wait for bomb explosive before execute next action
    if is_in_danger_area(board=board, x=my_player.col, y=my_player.row):
        logger.info("=============> IN POSITION DANGER")

        directions, dest_pos, _ = find_positions(
            board=board,
            items_type=[ItemType.WOOD, ItemType.STONE],
            not_condition=False
        )
        if directions and len(directions) > 1:
            cmd = "".join(directions[1:])
            IS_RUN_TO_SAFE_POS = True
            MY_PLAYER_POS_WHEN_RUNNING = dest_pos
            send_command(cmd)
            # IS_WAIT_BOMB_EXPLOSIVE = True
            # start_waiting_bomb_explosive()
            IS_RUN_TO_AVOID_BOMB = True
    else:
        # If near enemy then bomb it first
        if is_near_enemy(board):
            bom_enemy(board)  # Bomb enemy
        else:
            # Find shortest path from current position to one of the spoils
            # and run to it
            spoil_directions, dest_pos, spoil_paths = find_nearest_spoils(board, my_pos)

            if spoil_directions and len(spoil_directions) > 1:
                cmd = "".join(spoil_directions[1:])
                IS_RUN_TO_SAFE_POS = True
                MY_PLAYER_POS_WHEN_RUNNING = dest_pos
                IS_RUN_TO_AVOID_BOMB = False
                send_command(cmd)
            else:
                # If my position near wood then try setup bomb here
                # to destroy it
                is_near, pos_item = near_by_item(board, my_pos, ItemType.WOOD)
                if is_near:
                    bom_setup(board)
                else:
                    # If I didn't find neighbor wood then try find nearest wood
                    # and move to it
                    wood_directions, dest_pos, wood_paths = find_positions(
                        board=board,
                        items_type=[ItemType.WOOD],
                        not_condition=True
                    )
                    logger.info("PATHS WOOD: %s" % wood_paths)
                    if wood_directions and len(wood_directions) > 1:
                        cmd = "".join(wood_directions[1:])
                        IS_RUN_TO_SAFE_POS = True
                        IS_RUN_TO_AVOID_BOMB = False
                        MY_PLAYER_POS_WHEN_RUNNING = dest_pos
                        send_command(cmd)
                    else:
                        # If I didn't find any wood on board then
                        # I will find shortest path to enemy and bomb him
                        enemy_directions, dest_pos, enemy_paths = shortest_path_to_enemy(board)
                        if enemy_directions and len(enemy_directions) > 1:
                            cmd = "".join(enemy_directions[1:])
                            IS_RUN_TO_SAFE_POS = True
                            IS_RUN_TO_AVOID_BOMB = False
                            MY_PLAYER_POS_WHEN_RUNNING = dest_pos
                            send_command(cmd)
                        else:
                            pass  # Don't execute any action

    if board.tag_name == 'start-game':
        # start a thread to run character when don't have any event from server
        start_run_character_watch_dog()
        pass
    elif board.tag_name == 'player:start-moving':
        pass
    elif board.tag_name == 'player:stop-moving':
        pass
    elif board.tag_name == 'bomb:setup':
        pass
    elif board.tag_name == 'bomb:explosed':
        pass


def is_path_safe(board, paths):
    for path in paths:
        if is_in_danger_area(board, path.col, path.row):
            return False
    return True


def send_command(cmd):
    cmd_info = {"direction": cmd}
    sio.emit("drive player", cmd_info)


@sio.event
def connect():
    begin_join_game()


@sio.event
def disconnect():
    print("I'm disconnected!")
    sys.exit(0)


@sio.on('join game')
def join_game(data):
    print('Join game data: ', data)


@sio.on('ticktack player')
def ticktack_player(data):
    logger.info('Received data: %s', data)

    global BOARD
    global TIME_START
    global ENEMY_PLAYER_ID

    new_board = Board(data)
    BOARD = new_board
    TIME_START = datetime.utcnow()

    if not board_is_valid():
        logger.info("Ignore data =====> ")
        return

    # get enemy play id
    enemy_id = [x.id for x in new_board.players or [] if x.id != MY_PLAYER_ID]
    if not enemy_id:
        return

    ENEMY_PLAYER_ID = enemy_id[0]

    logger.info("=====> ENEMY_PLAYER_ID: %s", ENEMY_PLAYER_ID)

    if MY_PLAYER_ID == 'player1-xxx-xxx-xxx':
        handle_command(new_board)


def find_around_me(board, allowed, col, row):
    """
    Find all directions can move around my player
    :param board: game board
    :param allowed: callback to check allow move
    :param col: x position of my player
    :param row: y position of my player
    :return: directions set can move to
    """
    matrix = board.map
    h = board.rows
    w = board.cols

    direction = set()
    if row - 1 >= 0 and allowed(matrix, col, row - 1):
        direction.update({Commands.UP})
    if row + 1 < h and allowed(matrix, col, row + 1):
        direction.update({Commands.DOWN})
    if col - 1 >= 0 and allowed(matrix, col - 1, row):
        direction.update({Commands.LEFT})  # LEFT
    if col + 1 < w and allowed(matrix, col + 1, row):
        direction.update({Commands.RIGHT})  # RIGHT

    return direction


def is_in_danger_area(board, x, y):
    """
    Check my player is in danger area
    :param board: game board
    :param x: column to check
    :param y: row to check
    :return: True if in danger area
    """

    matrix = board.map
    bombs = board.bombs
    width = board.cols
    height = board.rows
    position = Position(y, x)

    for bomb in bombs or []:
        # If bomb position and checked position is same row and column
        if position.row == bomb.row and position.col == bomb.col:
            return True

        # If bomb position and checked position is not same row and column
        if position.col != bomb.col and position.row != bomb.row:
            continue

        radius = get_bomb_radius(board, bomb)
        # If checked position and bomb position is same row
        if bomb.row == position.row:
            # If checked position out range of row radius
            if position.col < max(0, bomb.col - radius) or position.col > min(width, bomb.col + radius):
                continue

            # Check
            is_left = True if position.col < bomb.col else False
            first_obstacles = None
            row = position.row
            for r in range(1, radius + 1):
                cc = max(0, bomb.col - r) if is_left else min(bomb.col + r, width)
                if (matrix[row][cc] == CellType.STONE or matrix[row][cc] == CellType.WOOD) and first_obstacles is None:
                    first_obstacles = cc
                    break

            if first_obstacles is not None:
                if is_left:
                    if position.col > first_obstacles:
                        return True
                else:
                    if position.col < first_obstacles:
                        return True
            else:
                return True

        # If checked position and bomb position is same column
        if bomb.col == position.col:
            # If checked position out range of col radius
            if position.row < max(0, bomb.row - radius) or position.row > min(height, bomb.row + radius):
                continue

            is_up = True if position.row < bomb.row else False
            first_obstacles = False
            col = position.col
            for r in range(1, radius + 1):
                rr = max(0, bomb.row - r) if is_up else min(height, bomb.row + r)
                if (matrix[rr][col] == CellType.STONE or matrix[rr][col] == CellType.WOOD) and first_obstacles is None:
                    first_obstacles = rr
                    break

            if first_obstacles is not None:
                if is_up:
                    if position.row > first_obstacles:
                        return True
                else:
                    if position.row < first_obstacles:
                        return True
            else:
                return True

    return False


if __name__ == '__main__':
    if len(sys.argv) <= 2:
        print("Use command: python socket.py <url> <game_id> <player_id>")
        exit(0)

    URL, GAME_ID, MY_PLAYER_ID = sys.argv[1], sys.argv[2], sys.argv[3]

    sio.connect(URL)
    sio.wait()
