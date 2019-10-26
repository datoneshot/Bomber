from timeloop import Timeloop
from collections import deque
from datetime import timedelta
from datetime import datetime
from random import sample

import logging
import sys
import numpy as np
import socketio
from src.board import Board
from src.bomb import Bomb
from src.node import Node, Position

log_format = '%(asctime)s (%(levelname)s) : [%(name)s],%(filename)s:%(lineno)d %(message)s'
logging.root.handlers = []
logging.basicConfig(level='INFO', format=log_format, datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

sio = socketio.Client()
URL, GAME_ID, MY_PLAYER_ID, ENEMY_PLAYER_ID = '', '', '', ''

ROW_NUM = [-1, 0, 0, 1]
COL_NUM = [0, -1, 1, 0]

TIME_START = datetime.utcnow()
BOARD = None

IS_WAIT_BOMB_EXPLOSIVE = False

RUNNING_TIME_LOOP = Timeloop()

IS_RUN_TO_SAFE_POS = False
SAFE_RUN_PATH = None
MY_PLAYER_POS_WHEN_RUNNING = None


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


@RUNNING_TIME_LOOP.job(interval=timedelta(seconds=0.15))
def run_character():
    global SAFE_RUN_PATH
    global IS_RUN_TO_SAFE_POS
    global MY_PLAYER_POS_WHEN_RUNNING

    if SAFE_RUN_PATH and len(SAFE_RUN_PATH) > 0:
        next_position = SAFE_RUN_PATH.pop(0)
        action = find_action(next_position, MY_PLAYER_POS_WHEN_RUNNING)
        send_command(action)
        MY_PLAYER_POS_WHEN_RUNNING = next_position
    else:
        IS_RUN_TO_SAFE_POS = False
        MY_PLAYER_POS_WHEN_RUNNING = None
        SAFE_RUN_PATH = None
        RUNNING_TIME_LOOP.stop()


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
            return node.dist, node.get_path()
        for i in range(4):
            row = node.row + ROW_NUM[i]
            col = node.col + COL_NUM[i]
            if is_valid(row, col, board.rows, board.cols) and matrix[row][col] not in [1, 2] and not visited[row][col]:
                visited[row][col] = True
                queue.append(Node(row=row, col=col, dist=node.dist + 1, parent=node))

    return -1, None


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

        if ItemType.WOOD in items_type and not_condition:
            item_cells_pos = near_by_pos_wood(board, item_cells_pos)

        for cor in item_cells_pos:
            if not is_in_danger_area(board=board, x=cor[1], y=cor[0]):
                des_pos = Position(cor[0], cor[1])
                distance, pth = bfs(board, my_pos, des_pos)
                if distance != -1 and distance < min_dist:
                    min_dist = distance
                    paths = pth
        if paths and len(paths) > 1:
            return paths

    return None


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
    paths = find_positions(
        board=board,
        items_type=[ItemType.STONE, ItemType.WOOD],
        not_condition=False
    )
    if paths:
        # setup bomb and run to safe position
        send_command(Commands.BOMB)

    board.bombs.remove(bomb)


def find_nearest_spoils(board, my_pos):
    """
    Find nearest spoils
    :param board: game board
    :param my_pos: current position of my player
    :return: shortest path from my player's position to one of the spoils
    """
    if len(board.spoils) <= 0:
        return None

    min_distance = sys.maxsize
    min_paths = None
    for spoil in board.spoils:
        spoil_pos = Position(spoil.row, spoil.col)
        if not is_in_danger_area(board=board, x=spoil.col, y=spoil.row):
            distance, paths = bfs(board, my_pos, spoil_pos)
            if distance > 0 and paths and len(paths) > 1:
                if distance < min_distance:
                    min_distance = distance
                    min_paths = paths

    return min_paths


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
    _, paths = bfs(board, my_pos, enemy_pos)
    return paths


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

    my_player = board.get_player(MY_PLAYER_ID)
    my_pos = Position(my_player.row, my_player.col)

    # If my player is running to safe position
    if IS_RUN_TO_SAFE_POS:
        return

    # If current my position not safe then I must find nearest safe position
    # and run to there as fast as possible
    # I must wait for bomb explosive before execute next action
    if is_in_danger_area(board=board, x=my_player.col, y=my_player.row):
        paths = find_positions(
            board=board,
            items_type=[ItemType.WOOD, ItemType.STONE],
            not_condition=False
        )
        if paths and len(paths) > 1:
            IS_RUN_TO_SAFE_POS = True
            paths.pop(0)
            SAFE_RUN_PATH = paths
            MY_PLAYER_POS_WHEN_RUNNING = my_pos
            RUNNING_TIME_LOOP.start()
    else:
        # If near enemy then bomb it first
        if is_near_enemy(board):
            bom_setup(board)  # Bomb enemy
        else:
            # Find shortest path from current position to one of the spoils
            # and run to it
            spoil_paths = find_nearest_spoils(board, my_pos)
            if spoil_paths and len(spoil_paths) > 1:
                action = find_action(spoil_paths[1], my_pos)
                send_command(action)
            else:
                # If my position near wood then try setup bomb here
                # to destroy it
                is_near, pos_item = near_by_item(board, my_pos, ItemType.WOOD)
                if is_near:
                    bom_setup(board)
                else:
                    # If I didn't find neighbor wood then try find nearest wood
                    # and move to it
                    wood_paths = find_positions(
                        board=board,
                        items_type=[ItemType.WOOD],
                        not_condition=True
                    )
                    logger.info("PATHS WOOD: %s" % wood_paths)
                    if wood_paths and len(wood_paths) > 1:
                        action = find_action(wood_paths[1], my_pos)
                        send_command(action)
                    else:
                        # If I didn't find any wood on board then
                        # I will find shortest path to enemy and bomb him
                        enemy_paths = shortest_path_to_enemy(board)
                        if enemy_paths and len(enemy_paths) > 1:
                            action = find_action(enemy_paths[1], my_pos)
                            send_command(action)
                        else:
                            pass  # Don't execute any action

    if board.tag_name == 'start-game':
        pass
    elif board.tag_name == 'player:start-moving':
        pass
    elif board.tag_name == 'player:stop-moving':
        pass
    elif board.tag_name == 'bomb:setup':
        pass
    elif board.tag_name == 'bomb:explosed':
        pass


def send_command(cmd):
    cmd_info = {"direction": cmd}
    sio.emit("drive player", cmd_info)


@sio.event
def connect():
    begin_join_game()


@sio.event
def disconnect():
    print("I'm disconnected!")
    RUNNING_TIME_LOOP.stop()
    sys.exit(0)


@sio.on('join game')
def join_game(data):
    print('Join game data: ', data)


@sio.on('ticktack player')
def ticktack_player(data):
    logger.info('Received data: ', data)

    global TIME_START
    global BOARD

    new_board = Board(data)
    BOARD = new_board

    if not board_is_valid():
        logger.info("Ignore data =====> ")
        return

    if MY_PLAYER_ID == "player1-xxx-xxx-xxx":
        handle_command(new_board)


def control_character(board):
    command = next_move(board=board)
    if command is not None:
        send_command(command)


"""
New logic
"""


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


def bfs_(board, matrix, x, y, stop, allowed, limit=False):
    """
    Find shortest path from position to any cell item
    :param board: game board
    :param matrix: map matrix
    :param x: col to check
    :param y: row to check
    :param stop:
    :param allowed:
    :param limit:
    :return: shortest path
    """
    ans = set()
    visited = np.full((board.rows, board.cols), False, dtype=bool)

    q = deque()
    init_dirs = find_around_me(board, allowed, x, y)

    if Commands.LEFT in init_dirs:
        q.append((x - 1, y, 1, Commands.LEFT))
    if Commands.RIGHT in init_dirs:
        q.append((x + 1, y, 1, Commands.RIGHT))
    if Commands.UP in init_dirs:
        q.append((x, y - 1, 1, Commands.UP))
    if Commands.DOWN in init_dirs:
        q.append((x, y + 1, 1, Commands.DOWN))

    visited[y][x] = True

    while q:
        curr = q.popleft()
        n_x = curr[0]
        n_y = curr[1]
        d = curr[2]
        i_dir = curr[3]

        if limit and d > limit:
            return limit - 1, ans

        if stop(matrix, n_x, n_y):
            ans.update({i_dir})
            limit = d + 1

        matrix[n_y][n_x] = True

        dirs = find_around_me(board, allowed, n_x, n_y)
        if Commands.LEFT in dirs:
            q.append((n_x - 1, n_y, d + 1, i_dir))
        if Commands.RIGHT in dirs:
            q.append((n_x + 1, n_y, d + 1, i_dir))
        if Commands.UP in dirs:
            q.append((n_x, n_y - 1, d + 1, i_dir))
        if Commands.DOWN in dirs:
            q.append((n_x, n_y + 1, d + 1, i_dir))

    return False, ans


def next_move(board):
    """
    Calculate action type after receive game board info
    :param board: game board
    :return: one of command:
        BOMB = "b"
        LEFT = "1"
        RIGHT = "2"
        UP = "3"
        DOWN = "4"
    """

    my_player = board.get_player(MY_PLAYER_ID)
    enemy_player = board.get_player(ENEMY_PLAYER_ID)
    bombs = board.bombs or []

    enemy_code = 555

    # copy board map matrix to change
    matrix = np.array(board.map, copy=True)

    # add bombs to map
    for bomb in bombs:
        x_bomb, y_bomb = bomb.col, bomb.row
        matrix[y_bomb][x_bomb] = 666

    # Look for nearby bombs
    if is_in_danger_area(board, my_player.col, my_player.row):
        # If in range, look for a safe place
        dist, path = bfs_(board, matrix, my_player.col, my_player.row,
                          lambda mat, x1, y1: not is_in_danger_area(board, x1, y1),
                          lambda mat, x2, y2: mat[y2][x2] == 0, limit=5)
        if path:
            direction = sample(path, 1)[0]
            return direction

    # find empty space around me
    dirs = find_around_me(board, lambda mat, x1, y1: mat[y1][x1] == 0, my_player.col, my_player.row)

    # add enemy to map
    matrix[enemy_player.row][enemy_player.col] = enemy_code

    # find nearest player and a safe root
    # higher priority - clear path
    dist, path = bfs_(board, matrix, my_player.col, my_player.row, lambda mat, x1, y1: mat[y1][x1] == enemy_code,
                      lambda mat, x1, y1: mat[y1][x1] in [0, enemy_code] and not is_in_danger_area(board, x1, y1))
    if dist and dist == 1:
        return Commands.BOMB
    path.intersection_update(dirs)
    if path:
        direction = sample(path, 1)[0]
        return direction

    # lower priority - blocked path
    dist, path = bfs_(board, matrix, my_player.col, my_player.row, lambda mat, x1, y1: mat[y1][x1] == 2,
                      lambda mat, x1, y1: mat[y1][x1] in [0, 2] and not is_in_danger_area(board, x1, y1))
    path.intersection_update(dirs)
    if path:
        direction = sample(path, 1)[0]
        return direction

    # way is blocked- drop a bomb only if you have somewhere to run
    dist, path = bfs_(board, matrix, my_player.col, my_player.row,
                      lambda mat, x1, y1: not is_in_danger_area(board, x1, y1),
                      lambda mat, x1, y1: mat[y1][x1] in [0, enemy_code] and not is_in_danger_area(board, x1, y1),
                      limit=5)
    if path:
        return Commands.BOMB

    # nothing to else do
    return None


if __name__ == '__main__':
    if len(sys.argv) <= 3:
        print("Use command: python socket.py <url> <game_id> <player_id>")
        exit(0)

    URL, GAME_ID, MY_PLAYER_ID, ENEMY_PLAYER_ID = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    sio.connect(URL)
    sio.wait()
