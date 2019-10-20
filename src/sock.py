import logging
import sys
from collections import deque
from datetime import datetime

import numpy as np
import socketio
from src.board import Board
from src.bomb import Bomb
from src.node import Node, Position

import threading

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


class ItemType(object):
    EMPTY = 0
    STONE = 1
    WOOD = 2


class Commands(object):
    BOMB = "b"
    LEFT = "1"
    RIGHT = "2"
    UP = "3"
    DOWN = "4"


def begin_join_game():
    game_con_info = {"game_id": GAME_ID,
                     "player_id": MY_PLAYER_ID}
    sio.emit("join game", game_con_info)


def get_bomb_radius(board, bomb):
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
    return 0 <= row < rows and 0 <= col < cols


def bfs(board, src, dest):
    """
    Find shortest path to src to dest
    :param board:
    :param src:
    :param dest:
    :return:
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
    relative_woods_pos = []

    for wood_pos in woods_pos or []:
        for i in range(4):
            row = wood_pos[0] + ROW_NUM[i]
            col = wood_pos[1] + COL_NUM[i]

            row = 0 if row < 0 else row
            row = board.rows - 1 if row >= board.rows else row

            col = 0 if col < 0 else col
            col = board.cols - 1 if col >= board.cols else col

            if check_safe(board, Position(row, col)) and board.map[row][col] not in [ItemType.WOOD, ItemType.STONE]:
                relative_woods_pos.append((row, col))

    return relative_woods_pos


def find_positions(board, my_pos, items_type, not_condition=True):
    """
    Find nearest safe position
    :return:
    """
    limit = max(board.cols, board.rows)

    for d in range(1, limit):
        r1 = max(0, my_pos.row - d)
        r2 = min(my_pos.row + d, board.rows)
        c1 = max(0, my_pos.col - d)
        c2 = min(my_pos.col + d, board.cols)

        sub_map = board.map[r1:r2 + 1, c1:c2 + 1]

        item_cells_pos_sub = find_pos(sub_map, items_type, not_condition)
        item_cells_pos = [(r1 + e[0], c1 + e[1]) for e in item_cells_pos_sub]

        if len(item_cells_pos_sub) == 0:
            continue

        min_dist = sys.maxsize
        paths = None

        my_player = board.get_player(MY_PLAYER_ID)
        pos_player = Position(my_player.row, my_player.col)

        if ItemType.WOOD in items_type and not_condition:
            item_cells_pos = near_by_pos_wood(board, item_cells_pos)

        for cor in item_cells_pos:
            pos = Position(cor[0], cor[1])
            if check_safe(board, pos) and (pos.row != pos_player.row or pos.col != pos_player.col):
                des_pos = Position(cor[0], cor[1])
                distance, pth = bfs(board, pos_player, des_pos)
                if distance != -1 and distance < min_dist:
                    min_dist = distance
                    paths = pth

        return paths

    return None


def find_action(dest_pos, my_pos):
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


def bom_setup(board):
    """
    Thu dat bomb va chay
    :param board:
    :return:
    """
    my_player = board.get_player(MY_PLAYER_ID)
    my_pos = Position(my_player.row, my_player.col)

    is_near, pos_item = near_by_item(board, my_pos, ItemType.WOOD)

    if is_near:
        bomb = Bomb({
            'row': my_pos.row,
            'col': my_pos.col,
            'playerId': MY_PLAYER_ID,
            'remainTime': 2000
        })

        board.bombs.append(bomb)
        paths = find_positions(
            board=board,
            my_pos=my_pos,
            items_type=[ItemType.STONE, ItemType.WOOD],
            not_condition=False
        )
        if paths:
            # dat bomb va chay
            send_command(Commands.BOMB)

        board.bombs.remove(bomb)

    else:
        paths = find_positions(
            board=board,
            my_pos=my_pos,
            items_type=[ItemType.WOOD],
            not_condition=True
        )

        logger.info("PATHS WOOD: %s" % paths)

        if paths and len(paths) > 1:

            action = find_action(paths[1], my_pos)
            send_command(action)
        else:
            paths = find_positions(
                board=board,
                my_pos=my_pos,
                items_type=[ItemType.EMPTY],
                not_condition=True
            )

            logger.info("PATHS EMPTY: %s" % paths)

            if paths and len(paths) > 1:
                action = find_action(paths[1], my_pos)
                send_command(action)


def handle_command(board):
    my_player = board.get_player(MY_PLAYER_ID)
    my_pos = Position(my_player.row, my_player.col)

    # Luon luon check bomb
    if not check_safe(board, my_pos):
        paths = find_positions(
            board=board,
            my_pos=my_pos,
            items_type=[ItemType.WOOD, ItemType.STONE],
            not_condition=False
        )

        if paths and len(paths) > 1:
            action = find_action(paths[1], my_pos)
            send_command(action)

    else:
        # Tim vat pham canh minh
        if len(board.spoils) > 0:
            is_spoil = False

            for spoil in board.spoils:
                spoil_pos = Position(spoil.row, spoil.col)
                distance, paths = bfs(board, my_pos, spoil_pos)
                if 0 < distance <= 3:
                    if paths and len(paths) > 1:
                        action = find_action(paths[1], my_pos)
                        send_command(action)
                        is_spoil = True

            if not is_spoil:
                bom_setup(board)

        else:
            bom_setup(board)

    if board.tag_name == 'start-game':
        thread = threading.Thread(target=check_time)
        thread.start()
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
    sys.exit(0)


@sio.on('join game')
def join_game(data):
    print('Join game data: ', data)


@sio.on('ticktack player')
def ticktack_player(data):
    print('Received data: ', data)

    global TIME_START
    global BOARD

    new_board = Board(data)
    BOARD = new_board

    if not new_board.cols or not new_board.rows or not data.get('map_info'):
        logger.info("Ignore data =====> ")
        return

    if MY_PLAYER_ID == "player1-xxx-xxx-xxx":
        TIME_START = datetime.utcnow()
        handle_command(new_board)


def check_time():
    global TIME_START
    global BOARD

    current_time = datetime.utcnow()
    logger.info("Checking time ====>>> %s" % current_time.isoformat())

    delta = current_time - TIME_START
    if delta.seconds > 2:
        TIME_START = datetime.utcnow()
        handle_command(BOARD)

if __name__ == '__main__':
    if len(sys.argv) <= 3:
        print("Use command: python socket.py <url> <game_id> <player_id>")
        exit(0)

    URL, GAME_ID, MY_PLAYER_ID, ENEMY_PLAYER_ID = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    sio.connect(URL)
    sio.wait()
