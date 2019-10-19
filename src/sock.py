import socketio
import sys
import logging
import numpy as np
from collections import deque
from src.node import Node
from src.bomb import Bomb
from src.player import Player
from src.board import Board

logger = logging.getLogger(__name__)

"""
1. da 
2. go 
0. loi di
"""


sio = socketio.Client()
URL, GAME_ID, MY_PLAYER_ID, ENEMY_PLAYER_ID = '', '', '', ''

ROW_NUM = [-1, 0, 0, 1]
COL_NUM = [0, -1, 1, 0]

BOARD = None


def begin_join_game():
    game_con_info = {"game_id": GAME_ID,
                     "player_id": MY_PLAYER_ID}
    sio.emit("join game", game_con_info)


def get_bomb_radius(bomb):
    my_player = BOARD.get_player(MY_PLAYER_ID)
    enemy_player = BOARD.get_player(ENEMY_PLAYER_ID)

    if bomb.player_id == MY_PLAYER_ID:
        return 1 + my_player.power_stone
    else:
        return 1 + enemy_player.power_stone


def check_safe(bombs, row, col):
    for bomb in bombs:
        radius = get_bomb_radius(bomb)
        up = bomb.row + radius
        down = bomb.row - radius
        left = bomb.col - radius
        right = bomb.col + radius
        if (bomb.col == col and up <= row <= down) or \
                (bomb.row == row and left <= col <= right):
            return False
    return True


def is_valid(row, col):
    return 0 <= row < BOARD.rows and 0 <= col < BOARD.cols


def bfs(src, dest):
    visited = np.full((BOARD.rows, BOARD.cols), False, dtype=bool)
    visited[src.row][src.col] = True
    queue = deque()
    queue.append(src)

    mapx = BOARD.map
    while queue:
        node = queue.popleft()
        if node.row == dest.row and node.col == dest.col:
            return node.dist, node.get_path()
        for i in range(4):
            row = node.row + ROW_NUM[i]
            col = node.col + COL_NUM[i]
            if is_valid(row, col) and mapx[row][col] == 1 and not visited[row][col]:
                visited[row][col] = True
                queue.append(Node(row=row, col=col, dist=node.dist + 1, parent=node))

    return -1, None


def find_safe_position(mapx, row, col):
    """
    Find nearest safe position
    :return:
    """
    limit = max(BOARD.cols, BOARD.rows)
    for d in range(limit):
        r1 = max(0, row - d)
        r2 = max(row + d, BOARD.rows)
        c1 = max(0, col - d)
        c2 = max(col + d, BOARD.cols)

        sub_map = mapx[r1:r2+1, c1:c2+1]
        empty_cells = np.where(sub_map != [1, 2])
        empty_cells_pos_sub = list(zip(empty_cells[0], empty_cells[1]))

        empty_cells_pos = [(r1 + e[0], c1 + e[1]) for e in empty_cells_pos_sub]

        if len(empty_cells) == 0:
            continue

        min_dist = sys.maxsize
        path = None

        for cor in empty_cells_pos:
            # map from sub array index to matrix index base on position
            if check_safe(pos_check):
                distance, pth = bfs(my_pos, pos_check)
                if distance < min_dist:
                    min_dist = distance
                    path = pth

        return path

    return None


def handle_command(data):
    global BOARD
    BOARD = Board(data)

    if BOARD.tag_name == 'start-game':
        my_player = BOARD.get_player(MY_PLAYER_ID)
        if not my_player:
            return

        if not check_safe(BOARD.bombs, my_player.row, my_player.col):
            find_safe_position()
        else:
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
    handle_command(data)


if __name__ == '__main__':
    if len(sys.argv) <= 3:
        print("Use command: python socket.py <url> <game_id> <player_id>")
        exit(0)

    URL, GAME_ID, MY_PLAYER_ID, ENEMY_PLAYER_ID = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    sio.connect(URL)
    sio.wait()
