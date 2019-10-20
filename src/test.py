#!/usr/bin/env python
# encoding: utf-8
"""
test

Copyright (c) 2019 __CGD Inc__. All rights reserved.
"""
from __future__ import absolute_import, unicode_literals
from src.node import Node, Position
import numpy as np
import sys
from collections import deque
from src.board import Board
import logging


class ItemType(object):
    EMPTY = 0
    STONE = 1
    WOOD = 2


log_format = '%(asctime)s (%(levelname)s) : [%(name)s],%(filename)s:%(lineno)d %(message)s'
logging.root.handlers = []
logging.basicConfig(level='INFO', format=log_format, datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)


data = {u'tag': u'player:start-moving', u'map_info': {u'map': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 2, 1, 0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 1, 0, 2, 0, 1], [1, 0, 0, 2, 0, 2, 2, 2, 0, 0, 2, 1, 0, 2, 0, 2, 0, 1, 2, 2, 0, 2, 2, 0, 2, 0, 0, 1], [1, 1, 0, 0, 2, 0, 2, 1, 2, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 1], [1, 2, 1, 2, 0, 0, 0, 2, 1, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2, 1], [1, 0, 2, 2, 2, 1, 0, 0, 2, 1, 2, 0, 0, 0, 0, 1, 0, 0, 1, 2, 2, 2, 1, 2, 1, 2, 0, 1], [1, 0, 0, 2, 1, 0, 2, 0, 0, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 0, 2, 0, 0, 1, 2, 0, 0, 1], [1, 0, 2, 0, 2, 0, 0, 1, 2, 0, 1, 2, 2, 2, 0, 0, 0, 2, 2, 2, 0, 2, 0, 0, 0, 2, 0, 1], [1, 0, 0, 0, 0, 2, 0, 1, 2, 0, 1, 1, 1, 0, 0, 2, 0, 0, 0, 2, 1, 0, 2, 0, 2, 1, 1, 1], [1, 0, 1, 1, 2, 0, 0, 1, 2, 0, 2, 2, 0, 2, 2, 1, 1, 1, 0, 0, 1, 2, 0, 2, 0, 0, 0, 1], [1, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 2, 1, 2, 0, 2, 0, 1, 0, 0, 2, 0, 2, 0, 1], [1, 0, 2, 2, 1, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 0, 2, 0, 2, 0, 2, 2, 2, 1, 2, 0, 2, 1], [1, 0, 2, 1, 2, 1, 2, 0, 0, 1, 2, 2, 1, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 2, 0, 1], [1, 2, 1, 0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2, 1], [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 2, 0, 1, 0, 2, 0, 0, 0, 1, 1], [1, 0, 0, 2, 2, 2, 0, 0, 2, 2, 1, 0, 2, 2, 2, 2, 1, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 1], [1, 0, 0, 2, 1, 2, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 2, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], u'spoils': [], u'players': [{u'soulStone': 1, u'spaceStone': 1, u'power': 1, u'powerStone': 0, u'delay': 2000, u'currentPosition': {u'col': 23, u'row': 13}, u'realityStone': 1, u'mindStone': 1, u'spawnBegin': {u'col': 23, u'row': 13}, u'timeStone': 0, u'speed': 100, u'id': u'player1-xxx-xxx-xxx'}, {u'soulStone': 0, u'spaceStone': 0, u'powerStone': 0, u'currentPosition': {u'col': 22, u'row': 3}, u'realityStone': 0, u'mindStone': 0, u'spawnBegin': {u'col': 22, u'row': 3}, u'timeStone': 0, u'id': u'player2-xxx-xxx-xxx'}], u'bombs': [], u'myId': u'player1-xxx-xxx-xxx', u'gameStatus': None, u'size': {u'rows': 18, u'cols': 28}}}

ROW_NUM = [-1, 0, 0, 1]
COL_NUM = [0, -1, 1, 0]
MY_PLAYER_ID = "player1-xxx-xxx-xxx"
ENEMY_PLAYER_ID = "player2-xxx-xxx-xxx"


def get_bomb_radius(board, bomb):
    my_player = board.get_player(MY_PLAYER_ID)
    enemy_player = board.get_player(ENEMY_PLAYER_ID)

    if bomb.player_id == MY_PLAYER_ID:
        return 1 + my_player.power_stone
    else:
        return 1 + enemy_player.power_stone


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

        empty_cells_pos_sub = find_pos(sub_map, items_type, not_condition)
        empty_cells_pos = [(r1 + e[0], c1 + e[1]) for e in empty_cells_pos_sub]

        if len(empty_cells_pos_sub) == 0:
            continue

        min_dist = sys.maxsize
        path = None

        my_player = board.get_player(MY_PLAYER_ID)
        pos_player = Position(my_player.row, my_player.col)

        for cor in empty_cells_pos:
            pos = Position(cor[0], cor[1])
            if check_safe(board, pos) and (pos.row != pos_player.row or pos.col != pos_player.col):
                des_pos = Position(cor[0], cor[1])
                distance, pth = bfs(board, pos_player, des_pos)
                if distance != -1 and distance < min_dist:
                    min_dist = distance
                    path = pth

        return path

    return None

board = Board(data)

logging.info(find_positions(
    board=board,
    my_pos=Position(13, 23),
    items_type=[ItemType.WOOD],
    not_condition=True
))
