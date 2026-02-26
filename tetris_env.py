import numpy as np
import copy

# 테트리스 보드 크기
BOARD_WIDTH = 10
BOARD_HEIGHT = 20

# 테트로미노 정의 (7가지)
TETROMINOES = {
    'I': [[(0,0),(0,1),(0,2),(0,3)],
          [(0,0),(1,0),(2,0),(3,0)]],
    'O': [[(0,0),(0,1),(1,0),(1,1)]],
    'T': [[(0,0),(0,1),(0,2),(1,1)],
          [(0,0),(1,0),(2,0),(1,1)],
          [(0,1),(1,0),(1,1),(1,2)],
          [(0,0),(1,0),(2,0),(1,-1)]],
    'S': [[(0,1),(0,2),(1,0),(1,1)],
          [(0,0),(1,0),(1,1),(2,1)]],
    'Z': [[(0,0),(0,1),(1,1),(1,2)],
          [(0,1),(1,0),(1,1),(2,0)]],
    'J': [[(0,0),(1,0),(1,1),(1,2)],
          [(0,0),(0,1),(1,0),(2,0)],
          [(0,0),(0,1),(0,2),(1,2)],
          [(0,0),(1,0),(2,0),(2,1)]],
    'L': [[(0,2),(1,0),(1,1),(1,2)],
          [(0,0),(1,0),(2,0),(2,1)],
          [(0,0),(0,1),(0,2),(1,0)],
          [(0,0),(0,1),(1,1),(2,1)]],
}
TETROMINO_KEYS = list(TETROMINOES.keys())


class TetrisEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False
        self.current_piece = self._new_piece()
        self.next_piece = self._new_piece()
        return self.get_state()

    def _new_piece(self):
        key = TETROMINO_KEYS[np.random.randint(len(TETROMINO_KEYS))]
        return {'type': key, 'rotation': 0, 'x': 0, 'y': BOARD_WIDTH // 2 - 1}

    def _get_cells(self, piece):
        """피스의 실제 보드 좌표 반환"""
        shape = TETROMINOES[piece['type']][piece['rotation'] % len(TETROMINOES[piece['type']])]
        return [(piece['x'] + dr, piece['y'] + dc) for dr, dc in shape]

    def _is_valid(self, piece):
        for r, c in self._get_cells(piece):
            if r < 0 or r >= BOARD_HEIGHT or c < 0 or c >= BOARD_WIDTH:
                return False
            if self.board[r][c] != 0:
                return False
        return True

    def _place_piece(self, piece):
        for r, c in self._get_cells(piece):
            if r < 0:
                self.game_over = True
                return 0
            self.board[r][c] = 1
        return self._clear_lines()

    def _clear_lines(self):
        full_rows = [r for r in range(BOARD_HEIGHT) if all(self.board[r])]
        cleared = len(full_rows)
        if cleared > 0:
            self.board = np.delete(self.board, full_rows, axis=0)
            self.board = np.vstack([np.zeros((cleared, BOARD_WIDTH), dtype=int), self.board])
        self.lines_cleared += cleared
        return cleared

    def get_possible_actions(self):
        """모든 가능한 (회전, 열) 조합 반환"""
        actions = []
        piece_type = self.current_piece['type']
        num_rotations = len(TETROMINOES[piece_type])
        for rot in range(num_rotations):
            for col in range(-2, BOARD_WIDTH + 2):
                test_piece = {'type': piece_type, 'rotation': rot, 'x': 0, 'y': col}
                if self._is_valid(test_piece):
                    # 바닥까지 드롭
                    dropped = self._hard_drop_sim(test_piece)
                    if dropped is not None:
                        actions.append((rot, col, dropped))
        return actions

    def _hard_drop_sim(self, piece):
        """시뮬레이션용 하드드롭 (보드 변경 없음)"""
        p = copy.deepcopy(piece)
        while True:
            p['x'] += 1
            if not self._is_valid(p):
                p['x'] -= 1
                return p
        return None

    def evaluate_board_after(self, piece):
        """피스 배치 후 보드 특성 계산 (신경망 입력용)"""
        sim_board = self.board.copy()
        cells = self._get_cells(piece)
        for r, c in cells:
            if 0 <= r < BOARD_HEIGHT and 0 <= c < BOARD_WIDTH:
                sim_board[r][c] = 1

        # 줄 제거 시뮬레이션
        full_rows = [r for r in range(BOARD_HEIGHT) if all(sim_board[r])]
        lines = len(full_rows)

        return self._extract_features(sim_board, lines)

    def _extract_features(self, board, lines_cleared):
        """보드에서 특성 추출 (14개)"""
        heights = []
        for c in range(BOARD_WIDTH):
            col = board[:, c]
            filled = np.where(col > 0)[0]
            heights.append(BOARD_HEIGHT - filled[0] if len(filled) > 0 else 0)

        # 글로벌 특성 4개
        agg_height = sum(heights)
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(BOARD_WIDTH-1))
        holes = self._count_holes(board)
        max_height = max(heights)

        # 열별 높이 10개 (핵심 개선!)
        col_heights = list(np.array(heights, dtype=float))

        return np.array([agg_height, lines_cleared, holes, bumpiness, max_height] + col_heights, dtype=float)

    def _count_holes(self, board):
        holes = 0
        for c in range(BOARD_WIDTH):
            col = board[:, c]
            filled = np.where(col > 0)[0]
            if len(filled) > 0:
                top = filled[0]
                holes += np.sum(col[top:] == 0)
        return holes

    def step(self, action):
        """action: (rotation, col, dropped_piece)"""
        rot, col, dropped_piece = action
        lines = self._place_piece(dropped_piece)

        # 점수 계산
        line_scores = [0, 100, 300, 500, 800]
        self.score += line_scores[min(lines, 4)]
        self.pieces_placed += 1

        if not self.game_over:
            self.current_piece = self.next_piece
            self.next_piece = self._new_piece()
            # 새 피스도 유효하지 않으면 게임오버
            if not self._is_valid(self.current_piece):
                self.game_over = True

        return self.get_state(), self.score, self.game_over

    def get_state(self):
        return self.board.copy()

    def get_board_features(self):
        return self._extract_features(self.board, 0)
