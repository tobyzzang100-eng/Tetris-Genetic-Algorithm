import pygame
import numpy as np
import sys
import os
import time
import threading
import copy

# 경로 추가
sys.path.insert(0, os.path.dirname(__file__))
from tetris_env import TetrisEnv, BOARD_WIDTH, BOARD_HEIGHT, TETROMINOES
from genetic_algorithm import GeneticAlgorithm, POPULATION_SIZE

# ─── 화면 설정 ───────────────────────────────────────────────
CELL = 28                        # 셀 크기 (픽셀)
PANEL_W = 280                    # 오른쪽 정보 패널 너비
SCREEN_W = BOARD_WIDTH * CELL + PANEL_W
SCREEN_H = BOARD_HEIGHT * CELL
FPS = 60

# 색상
BLACK   = (10, 10, 15)
GRAY    = (40, 40, 50)
WHITE   = (220, 220, 230)
CYAN    = (0, 220, 220)
YELLOW  = (240, 200, 0)
PURPLE  = (160, 0, 200)
GREEN   = (0, 200, 80)
RED     = (220, 40, 40)
ORANGE  = (240, 140, 0)
BLUE    = (30, 80, 220)
LIGHT_BLUE = (100, 180, 255)
DARK_GRAY = (25, 25, 35)
ACCENT  = (80, 200, 140)

PIECE_COLORS = {
    'I': CYAN, 'O': YELLOW, 'T': PURPLE,
    'S': GREEN, 'Z': RED, 'J': BLUE, 'L': ORANGE
}


class TetrisVisualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("🧬 Tetris AI - Genetic Algorithm + Neural Network")
        self.clock = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_md = pygame.font.SysFont("consolas", 14)
        self.font_sm = pygame.font.SysFont("consolas", 12)

        self.ga = GeneticAlgorithm()

        # 현재 시각화용 상태
        self.vis_env = None
        self.vis_chrom_idx = 0
        self.vis_fitness = 0
        self.current_gen_fitnesses = []
        self.eval_progress = 0
        self.training = False
        self.paused = False
        self.best_record = 0
        self.speed = 1  # 0=느림, 1=보통, 2=빠름

        # 그래프용
        self.fitness_history = []

    def draw_board(self, env, x_offset=0, y_offset=0):
        """테트리스 보드 그리기"""
        # 배경
        board_rect = pygame.Rect(x_offset, y_offset, BOARD_WIDTH * CELL, BOARD_HEIGHT * CELL)
        pygame.draw.rect(self.screen, DARK_GRAY, board_rect)

        # 그리드 선
        for r in range(BOARD_HEIGHT + 1):
            pygame.draw.line(self.screen, GRAY,
                             (x_offset, y_offset + r * CELL),
                             (x_offset + BOARD_WIDTH * CELL, y_offset + r * CELL), 1)
        for c in range(BOARD_WIDTH + 1):
            pygame.draw.line(self.screen, GRAY,
                             (x_offset + c * CELL, y_offset),
                             (x_offset + c * CELL, y_offset + BOARD_HEIGHT * CELL), 1)

        # 쌓인 블록
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                if env.board[r][c]:
                    rect = pygame.Rect(x_offset + c*CELL+1, y_offset + r*CELL+1, CELL-2, CELL-2)
                    pygame.draw.rect(self.screen, LIGHT_BLUE, rect)
                    pygame.draw.rect(self.screen, WHITE, rect, 1)

        # 현재 피스
        if not env.game_over and env.current_piece:
            piece = env.current_piece
            ptype = piece['type']
            color = PIECE_COLORS.get(ptype, WHITE)
            cells = env._get_cells(piece)
            for r, c in cells:
                if 0 <= r < BOARD_HEIGHT and 0 <= c < BOARD_WIDTH:
                    rect = pygame.Rect(x_offset + c*CELL+1, y_offset + r*CELL+1, CELL-2, CELL-2)
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, WHITE, rect, 1)

            # 고스트 피스 (드롭 위치 미리보기)
            ghost = env._hard_drop_sim(copy.deepcopy(piece))
            if ghost:
                for r, c in env._get_cells(ghost):
                    if 0 <= r < BOARD_HEIGHT and 0 <= c < BOARD_WIDTH:
                        rect = pygame.Rect(x_offset + c*CELL+1, y_offset + r*CELL+1, CELL-2, CELL-2)
                        ghost_color = tuple(max(0, v - 150) for v in color)
                        pygame.draw.rect(self.screen, ghost_color, rect)
                        pygame.draw.rect(self.screen, color, rect, 1)

    def draw_panel(self):
        """오른쪽 정보 패널"""
        px = BOARD_WIDTH * CELL + 10
        py = 10

        def text(s, x, y, color=WHITE, font=None):
            f = font or self.font_md
            surf = f.render(s, True, color)
            self.screen.blit(surf, (x, y))

        # 패널 배경
        panel_rect = pygame.Rect(BOARD_WIDTH * CELL, 0, PANEL_W, SCREEN_H)
        pygame.draw.rect(self.screen, (18, 18, 28), panel_rect)
        pygame.draw.line(self.screen, ACCENT, (BOARD_WIDTH * CELL, 0), (BOARD_WIDTH * CELL, SCREEN_H), 2)

        # 타이틀
        text("TETRIS AI", px, py, ACCENT, self.font_lg)
        text("Genetic + Neural Net", px, py+22, GRAY)
        py += 55

        # 세대 정보
        text(f"Generation : {self.ga.generation}", px, py, YELLOW)
        py += 22
        text(f"Population : {POPULATION_SIZE}", px, py, WHITE)
        py += 22
        text(f"Progress   : {self.eval_progress}/{POPULATION_SIZE}", px, py,
             GREEN if self.eval_progress == POPULATION_SIZE else WHITE)
        py += 30

        # 현재 개체
        text("── Current Individual ──", px, py, ACCENT)
        py += 20
        text(f"  Index   : #{self.vis_chrom_idx+1}", px, py, WHITE)
        py += 18
        text(f"  Fitness : {int(self.vis_fitness)}", px, py, CYAN)
        py += 18
        if self.vis_env:
            text(f"  Score   : {self.vis_env.score}", px, py, WHITE)
            py += 18
            text(f"  Pieces  : {self.vis_env.pieces_placed}", px, py, WHITE)
            py += 18
            text(f"  Lines   : {self.vis_env.lines_cleared}", px, py, WHITE)
            py += 18
        py += 10

        # 최고 기록
        text("── Best Record ──────────", px, py, ACCENT)
        py += 20
        text(f"  Best Fitness : {int(self.best_record)}", px, py, YELLOW)
        py += 18
        if self.ga.history:
            last = self.ga.history[-1]
            text(f"  Last Avg     : {int(last['avg'])}", px, py, WHITE)
            py += 18
        py += 10

        # 피트니스 그래프 (간단한 바 그래프)
        text("── Fitness Graph ────────", px, py, ACCENT)
        py += 20
        graph_h = 80
        graph_w = PANEL_W - 20
        graph_rect = pygame.Rect(px, py, graph_w, graph_h)
        pygame.draw.rect(self.screen, (25, 25, 40), graph_rect)
        pygame.draw.rect(self.screen, GRAY, graph_rect, 1)

        if len(self.ga.history) > 1:
            bests = [h['best'] for h in self.ga.history[-30:]]
            avgs = [h['avg'] for h in self.ga.history[-30:]]
            max_val = max(bests) if max(bests) > 0 else 1
            n = len(bests)
            bw = max(2, graph_w // max(n, 1))
            for i in range(n):
                # best (노란색)
                bh = int(bests[i] / max_val * (graph_h - 4))
                pygame.draw.rect(self.screen, YELLOW,
                    (px + i * bw, py + graph_h - bh - 2, max(1, bw-1), bh))
                # avg (청록색)
                ah = int(avgs[i] / max_val * (graph_h - 4))
                pygame.draw.rect(self.screen, CYAN,
                    (px + i * bw, py + graph_h - ah - 2, max(1, bw-1), 2))
        py += graph_h + 8
        text("  YELLOW=best  CYAN=avg", px, py, GRAY, self.font_sm)
        py += 20

        # 조작 키
        text("── Controls ─────────────", px, py, ACCENT)
        py += 18
        text("  SPACE : Pause/Resume", px, py, GRAY, self.font_sm)
        py += 15
        text("  S     : Speed toggle", px, py, GRAY, self.font_sm)
        py += 15
        text("  Q/ESC : Quit", px, py, GRAY, self.font_sm)
        py += 15

        speed_labels = ["SLOW", "NORMAL", "FAST"]
        text(f"  Speed : {speed_labels[self.speed]}", px, py, GREEN, self.font_sm)

    def run_visualized(self):
        """메인 루프: GA 학습을 시각화하며 실행"""
        print("🧬 Tetris GA 학습 시작!")
        print(f"   개체 수: {POPULATION_SIZE} | 최대 피스/게임: 500")
        print("   (Pygame 창을 보며 학습 진행 확인 가능)")
        print()

        running = True
        gen_thread = None
        gen_done = threading.Event()
        gen_done.set()  # 처음은 바로 시작

        speeds = [30, 120, 2000]  # 프레임당 처리 속도 (높을수록 빠름)

        while running:
            # ── 이벤트 처리 ──
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    if event.key == pygame.K_s:
                        self.speed = (self.speed + 1) % 3

            if self.paused:
                self.screen.fill(BLACK)
                self.draw_panel()
                pause_surf = self.font_lg.render("⏸  PAUSED  (SPACE to resume)", True, YELLOW)
                self.screen.blit(pause_surf, (10, SCREEN_H // 2))
                pygame.display.flip()
                self.clock.tick(FPS)
                continue

            # ── 한 세대 평가 ──
            if gen_done.is_set():
                # 세대 완료 처리 (평가가 한 번이라도 끝났으면)
                if self.eval_progress == POPULATION_SIZE:
                    best_f = max(c.fitness() for c in self.ga.chromosomes)
                    avg_f = np.mean([c.fitness() for c in self.ga.chromosomes])
                    self.best_record = max(self.best_record, best_f)
                    print(f"  세대 {self.ga.generation:4d} | Best: {int(best_f):6d} | Avg: {int(avg_f):6d} | All-time Best: {int(self.best_record):6d}")
                    self.ga.history.append({'gen': self.ga.generation, 'best': best_f, 'avg': avg_f})
                    self.ga.next_generation()

                gen_done.clear()
                self.eval_progress = 0

                # 백그라운드 스레드로 평가
                def eval_thread():
                    for i, chrom in enumerate(self.ga.chromosomes):
                        if not running:
                            break
                        f = self.ga.evaluate(chrom)
                        chrom.set_fitness(f)
                        self.eval_progress = i + 1
                    gen_done.set()

                gen_thread = threading.Thread(target=eval_thread, daemon=True)
                gen_thread.start()

            # ── 시각화: 현재 베스트 염색체로 한 스텝 플레이 ──
            # 간단한 시각화: 가장 높은 fitness 염색체의 현재 상태
            sorted_chroms = sorted(self.ga.chromosomes, key=lambda x: x.fitness(), reverse=True)
            best_chrom = sorted_chroms[0]

            # 새 환경에서 몇 스텝 시뮬레이션
            if self.vis_env is None or self.vis_env.game_over:
                self.vis_env = TetrisEnv()
                self.vis_env.reset()
                self.vis_chrom_idx = 0
                self.vis_fitness = best_chrom.fitness()

            # 현재 env에서 한 스텝
            if not self.vis_env.game_over:
                actions = self.vis_env.get_possible_actions()
                if actions:
                    best_val = -float('inf')
                    best_act = None
                    for action in actions:
                        rot, col, dropped = action
                        features = self.vis_env.evaluate_board_after(dropped)
                        features = features / np.array([200.0, 4.0, 100.0, 100.0, 20.0,
                                                        20.0, 20.0, 20.0, 20.0, 20.0,
                                                        20.0, 20.0, 20.0, 20.0, 20.0])
                        val = best_chrom.forward(features)
                        if val > best_val:
                            best_val = val
                            best_act = action
                    if best_act:
                        self.vis_env.step(best_act)

            # ── 화면 그리기 ──
            self.screen.fill(BLACK)
            self.draw_board(self.vis_env)
            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(speeds[self.speed])

        pygame.quit()
        print("\n학습 종료!")
        if self.ga.history:
            print(f"최종 세대: {self.ga.generation}")
            print(f"최고 Fitness: {int(self.best_record)}")


def main():
    visualizer = TetrisVisualizer()
    visualizer.run_visualized()


if __name__ == "__main__":
    main()
