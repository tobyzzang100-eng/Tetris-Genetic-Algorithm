import numpy as np
import random
from neural_network import Chromosome
from tetris_env import TetrisEnv

POPULATION_SIZE = 50       # 염색체(개체) 수
MAX_PIECES = 1000          # 한 게임당 최대 피스 수
MUTATION_RATE = 0.08       # 돌연변이 확률


class GeneticAlgorithm:
    def __init__(self):
        self.chromosomes = [Chromosome() for _ in range(POPULATION_SIZE)]
        self.generation = 0
        self.best_fitness = 0
        self.history = []  # 세대별 최고 fitness 기록

    def evaluate(self, chromosome):
        """염색체(신경망)로 테트리스 플레이 -> fitness 계산"""
        env = TetrisEnv()
        env.reset()
        total_score = 0

        for _ in range(MAX_PIECES):
            if env.game_over:
                break
            actions = env.get_possible_actions()
            if not actions:
                break

            # 각 action의 보드 특성을 신경망으로 평가
            best_action = None
            best_value = -float('inf')
            for action in actions:
                rot, col, dropped = action
                features = env.evaluate_board_after(dropped)
                # 특성 정규화
                features = features / np.array([200.0, 4.0, 100.0, 100.0, 20.0,
                                               20.0, 20.0, 20.0, 20.0, 20.0,
                                               20.0, 20.0, 20.0, 20.0, 20.0])
                value = chromosome.forward(features)
                if value > best_value:
                    best_value = value
                    best_action = action

            _, score, done = env.step(best_action)
            total_score = score
            if done:
                break

        # fitness = 점수 + 배치한 피스 수 (오래 살수록 좋음)
        fitness = total_score + env.pieces_placed * 2 + env.lines_cleared * 50
        return fitness

    def evaluate_all(self, callback=None):
        """전체 염색체 평가"""
        for i, chrom in enumerate(self.chromosomes):
            f = self.evaluate(chrom)
            chrom.set_fitness(f)
            if callback:
                callback(i, f)

        fitnesses = [c.fitness() for c in self.chromosomes]
        best = max(fitnesses)
        avg = np.mean(fitnesses)
        self.best_fitness = best
        self.history.append({'gen': self.generation, 'best': best, 'avg': avg})
        return best, avg

    # ─── 선택 ───────────────────────────────────────────────
    def elitist_preserve_selection(self):
        """엘리트 보존 선택 - 상위 2개"""
        sorted_chroms = sorted(self.chromosomes, key=lambda x: x.fitness(), reverse=True)
        return sorted_chroms[:2]

    def roulette_wheel_selection(self):
        """룰렛 휠 선택"""
        result = []
        fitnesses = [max(c.fitness(), 0.01) for c in self.chromosomes]
        fitness_sum = sum(fitnesses)
        for _ in range(2):
            pick = random.uniform(0, fitness_sum)
            current = 0
            for i, chrom in enumerate(self.chromosomes):
                current += fitnesses[i]
                if current >= pick:
                    result.append(chrom)
                    break
        return result if len(result) == 2 else self.chromosomes[:2]

    # ─── 교배 ───────────────────────────────────────────────
    def SBX(self, p1, p2, eta=100):
        """Simulated Binary Crossover"""
        rand = np.random.random(p1.shape)
        gamma = np.empty(p1.shape)
        gamma[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1.0 / (eta + 1))
        gamma[rand > 0.5] = (1.0 / (2.0 * (1.0 - rand[rand > 0.5]))) ** (1.0 / (eta + 1))
        c1 = 0.5 * ((1 + gamma) * p1 + (1 - gamma) * p2)
        c2 = 0.5 * ((1 - gamma) * p1 + (1 + gamma) * p2)
        return c1, c2

    def crossover(self, chrom1, chrom2):
        """교배 - 두 자식 생성"""
        child1 = Chromosome()
        child2 = Chromosome()
        child1.w1, child2.w1 = self.SBX(chrom1.w1, chrom2.w1)
        child1.b1, child2.b1 = self.SBX(chrom1.b1, chrom2.b1)
        child1.w2, child2.w2 = self.SBX(chrom1.w2, chrom2.w2)
        child1.b2, child2.b2 = self.SBX(chrom1.b2, chrom2.b2)
        return child1, child2

    # ─── 변이 ───────────────────────────────────────────────
    def static_mutation(self, data):
        """정적 돌연변이 (가우시안)"""
        mutation_mask = np.random.random(data.shape) < MUTATION_RATE
        gaussian_noise = np.random.normal(size=data.shape)
        data[mutation_mask] += gaussian_noise[mutation_mask]

    def mutate(self, chromosome):
        """염색체 변이"""
        self.static_mutation(chromosome.w1)
        self.static_mutation(chromosome.b1)
        self.static_mutation(chromosome.w2)
        self.static_mutation(chromosome.b2)

    # ─── 다음 세대 생성 ─────────────────────────────────────
    def next_generation(self):
        """새 세대 생성"""
        new_chromosomes = []

        # 1. 엘리트 2개 보존 (변이 없이)
        elites = self.elitist_preserve_selection()
        new_chromosomes.extend([e.clone() for e in elites])

        # 2. 나머지는 룰렛 휠 선택 + 교배 + 변이
        while len(new_chromosomes) < POPULATION_SIZE:
            parents = self.roulette_wheel_selection()
            if len(parents) < 2:
                parents = elites
            child1, child2 = self.crossover(parents[0], parents[1])
            self.mutate(child1)
            self.mutate(child2)
            new_chromosomes.append(child1)
            if len(new_chromosomes) < POPULATION_SIZE:
                new_chromosomes.append(child2)

        self.chromosomes = new_chromosomes[:POPULATION_SIZE]
        self.generation += 1

    def get_best(self):
        return max(self.chromosomes, key=lambda x: x.fitness())
