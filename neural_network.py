import numpy as np

# 입력: 보드 특성 14개 (총높이, 줄제거, 구멍, 울퉁불퉁함, 최대높이 + 열별높이 10개)
# 출력: 점수 (1개) -> 가장 좋은 action 선택
INPUT_SIZE = 15
HIDDEN_SIZE = 32
OUTPUT_SIZE = 1


class Chromosome:
    """유전 알고리즘의 염색체 = 신경망 가중치"""
    def __init__(self):
        # Xavier 초기화
        self.w1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * np.sqrt(2.0 / INPUT_SIZE)
        self.b1 = np.zeros(HIDDEN_SIZE)
        self.w2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * np.sqrt(2.0 / HIDDEN_SIZE)
        self.b2 = np.zeros(OUTPUT_SIZE)
        self._fitness = 0

    def forward(self, x):
        """순전파 - 보드 특성 입력 -> 점수 출력"""
        relu = lambda X: np.maximum(0, X)
        l1 = relu(np.dot(x, self.w1) + self.b1)
        output = np.dot(l1, self.w2) + self.b2
        return output[0]

    def fitness(self):
        return self._fitness

    def set_fitness(self, val):
        self._fitness = val

    def clone(self):
        c = Chromosome()
        c.w1 = self.w1.copy()
        c.b1 = self.b1.copy()
        c.w2 = self.w2.copy()
        c.b2 = self.b2.copy()
        c._fitness = self._fitness
        return c
