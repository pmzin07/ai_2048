"""Deterministic probability-limited Expectimax (no wall-clock cutoff)."""
from student.fastboard import evaluate, moves, empty_shifts

class SearchStopped(Exception):
    pass

class Search:
    def __init__(self, limit, probability_cutoff=0.01, cache=None):
        self.limit = limit
        self.cutoff = probability_cutoff
        self.nodes = 0
        self.cache = {} if cache is None else cache

    def _key(self, board, depth, probability):
        # Exact probabilities almost never repeat. This bucket preserves the
        # important cutoff distinction while allowing transposition reuse.
        return board, depth, min(100, int(probability * 100))

    def player(self, board, depth, probability):
        self.nodes += 1
        if self.nodes > self.limit:
            raise SearchStopped
        key = self._key(board, depth, probability)
        if key in self.cache:
            return self.cache[key]
        if depth == 0 or probability < self.cutoff:
            if not empty_shifts(board) and not moves(board):
                return -1e12
            return evaluate(board)
        options = moves(board)
        if not options:
            return -1e12
        value = float('-inf')
        for _, child, _ in options:
            candidate = self.chance(child, depth-1, probability)
            if candidate > value:
                value = candidate
        self.cache[key] = value
        return value

    def chance(self, board, depth, probability):
        self.nodes += 1
        if self.nodes > self.limit:
            raise SearchStopped
        if depth == 0 or probability < self.cutoff:
            # Afterstate leaf: intentional approximation, before random spawn.
            return evaluate(board)
        shifts = empty_shifts(board)
        if not shifts:
            return self.player(board, depth, probability)
        p2, p4 = probability*.9/len(shifts), probability*.1/len(shifts)
        value = 0.0
        for shift in shifts:
            value += .9*self.player(board | 1 << shift, depth, p2)
            value += .1*self.player(board | 2 << shift, depth, p4)
        return value/len(shifts)

    def root(self, options, depth):
        best, best_value = options[0][0], float('-inf')
        for direction, child, _ in options:
            value = self.chance(child, depth, 1.0)
            if value > best_value:
                best, best_value = direction, value
        return best
