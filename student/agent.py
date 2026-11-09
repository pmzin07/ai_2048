"""No-tablebase agent: deterministic bounded Expectimax on packed boards."""
from game2048.types import Move
from student.fastboard import encode, moves, evaluate, board_risk, clear_caches
from student.fastsearch import Search, SearchStopped

def _time_factor(obs):
    """Coarse tiers so normal games stay reproducible; only long games brake."""
    remaining, total = obs.time_remaining_ms, obs.total_time_limit_ms
    if remaining is None or not total:
        return 1.0
    if remaining < min(1500.0, 0.03*total):   # emergency: greedy, no search
        return 0.0
    frac = remaining/total
    if frac > 0.30:
        return 1.0
    if frac > 0.15:
        return 0.6
    if frac > 0.07:
        return 0.35
    return 0.15

class Agent:
    name = 'Thu Do Hung Yen'
    BASE_NODES = 9000          # chỉnh tại đây sau khi đo
    CACHE_MAX = 200_000

    def reset(self, seed=None):
        clear_caches()
        self.search_cache = {}
        self.last_nodes = self.last_completed_depth = 0

    def choose_move(self, obs, time_limit_ms):
        self.last_nodes = self.last_completed_depth = 0
        if not obs.legal_moves:
            raise ValueError('choose_move called on terminal board')
        if len(obs.legal_moves) == 1:
            return obs.legal_moves[0]
        board = encode(obs.board)
        options = [item for item in moves(board) if Move(item[0]) in obs.legal_moves]
        options.sort(key=lambda item: evaluate(item[1]), reverse=True)
        best = options[0][0]

        factor = _time_factor(obs)
        if factor == 0.0:
            return Move(best)

        if len(self.search_cache) > self.CACHE_MAX:
            self.search_cache.clear()

        scale = min(1.0, obs.total_time_limit_ms/60000) if obs.total_time_limit_ms else min(1.0, time_limit_ms/50)
        crowded, trapped = board_risk(board)
        danger_factor = 0.72 + 0.65*crowded + 0.35*trapped*crowded
        limit = max(64, int(self.BASE_NODES*scale*danger_factor*factor
                            /(1+obs.move_count/2200)**2))
        search = Search(limit, cache=self.search_cache)
        max_depth = 8
        for depth in range(0, max_depth):
            try:
                candidate = search.root(options, depth)
            except SearchStopped:
                break
            best = candidate
            self.last_completed_depth = depth+1
        self.last_nodes = search.nodes
        return Move(best)
