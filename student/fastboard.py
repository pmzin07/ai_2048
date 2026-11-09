from functools import lru_cache

MASK = (1 << 20) - 1
TRANSPOSE_MASKS = tuple(
    (delta, sum(31 << (5*(4*r+c)) for r in range(4) for c in range(4)
                if 15*(c-r) == delta))
    for delta in (-45, -30, -15, 0, 15, 30, 45)
)
T0, T1, T2, T3, T4, T5, T6 = (mask for _, mask in TRANSPOSE_MASKS)
POWER = tuple(float(e)**3.5 for e in range(32))
MONO = tuple(e**4 for e in range(32))

def encode(board):
    packed = 0
    for i, value in enumerate(board.flat):
        exponent = int(value).bit_length()-1 if value else 0
        if exponent > 31:
            raise OverflowError('tile exceeds 80-bit representation')
        packed |= exponent << (5*i)
    return packed

def transpose(board):
    return ((board & T0) >> 45 | (board & T1) >> 30 |
            (board & T2) >> 15 | (board & T3) | (board & T4) << 15 |
            (board & T5) << 30 | (board & T6) << 45)

def reverse(row):
    return ((row & 31) << 15 | (row & 992) << 5 |
            (row >> 5) & 992 | (row >> 15) & 31)

@lru_cache(maxsize=131072)
def row_left(row):
    tiles = [v for v in (row & 31, (row >> 5) & 31,
                         (row >> 10) & 31, row >> 15) if v]
    result = reward = index = position = 0
    while index < len(tiles):
        value = tiles[index]
        if index+1 < len(tiles) and value == tiles[index+1]:
            value += 1
            if value > 31:
                raise OverflowError('merged tile exceeds 80-bit representation')
            reward += 1 << value
            index += 1
        result |= value << position
        position += 5
        index += 1
    return result, reward

@lru_cache(maxsize=131072)
def row_right(row):
    result, reward = row_left(reverse(row))
    return reverse(result), reward

@lru_cache(maxsize=131072)
def moves(board):
    vertical = transpose(board)
    result = []
    for direction, source, function in ((0, vertical, row_left),
                                         (1, vertical, row_right),
                                         (2, board, row_left),
                                         (3, board, row_right)):
        a, ra = function(source & MASK)
        b, rb = function((source >> 20) & MASK)
        c, rc = function((source >> 40) & MASK)
        d, rd = function(source >> 60)
        child = a | b << 20 | c << 40 | d << 60
        if child == source:
            continue
        if direction < 2:
            child = transpose(child)
        result.append((direction, child, ra+rb+rc+rd))
    return tuple(result)

@lru_cache(maxsize=131072)
def row_value(row):
    line = (row & 31, (row >> 5) & 31, (row >> 10) & 31, row >> 15)
    increasing = decreasing = merges = 0
    previous = 0
    for value in line:
        if value:
            if previous == value:
                merges += 1
            previous = value
    for a, b in zip(line, line[1:]):
        difference = MONO[a]-MONO[b]
        if difference > 0:
            increasing += difference
        else:
            decreasing -= difference
    return (100000.0 + 270*line.count(0) + 700*merges
            - 47*min(increasing, decreasing) - 11*sum(POWER[v] for v in line))

@lru_cache(maxsize=65536)
def evaluate(board):
    other = transpose(board)
    return (row_value(board & MASK) + row_value((board >> 20) & MASK)
            + row_value((board >> 40) & MASK) + row_value(board >> 60)
            + row_value(other & MASK) + row_value((other >> 20) & MASK)
            + row_value((other >> 40) & MASK) + row_value(other >> 60))

@lru_cache(maxsize=131072)
def empty_shifts(board):
    return tuple(shift for shift in range(0, 80, 5)
                 if not ((board >> shift) & 31))

@lru_cache(maxsize=131072)
def board_risk(board):
    empty = 0
    maximum = 0
    for shift in range(0, 80, 5):
        exponent = (board >> shift) & 31
        if not exponent:
            empty += 1
        elif exponent > maximum:
            maximum = exponent
    corner_max = max(board & 31, (board >> 15) & 31,
                     (board >> 60) & 31, (board >> 75) & 31)
    crowded = (16 - empty) / 16.0
    trapped = 1.0 if corner_max < maximum else 0.0
    return crowded, trapped

def clear_caches():
    moves.cache_clear()
    row_left.cache_clear()
    row_right.cache_clear()
    row_value.cache_clear()
    evaluate.cache_clear()
    empty_shifts.cache_clear()
    board_risk.cache_clear()
