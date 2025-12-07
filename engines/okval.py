#!/usr/bin/env python3
"""
Okval - a simple UCI chess engine in Python
Author: Classic
This engine uses python-chess for move generation and implements a negamax
alpha-beta search with iterative deepening, quiescence search, simple
evaluation (material + piece-square tables + mobility), and a transposition
table. It supports basic UCI commands and a "SkillLevel" option to tune
playing strength (approx Lichess ~1700 for SkillLevel 12-14 depending on time).

Note: The engine is intended as educational and tuned for solid play rather
than extreme performance. It intentionally keeps dependencies minimal.
"""

import sys
import time
import random
import math
from collections import defaultdict

import chess

# ---------------- Engine configuration / UCI options ----------------
ENGINE_NAME = "Okval"
ENGINE_AUTHOR = "Classic"

# Default options (modifiable via 'setoption name <name> value <value>')
OPTIONS = {
    "Hash": 64,  # MB (used only to size the transposition table modestly)
    "SkillLevel": 12,  # 0..20, roughly controls search depth/time/aggressiveness
}

# Map SkillLevel to search parameters
def skill_to_depth(skill):
    # skill 0 -> depth 1-2, skill 20 -> depth 6-12 depending on time
    return max(1, 2 + skill // 3)

# ---------------- Basic evaluation parameters ----------------
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Piece-square tables (simple, middle-game oriented)
# Indexed by square (0..63) in python-chess board order (a1=0 .. h8=63)
# For black pieces we will mirror the table.
PST = {
    chess.PAWN: [
         0,  0,  0,   0,   0,  0,  0,   0,
         5, 10, 10, -20, -20, 10, 10,   5,
         5, -5,-10,   0,   0,-10, -5,   5,
         0,  0,  0,  20,  20,  0,  0,   0,
         5,  5, 10,  25,  25, 10,  5,   5,
        10, 10, 20,  30,  30, 20, 10,  10,
        50, 50, 50,  50,  50, 50, 50,  50,
         0,  0,  0,   0,   0,  0,  0,   0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ],
    chess.ROOK: [
         0,  0,  5, 10, 10,  5,  0,  0,
         0,  0,  5, 10, 10,  5,  0,  0,
         0,  0,  5, 10, 10,  5,  0,  0,
         0,  0,  5, 10, 10,  5,  0,  0,
         0,  0,  5, 10, 10,  5,  0,  0,
         0,  0,  5, 10, 10,  5,  0,  0,
        25, 25, 25, 25, 25, 25, 25, 25,
         0,  0,  5, 10, 10,  5,  0,  0
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20
    ]
}

# Mirror function for black
def pst_score(piece_type, square, color):
    table = PST[piece_type]
    if color == chess.WHITE:
        return table[square]
    else:
        # mirror vertically for black
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        mirrored = chess.square(file, 7 - rank)
        return table[mirrored]

# ---------------- Transposition Table ----------------
class TTEntry:
    def __init__(self, depth, score, flag, best):
        self.depth = depth
        self.score = score
        self.flag = flag  # 'EXACT', 'LOWER', 'UPPER'
        self.best = best

class TranspositionTable:
    def __init__(self, mb=64):
        # crude sizing based on MB
        approx_entries = max(1000, (mb * 1024 * 1024) // 64)
        self.table = {}
        self.max_entries = approx_entries

    def get(self, key):
        return self.table.get(key)

    def store(self, key, entry):
        if len(self.table) > self.max_entries:
            # random prune
            for k in list(self.table)[:len(self.table)//10]:
                del self.table[k]
        self.table[key] = entry

# ---------------- Move ordering helpers ----------------
def mvv_lva(move, board):
    # higher is better
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        mover = board.piece_at(move.from_square)
        if captured and mover:
            return PIECE_VALUES[captured.piece_type] - (PIECE_VALUES[mover.piece_type] // 10)
        else:
            return 1000
    # promotions
    if move.promotion:
        return PIECE_VALUES[move.promotion] + 800
    return 0

# ---------------- Evaluation ----------------

def evaluate(board):
    """Simple evaluation: material + piece-square + mobility + small king safety"""
    if board.is_checkmate():
        # if side to move is checkmated -> large negative
        return -99999 if board.turn == chess.WHITE else 99999

    material = 0
    pst = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            val = PIECE_VALUES[piece.piece_type]
            if piece.color == chess.WHITE:
                material += val
                pst += pst_score(piece.piece_type, square, chess.WHITE)
            else:
                material -= val
                pst -= pst_score(piece.piece_type, square, chess.BLACK)
    # mobility (legal moves count)
    mobility = len(list(board.legal_moves))
    # small king safety: penalty for open files near king
    king_safety = 0
    # combine
    score = material + pst + 10 * (mobility if board.turn == chess.WHITE else -mobility) + king_safety
    return score

# ---------------- Search ----------------

INFINITY = 100000

class Search:
    def __init__(self, board, tt_mb=64):
        self.board = board
        self.tt = TranspositionTable(mb=tt_mb)
        self.start_time = 0
        self.stop_time = False
        self.nodes = 0
        self.best_move = None
        self.killer = defaultdict(lambda: [None, None])
        self.history = defaultdict(int)

    def time_up(self):
        return self.stop_time

    def think(self, wtime=None, btime=None, movetime=None, depth=None, skill=12):
        self.start_time = time.time()
        self.stop_time = False
        self.nodes = 0
        self.best_move = None

        max_depth = depth if depth is not None else skill_to_depth(skill)
        if movetime:
            end_time = self.start_time + movetime / 1000.0
        elif wtime is not None or btime is not None:
            # simplistic time allocation
            remaining = wtime if self.board.turn == chess.WHITE else btime
            if remaining is None:
                end_time = None
            else:
                # spend small fraction
                spend = max(0.01, min(0.2, 2000.0 / (remaining + 1)))
                # better: use skill to decide how much
                end_time = self.start_time + remaining / 1000.0 * 0.02 * (1 + skill/10)
        else:
            end_time = None

        best = None
        best_score = -INFINITY
        for d in range(1, max_depth + 1):
            try:
                score = self.alphabeta_root(d, end_time)
            except TimeoutError:
                break
            if self.time_up():
                break
            if self.best_move:
                best = self.best_move
                best_score = score
            # small iterative print (not required by UCI but useful if run interactively)
        return best, best_score, self.nodes

    def alphabeta_root(self, depth, end_time):
        alpha = -INFINITY
        beta = INFINITY
        best_score = -INFINITY
        best_move = None
        moves = list(self.board.legal_moves)
        # order moves by MVV/LVA + captures
        moves.sort(key=lambda m: -mvv_lva(m, self.board))
        for move in moves:
            if end_time and time.time() > end_time:
                self.stop_time = True
                raise TimeoutError
            self.board.push(move)
            self.nodes += 1
            score = -self.alphabeta(depth - 1, -beta, -alpha, end_time)
            self.board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        self.best_move = best_move
        return best_score

    def quiescence(self, alpha, beta, end_time):
        if end_time and time.time() > end_time:
            self.stop_time = True
            raise TimeoutError
        stand_pat = evaluate(self.board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        # consider captures
        moves = [m for m in self.board.legal_moves if self.board.is_capture(m)]
        # order captures
        moves.sort(key=lambda m: -mvv_lva(m, self.board))
        for move in moves:
            self.board.push(move)
            self.nodes += 1
            score = -self.quiescence(-beta, -alpha, end_time)
            self.board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def alphabeta(self, depth, alpha, beta, end_time):
        if end_time and time.time() > end_time:
            self.stop_time = True
            raise TimeoutError
        self.nodes += 1

        # Check for terminal states
        if self.board.is_checkmate():
            return -99999 + (99 - depth)
        if self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_threefold():
            return 0

        # TT lookup
        key = self.board.fen()
        entry = self.tt.get(key)
        if entry and entry.depth >= depth:
            if entry.flag == 'EXACT':
                return entry.score
            elif entry.flag == 'LOWER' and entry.score > alpha:
                alpha = entry.score
            elif entry.flag == 'UPPER' and entry.score < beta:
                beta = entry.score
            if alpha >= beta:
                return entry.score

        if depth == 0:
            return self.quiescence(alpha, beta, end_time)

        legal_moves = list(self.board.legal_moves)
        # move ordering: try TT best, captures, killers, history
        if entry and entry.best in legal_moves:
            legal_moves.remove(entry.best)
            legal_moves.insert(0, entry.best)

        legal_moves.sort(key=lambda m: (-mvv_lva(m, self.board), -self.history[(m.from_square, m.to_square)]))

        best_score = -INFINITY
        best_move = None
        for move in legal_moves:
            self.board.push(move)
            try:
                score = -self.alphabeta(depth - 1, -beta, -alpha, end_time)
            except TimeoutError:
                self.board.pop()
                raise
            self.board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                # store killer/history heuristics
                ply = depth
                if move not in [self.killer[ply][0], self.killer[ply][1]]:
                    self.killer[ply][1] = self.killer[ply][0]
                    self.killer[ply][0] = move
                self.history[(move.from_square, move.to_square)] += 2 ** depth
                break
        # store in TT
        flag = 'EXACT'
        if best_score <= alpha_orig := alpha:
            flag = 'UPPER'
        elif best_score >= beta:
            flag = 'LOWER'
        # Normalize best_score into int
        entry = TTEntry(depth, int(best_score), flag, best_move)
        self.tt.store(key, entry)
        return best_score

# ---------------- UCI Loop ----------------

class UCIEngine:
    def __init__(self):
        self.board = chess.Board()
        self.searcher = Search(self.board, tt_mb=OPTIONS['Hash'])
        self.uci_options = OPTIONS.copy()

    def loop(self):
        while True:
            try:
                line = sys.stdin.readline()
            except KeyboardInterrupt:
                break
            if not line:
                break
            line = line.strip()
            if line == "":
                continue
            parts = line.split()
            cmd = parts[0]
            if cmd == 'uci':
                print(f"id name {ENGINE_NAME}")
                print(f"id author {ENGINE_AUTHOR}")
                # options
                print("option name Hash type spin default {} min 1 max 1024".format(self.uci_options['Hash']))
                print("option name SkillLevel type spin default {} min 0 max 20".format(self.uci_options['SkillLevel']))
                print("uciok")
            elif cmd == 'isready':
                print('readyok')
            elif cmd == 'setoption':
                # setoption name <name> value <val>
                try:
                    name_idx = parts.index('name') + 1
                    value_idx = parts.index('value') + 1
                    name = ' '.join(parts[name_idx:value_idx - 1])
                    value = ' '.join(parts[value_idx:])
                except ValueError:
                    # fallback simple
                    # assume format: setoption name X value Y
                    name = parts[2]
                    value = parts[4]
                name = name.strip()
                value = value.strip()
                if name == 'Hash':
                    self.uci_options['Hash'] = int(value)
                    self.searcher.tt = TranspositionTable(mb=self.uci_options['Hash'])
                elif name == 'SkillLevel':
                    self.uci_options['SkillLevel'] = int(value)
                # ignore others for now
            elif cmd == 'ucinewgame':
                self.board.reset()
                self.searcher = Search(self.board, tt_mb=self.uci_options['Hash'])
            elif cmd == 'position':
                # position [fen <fen> | startpos ]  moves <move1> ...
                if 'startpos' in parts:
                    self.board.reset()
                    idx = parts.index('startpos') + 1
                elif 'fen' in parts:
                    fen_idx = parts.index('fen') + 1
                    # fen is until 'moves' or end
                    if 'moves' in parts:
                        moves_idx = parts.index('moves')
                        fen = ' '.join(parts[fen_idx:moves_idx])
                        idx = moves_idx + 1
                        self.board.set_fen(fen)
                    else:
                        fen = ' '.join(parts[fen_idx:])
                        self.board.set_fen(fen)
                        idx = len(parts)
                else:
                    idx = 1
                if 'moves' in parts:
                    moves_idx = parts.index('moves') + 1
                    moves = parts[moves_idx:]
                    for m in moves:
                        try:
                            move = chess.Move.from_uci(m)
                            if move in self.board.legal_moves:
                                self.board.push(move)
                            else:
                                # illegal move in provided list -> ignore
                                pass
                        except Exception:
                            pass
            elif cmd == 'go':
                # parse common args: wtime, btime, movetime, depth
                wtime = btime = movetime = depth = None
                if 'wtime' in parts:
                    wtime = int(parts[parts.index('wtime') + 1])
                if 'btime' in parts:
                    btime = int(parts[parts.index('btime') + 1])
                if 'movetime' in parts:
                    movetime = int(parts[parts.index('movetime') + 1])
                if 'depth' in parts:
                    depth = int(parts[parts.index('depth') + 1])
                # start thinking
                best, score, nodes = self.searcher.think(wtime=wtime, btime=btime, movetime=movetime, depth=depth, skill=self.uci_options['SkillLevel'])
                if best is None:
                    # no move (maybe game over)
                    print('bestmove (none)')
                else:
                    print(f"bestmove {best.uci()}")
            elif cmd == 'quit':
                break
            # flush stdout explicitly
            sys.stdout.flush()


if __name__ == '__main__':
    engine = UCIEngine()
    # If run directly, optionally allow running a quick play vs human via UCI-like CLI
    if sys.stdin.isatty():
        print(f"{ENGINE_NAME} by {ENGINE_AUTHOR} - running in UCI mode. Type 'uci' to start.")
    engine.loop()
