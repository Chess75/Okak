#!/usr/bin/env python3
"""
Simple UCI-compatible chess engine "Okval" by author "classic".
- Language: Python 3
- Dependency: python-chess (pip install python-chess)

Features:
- UCI protocol: responds to 'uci', 'isready', 'position', 'go', 'stop', 'quit', 'ucinewgame', basic 'setoption'
- Search: minimax with alpha-beta + simple move ordering + quiescence search.
- Evaluation: material + simple piece-square tables + mobility bonus.
- Not a world-beater but plays reasonably and "tries to win" by preferring aggressive moves and captures.

Use with lichess-bot: point engine.command to `python3 okval_engine.py`

Author: classic
Engine name: Okval
"""

import sys
import time
import math
import threading
from collections import defaultdict

import chess
import chess.polyglot

# UCI identity
ENGINE_NAME = "Okval"
ENGINE_AUTHOR = "Classic"

# Default search parameters
MAX_DEPTH = 4  # iterative deepening can be used by wrapper (keep small by default)
TIME_BUFFER = 0.05  # seconds reserved

# Piece values
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Simple piece-square tables (from common simplified heuristics)
# indexed by square for white; for black we mirror vertically
PST = {
    chess.PAWN: [0,0,0,0,0,0,0,0,
                 5,10,10,-20,-20,10,10,5,
                 5,-5,-10,0,0,-10,-5,5,
                 0,0,0,20,20,0,0,0,
                 5,5,10,25,25,10,5,5,
                 10,10,20,30,30,20,10,10,
                 50,50,50,50,50,50,50,50,
                 0,0,0,0,0,0,0,0],
    chess.KNIGHT: [-50,-40,-30,-30,-30,-30,-40,-50,
                   -40,-20,0,5,5,0,-20,-40,
                   -30,5,10,15,15,10,5,-30,
                   -30,0,15,20,20,15,0,-30,
                   -30,5,15,20,20,15,5,-30,
                   -30,0,10,15,15,10,0,-30,
                   -40,-20,0,0,0,0,-20,-40,
                   -50,-90,-30,-30,-30,-30,-90,-50],
    chess.BISHOP: [-20,-10,-10,-10,-10,-10,-10,-20,
                   -10,5,0,0,0,0,5,-10,
                   -10,10,10,10,10,10,10,-10,
                   -10,0,10,10,10,10,0,-10,
                   -10,5,5,10,10,5,5,-10,
                   -10,0,5,10,10,5,0,-10,
                   -10,0,0,0,0,0,0,-10,
                   -20,-10,-40,-10,-40,-10,-10,-20],
    chess.ROOK: [0,0,5,10,10,5,0,0,
                 -5,0,0,0,0,0,0,-5,
                 -5,0,0,0,0,0,0,-5,
                 -5,0,0,0,0,0,0,-5,
                 -5,0,0,0,0,0,0,-5,
                 -5,0,0,0,0,0,0,-5,
                 5,10,10,10,10,10,10,5,
                 0,0,0,0,0,0,0,0],
    chess.QUEEN: [-20,-10,-10,-5,-5,-10,-10,-20,
                  -10,0,5,0,0,0,0,-10,
                  -10,5,5,5,5,5,0,-10,
                  0,0,5,5,5,5,0,-5,
                  -5,0,5,5,5,5,0,-5,
                  -10,0,5,0,0,0,0,-10,
                  -10,0,0,0,0,0,0,-10,
                  -20,-10,-10,-5,-5,-10,-10,-20],
    chess.KING: [20,30,10,0,0,10,30,20,
                 20,20,0,0,0,0,20,20,
                 -10,-20,-20,-20,-20,-20,-20,-10,
                 -20,-30,-30,-40,-40,-30,-30,-20,
                 -30,-40,-40,-50,-50,-40,-40,-30,
                 -30,-40,-40,-50,-50,-40,-40,-30,
                 -30,-40,-40,-50,-50,-40,-40,-30,
                 -30,-40,-40,-50,-50,-40,-40,-30],
}

# Transposition table
TT = {}

class SearchTimeout(Exception):
    pass

class Engine:
    def __init__(self):
        self.board = chess.Board()
        self.stop_search = False
        self.nodes = 0
        self.start_time = 0
        self.time_limit = None
        self.go_event = threading.Event()
        self.search_thread = None
        self.ponder = False
        self.max_depth = MAX_DEPTH
        self.lock = threading.Lock()

    def uci(self):
        print(f"id name {ENGINE_NAME}")
        print(f"id author {ENGINE_AUTHOR}")
        print("uciok")
        sys.stdout.flush()

    def isready(self):
        print("readyok")
        sys.stdout.flush()

    def ucinewgame(self):
        with self.lock:
            self.board.reset()
            TT.clear()

    def setoption(self, name, value):
        # support custom option: SkillLevel or MaxDepth
        if name.lower() == "maxdepth":
            try:
                self.max_depth = int(value)
            except:
                pass

    def position(self, tokens):
        # tokens: list starting from 'position'
        # examples: position startpos moves e2e4 e7e5
        idx = 1
        if tokens[1] == 'startpos':
            self.board.reset()
            idx = 2
        elif tokens[1] == 'fen':
            fen_parts = tokens[2:2+6]
            fen = ' '.join(fen_parts)
            self.board.set_fen(fen)
            idx = 8
        # process moves
        if idx < len(tokens) and tokens[idx] == 'moves':
            moves = tokens[idx+1:]
            for mv in moves:
                try:
                    self.board.push_uci(mv)
                except Exception:
                    pass

    def go(self, tokens):
        # handle simple time controls: movetime, wtime/btime/ponder
        wtime = None
        btime = None
        movetime = None
        depth = None
        for i, t in enumerate(tokens):
            if t == 'movetime':
                movetime = int(tokens[i+1]) / 1000.0
            if t == 'wtime':
                wtime = int(tokens[i+1]) / 1000.0
            if t == 'btime':
                btime = int(tokens[i+1]) / 1000.0
            if t == 'depth':
                depth = int(tokens[i+1])
            if t == 'ponder':
                self.ponder = True
        # decide time limit
        if movetime:
            self.time_limit = movetime
        else:
            # simple allocation: use 1/30 of remaining clock for a move
            if self.board.turn == chess.WHITE and wtime:
                self.time_limit = max(0.01, wtime / 30.0)
            elif self.board.turn == chess.BLACK and btime:
                self.time_limit = max(0.01, btime / 30.0)
            else:
                self.time_limit = None
        if depth:
            self.max_depth = depth
        # start search thread
        self.stop_search = False
        self.go_event.clear()
        self.search_thread = threading.Thread(target=self.search_root)
        self.search_thread.start()

    def stop(self):
        self.stop_search = True
        self.go_event.set()
        if self.search_thread:
            self.search_thread.join()

    # Evaluation
    def evaluate(self, board: chess.Board):
        if board.is_checkmate():
            return -999999 if board.turn else 999999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        val = 0
        # material + pst
        for piece_type in PIECE_VALUES:
            for sq in board.pieces(piece_type, chess.WHITE):
                val += PIECE_VALUES[piece_type]
                val += PST[piece_type][sq]
            for sq in board.pieces(piece_type, chess.BLACK):
                val -= PIECE_VALUES[piece_type]
                # mirror for black
                val -= PST[piece_type][chess.square_mirror(sq)]
        # mobility
        val += 5 * (len(list(board.legal_moves)) if board.turn == chess.WHITE else -len(list(board.legal_moves)))
        return val

    def quiescence(self, board, alpha, beta):
        stand_pat = self.evaluate(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        # consider captures
        for move in sorted(board.legal_moves, key=lambda m: -self.mv_capture_value(board, m)):
            if not board.is_capture(move):
                continue
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def mv_capture_value(self, board, move):
        if not board.is_capture(move):
            return 0
        if board.is_en_passant(move):
            return PIECE_VALUES[chess.PAWN]
        captured = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if captured:
            return PIECE_VALUES[captured.piece_type] - PIECE_VALUES[attacker.piece_type]
        return 0

    def order_moves(self, board, moves):
        # order by captures (most valuable victim - least valuable attacker), then promotions, then killer
        def key(m):
            score = 0
            if board.is_capture(m):
                captured = board.piece_at(m.to_square)
                attacker = board.piece_at(m.from_square)
                if captured:
                    score += 1000 * PIECE_VALUES[captured.piece_type] - PIECE_VALUES[attacker.piece_type]
            if m.promotion:
                score += 800
            # prefer checks (heuristic)
            board.push(m)
            if board.is_check():
                score += 50
            board.pop()
            return -score
        return sorted(moves, key=key)

    def alpha_beta(self, board, depth, alpha, beta):
        # timeout check
        if self.time_limit is not None and time.time() - self.start_time > self.time_limit - TIME_BUFFER:
            raise SearchTimeout()
        self.nodes += 1
        key = (board.zobrist_hash(), depth, board.turn)
        if key in TT:
            return TT[key]
        if depth == 0:
            val = self.quiescence(board, alpha, beta)
            TT[key] = val
            return val
        if board.is_checkmate():
            return -999999
        if board.is_stalemate():
            return 0
        max_eval = -9999999
        moves = list(board.legal_moves)
        moves = self.order_moves(board, moves)
        for mv in moves:
            board.push(mv)
            score = -self.alpha_beta(board, depth-1, -beta, -alpha)
            board.pop()
            if score > max_eval:
                max_eval = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        TT[key] = max_eval
        return max_eval

    def search_root(self):
        best_move = None
        best_score = -9999999
        self.start_time = time.time()
        self.nodes = 0
        try:
            for depth in range(1, self.max_depth + 1):
                if self.stop_search:
                    break
                # aspiration windows could be used; we keep simple
                moves = list(self.board.legal_moves)
                moves = self.order_moves(self.board, moves)
                local_best = None
                local_best_score = -9999999
                for mv in moves:
                    if self.stop_search:
                        break
                    self.board.push(mv)
                    try:
                        score = -self.alpha_beta(self.board, depth-1, -99999999, 99999999)
                    except SearchTimeout:
                        self.board.pop()
                        raise
                    self.board.pop()
                    if score > local_best_score:
                        local_best_score = score
                        local_best = mv
                if local_best is not None:
                    best_move = local_best
                    best_score = local_best_score
                # report best move so far (UCI info)
                elapsed = time.time() - self.start_time
                nps = int(self.nodes / max(1.0, elapsed))
                print(f"info depth {depth} score cp {best_score} nodes {self.nodes} nps {nps} time {int(elapsed*1000)} pv {best_move}")
                sys.stdout.flush()
        except SearchTimeout:
            pass
        if best_move is None:
            # fallback: random legal move
            try:
                best_move = next(iter(self.board.legal_moves))
            except StopIteration:
                print("bestmove (none)")
                sys.stdout.flush()
                return
        print(f"bestmove {best_move.uci()}")
        sys.stdout.flush()

    def run(self):
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if line == '':
                    continue
                tokens = line.split()
                cmd = tokens[0]
                if cmd == 'uci':
                    self.uci()
                elif cmd == 'isready':
                    self.isready()
                elif cmd == 'ucinewgame':
                    self.ucinewgame()
                elif cmd == 'position':
                    self.position(tokens)
                elif cmd == 'go':
                    self.go(tokens)
                elif cmd == 'stop':
                    self.stop()
                elif cmd == 'quit':
                    self.stop()
                    break
                elif cmd == 'setoption':
                    # setoption name <name> value <value>
                    # naive parse
                    try:
                        name_idx = tokens.index('name') + 1
                        val_idx = tokens.index('value') + 1
                        name = ' '.join(tokens[name_idx:val_idx-1])
                        value = ' '.join(tokens[val_idx:])
                        self.setoption(name, value)
                    except Exception:
                        pass
                else:
                    # ignore unhandled commands
                    pass
            except Exception as e:
                # keep engine alive on errors
                print(f"debug {e}")
                sys.stdout.flush()


if __name__ == '__main__':
    engine = Engine()
    engine.run()
