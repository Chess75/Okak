#!/usr/bin/env python3


import copy
import time
import chess

############################################
# ============ PeSTO TABLES ================
############################################

def flip_table(table):
    flip_temp_table = list(table)
    for i in range(len(table)):
        flip_temp_table[i] = table[i ^ 56]
    return tuple(flip_temp_table)


def init_tables():
    # --- piece-square tables (PeSTO) ---
    middlegame_pawn_table = (
        0, 0, 0, 0, 0, 0, 0, 0,
        98, 134, 61, 95, 68, 126, 34, -11,
        -6, 7, 26, 31, 65, 56, 25, -20,
        -14, 13, 6, 21, 23, 12, 17, -23,
        -27, -2, -5, 12, 17, 6, 10, -25,
        -26, -4, -4, -10, 3, 3, 33, -12,
        -35, -1, -20, -23, -15, 24, 38, -22,
        0, 0, 0, 0, 0, 0, 0, 0,
    )
    endgame_pawn_table = (
        0, 0, 0, 0, 0, 0, 0, 0,
        178, 173, 158, 134, 147, 132, 165, 187,
        94, 100, 85, 67, 56, 53, 82, 84,
        32, 24, 13, 5, -2, 4, 17, 17,
        13, 9, -3, -7, -7, -8, 3, -1,
        4, 7, -6, 1, 0, -5, -1, -8,
        13, 8, 8, 10, 13, 0, 2, -7,
        0, 0, 0, 0, 0, 0, 0, 0,
    )
    middlegame_knight_table = (
        -167, -89, -34, -49, 61, -97, -15, -107,
        -73, -41, 72, 36, 23, 62, 7, -17,
        -47, 60, 37, 65, 84, 129, 73, 44,
        -9, 17, 19, 53, 37, 69, 18, 22,
        -13, 4, 16, 13, 28, 19, 21, -8,
        -23, -9, 12, 10, 19, 17, 25, -16,
        -29, -53, -12, -3, -1, 18, -14, -19,
        -105, -21, -58, -33, -17, -28, -19, -23,
    )
    endgame_knight_table = (
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25, -8, -25, -2, -9, -25, -24, -52,
        -24, -20, 10, 9, -1, -9, -19, -41,
        -17, 3, 22, 22, 22, 11, 8, -18,
        -18, -6, 16, 25, 16, 17, 4, -18,
        -23, -3, -1, 15, 10, -3, -20, -22,
        -42, -20, -10, -5, -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    )
    middlegame_tables = []
    endgame_tables = []
    values_mg = {'p':82,'n':337,'b':365,'r':477,'q':1025,'k':0}
    values_eg = {'p':94,'n':281,'b':297,'r':512,'q':936,'k':0}
    pesto_mg = {'p':middlegame_pawn_table,'n':middlegame_knight_table}
    pesto_eg = {'p':endgame_pawn_table,'n':endgame_knight_table}
    for p in ('p','n'):
        t=[]
        for i in range(64): t.append(values_mg[p]+pesto_mg[p][i])
        middlegame_tables.append(tuple(t))
        middlegame_tables.append(flip_table(t))
        t=[]
        for i in range(64): t.append(values_eg[p]+pesto_eg[p][i])
        endgame_tables.append(tuple(t))
        endgame_tables.append(flip_table(t))
    return tuple(middlegame_tables), tuple(endgame_tables)

############################################
# ============== ENGINE ====================
############################################

class Engine:
    def __init__(self):
        self.name = "Okval"
        self.author = "classic"
        self.board = chess.Board()
        self.nodes = 0
        self.start_time = 0
        self.stop_time = 0
        self.mg, self.eg = init_tables()

    def eval(self):
        if self.board.is_checkmate():
            return -999999
        if self.board.is_stalemate():
            return 0
        score = 0
        for p in chess.PIECE_TYPES:
            for c in chess.COLORS:
                idx = 2*(p-1)+int(c)
                for sq in self.board.pieces(p,c):
                    score += self.mg[idx][sq] if c else -self.mg[idx][sq]
        return score if self.board.turn else -score

    def negamax(self, depth, alpha, beta):
        if depth == 0 or self.board.is_game_over():
            self.nodes += 1
            return self.eval()
        best = -1_000_000
        for mv in self.board.legal_moves:
            self.board.push(mv)
            val = -self.negamax(depth-1, -beta, -alpha)
            self.board.pop()
            best = max(best, val)
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return best

    def search(self, movetime):
        self.start_time = time.time()
        best_move = None
        depth = 1
        while time.time()-self.start_time < movetime:
            best = -1_000_000
            for mv in self.board.legal_moves:
                self.board.push(mv)
                val = -self.negamax(depth-1, -1_000_000, 1_000_000)
                self.board.pop()
                if val > best:
                    best = val
                    best_move = mv
            print(f"info depth {depth} score cp {best}", flush=True)
            depth += 1
        return best_move

############################################
# ================ UCI =====================
############################################

engine = Engine()

while True:
    cmd = input().split()
    if not cmd:
        continue
    if cmd[0] == 'uci':
        print(f"id name {engine.name}")
        print(f"id author {engine.author}")
        print("uciok")
    elif cmd[0] == 'isready':
        print("readyok")
    elif cmd[0] == 'ucinewgame':
        engine.board.reset()
    elif cmd[0] == 'position':
        engine.board.reset()
        if cmd[1] == 'fen':
            engine.board.set_fen(' '.join(cmd[2:]))
        elif cmd[1] == 'startpos' and 'moves' in cmd:
            for m in cmd[cmd.index('moves')+1:]: engine.board.push_uci(m)
    elif cmd[0] == 'go':
        movetime = 1
        if 'movetime' in cmd:
            movetime = int(cmd[cmd.index('movetime')+1]) / 1000
        move = engine.search(movetime)
        print(f"bestmove {move.uci() if move else '(none)'}", flush=True)
    elif cmd[0] == 'quit':
        break
