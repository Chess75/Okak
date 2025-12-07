#!/usr/bin/env python3
# simple_engine.py
# UCI-compatible simple chess engine using python-chess

import chess
import sys
import time
import threading
from collections import defaultdict, namedtuple

# ---- Константы ----
INF = 10**9

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Простейшие PST (пример)
PST = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10,-20,-20, 10, 10,  5,
         5, -5,-10,  0,  0,-10, -5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5,  5, 10, 25, 25, 10,  5,  5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
         0,  0,  0,  0,  0,  0,  0,  0
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
    ]
}

# Transposition table entry
TTEntry = namedtuple("TTEntry", ["depth", "flag", "score", "best_move"])
# flag: 'EXACT', 'LOWER', 'UPPER'

# ---- Утилиты ----

def fast_board_key(board: chess.Board):
    """
    Ключ для TT: используем полную FEN + ход (turn) + castling + ep + halfmove.
    Это надёжнее, чем только board_fen().
    """
    # board.fen() уже содержит всю необходимую информацию,
    # но она включает счетчик ply — используем именно полную FEN,
    # чтобы ключ был детерминирован по позиции.
    try:
        fen = board.board_fen()
        turn = board.turn
        castling = board.castling_xfen()
        ep = board.ep_square
        half = board.halfmove_clock
        key = (fen, turn, castling, ep, half)
    except Exception:
        key = (board.fen(), board.turn)
    return key

def mvv_lva_score(board, move):
    """
    MVV-LVA-like score: приоритет захватов, промоции придают бонус.
    """
    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim and attacker:
            score += PIECE_VALUES.get(victim.piece_type, 0) * 10 - PIECE_VALUES.get(attacker.piece_type, 0)
    if move.promotion:
        # побуждаем к промоции
        score += PIECE_VALUES[chess.QUEEN] // 2
    return score

# ---- Оценка позиции ----

def evaluate(board: chess.Board):
    """
    Оценка: материал + PST + мобильность + штраф за шах.
    Возвращается оценка **отн. стороны, которая ходит** (положительно = хорошо для side-to-move).
    """
    if board.is_checkmate():
        # если мат — очень плохая оценка для стороны, которая ходит
        return -INF + 1
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    material = 0
    pst_score = 0

    for piece_type, value in PIECE_VALUES.items():
        for sq in board.pieces(piece_type, chess.WHITE):
            material += value
            if piece_type in PST:
                pst_score += PST[piece_type][sq]
        for sq in board.pieces(piece_type, chess.BLACK):
            material -= value
            if piece_type in PST:
                # mirror для чёрных
                pst_score -= PST[piece_type][chess.square_mirror(sq)]

    # простая мобильность: число легальных ходов
    try:
        mobility_count = board.legal_moves.count()
    except Exception:
        mobility_count = len(list(board.legal_moves))
    mobility = 10 * mobility_count

    check_penalty = -50 if board.is_check() else 0

    score_white = material + pst_score + mobility + check_penalty
    return score_white if board.turn == chess.WHITE else -score_white

# ---- Search State ----

class SearchState:
    def __init__(self):
        self.tt = {}  # key -> TTEntry
        self.nodes = 0
        self.start_time = 0.0
        self.time_limit = 0.0
        self.history = defaultdict(int)

# ---- Исключение прерывания ----
class SearchAbort(Exception):
    pass

# ---- Quiescence ----

def quiescence(board: chess.Board, alpha: int, beta: int, state: SearchState, stop_event: threading.Event):
    if stop_event.is_set():
        raise SearchAbort()
    if state.start_time and (time.time() - state.start_time) > state.time_limit:
        raise SearchAbort()

    state.nodes += 1
    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    # рассматриваем только захваты (и промоции — они уже в legal_moves)
    captures = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    if not captures:
        return alpha
    captures.sort(key=lambda mv: -mvv_lva_score(board, mv))

    for move in captures:
        if stop_event.is_set():
            raise SearchAbort()
        board.push(move)
        try:
            score = -quiescence(board, -beta, -alpha, state, stop_event)
        finally:
            board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha

# ---- Negamax + AlphaBeta + TT + Ordering ----

def negamax(board: chess.Board, depth: int, alpha: int, beta: int, state: SearchState, stop_event: threading.Event):
    if stop_event.is_set():
        raise SearchAbort()
    if state.start_time and (time.time() - state.start_time) > state.time_limit:
        raise SearchAbort()

    state.nodes += 1

    if depth == 0:
        return quiescence(board, alpha, beta, state, stop_event)

    key = fast_board_key(board)
    tt_entry = state.tt.get(key)
    if tt_entry and tt_entry.depth >= depth:
        if tt_entry.flag == 'EXACT':
            return tt_entry.score
        elif tt_entry.flag == 'LOWER':
            alpha = max(alpha, tt_entry.score)
        elif tt_entry.flag == 'UPPER':
            beta = min(beta, tt_entry.score)
        if alpha >= beta:
            return tt_entry.score

    alpha_orig = alpha
    beta_orig = beta

    best_score = -INF
    best_move = None

    # prepare moves and ordering
    moves = list(board.legal_moves)

    def move_key(mv):
        # TT move highest priority
        if tt_entry and tt_entry.best_move and mv == tt_entry.best_move:
            return (0, 0, 0)
        cap = 0 if board.is_capture(mv) else 1
        mvv = -mvv_lva_score(board, mv)
        # history: more положительное -> выше приоритет (используем отрицание для сортировки)
        hist = -state.history[(board.turn, mv.from_square, mv.to_square, mv.promotion)]
        return (cap, mvv, hist)

    moves.sort(key=move_key)

    for move in moves:
        if stop_event.is_set():
            raise SearchAbort()

        mover = board.turn  # сторона, которая сделала ход
        board.push(move)
        try:
            score = -negamax(board, depth - 1, -beta, -alpha, state, stop_event)
        finally:
            # всегда один pop (восстанавливаем позицию после root-хода)
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score
            # history update для непойманного хода
            if not board.is_capture(move):
                state.history[(mover, move.from_square, move.to_square, move.promotion)] += (1 << depth)

        if alpha >= beta:
            # beta cutoff: history update
            state.history[(mover, move.from_square, move.to_square, move.promotion)] += (1 << depth)
            break

    # set TT flag relative to originals
    if best_score >= beta_orig:
        flag = 'LOWER'
    elif best_score <= alpha_orig:
        flag = 'UPPER'
    else:
        flag = 'EXACT'

    state.tt[key] = TTEntry(depth=depth, flag=flag, score=best_score, best_move=best_move)
    return best_score

# ---- Search Thread (iterative deepening) ----

class SearchThread(threading.Thread):
    def __init__(self, root_board: chess.Board, wtime=None, btime=None, winc=0, binc=0, movetime=None, max_depth=None, stop_event=None, options=None):
        super().__init__()
        self.root_board = root_board.copy()
        self.wtime = wtime
        self.btime = btime
        self.winc = winc or 0
        self.binc = binc or 0
        self.movetime = movetime
        self.max_depth = max_depth
        self.stop_event = stop_event or threading.Event()

        self.best_move = None
        self.best_score = None
        self.depth_reached = 0

        self.state = SearchState()
        self.state.time_limit = 0.0
        self.state.start_time = 0.0

        self.options = options or {}

    def time_remaining_ms(self):
        if self.movetime:
            return self.movetime
        # простая эвристика
        if self.root_board.turn == chess.WHITE:
            if self.wtime is None:
                return 10000
            if self.wtime < 2000:
                return 50
            return max(20, self.wtime // 20 + self.winc * 2)
        else:
            if self.btime is None:
                return 10000
            if self.btime < 2000:
                return 50
            return max(20, self.btime // 20 + self.binc * 2)

    def run(self):
        ms = self.time_remaining_ms()
        self.state.time_limit = ms / 1000.0
        self.state.start_time = time.time()
        depth = 1

        try:
            while not self.stop_event.is_set():
                if self.max_depth and depth > self.max_depth:
                    break
                self.depth_reached = depth

                # root ordering: prefer TT move then MVV-LVA
                moves = list(self.root_board.legal_moves)
                root_key = fast_board_key(self.root_board)
                root_tt = self.state.tt.get(root_key)

                def root_key_fn(mv):
                    if root_tt and root_tt.best_move and mv == root_tt.best_move:
                        return (0, 0)
                    cap = 0 if self.root_board.is_capture(mv) else 1
                    mvv = -mvv_lva_score(self.root_board, mv)
                    return (cap, mvv)

                moves.sort(key=root_key_fn)

                best_for_depth = None
                best_score_for_depth = -INF

                for mv in moves:
                    if self.stop_event.is_set():
                        break

                    # push root move, search depth-1, then pop exactly once
                    self.root_board.push(mv)
                    try:
                        # search from the resulting position
                        score = -negamax(self.root_board, depth - 1, -INF, INF, self.state, self.stop_event)
                    except SearchAbort:
                        # прерывание — просто восстановим и пробросим дальше чтобы выйти корректно
                        self.root_board.pop()
                        raise
                    else:
                        # восстановим позицию после поиска по корневому ходу
                        self.root_board.pop()

                    if score > best_score_for_depth:
                        best_score_for_depth = score
                        best_for_depth = mv

                    # time check
                    if (time.time() - self.state.start_time) > self.state.time_limit:
                        break

                if best_for_depth is not None:
                    self.best_move = best_for_depth
                    self.best_score = best_score_for_depth
                    elapsed = time.time() - self.state.start_time
                    nps = int(self.state.nodes / elapsed) if elapsed > 0 else 0
                    try:
                        pv_str = self.best_move.uci()
                    except Exception:
                        pv_str = "-"
                    # печатаем только базовую info (UCI-парсеры ожидают корректный PV)
                    print(f"info depth {depth} score cp {best_score_for_depth} time {int(elapsed*1000)} nodes {self.state.nodes} nps {nps} pv {pv_str}")
                    sys.stdout.flush()

                # таймаут/время
                if (time.time() - self.state.start_time) > self.state.time_limit:
                    break
                depth += 1

        except SearchAbort:
            # прервано по таймауту/stop
            pass
        except Exception as e:
            print("Search error:", e, file=sys.stderr)
            sys.stderr.flush()

# ---- UCI loop ----

def uci_loop():
    board = chess.Board()
    search_thread = None
    stop_event = threading.Event()
    engine_options = {"Threads": 1}

    # announce engine once at start
    print("id name Okval")
    print("id author Classic")
    # declare supported options (minimal)
    print("option name Threads type spin default 1 min 1 max 64")
    print("uciok")
    sys.stdout.flush()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if line == "":
                continue
            parts = line.split()
            cmd = parts[0]

            if cmd == "uci":
                # respond with engine identity
                print("id name Okval")
                print("id author Classic")
                print("option name Threads type spin default 1 min 1 max 64")
                print("uciok")
                sys.stdout.flush()

            elif cmd == "isready":
                print("readyok")
                sys.stdout.flush()

            elif cmd == "setoption":
                # format: setoption name <name> value <value>
                # Простая обработка Threads — запомним, но не используем многопоточность.
                try:
                    # naive parse
                    if "name" in parts:
                        idx = parts.index("name") + 1
                        name_parts = []
                        while idx < len(parts) and parts[idx] != "value":
                            name_parts.append(parts[idx]); idx += 1
                        name = " ".join(name_parts)
                        value = None
                        if idx < len(parts) and parts[idx] == "value":
                            value = " ".join(parts[idx+1:]) if idx+1 < len(parts) else ""
                        if name.lower() == "threads":
                            try:
                                val = int(value)
                                engine_options["Threads"] = max(1, val)
                                # сообщаем GUI, что опция принята (но мы её игнорируем).
                                # Some GUIs warn if engine doesn't respond to setoption; so print nothing else.
                            except Exception:
                                pass
                        else:
                            # store generically
                            engine_options[name] = value
                except Exception:
                    pass

            elif cmd == "ucinewgame":
                board = chess.Board()

            elif cmd == "position":
                # position [fen <fen> | startpos ] moves <moves>...
                # robust parsing
                try:
                    idx = 1
                    if len(parts) >= 2 and parts[1] == "startpos":
                        board = chess.Board()
                        idx = 2
                    elif len(parts) >= 2 and parts[1] == "fen":
                        # fen is 6 fields
                        if len(parts) >= 8:
                            fen = " ".join(parts[2:8])
                            try:
                                board = chess.Board(fen)
                            except Exception:
                                board = chess.Board()
                            idx = 8
                        else:
                            # bad fen: ignore and leave board unchanged
                            idx = len(parts)
                    # moves
                    if idx < len(parts) and parts[idx] == "moves":
                        for mv in parts[idx+1:]:
                            try:
                                board.push_uci(mv)
                            except Exception:
                                # пропускаем некорректные/нелегальные ходы
                                pass
                except Exception:
                    # любая ошибка — сохраняем прежнюю позицию
                    pass

            elif cmd == "go":
                # parse time control args
                wtime = btime = winc = binc = movetime = None
                depth = None
                i = 1
                while i < len(parts):
                    if parts[i] == "wtime":
                        try:
                            wtime = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "btime":
                        try:
                            btime = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "winc":
                        try:
                            winc = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "binc":
                        try:
                            binc = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "movetime":
                        try:
                            movetime = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "depth":
                        try:
                            depth = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    else:
                        i += 1

                # stop previous search thread
                if search_thread and search_thread.is_alive():
                    stop_event.set()
                    search_thread.join(timeout=1.0)
                    stop_event.clear()

                stop_event = threading.Event()
                search_thread = SearchThread(board, wtime=wtime, btime=btime, winc=winc or 0, binc=binc or 0,
                                             movetime=movetime, max_depth=depth, stop_event=stop_event, options=engine_options)
                search_thread.start()

                # wait for thread to finish (он сам завершится по таймауту/глубине/stop)
                while search_thread.is_alive():
                    time.sleep(0.02)

                if search_thread.best_move:
                    print(f"bestmove {search_thread.best_move.uci()}")
                    sys.stdout.flush()
                else:
                    # fallback: любой легальный ход
                    try:
                        fallback = next(iter(board.legal_moves))
                        print(f"bestmove {fallback.uci()}")
                        sys.stdout.flush()
                    except StopIteration:
                        print("bestmove 0000")
                        sys.stdout.flush()

            elif cmd == "stop":
                if search_thread and search_thread.is_alive():
                    stop_event.set()
                    search_thread.join(timeout=1.0)
                if search_thread and search_thread.best_move:
                    print(f"bestmove {search_thread.best_move.uci()}")
                    sys.stdout.flush()
                else:
                    print("bestmove 0000")
                    sys.stdout.flush()

            elif cmd == "quit":
                if search_thread and search_thread.is_alive():
                    stop_event.set()
                    search_thread.join(timeout=1.0)
                break

            else:
                # ignore unknown commands
                pass

        except Exception as e:
            print("error:", e, file=sys.stderr)
            sys.stderr.flush()
            break

if __name__ == "__main__":
    uci_loop()
