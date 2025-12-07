#!/usr/bin/env python3
# simple_engine_fixed.py
# Исправленная и улучшенная версия простого UCI-совместимого шахматного движка
# Требуется: python-chess

import chess
import sys
import time
import threading
from collections import defaultdict, namedtuple

INF = 99999999

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Простые PST (примерные), можно расширять
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

TTEntry = namedtuple("TTEntry", ["depth", "flag", "score", "best_move"])
# flag: 'EXACT', 'LOWER', 'UPPER'

def fast_board_key(board: chess.Board):
    """
    Быстрый ключ для TT — tuple из частей позиции.
    """
    # board_fen + turn + castling rights + ep square + halfmove
    return (board.board_fen(), board.turn, board.castling_xfen(), board.ep_square, board.halfmove_clock)

def mvv_lva_score(board, move):
    """
    Простая MVV-LVA: больше значит лучше для сортировки (higher = earlier).
    Выполняется на позиции ДО выполнения хода.
    """
    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim and attacker:
            score += PIECE_VALUES.get(victim.piece_type, 0) * 10 - PIECE_VALUES.get(attacker.piece_type, 0)
    if move.promotion:
        # поощряем промоцию (особенно ферзя)
        score += PIECE_VALUES[chess.QUEEN] // 2
    return score

def evaluate(board: chess.Board):
    """
    Возвращает оценку в центопешках в пользу стороны, которая на ходe (positive = хорошо для side-to-move).
    Простая: материал + PST + мобильность + штраф за шах.
    """
    if board.is_checkmate():
        # если side-to-move мат — очень плохо
        return -INF + 1
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    material = 0
    pst_score = 0

    for piece_type in PIECE_VALUES:
        for sq in board.pieces(piece_type, chess.WHITE):
            material += PIECE_VALUES[piece_type]
            if piece_type in PST:
                pst_score += PST[piece_type][sq]
        for sq in board.pieces(piece_type, chess.BLACK):
            material -= PIECE_VALUES[piece_type]
            if piece_type in PST:
                # mirror для черных
                pst_score -= PST[piece_type][chess.square_mirror(sq)]

    # мобильность (кол-во легальных ходов)
    try:
        mobility_count = board.legal_moves.count()
    except Exception:
        mobility_count = len(list(board.legal_moves))
    mobility = 10 * mobility_count

    check_penalty = -50 if board.is_check() else 0

    score_white = material + pst_score + mobility + check_penalty
    # возвращаем оценку с точки зрения side-to-move
    return score_white if board.turn == chess.WHITE else -score_white

class SearchState:
    def __init__(self):
        self.tt = {}  # key -> TTEntry
        self.nodes = 0
        self.start_time = 0.0
        self.time_limit = 0.0
        # history: key -> score (for move ordering)
        self.history = defaultdict(int)

class SearchAbort(Exception):
    pass

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

    # используем только легальные захваты
    captures = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    if not captures:
        return alpha
    # сортируем по MVV-LVA (descending: лучшие вначале)
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
        # использование TT
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

    moves = list(board.legal_moves)

    # сортируем: TT move, захваты (MVV-LVA), затем history
    def move_key(mv):
        # если TT есть и mv совпадает с ним — максимум приоритета
        if tt_entry and tt_entry.best_move and mv == tt_entry.best_move:
            return (0, 0, 0)
        cap = 0 if board.is_capture(mv) or mv.promotion else 1
        mvv = -mvv_lva_score(board, mv)
        # отрицательное, потому что более высокий history должен идти раньше
        hist = -state.history[(board.turn, mv.from_square, mv.to_square, mv.promotion)]
        return (cap, mvv, hist)

    moves.sort(key=move_key)

    for move in moves:
        if stop_event.is_set():
            raise SearchAbort()

        mover = board.turn
        board.push(move)
        try:
            score = -negamax(board, depth - 1, -beta, -alpha, state, stop_event)
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score
            # обновляем history для не-захвата
            if not board.is_capture(move) and not move.promotion:
                state.history[(mover, move.from_square, move.to_square, move.promotion)] += 2 ** depth

        if alpha >= beta:
            # бета-отсечка: усиливаем историю
            state.history[(mover, move.from_square, move.to_square, move.promotion)] += 2 ** depth
            break

    # сохраняем в TT
    if best_score >= beta_orig:
        flag = 'LOWER'
    elif best_score <= alpha_orig:
        flag = 'UPPER'
    else:
        flag = 'EXACT'

    state.tt[key] = TTEntry(depth=depth, flag=flag, score=best_score, best_move=best_move)
    return best_score

def extract_pv(state: SearchState, board: chess.Board, max_depth: int):
    """
    Собирает PV (principal variation) используя TT, начиная с позиции board.
    Ограничиваем глубину по max_depth.
    """
    pv = []
    b = board.copy()
    for _ in range(max_depth):
        key = fast_board_key(b)
        entry = state.tt.get(key)
        if not entry or not entry.best_move:
            break
        mv = entry.best_move
        pv.append(mv)
        if mv not in b.legal_moves:
            break
        b.push(mv)
    return pv

class SearchThread(threading.Thread):
    def __init__(self, root_board: chess.Board, wtime=None, btime=None, winc=0, binc=0, movetime=None, max_depth=None, stop_event=None):
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

    def time_remaining_ms(self):
        # определяем лимит времени для поиска (в мс)
        if self.movetime:
            return self.movetime
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

                # корневая сортировка: TT-ход в приоритете, затем MVV-LVA
                moves = list(self.root_board.legal_moves)
                root_key = fast_board_key(self.root_board)
                root_tt = self.state.tt.get(root_key)

                def root_key_fn(mv):
                    if root_tt and root_tt.best_move and mv == root_tt.best_move:
                        return (0, 0)
                    cap = 0 if self.root_board.is_capture(mv) or mv.promotion else 1
                    mvv = -mvv_lva_score(self.root_board, mv)
                    return (cap, mvv)

                moves.sort(key=root_key_fn)

                best_for_depth = None
                best_score_for_depth = -INF

                for mv in moves:
                    if self.stop_event.is_set():
                        break

                    self.root_board.push(mv)
                    try:
                        score = -negamax(self.root_board, depth - 1, -INF, INF, self.state, self.stop_event)
                    except SearchAbort:
                        # прерывание — вернёмся к корню и остановим итерацию
                        self.root_board.pop()
                        raise
                    finally:
                        # нормальное одинарное восстановление позиции
                        if self.root_board.move_stack:
                            self.root_board.pop()

                    if score > best_score_for_depth:
                        best_score_for_depth = score
                        best_for_depth = mv

                    if (time.time() - self.state.start_time) > self.state.time_limit:
                        break

                # если нашли лучший ход на данном уровне — сохраним
                if best_for_depth is not None:
                    self.best_move = best_for_depth
                    self.best_score = best_score_for_depth
                    elapsed = time.time() - self.state.start_time
                    nps = int(self.state.nodes / elapsed) if elapsed > 0 else 0
                    pv_moves = extract_pv(self.state, self.root_board, depth)
                    pv_str = " ".join(m.uci() for m in pv_moves) if pv_moves else self.best_move.uci()
                    # выводим info UCI
                    print(f"info depth {depth} score cp {best_score_for_depth} time {int(elapsed*1000)} nodes {self.state.nodes} nps {nps} pv {pv_str}")
                    sys.stdout.flush()

                if (time.time() - self.state.start_time) > self.state.time_limit:
                    break
                depth += 1

        except SearchAbort:
            # ожидаемо — просто выходим
            pass
        except Exception as e:
            print("Search error:", e, file=sys.stderr)
            sys.stderr.flush()

def uci_loop():
    board = chess.Board()
    search_thread = None
    stop_event = threading.Event()

    # announce engine once at start (name/author по пожеланию)
    print("id name Okval")
    print("id author Classic")
    # можно добавить опции здесь (например, Hash, Threads) если реализовать соответствующую логику
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
                print("id name Okval")
                print("id author Classic")
                print("uciok")
                sys.stdout.flush()

            elif cmd == "isready":
                print("readyok")
                sys.stdout.flush()

            elif cmd == "ucinewgame":
                board = chess.Board()

            elif cmd == "position":
                # position [fen <fen> | startpos ] moves <moves>...
                idx = 1
                if len(parts) >= 2 and parts[1] == "startpos":
                    board = chess.Board()
                    idx = 2
                elif len(parts) >= 2 and parts[1] == "fen":
                    # fen состоит из 6 полей
                    if len(parts) >= 8:
                        fen = " ".join(parts[2:8])
                        try:
                            board = chess.Board(fen)
                        except Exception:
                            # если fen неверен — оставляем текущую доску
                            board = chess.Board()
                        idx = 8
                    else:
                        idx = len(parts)

                # moves
                if idx < len(parts) and parts[idx] == "moves":
                    for mv in parts[idx+1:]:
                        try:
                            board.push_uci(mv)
                        except Exception:
                            # пропускаем некорректные ходы
                            pass

            elif cmd == "go":
                # parse time control args
                wtime = btime = winc = binc = movetime = None
                depth = None
                i = 1
                while i < len(parts):
                    if parts[i] == "wtime" and i + 1 < len(parts):
                        try:
                            wtime = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "btime" and i + 1 < len(parts):
                        try:
                            btime = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "winc" and i + 1 < len(parts):
                        try:
                            winc = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "binc" and i + 1 < len(parts):
                        try:
                            binc = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "movetime" and i + 1 < len(parts):
                        try:
                            movetime = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "depth" and i + 1 < len(parts):
                        try:
                            depth = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    else:
                        i += 1

                # остановим предыдущий поисковый поток (если есть)
                if search_thread and search_thread.is_alive():
                    stop_event.set()
                    search_thread.join(timeout=1.0)
                    stop_event.clear()

                stop_event = threading.Event()
                search_thread = SearchThread(board, wtime=wtime, btime=btime, winc=winc or 0, binc=binc or 0, movetime=movetime, max_depth=depth, stop_event=stop_event)
                search_thread.start()

                # ждём завершения потока (он сам остановится по time_limit или по depth)
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
                # неизвестные команды — игнорируем
                pass

        except Exception as e:
            print("error:", e, file=sys.stderr)
            sys.stderr.flush()
            break

if __name__ == "__main__":
    uci_loop()
