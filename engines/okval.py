#!/usr/bin/env python3
# simple_engine_fixed.py
# Исправленная версия простого UCI-совместимого шахматного движка на python-chess
# NOTE: requires python-chess (pip install python-chess)

import chess
import sys
import time
import threading
from collections import defaultdict, namedtuple

# ---- Константы ----
INF = 99999999

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Небольшая PST (пример) — можно расширять/тонить позже
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
    Быстрый (но не Zobrist) ключ для TT.
    """
    return (board.board_fen(), board.turn, board.castling_xfen(), board.ep_square, board.halfmove_clock)

def mvv_lva_score(board, move):
    """
    Простая MVV-LVA оценка: (victim_value * 10 - attacker_value) + премия за промоцию.
    Чем больше — тем более приоритетный ход (по убыванию сортируем).
    """
    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim and attacker:
            score += PIECE_VALUES.get(victim.piece_type, 0) * 10 - PIECE_VALUES.get(attacker.piece_type, 0)
    if move.promotion:
        score += PIECE_VALUES[chess.QUEEN] // 2
    return score

# ---- Оценка позиции ----

def evaluate(board: chess.Board):
    """
    Возвращает оценку в центопешках для стороны, которая ходит (положительно — хорошо для side-to-move).
    Простая: материал + PST + мобильность + штраф за шах.
    """
    # Быстрая разборка матов/патов
    if board.is_checkmate():
        # если side-to-move оказался под матом, оценка очень плохая
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
                # зеркалим для черных
                pst_score -= PST[piece_type][chess.square_mirror(sq)]

    # мобильность: количество легальных ходов (простой подход)
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

    # только захватывающие ходы (и промоции)
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

    # подготовка ходов и сортировка
    moves = list(board.legal_moves)

    def move_key(mv):
        # TT-ход приоритет
        if tt_entry and tt_entry.best_move and mv == tt_entry.best_move:
            return (0, 0, 0)
        cap = 0 if board.is_capture(mv) else 1
        # MVV-LVA: большие значения должны идти раньше -> сортируем по -mvv
        mvv = -mvv_lva_score(board, mv)
        # history: большее значение значит более успешный ход в прошлом
        hist = -state.history[(board.turn, mv.from_square, mv.to_square)]
        return (cap, mvv, hist)

    moves.sort(key=move_key)

    for move in moves:
        if stop_event.is_set():
            raise SearchAbort()

        mover = board.turn  # сторона, которая делает этот ход (до push)
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
            # обновляем history только для некрупных жертв (non-capture)
            if not board.is_capture(move):
                state.history[(mover, move.from_square, move.to_square)] += 2 ** depth

        if alpha >= beta:
            # beta cutoff: усиленная запись в history
            state.history[(mover, move.from_square, move.to_square)] += 2 ** depth
            break

    # определяем флаг TT
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
    def __init__(self, root_board: chess.Board, wtime=None, btime=None, winc=0, binc=0, movetime=None, max_depth=None, stop_event=None):
        super().__init__()
        # сохраняем ссылку на исходную позицию, но поиски делаем на копии в run()
        self.root_board = root_board
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
        # используем копию позиции для поиска — защищаем UCI-loop от гонок
        search_root = self.root_board.copy()
        ms = self.time_remaining_ms()
        self.state.time_limit = ms / 1000.0 if ms is not None else 5.0
        self.state.start_time = time.time()
        depth = 1

        try:
            while not self.stop_event.is_set():
                if self.max_depth and depth > self.max_depth:
                    break
                self.depth_reached = depth

                # получаем список корневых ходов и сортируем
                moves = list(search_root.legal_moves)
                root_key = fast_board_key(search_root)
                root_tt = self.state.tt.get(root_key)

                def root_key_fn(mv):
                    if root_tt and root_tt.best_move and mv == root_tt.best_move:
                        return (0, 0)
                    cap = 0 if search_root.is_capture(mv) else 1
                    mvv = -mvv_lva_score(search_root, mv)
                    return (cap, mvv)

                moves.sort(key=root_key_fn)

                best_for_depth = None
                best_score_for_depth = -INF

                for mv in moves:
                    if self.stop_event.is_set():
                        break

                    # push на локальной копии
                    search_root.push(mv)
                    try:
                        score = -negamax(search_root, depth - 1, -INF, INF, self.state, self.stop_event)
                    except SearchAbort:
                        # при прерывании просто восстановим позицию и прервем внешний цикл
                        search_root.pop()
                        raise
                    finally:
                        # гарантированно попаем корневой ход (если он ещё там)
                        if search_root.move_stack:
                            search_root.pop()

                    if score > best_score_for_depth:
                        best_score_for_depth = score
                        best_for_depth = mv

                    # проверка времени
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
                    # вывод информации в UCI-формате (info ...)
                    print(f"info depth {depth} score cp {best_score_for_depth} time {int(elapsed*1000)} nodes {self.state.nodes} nps {nps} pv {pv_str}")
                    sys.stdout.flush()

                # остановились по времени?
                if (time.time() - self.state.start_time) > self.state.time_limit:
                    break
                depth += 1

        except SearchAbort:
            # остановка по внешнему событию — тихо выходим
            pass
        except Exception as e:
            # лог ошибок в stderr
            print("Search error:", e, file=sys.stderr)
            sys.stderr.flush()

# ---- UCI loop ----

def uci_loop():
    board = chess.Board()
    search_thread = None
    stop_event = threading.Event()

    # announce engine once at start
    print("id name Okval")
    print("id author Classic")
    # Поддерживаем опции? (минимально)
    print("option name Hash type spin default 0 min 0 max 4096")
    print("option name Threads type spin default 1 min 1 max 128")
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
                print("option name Hash type spin default 0 min 0 max 4096")
                print("option name Threads type spin default 1 min 1 max 128")
                print("uciok")
                sys.stdout.flush()

            elif cmd == "isready":
                print("readyok")
                sys.stdout.flush()

            elif cmd == "ucinewgame":
                board = chess.Board()

            elif cmd == "position":
                # position [fen <fen> | startpos ] moves <moves>...
                # robust parsing
                # Возможные формы:
                #  position startpos
                #  position startpos moves e2e4 e7e5 ...
                #  position fen <FEN>
                #  position fen <FEN> moves ...
                idx = 1
                if len(parts) >= 2 and parts[1] == "startpos":
                    board = chess.Board()
                    idx = 2
                elif len(parts) >= 2 and parts[1] == "fen":
                    # fen состоит ровно из 6 полей
                    if len(parts) >= 8:
                        fen = " ".join(parts[2:8])
                        try:
                            board = chess.Board(fen)
                        except Exception:
                            # если некорректная FEN — оставляем доску прежней (без изменений)
                            board = chess.Board()
                        idx = 8
                    else:
                        # неверная команда — игнорируем
                        idx = len(parts)

                # moves (если есть)
                if idx < len(parts) and parts[idx] == "moves":
                    # применяем последовательность ходов (после startpos/fen)
                    for mv in parts[idx+1:]:
                        try:
                            # push_uci проверяет корректность хода и бросает исключение если неверно
                            board.push_uci(mv)
                        except Exception:
                            # пропускаем некорректный ход
                            pass

            elif cmd == "go":
                # parse time control args
                wtime = btime = winc = binc = movetime = None
                depth = None
                i = 1
                while i < len(parts):
                    if parts[i] == "wtime" and i+1 < len(parts):
                        try:
                            wtime = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "btime" and i+1 < len(parts):
                        try:
                            btime = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "winc" and i+1 < len(parts):
                        try:
                            winc = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "binc" and i+1 < len(parts):
                        try:
                            binc = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "movetime" and i+1 < len(parts):
                        try:
                            movetime = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "depth" and i+1 < len(parts):
                        try:
                            depth = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    else:
                        i += 1

                # stop previous search thread if жив
                if search_thread and search_thread.is_alive():
                    stop_event.set()
                    search_thread.join(timeout=1.0)
                    stop_event.clear()

                # создаем новый stop_event специально для потока
                stop_event = threading.Event()
                search_thread = SearchThread(board.copy(), wtime=wtime, btime=btime, winc=winc or 0, binc=binc or 0, movetime=movetime, max_depth=depth, stop_event=stop_event)
                search_thread.start()

                # Ожидаем завершения (поток остановится по времени или досрочно при stop)
                while search_thread.is_alive():
                    # дежурная пауза — но при этом можем принять команду "stop" из stdin
                    # поэтому читаем stdin неблокирующе? Для простоты — sleep короткий и loop
                    time.sleep(0.05)

                # после завершения — выбираем лучший ход
                if search_thread.best_move:
                    best_uci = search_thread.best_move.uci()
                    # Обновляем нашу внутреннюю доску — это помогает, если среда полагается на движок
                    try:
                        board.push_uci(best_uci)
                    except Exception:
                        # если push_uci не проходит — просто игнорируем (GUI должен принять ход)
                        pass
                    print(f"bestmove {best_uci}")
                    sys.stdout.flush()
                else:
                    # fallback: любой легальный ход
                    try:
                        fallback = next(iter(board.legal_moves))
                        # применяем локально
                        try:
                            board.push(fallback)
                        except Exception:
                            pass
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
                    try:
                        print(f"bestmove {search_thread.best_move.uci()}")
                    except Exception:
                        print("bestmove 0000")
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
                # игнорируем неизвестные команды
                pass

        except Exception as e:
            print("error:", e, file=sys.stderr)
            sys.stderr.flush()
            break

if __name__ == "__main__":
    uci_loop()
