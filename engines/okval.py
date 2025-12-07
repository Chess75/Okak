#!/usr/bin/env python3
# simple_engine.py
# UCI-compatible simple chess engine using python-chess
# Исправленная и улучшенная версия (фикс double-pop, корректные TT-флаги,
# улучшенное упорядочивание ходов, безопасный push/pop, улучшенный тайм-менеджмент)

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

# Небольшие PST для примера
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
# best_move — UCI string или None (надёжно сравнивать между досками)
TTEntry = namedtuple("TTEntry", ["depth", "flag", "score", "best_move"])
# flag: 'EXACT', 'LOWER', 'UPPER'

# ---- Утилиты ----

def fast_board_key(board: chess.Board):
    """
    Быстрый ключ для TT — tuple из частей позиции (быстрее, чем fen()).
    Не идеален (не Zobrist), но достаточно для простого TT.
    """
    return (board.board_fen(), board.turn, board.castling_xfen(), board.ep_square, board.halfmove_clock)

def mvv_lva_score(board, move):
    """
    Простая MVV-LVA: (victim_value * 10 - attacker_value) + премия за промоцию.
    Чем больше — тем раньше.
    """
    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim and attacker:
            score += PIECE_VALUES.get(victim.piece_type, 0) * 10 - PIECE_VALUES.get(attacker.piece_type, 0)
    if move.promotion:
        # Promote to queen is best
        score += PIECE_VALUES[chess.QUEEN] // 2
    return score

# ---- Оценка позиции ----

def evaluate(board: chess.Board):
    """
    Оценка в сотых шахматной единицы (centipawns).
    Возвращает оценку с точки зрения стороны, которая ходит (положительно — хорошо для side-to-move).
    """
    if board.is_checkmate():
        # если мат — ужас для стороны, которой сейчас ходить
        return -INF + 1
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    material = 0
    pst_score = 0

    # material & PST (белая перспектива)
    for piece_type in PIECE_VALUES:
        for sq in board.pieces(piece_type, chess.WHITE):
            material += PIECE_VALUES[piece_type]
            if piece_type in PST:
                pst_score += PST[piece_type][sq]
        for sq in board.pieces(piece_type, chess.BLACK):
            material -= PIECE_VALUES[piece_type]
            if piece_type in PST:
                pst_score -= PST[piece_type][chess.square_mirror(sq)]

    # mobility: количество легальных ходов (для стороны, которая ходит)
    try:
        mobility_count = board.legal_moves.count()
    except Exception:
        mobility_count = len(list(board.legal_moves))
    mobility = 10 * mobility_count

    # штраф/бонус за шах
    check_bonus = -50 if board.is_check() else 0

    score_white = material + pst_score + mobility + check_bonus
    return score_white if board.turn == chess.WHITE else -score_white

# ---- Транспозиционная таблица и счётчики ----
class SearchState:
    def __init__(self):
        self.tt = {}  # maps key -> TTEntry
        self.nodes = 0
        self.start_time = 0.0
        self.time_limit = 0.0
        # простой history heuristic: ключ (color, from, to) -> score
        self.history = defaultdict(int)

# ---- Исключение для прерывания поиска ----
class SearchAbort(Exception):
    pass

# ---- Кви-Поиск ----

def quiescence(board: chess.Board, alpha: int, beta: int, state: SearchState, stop_event: threading.Event):
    """
    Кви-поиск: stand-pat и рассмотрение только захватов (MVV-LVA порядок).
    """
    if stop_event.is_set():
        raise SearchAbort()

    # тайм-аут
    if state.start_time and (time.time() - state.start_time) > state.time_limit:
        raise SearchAbort()

    state.nodes += 1
    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    # generate capture moves and sort by MVV-LVA
    captures = [m for m in board.legal_moves if board.is_capture(m)]
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
            # гарантированно снимаем ход — единственный pop для соответствующего push
            board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha

# ---- Negamax с альфа-бета, TT и упорядочиванием ходов ----

def negamax(board: chess.Board, depth: int, alpha: int, beta: int, state: SearchState, stop_event: threading.Event):
    """
    Negamax с альфа-бета, использование TT.
    Возвращает оценку с точки зрения стороны, которая ходит.
    """
    if stop_event.is_set():
        raise SearchAbort()

    # тайм-аут
    if state.start_time and (time.time() - state.start_time) > state.time_limit:
        raise SearchAbort()

    state.nodes += 1

    # terminal
    if depth == 0:
        return quiescence(board, alpha, beta, state, stop_event)

    key = fast_board_key(board)
    tt_entry = state.tt.get(key)
    if tt_entry and tt_entry.depth >= depth:
        # TT содержит оценку, можно использовать
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

    # Генерация ходов и порядок:
    # 1) TT move
    # 2) captures sorted by MVV-LVA
    # 3) промоции (частично покрыты MVV-LVA)
    # 4) others, отсортированные по history heuristic
    moves = list(board.legal_moves)

    # Сборка ключей сортировки
    def move_key(mv):
        # TT-ход — самый высокий приоритет (сравниваем по UCI)
        if tt_entry and tt_entry.best_move and mv.uci() == tt_entry.best_move:
            return (0, 0, 0)
        # captures: primary by -MVV_LVA so larger score goes first
        cap = 0 if board.is_capture(mv) else 1
        mvv = -mvv_lva_score(board, mv)
        # history heuristic (больше -> раньше) — используем текущее значение из state.history
        hist = -state.history[(board.turn, mv.from_square, mv.to_square)]
        return (cap, mvv, hist)

    moves.sort(key=move_key)

    for move in moves:
        if stop_event.is_set():
            raise SearchAbort()

        mover = board.turn  # кто делает этот ход (до push)
        board.push(move)
        try:
            score = -negamax(board, depth - 1, -beta, -alpha, state, stop_event)
        finally:
            # гарантированно снимаем ход — единственный pop для соответствующего push
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score
            # update history heuristic for non-capture moves (captures руководствуются MVV-LVA)
            if not board.is_capture(move):
                # ключ — цвет, который совершил ход
                state.history[(mover, move.from_square, move.to_square)] += 2 ** depth

        if alpha >= beta:
            # beta-cutoff: запомним ход в history для ускорения порядка
            state.history[(mover, move.from_square, move.to_square)] += 2 ** depth
            break

    # вычисление флага TT корректно относительно исходных alpha_orig/beta_orig
    if best_score >= beta_orig:
        flag = 'LOWER'   # занял или превысил beta => lower bound
    elif best_score <= alpha_orig:
        flag = 'UPPER'   # не смог превысить alpha => upper bound
    else:
        flag = 'EXACT'

    # Сохраним entry. best_move сохраняем как UCI (строку) или None
    best_move_uci = best_move.uci() if best_move is not None else None
    state.tt[key] = TTEntry(depth=depth, flag=flag, score=best_score, best_move=best_move_uci)
    return best_score

# ---- Поисковый поток с итеративным углублением ----

class SearchThread(threading.Thread):
    """
    Поток поиска с итеративным углублением и тайм-аутом.
    Повторно использует одну структуру SearchState (TT, history).
    """
    def __init__(self, root_board: chess.Board, wtime=None, btime=None, winc=0, binc=0, movetime=None, max_depth=None, stop_event=None):
        super().__init__()
        # делаем копию — чтобы внешний board можно было безопасно изменять
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

    def time_remaining_ms(self):
        """
        Простая логика распределения времени:
        - Если movetime задан — использовать его.
        - Иначе делим оставшееся время на некоторое число ходов (агрессивно).
        """
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

                # корневое упорядочивание: используем TT.best_move сначала, затем MVV-LVA
                moves = list(self.root_board.legal_moves)
                # если в TT есть best_move для root position, ставим его первым
                root_key = fast_board_key(self.root_board)
                root_tt = self.state.tt.get(root_key)

                def root_key_fn(mv):
                    if root_tt and root_tt.best_move and mv.uci() == root_tt.best_move:
                        return (0, 0)
                    # captures first by MVV-LVA
                    cap = 0 if self.root_board.is_capture(mv) else 1
                    mvv = -mvv_lva_score(self.root_board, mv)
                    return (cap, mvv)

                moves.sort(key=root_key_fn)

                best_for_depth = None
                best_score_for_depth = -INF

                for mv in moves:
                    if self.stop_event.is_set():
                        break
                    # push once per move — pop в finally (гарантированно один pop)
                    self.root_board.push(mv)
                    try:
                        # поиск на depth-1; границы широкие, используем negamax
                        score = -negamax(self.root_board, depth - 1, -INF, INF, self.state, self.stop_event)
                    except SearchAbort:
                        # при прерывании просто выйти наружу (доска будет восстановлена в finally)
                        # не делаем дополнительных pop здесь (finally сделает единственный pop)
                        raise
                    finally:
                        # гарантированно снимаем ход — единственный pop для соответствующего push
                        self.root_board.pop()

                    if score > best_score_for_depth:
                        best_score_for_depth = score
                        best_for_depth = mv

                    # тайм-чек между ходами
                    if (time.time() - self.state.start_time) > self.state.time_limit:
                        break

                # сохранить лучший найденный ход
                if best_for_depth is not None:
                    self.best_move = best_for_depth
                    self.best_score = best_score_for_depth
                    elapsed = time.time() - self.state.start_time
                    nps = int(self.state.nodes / elapsed) if elapsed > 0 else 0
                    # UCI-style info (pv как минимум первый ход)
                    try:
                        pv_str = self.best_move.uci()
                    except Exception:
                        pv_str = "-"
                    print(f"info depth {depth} score cp {best_score_for_depth} time {int(elapsed*1000)} nodes {self.state.nodes} nps {nps} pv {pv_str}")
                    sys.stdout.flush()

                # стоп по таймауту
                if (time.time() - self.state.start_time) > self.state.time_limit:
                    break

                depth += 1

        except SearchAbort:
            # корректное окончание поиска (тайм-аут или stop_event)
            pass
        except Exception as e:
            # логируем исключение, не даём упасть процессу
            print("Search error:", e, file=sys.stderr)
            sys.stderr.flush()

# ---- UCI loop ----

def uci_loop():
    board = chess.Board()
    search_thread = None
    stop_event = threading.Event()
    print("id name Okval")
    print("id author Classic")
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
                # position [fen <fen> | startpos ]  moves <moves>...
                # более устойчивый парсинг
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
                            board = chess.Board()  # fallback
                        idx = 8
                    else:
                        # некорректный fen -> ignore
                        pass
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

                # stop previous search if alive
                if search_thread and search_thread.is_alive():
                    stop_event.set()
                    search_thread.join(timeout=1.0)
                    stop_event.clear()

                stop_event = threading.Event()
                # создаём поток с текущей позицией (копируется внутри)
                search_thread = SearchThread(board, wtime=wtime, btime=btime, winc=winc or 0, binc=binc or 0, movetime=movetime, max_depth=depth, stop_event=stop_event)
                search_thread.start()

                # ждать завершения потока — но не блокируем бесконечно
                while search_thread.is_alive():
                    time.sleep(0.05)

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
                # игнорируем неизвестные команды
                pass

        except Exception as e:
            print("error:", e, file=sys.stderr)
            sys.stderr.flush()
            break

if __name__ == "__main__":
    uci_loop()
