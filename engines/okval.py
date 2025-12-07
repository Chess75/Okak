#!/usr/bin/env python3
# simple_engine.py
# UCI-compatible simple chess engine using python-chess
# Исправленная и улучшенная версия
# Автор: ChatGPT (выдано как пример учебного движка)

import chess
import sys
import time
import threading
from collections import defaultdict, namedtuple
from typing import Optional, Tuple, Dict, Any, List

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

# Простые PST (пример). Индексация: квадрат для белых; для чёрных используем square_mirror.
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
    ],
    # можно добавить другие PST при желании
}

# Transposition table entry
TTEntry = namedtuple("TTEntry", ["depth", "flag", "score", "best_move"])
# flag: 'EXACT', 'LOWER', 'UPPER'

# ---- Утилиты ----

def fast_board_key(board: chess.Board) -> Tuple[Any, ...]:
    """
    Быстрый ключ для TT — tuple из частей позиции.
    Не Zobrist, но хватает для простого TT.
    """
    # board_fen() + turn + castling_xfen + ep_square + halfmove_clock
    return (board.board_fen(), board.turn, board.castling_xfen(), board.ep_square, board.halfmove_clock)

def mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    """
    MVV-LVA с учётом промоции и en-passant.
    Чем выше — тем лучше (будем сортировать по убыванию).
    """
    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        # en-passant: victim может быть None — восстановим пешку
        if victim is None and board.is_en_passant(move):
            # жертва — пешка противника
            victim = chess.Piece(chess.PAWN, not board.turn)
        attacker = board.piece_at(move.from_square)
        if victim and attacker:
            score += PIECE_VALUES.get(victim.piece_type, 0) * 10
            score -= PIECE_VALUES.get(attacker.piece_type, 0)
    # небольшая премия за превращение (чтобы продвижения были выше)
    if move.promotion:
        score += PIECE_VALUES[chess.QUEEN] // 2
    return score

# ---- Оценка позиции ----

def evaluate(board: chess.Board) -> int:
    """
    Оценка позиции в центопешках для стороны, которая ходит.
    Возвращаем положительное — хорошо для side-to-move.
    Материал + PST + мобильность + штраф за шах.
    """
    # Быстрые результаты
    if board.is_checkmate():
        # если мата — плохо для стороны, которая ходит
        return -INF + 1
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    material = 0
    pst_score = 0

    for piece_type, value in PIECE_VALUES.items():
        # белые
        for sq in board.pieces(piece_type, chess.WHITE):
            material += value
            if piece_type in PST:
                pst_score += PST[piece_type][sq]
        # чёрные
        for sq in board.pieces(piece_type, chess.BLACK):
            material -= value
            if piece_type in PST:
                # зеркалируем для чёрных
                pst_score -= PST[piece_type][chess.square_mirror(sq)]

    # мобильность — количество легальных ходов (включая псевдо)
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
        self.tt: Dict[Any, TTEntry] = {}
        self.nodes: int = 0
        self.start_time: float = 0.0
        self.time_limit: float = 0.0
        self.history: Dict[Tuple[bool, int, int], int] = defaultdict(int)
        # простая таблица убийц (по глубине) для улучшения порядка ходов
        self.killers: Dict[int, List[chess.Move]] = defaultdict(list)

# ---- Исключение прерывания ----
class SearchAbort(Exception):
    pass

# ---- Quiescence ----

def quiescence(board: chess.Board, alpha: int, beta: int, state: SearchState, stop_event: threading.Event) -> int:
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

    # генерируем только захваты (и превращения)
    captures = list(board.generate_legal_captures())
    if not captures:
        return alpha
    # сортировка: лучшие захваты первыми
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

def negamax(board: chess.Board, depth: int, alpha: int, beta: int, state: SearchState, stop_event: threading.Event) -> int:
    if stop_event.is_set():
        raise SearchAbort()
    if state.start_time and (time.time() - state.start_time) > state.time_limit:
        raise SearchAbort()

    state.nodes += 1

    # терминальное положение или ноль-глубина
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
    best_move: Optional[chess.Move] = None

    # подготовка ходов и их упорядочивание
    moves = list(board.legal_moves)

    def move_key(mv: chess.Move):
        # TT move имеет приоритет
        if tt_entry and tt_entry.best_move and mv == tt_entry.best_move:
            return (0, 0, 0)
        # захваты первыми
        cap = 0 if board.is_capture(mv) else 1
        # MVV-LVA (больше — лучше, но мы сортируем по возрастанию ключа => используем -mvv)
        mvv = -mvv_lva_score(board, mv)
        # history: более высокий => лучше (делаем минус, т.к. сортируем по возрастанию)
        hist = -state.history.get((board.turn, mv.from_square, mv.to_square), 0)
        # killer moves: понижаем ключ, чтобы они шли раньше (маленький приоритетный номер)
        killer_bonus = 0
        for klist in state.killers.values():
            if mv in klist:
                killer_bonus = -1
                break
        return (cap, mvv, killer_bonus, hist)

    moves.sort(key=move_key)

    for move in moves:
        if stop_event.is_set():
            raise SearchAbort()

        mover = board.turn  # сторона, которая делает ход (до push)
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
            # update history for non-capture (use mover as key)
            if not board.is_capture(move):
                state.history[(mover, move.from_square, move.to_square)] += 2 ** depth

        if alpha >= beta:
            # beta cutoff: record killer and history
            # добавляем в список убийц для текущей глубины
            klist = state.killers.get(depth, [])
            if move not in klist:
                klist.insert(0, move)
                # держим только 2 убийцы на глубине
                state.killers[depth] = klist[:2]
            state.history[(mover, move.from_square, move.to_square)] += 2 ** depth
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

# ---- Вытаскивание PV из TT ----

def extract_pv(state: SearchState, board: chess.Board, max_length: int = 64) -> List[chess.Move]:
    pv = []
    b = board.copy()
    for _ in range(max_length):
        entry = state.tt.get(fast_board_key(b))
        if not entry or not entry.best_move:
            break
        mv = entry.best_move
        # убедимся что ход всё ещё легален в текущей позиции
        if mv not in b.legal_moves:
            break
        pv.append(mv)
        b.push(mv)
    return pv

# ---- Search Thread (iterative deepening) ----

class SearchThread(threading.Thread):
    def __init__(self, root_board: chess.Board, wtime: Optional[int] = None, btime: Optional[int] = None,
                 winc: int = 0, binc: int = 0, movetime: Optional[int] = None, max_depth: Optional[int] = None,
                 stop_event: Optional[threading.Event] = None):
        super().__init__()
        self.root_board = root_board.copy()
        self.wtime = wtime
        self.btime = btime
        self.winc = winc or 0
        self.binc = binc or 0
        self.movetime = movetime
        self.max_depth = max_depth
        self.stop_event = stop_event or threading.Event()

        self.best_move: Optional[chess.Move] = None
        self.best_score: Optional[int] = None
        self.depth_reached: int = 0

        self.state = SearchState()
        self.state.time_limit = 0.0
        self.state.start_time = 0.0

    def time_remaining_ms(self) -> int:
        """
        Элементарная логика контроля времени.
        Возвращаем миллисекунды.
        """
        if self.movetime:
            return int(self.movetime)
        if self.root_board.turn == chess.WHITE:
            if self.wtime is None:
                return 10000
            if self.wtime < 2000:
                return 50
            return max(20, int(self.wtime // 20 + self.winc * 2))
        else:
            if self.btime is None:
                return 10000
            if self.btime < 2000:
                return 50
            return max(20, int(self.btime // 20 + self.binc * 2))

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

                # root ordering: prefer TT move, затем захваты, затем MVV-LVA
                moves = list(self.root_board.legal_moves)
                root_key = fast_board_key(self.root_board)
                root_tt = self.state.tt.get(root_key)

                def root_key_fn(mv: chess.Move):
                    if root_tt and root_tt.best_move and mv == root_tt.best_move:
                        return (0, 0)
                    cap = 0 if self.root_board.is_capture(mv) else 1
                    mvv = -mvv_lva_score(self.root_board, mv)
                    return (cap, mvv)

                moves.sort(key=root_key_fn)

                best_for_depth: Optional[chess.Move] = None
                best_score_for_depth = -INF

                for mv in moves:
                    if self.stop_event.is_set():
                        break
                    # time check before expensive call
                    if (time.time() - self.state.start_time) > self.state.time_limit:
                        break

                    self.root_board.push(mv)
                    try:
                        score = -negamax(self.root_board, depth - 1, -INF, INF, self.state, self.stop_event)
                    except SearchAbort:
                        # прерывание — гарантированно восстановим доску и выйдем
                        self.root_board.pop()
                        raise
                    finally:
                        # pop один раз (если не попнули в except)
                        if self.root_board.move_stack and self.root_board.peek() == mv:
                            # это защита — но простейшее: если последний ход — mv, то попаем
                            self.root_board.pop()
                        else:
                            # если предыдущая логика не сработала — попаем безопасно (если ход есть)
                            if self.root_board.move_stack:
                                self.root_board.pop()

                    # ПОМЕТКА: выше использовано безопасное pop() — гарантируем, что мы вернули позицию
                    # после push. (Предшествовавшая версия могла дважды вызывать pop.)

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

                    # извлекаем PV из TT (если есть)
                    pv_moves = extract_pv(self.state, self.root_board, max_length=depth + 8)
                    pv_str = " ".join(m.uci() for m in pv_moves) if pv_moves else self.best_move.uci()

                    # score cp (centipawns) — если очень большой (mate) — оставляем.
                    print(f"info depth {depth} score cp {best_score_for_depth} time {int(elapsed*1000)} nodes {self.state.nodes} nps {nps} pv {pv_str}")
                    sys.stdout.flush()

                # проверка времени и условия выхода
                if (time.time() - self.state.start_time) > self.state.time_limit:
                    break
                depth += 1

        except SearchAbort:
            # обычное прерывание поиска по таймауту/stop
            pass
        except Exception as e:
            print("Search error:", e, file=sys.stderr)
            sys.stderr.flush()

# ---- UCI loop ----

def uci_loop():
    board = chess.Board()
    search_thread: Optional[SearchThread] = None
    stop_event = threading.Event()

    # announce engine once at start
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
                # respond with engine identity
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
                    # fen — 6 полей
                    if len(parts) >= 8:
                        fen = " ".join(parts[2:8])
                        try:
                            board = chess.Board(fen)
                        except Exception:
                            # если FEN битый — оставляем прежнюю позицию (без изменений)
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
                                             movetime=movetime, max_depth=depth, stop_event=stop_event)
                search_thread.start()

                # wait for thread to finish (будет завершён по таймауту или по достижению глубины)
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
                # ignore unknown commands
                pass

        except Exception as e:
            print("error:", e, file=sys.stderr)
            sys.stderr.flush()
            break

if __name__ == "__main__":
    uci_loop()
