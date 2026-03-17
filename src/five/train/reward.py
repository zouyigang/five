from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from five.common.config import RewardConfig
from five.core.board import Board
from five.core.move import Move
from five.core.rules import DIRECTIONS, in_bounds


@dataclass(slots=True)
class LineInfo:
    count: int
    open_ends: int
    is_blocked: bool
    has_jump: bool
    jump_count: int
    # 活三/跳活三时：两端中「再延一步仍是空」的端点数，0 表示两端都只能成冲四（restricted）
    extendable_ends: int = 0


@dataclass(slots=True)
class ThreatInfo:
    living_fours: int
    living_threes: int
    blocked_fours: int
    blocked_threes: int
    jump_living_fours: int
    jump_living_threes: int
    jump_blocked_fours: int
    winning_moves: int
    # 仅能成冲四的活三（连子活三且两端再延一步即边线或敌子；跳活三中间填跳必成活四，无此类）
    restricted_living_threes: int = 0


@dataclass(slots=True)
class RewardDetail:
    amount: float
    reason: str


@dataclass(slots=True)
class RewardResult:
    total_reward: float
    details: list[RewardDetail]
    missed_own_win: bool = False


def analyze_line(
    board: Board,
    move: Move,
    player: int,
    delta_row: int,
    delta_col: int,
) -> LineInfo:
    count = 1
    open_ends = 0
    has_jump = False
    jump_count = 0
    extendable_ends = 0

    for sign in [1, -1]:
        dr, dc = sign * delta_row, sign * delta_col
        row, col = move.row + dr, move.col + dc

        while in_bounds(board.size, row, col):
            cell = board.grid[row, col]
            if cell == player:
                count += 1
                row += dr
                col += dc
            elif cell == 0:
                # Only allow ONE jump total across both directions
                if jump_count == 0:
                    # Check if there's a player piece after this empty cell (jump)
                    next_row, next_col = row + dr, col + dc
                    if in_bounds(board.size, next_row, next_col) and board.grid[next_row, next_col] == player:
                        # This is a valid jump
                        count += 1
                        has_jump = True
                        jump_count += 1
                        row = next_row + dr
                        col = next_col + dc
                        while in_bounds(board.size, row, col) and board.grid[row, col] == player:
                            count += 1
                            row += dr
                            col += dc
                    else:
                        # Not a valid jump, this is an open end; check if one more step is empty (能成活四)
                        open_ends += 1
                        out_r, out_c = row + dr, col + dc
                        if in_bounds(board.size, out_r, out_c) and board.grid[out_r, out_c] == 0:
                            extendable_ends += 1
                        break
                else:
                    # Already has a jump, this is an open end
                    open_ends += 1
                    out_r, out_c = row + dr, col + dc
                    if in_bounds(board.size, out_r, out_c) and board.grid[out_r, out_c] == 0:
                        extendable_ends += 1
                    break
            else:
                # Blocked by opponent
                break

        # 棋盘边界视为被堵，不计为开放端；遇空位或对手时已在循环内处理，此处不再重复计数

    is_blocked = open_ends == 0
    return LineInfo(
        count=count,
        open_ends=open_ends,
        is_blocked=is_blocked,
        has_jump=has_jump,
        jump_count=jump_count,
        extendable_ends=extendable_ends,
    )


def get_threat_info(board: Board, move: Move, player: int) -> ThreatInfo:
    living_fours = 0
    living_threes = 0
    blocked_fours = 0
    blocked_threes = 0
    jump_living_fours = 0
    jump_living_threes = 0
    jump_blocked_fours = 0
    winning_moves = 0
    restricted_living_threes = 0

    for delta_row, delta_col in DIRECTIONS:
        line = analyze_line(board, move, player, delta_row, delta_col)

        if line.count >= 5:
            winning_moves += 1
        elif line.count == 4:
            if line.has_jump:
                if line.open_ends == 2:
                    jump_living_fours += 1
                elif line.open_ends == 1:
                    jump_blocked_fours += 1
            else:
                if line.open_ends == 2:
                    living_fours += 1
                elif line.open_ends == 1:
                    blocked_fours += 1
        elif line.count == 3:
            if line.open_ends == 2:
                # 跳活三：中间填跳必成活四，故不区分 restricted，一律按跳活三计
                restricted = not line.has_jump and line.extendable_ends == 0
                if line.has_jump:
                    jump_living_threes += 1
                elif restricted:
                    restricted_living_threes += 1
                else:
                    living_threes += 1
            elif line.open_ends == 1:
                blocked_threes += 1

    return ThreatInfo(
        living_fours=living_fours,
        living_threes=living_threes,
        blocked_fours=blocked_fours,
        blocked_threes=blocked_threes,
        jump_living_fours=jump_living_fours,
        jump_living_threes=jump_living_threes,
        jump_blocked_fours=jump_blocked_fours,
        winning_moves=winning_moves,
        restricted_living_threes=restricted_living_threes,
    )


def find_winning_moves(board: Board, player: int) -> list[Move]:
    winning_moves = []
    for move in board.legal_moves():
        board.grid[move.row, move.col] = player
        winner = board.check_winner(move)
        board.grid[move.row, move.col] = 0

        if winner == player:
            winning_moves.append(move)

    return winning_moves


def find_living_three_moves(board: Board, player: int) -> list[Move]:
    """Find moves that would create a living three for the player."""
    living_three_moves = []
    for move in board.legal_moves():
        board.grid[move.row, move.col] = player
        threat = get_threat_info(board, move, player)
        board.grid[move.row, move.col] = 0

        if threat.living_threes > 0:
            living_three_moves.append(move)

    return living_three_moves


def find_blocked_four_moves(board: Board, player: int) -> list[Move]:
    """Find moves that would create a blocked four for the player."""
    blocked_four_moves = []
    for move in board.legal_moves():
        board.grid[move.row, move.col] = player
        threat = get_threat_info(board, move, player)
        board.grid[move.row, move.col] = 0

        if threat.blocked_fours > 0:
            blocked_four_moves.append(move)

    return blocked_four_moves


def find_jump_living_three_moves(board: Board, player: int) -> list[Move]:
    """Find moves that would create a jump living three for the player."""
    jump_living_three_moves = []
    for move in board.legal_moves():
        board.grid[move.row, move.col] = player
        threat = get_threat_info(board, move, player)
        board.grid[move.row, move.col] = 0

        if threat.jump_living_threes > 0:
            jump_living_three_moves.append(move)

    return jump_living_three_moves


def find_jump_blocked_four_moves(board: Board, player: int) -> list[Move]:
    """Find moves that would create a jump blocked four for the player."""
    jump_blocked_four_moves = []
    for move in board.legal_moves():
        board.grid[move.row, move.col] = player
        threat = get_threat_info(board, move, player)
        board.grid[move.row, move.col] = 0

        if threat.jump_blocked_fours > 0:
            jump_blocked_four_moves.append(move)

    return jump_blocked_four_moves


def find_jump_living_four_moves(board: Board, player: int) -> list[Move]:
    """Find moves that would create a jump living four for the player."""
    jump_living_four_moves = []
    for move in board.legal_moves():
        board.grid[move.row, move.col] = player
        threat = get_threat_info(board, move, player)
        board.grid[move.row, move.col] = 0

        if threat.jump_living_fours > 0:
            jump_living_four_moves.append(move)

    return jump_living_four_moves


def scan_existing_threats(
    board: Board, player: int
) -> tuple[
    list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]],
    list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]],
    list[tuple[int, int]],
]:
    """
    Scan the board for existing threats including jump threats.
    Returns: (living_four_positions, blocked_four_positions, living_three_positions,
              jump_living_four_positions, jump_blocked_four_positions, jump_living_three_positions,
              restricted_living_three_positions)
    Jump 活三 always has 活四 at the jump cell, so no restricted_jump. Each position is (row, col).
    """
    living_four_positions = []
    blocked_four_positions = []
    living_three_positions = []
    jump_living_four_positions = []
    jump_blocked_four_positions = []
    jump_living_three_positions = []
    restricted_living_three_positions = []

    for delta_row, delta_col in DIRECTIONS:
        for row in range(board.size):
            for col in range(board.size):
                if board.grid[row, col] != player:
                    continue

                positions = []
                # (r, c, outward_dr, outward_dc) for checking extendable (能成活四)
                empty_with_dir: list[tuple[int, int, int, int]] = []
                blocked_ends = 0
                jump_positions: list[tuple[int, int]] = []
                jump_count = 0

                for sign in [1, -1]:
                    dr, dc = sign * delta_row, sign * delta_col
                    r, c = row + dr, col + dc
                    dir_jump_count = 0

                    while in_bounds(board.size, r, c):
                        if board.grid[r, c] == player:
                            positions.append((r, c))
                            r += dr
                            c += dc
                        elif board.grid[r, c] == 0:
                            nr, nc = r + dr, c + dc
                            if in_bounds(board.size, nr, nc) and board.grid[nr, nc] == player:
                                if jump_count + dir_jump_count == 0:
                                    dir_jump_count = 1
                                    jump_positions.append((r, c))
                                    r = nr
                                    c = nc
                                else:
                                    empty_with_dir.append((r, c, dr, dc))
                                    break
                            else:
                                empty_with_dir.append((r, c, dr, dc))
                                break
                        else:
                            blocked_ends += 1
                            break
                    else:
                        blocked_ends += 1

                    jump_count += dir_jump_count

                total_count = 1 + len(positions)
                has_jump = len(jump_positions) > 0

                if total_count == 4:
                    if has_jump:
                        if len(empty_with_dir) == 2:
                            for pos in jump_positions:
                                if pos not in jump_living_four_positions:
                                    jump_living_four_positions.append(pos)
                        elif len(empty_with_dir) == 1 and blocked_ends == 1:
                            for pos in jump_positions:
                                if pos not in jump_blocked_four_positions:
                                    jump_blocked_four_positions.append(pos)
                    else:
                        if len(empty_with_dir) == 2:
                            for e in empty_with_dir:
                                pos = (e[0], e[1])
                                if pos not in living_four_positions:
                                    living_four_positions.append(pos)
                        elif len(empty_with_dir) == 1 and blocked_ends == 1:
                            for e in empty_with_dir:
                                pos = (e[0], e[1])
                                if pos not in blocked_four_positions:
                                    blocked_four_positions.append(pos)
                elif total_count == 3 and len(empty_with_dir) == 2:
                    block_cells = [(e[0], e[1]) for e in empty_with_dir] + list(jump_positions)
                    if has_jump:
                        # 跳活三：中间填跳必成活四，不区分 restricted
                        for pos in block_cells:
                            if pos not in jump_living_three_positions:
                                jump_living_three_positions.append(pos)
                    else:
                        extendable = sum(
                            1
                            for (r, c, odr, odc) in empty_with_dir
                            if in_bounds(board.size, r + odr, c + odc) and board.grid[r + odr, c + odc] == 0
                        )
                        if extendable == 0:
                            for pos in block_cells:
                                if pos not in restricted_living_three_positions:
                                    restricted_living_three_positions.append(pos)
                        else:
                            for pos in block_cells:
                                if pos not in living_three_positions:
                                    living_three_positions.append(pos)

    return (
        living_four_positions,
        blocked_four_positions,
        living_three_positions,
        jump_living_four_positions,
        jump_blocked_four_positions,
        jump_living_three_positions,
        restricted_living_three_positions,
    )


def find_existing_living_four_moves(board: Board, player: int) -> list[Move]:
    """Find positions that would block an existing living four for the player."""
    living_fours, _, _, _, _, _, _ = scan_existing_threats(board, player)
    return [Move(row, col) for row, col in living_fours]


def find_existing_blocked_four_moves(board: Board, player: int) -> list[Move]:
    """Find positions that would block an existing blocked four for the player."""
    _, blocked_fours, _, _, _, _, _ = scan_existing_threats(board, player)
    return [Move(row, col) for row, col in blocked_fours]


def find_existing_living_three_moves(board: Board, player: int) -> list[Move]:
    """Find positions that would block an existing living three (能成活四) for the player."""
    _, _, living_threes, _, _, _, _ = scan_existing_threats(board, player)
    return [Move(row, col) for row, col in living_threes]


def find_existing_jump_living_four_moves(board: Board, player: int) -> list[Move]:
    """Find positions that would block an existing jump living four for the player."""
    _, _, _, jump_living_fours, _, _, _ = scan_existing_threats(board, player)
    return [Move(row, col) for row, col in jump_living_fours]


def find_existing_jump_blocked_four_moves(board: Board, player: int) -> list[Move]:
    """Find positions that would block an existing jump blocked four for the player."""
    _, _, _, _, jump_blocked_fours, _, _ = scan_existing_threats(board, player)
    return [Move(row, col) for row, col in jump_blocked_fours]


def find_existing_jump_living_three_moves(board: Board, player: int) -> list[Move]:
    """Find positions that would block an existing jump living three (中间填跳必成活四)."""
    _, _, _, _, _, jump_living_threes, _ = scan_existing_threats(board, player)
    return [Move(row, col) for row, col in jump_living_threes]


def find_existing_restricted_living_three_moves(board: Board, player: int) -> list[Move]:
    """Find positions that would block an existing restricted living three (仅能成冲四的连子活三)."""
    _, _, _, _, _, _, restricted = scan_existing_threats(board, player)
    return [Move(row, col) for row, col in restricted]


@dataclass(slots=True)
class ShapeFeatures:
    five: int = 0
    open_four: int = 0
    rush_four: int = 0
    open_three: int = 0
    jump_open_three: int = 0
    restricted_open_three: int = 0
    sleep_three: int = 0
    double_four: int = 0
    four_three: int = 0
    double_three: int = 0


@dataclass(slots=True)
class ThreatInventory:
    immediate_win: int = 0
    open_four: int = 0
    double_four: int = 0
    four_three: int = 0
    double_three: int = 0
    rush_four: int = 0
    open_three: int = 0
    jump_open_three: int = 0
    restricted_open_three: int = 0
    sleep_three: int = 0

    def increment(self, category: str) -> None:
        setattr(self, category, getattr(self, category) + 1)

    def get(self, category: str) -> int:
        return int(getattr(self, category))


PRIMARY_CATEGORY_ORDER = (
    "immediate_win",
    "double_four",
    "open_four",
    "four_three",
    "double_three",
    "rush_four",
    "open_three",
    "jump_open_three",
    "restricted_open_three",
    "sleep_three",
)

# 直接消除制胜手（冲四/活三等）优先于堵了对方还有（活四/双四等）
BLOCK_MISS_CATEGORY_ORDER = (
    "rush_four",
    "open_three",
    "jump_open_three",
    "restricted_open_three",
    "open_four",
    "double_four",
    "four_three",
    "double_three",
)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _shape_weight_map(config: RewardConfig) -> dict[str, float]:
    return {
        "immediate_win": config.immediate_win_score,
        "open_four": config.open_four_score,
        "double_four": config.double_four_score,
        "four_three": config.four_three_score,
        "double_three": config.double_three_score,
        "rush_four": config.rush_four_score,
        "open_three": config.open_three_score,
        "jump_open_three": config.jump_open_three_score,
        "restricted_open_three": config.restricted_open_three_score,
        "sleep_three": config.sleep_three_score,
    }


def _block_weight_map(config: RewardConfig) -> dict[str, float]:
    """封堵专用分数：直接消除制胜手（冲四/活三等）高于堵了对方还有（活四/双四等）。"""
    return {
        "open_four": config.block_open_four_score,
        "double_four": config.block_double_four_score,
        "four_three": config.block_four_three_score,
        "double_three": config.block_double_three_score,
        "rush_four": config.block_rush_four_score,
        "open_three": config.block_open_three_score,
        "jump_open_three": config.block_jump_open_three_score,
        "restricted_open_three": config.block_restricted_open_three_score,
    }


def _shape_label_map() -> dict[str, str]:
    return {
        "open_four": "形成活四",
        "rush_four": "形成冲四/跳四",
        "open_three": "形成活三",
        "jump_open_three": "形成跳活三",
        "restricted_open_three": "形成仅能成冲四的活三",
        "sleep_three": "形成眠三",
        "double_four": "形成双四",
        "four_three": "形成冲四活三/四三",
        "double_three": "形成双活三",
    }


def _extract_shape_features(threat: ThreatInfo) -> ShapeFeatures:
    total_fours = threat.living_fours + threat.blocked_fours + threat.jump_living_fours + threat.jump_blocked_fours
    active_threes = threat.living_threes + threat.jump_living_threes
    restricted_threes = threat.restricted_living_threes
    return ShapeFeatures(
        five=1 if threat.winning_moves > 0 else 0,
        open_four=threat.living_fours,
        rush_four=threat.blocked_fours + threat.jump_living_fours + threat.jump_blocked_fours,
        open_three=threat.living_threes,
        jump_open_three=threat.jump_living_threes,
        restricted_open_three=threat.restricted_living_threes,
        sleep_three=threat.blocked_threes,
        double_four=1 if total_fours >= 2 else 0,
        four_three=1 if total_fours >= 1 and (active_threes + restricted_threes) >= 1 else 0,
        double_three=1 if (active_threes + restricted_threes) >= 2 else 0,
    )


def _primary_category(features: ShapeFeatures) -> str | None:
    counts = {
        "immediate_win": features.five,
        "double_four": features.double_four,
        "open_four": features.open_four,
        "four_three": features.four_three,
        "double_three": features.double_three,
        "rush_four": features.rush_four,
        "open_three": features.open_three,
        "jump_open_three": features.jump_open_three,
        "restricted_open_three": features.restricted_open_three,
        "sleep_three": features.sleep_three,
    }
    for category in PRIMARY_CATEGORY_ORDER:
        if counts[category] > 0:
            return category
    return None


def _evaluate_move_features(board: Board, move: Move, player: int) -> ShapeFeatures:
    board.grid[move.row, move.col] = player
    threat = get_threat_info(board, move, player)
    board.grid[move.row, move.col] = 0
    return _extract_shape_features(threat)


def _scan_threat_inventory(board: Board, player: int) -> ThreatInventory:
    inventory = ThreatInventory()
    for move in board.legal_moves():
        features = _evaluate_move_features(board, move, player)
        category = _primary_category(features)
        if category is not None:
            inventory.increment(category)
    return inventory


def _scan_existing_threat_inventory(board: Board, player: int) -> ThreatInventory:
    inventory = ThreatInventory()

    winning_moves = find_winning_moves(board, player)
    living_four_moves = find_existing_living_four_moves(board, player)
    blocked_four_moves = find_existing_blocked_four_moves(board, player)
    jump_living_four_moves = find_existing_jump_living_four_moves(board, player)
    jump_blocked_four_moves = find_existing_jump_blocked_four_moves(board, player)
    living_three_moves = find_existing_living_three_moves(board, player)
    jump_living_three_moves = find_existing_jump_living_three_moves(board, player)
    restricted_living_three_moves = find_existing_restricted_living_three_moves(board, player)

    inventory.immediate_win = 1 if winning_moves else 0
    inventory.open_four = 1 if living_four_moves else 0
    inventory.rush_four = 1 if (blocked_four_moves or jump_living_four_moves or jump_blocked_four_moves) else 0
    inventory.open_three = 1 if living_three_moves else 0
    inventory.jump_open_three = 1 if jump_living_three_moves else 0
    inventory.restricted_open_three = 1 if restricted_living_three_moves else 0

    total_fours = inventory.open_four + inventory.rush_four
    total_threes = inventory.open_three + inventory.jump_open_three + inventory.restricted_open_three
    inventory.double_four = 1 if total_fours >= 2 else 0
    inventory.four_three = 1 if total_fours >= 1 and total_threes >= 1 else 0
    inventory.double_three = 1 if total_threes >= 2 else 0

    return inventory


def _opponent_has_move_to_double_three(board: Board, opponent: int) -> bool:
    """检测对方是否存在一手落子即可形成双活三的着法。"""
    for m in board.legal_moves():
        features = _evaluate_move_features(board, m, opponent)
        if features.double_three > 0:
            return True
    return False


def _opponent_has_move_to_four_three(board: Board, opponent: int) -> bool:
    """检测对方是否存在一手落子即可形成冲四活三/四三的着法。"""
    for m in board.legal_moves():
        features = _evaluate_move_features(board, m, opponent)
        if features.four_three > 0:
            return True
    return False


def _has_tactical_threat(inventory: ThreatInventory) -> bool:
    return any(
        (
            inventory.immediate_win,
            inventory.open_four,
            inventory.rush_four,
            inventory.four_three,
            inventory.double_three,
            inventory.open_three,
            inventory.jump_open_three,
            inventory.restricted_open_three,
        )
    )


def _is_corner_move(board: Board, move: Move) -> bool:
    last_index = board.size - 1
    return (move.row, move.col) in {
        (0, 0),
        (0, last_index),
        (last_index, 0),
        (last_index, last_index),
    }


def _edge_tier(board: Board, move: Move) -> int:
    """Return 1 for outermost ring, 2 for second ring, 0 otherwise."""
    last = board.size - 1
    if move.row in (0, last) or move.col in (0, last):
        return 1
    if move.row in (1, last - 1) or move.col in (1, last - 1):
        return 2
    return 0


def _shape_position_scale(board: Board, move: Move, category: str, config: RewardConfig) -> tuple[float, str | None]:
    if category not in {"open_three", "jump_open_three", "restricted_open_three", "sleep_three"}:
        return 1.0, None
    if _is_corner_move(board, move):
        return config.corner_shape_decay, "角落棋型价值折减"
    tier = _edge_tier(board, move)
    if tier == 1:
        return config.edge_shape_decay, "边线棋型价值折减"
    if tier == 2:
        decay = 1.0 - (1.0 - config.edge_shape_decay) * 0.5
        return decay, "次边线棋型价值折减"
    return 1.0, None


def _accumulate_shape_reward(
    details: list[RewardDetail],
    board: Board,
    move: Move,
    features: ShapeFeatures,
    scale: float,
    config: RewardConfig,
) -> float:
    category = _primary_category(features)
    if category is None or category == "immediate_win":
        return 0.0

    counts = {
        "open_four": features.open_four,
        "rush_four": features.rush_four,
        "open_three": features.open_three,
        "jump_open_three": features.jump_open_three,
        "restricted_open_three": features.restricted_open_three,
        "sleep_three": features.sleep_three,
        "double_four": features.double_four,
        "four_three": features.four_three,
        "double_three": features.double_three,
    }
    count = counts[category]
    if count <= 0:
        return 0.0

    weight = _shape_weight_map(config)[category]
    reason = _shape_label_map()[category]
    position_scale, scale_reason = _shape_position_scale(board, move, category, config)
    amount = count * weight * scale * position_scale
    detail_reason = f"{reason} x{count}"
    if scale_reason is not None and position_scale < 1.0:
        detail_reason = f"{detail_reason}（{scale_reason} {position_scale:.2f}）"
    details.append(RewardDetail(amount=amount, reason=detail_reason))
    return amount


def _accumulate_block_reward(
    details: list[RewardDetail],
    before: ThreatInventory,
    after: ThreatInventory,
    config: RewardConfig,
) -> float:
    weights = _block_weight_map(config)
    labels = {
        "open_four": "封堵对方活四",
        "double_four": "封堵对方双四",
        "four_three": "封堵对方冲四活三/四三",
        "double_three": "封堵对方双活三",
        "rush_four": "封堵对方冲四/跳四",
        "open_three": "封堵对方活三",
        "jump_open_three": "封堵对方跳活三",
        "restricted_open_three": "封堵对方仅能成冲四的活三",
    }
    reward = 0.0
    for category in BLOCK_MISS_CATEGORY_ORDER:
        removed = max(0, before.get(category) - after.get(category))
        if removed <= 0:
            continue
        amount = removed * weights[category] * config.block_scale
        details.append(RewardDetail(amount=amount, reason=f"{labels[category]} x{removed}"))
        reward += amount
    return reward


def _accumulate_miss_penalty(
    details: list[RewardDetail],
    before: ThreatInventory,
    after: ThreatInventory,
    config: RewardConfig,
    *,
    my_strong_attack: bool = False,
    opp_has_move_to_double_three_before: bool = False,
    opp_has_move_to_double_three_after: bool = False,
    opp_has_move_to_four_three_before: bool = False,
    opp_has_move_to_four_three_after: bool = False,
) -> float:
    # Highest priority: opponent had a blockable winning move (e.g. rush four)
    # and it was not blocked.
    if (
        before.immediate_win > 0
        and after.immediate_win > 0
        and before.open_four == 0
    ):
        amount = -config.miss_immediate_win_penalty
        details.append(RewardDetail(amount=amount, reason="未阻止对方制胜手"))
        return amount

    total_penalty = 0.0
    # 对方有活四（双赢点）且未堵任何一端时，同样扣漏防惩罚；堵了由 block_reward 加分
    if (
        before.immediate_win > 0
        and after.immediate_win > 0
        and before.open_four > 0
        and after.open_four > 0
    ):
        amount = -config.miss_immediate_win_penalty
        details.append(RewardDetail(amount=amount, reason="未阻止对方制胜手"))
        total_penalty += amount

    # 本手强攻时豁免对一手成活四（活三/跳活三）的漏防惩罚
    waive_lower_threats = my_strong_attack
    # 顺序：未阻止（一手成活四）；冲四/跳四未堵已由上方「未阻止对方制胜手」覆盖并 return；活四由「封堵对方活四」体现，不在此重复扣分
    penalties = (
        ("open_three", config.miss_open_three_penalty, "未阻止对方一手成活四"),
        ("jump_open_three", config.miss_jump_open_three_penalty, "未阻止对方一手成活四"),
    )
    for category, unit_penalty, reason in penalties:
        if waive_lower_threats and category in (
            "open_three",
            "jump_open_three",
        ):
            continue
        unresolved = min(before.get(category), after.get(category))
        if unresolved <= 0:
            continue
        amount = -unit_penalty * unresolved
        details.append(RewardDetail(amount=amount, reason=f"{reason} x{unresolved}"))
        total_penalty += amount

    # 堵截活四不彻底仍留冲四：不再单独扣分，堵了总比不堵强。

    # 未阻止：对方存在一手成冲四活三/四三或一手成双活三的着法且未消除时扣分，己方强攻可豁免
    if (
        opp_has_move_to_four_three_before
        and opp_has_move_to_four_three_after
        and not my_strong_attack
    ):
        amount = -config.miss_one_move_four_three_penalty
        details.append(RewardDetail(amount=amount, reason="未阻止对方一手成冲四活三/四三"))
        total_penalty += amount
    if (
        opp_has_move_to_double_three_before
        and opp_has_move_to_double_three_after
        and not my_strong_attack
    ):
        amount = -config.miss_one_move_double_three_penalty
        details.append(RewardDetail(amount=amount, reason="未阻止对方一手成双活三"))
        total_penalty += amount

    return total_penalty


def _is_winning_move(board: Board, move: Move, player: int) -> bool:
    board.grid[move.row, move.col] = player
    winner = board.check_winner(move)
    board.grid[move.row, move.col] = 0
    return winner == player


def _find_own_open_four_moves(board: Board, player: int) -> list[Move]:
    """Find moves that would create an open four (or better) for the player."""
    results = []
    for move in board.legal_moves():
        features = _evaluate_move_features(board, move, player)
        if features.open_four > 0 or features.double_four > 0 or features.four_three > 0:
            results.append(move)
    return results


def _accumulate_missed_own_win_penalty(
    details: list[RewardDetail],
    board: Board,
    move: Move,
    player: int,
    config: RewardConfig,
    opponent_before: ThreatInventory | None = None,
) -> float:
    winning_moves = find_winning_moves(board, player)
    if winning_moves:
        if move in winning_moves:
            return 0.0
        amount = -config.miss_own_immediate_win_penalty
        details.append(RewardDetail(amount=amount, reason="错失直接获胜落点"))
        return amount

    # When the opponent has an immediate winning move, forming our own open
    # four is useless — the opponent wins before we can use it.  The correct
    # play is to block, so don't penalise for "missing" an own open four.
    if opponent_before is not None and opponent_before.immediate_win > 0:
        return 0.0

    my_features = _evaluate_move_features(board, move, player)
    move_forms_open_four = (
        my_features.open_four > 0
        or my_features.double_four > 0
        or my_features.four_three > 0
    )
    if move_forms_open_four:
        return 0.0

    open_four_moves = _find_own_open_four_moves(board, player)
    if not open_four_moves:
        return 0.0

    amount = -config.miss_own_open_four_penalty
    details.append(RewardDetail(amount=amount, reason="错失形成活四/必胜棋型"))
    return amount


def _accumulate_opening_position_reward(
    details: list[RewardDetail],
    board: Board,
    move: Move,
    config: RewardConfig,
) -> float:
    stones_played = int(np.count_nonzero(board.grid))
    if stones_played >= config.opening_position_horizon:
        return 0.0

    row, col = move.row, move.col
    last_index = board.size - 1
    center = (board.size - 1) / 2.0
    max_distance = max(math.dist((0.0, 0.0), (center, center)), 1.0)
    distance = math.dist((float(row), float(col)), (center, center))
    centrality = max(0.0, 1.0 - distance / max_distance)
    center_bias = config.opening_center_bonus * (centrality**2)
    if center_bias > 1e-8:
        details.append(RewardDetail(amount=center_bias, reason="开局中心趋向奖励"))

    is_corner = _is_corner_move(board, move)
    if is_corner:
        penalty = -config.opening_corner_penalty
        details.append(RewardDetail(amount=penalty, reason="开局角落落子惩罚"))
        return center_bias + penalty

    tier = _edge_tier(board, move)
    if tier == 1:
        penalty = -config.opening_edge_penalty
        details.append(RewardDetail(amount=penalty, reason="开局边线落子惩罚"))
        return center_bias + penalty
    if tier == 2:
        penalty = -config.opening_edge_penalty * 0.5
        details.append(RewardDetail(amount=penalty, reason="开局次边线落子惩罚"))
        return center_bias + penalty

    radius = max(1.0, (board.size - 1) * config.opening_center_radius_ratio)
    distance_sq = (row - center) ** 2 + (col - center) ** 2
    if distance_sq <= radius ** 2:
        bonus = config.opening_center_bonus
        details.append(RewardDetail(amount=bonus, reason="开局中心落子奖励"))
        return center_bias + bonus

    return center_bias


def _accumulate_opening_edge_corner_penalty_only(
    details: list[RewardDetail],
    board: Board,
    move: Move,
    config: RewardConfig,
) -> float:
    """仅累加开局边线/角惩罚（无中心奖励）。用于漏防时仍对走边/走角单独扣分。"""
    stones_played = int(np.count_nonzero(board.grid))
    if stones_played >= config.opening_position_horizon:
        return 0.0
    if _is_corner_move(board, move):
        penalty = -config.opening_corner_penalty
        details.append(RewardDetail(amount=penalty, reason="开局角落落子惩罚"))
        return penalty
    tier = _edge_tier(board, move)
    if tier == 1:
        penalty = -config.opening_edge_penalty
        details.append(RewardDetail(amount=penalty, reason="开局边线落子惩罚"))
        return penalty
    if tier == 2:
        penalty = -config.opening_edge_penalty * 0.5
        details.append(RewardDetail(amount=penalty, reason="开局次边线落子惩罚"))
        return penalty
    return 0.0


def _opening_position_scale(
    details: list[RewardDetail],
    opponent_before: ThreatInventory,
    miss_penalty: float,
    missed_own_win_penalty: float,
    is_winning_move: bool,
    config: RewardConfig,
) -> float:
    if is_winning_move:
        return 0.0
    if miss_penalty != 0.0 or missed_own_win_penalty != 0.0:
        return 0.0
    if (
        opponent_before.immediate_win
        or opponent_before.open_four
        or opponent_before.rush_four
        or opponent_before.four_three
    ):
        if config.opening_major_threat_scale < 1.0:
            details.append(
                RewardDetail(
                    amount=0.0,
                    reason=f"开局位置权重降低（对手强威胁，x{config.opening_major_threat_scale:.2f}）",
                )
            )
        return config.opening_major_threat_scale
    if (
        opponent_before.double_three
        or opponent_before.open_three
        or opponent_before.jump_open_three
        or _has_tactical_threat(opponent_before)
    ):
        if config.opening_minor_threat_scale < 1.0:
            details.append(
                RewardDetail(
                    amount=0.0,
                    reason=f"开局位置权重降低（对手牵制威胁，x{config.opening_minor_threat_scale:.2f}）",
                )
            )
        return config.opening_minor_threat_scale
    return 1.0


def compute_outcome_tail_bonus(
    player: int,
    winner: int,
    plies_from_end: int,
    config: RewardConfig | None = None,
) -> RewardDetail | None:
    if config is None:
        config = RewardConfig()
    if winner == 0:
        return None
    if plies_from_end <= 0 or plies_from_end >= config.outcome_horizon:
        return None
    magnitude = config.outcome_tail_bonus * (config.outcome_decay ** (plies_from_end - 1))
    signed_amount = magnitude if player == winner else -magnitude
    if abs(signed_amount) < 1e-8:
        return None
    reason = f"终局结果回传（距终局 {plies_from_end} 手）"
    return RewardDetail(amount=signed_amount, reason=reason)


def compute_process_reward_with_details(
    board: Board,
    move: Move,
    player: int,
    config: RewardConfig | None = None,
) -> RewardResult:
    if config is None:
        config = RewardConfig()

    opponent = -player
    details: list[RewardDetail] = []

    opponent_before = _scan_existing_threat_inventory(board, opponent)
    opp_can_double_three_before = _opponent_has_move_to_double_three(board, opponent)
    opp_can_four_three_before = _opponent_has_move_to_four_three(board, opponent)
    my_features = _evaluate_move_features(board, move, player)
    is_winning_move = my_features.five > 0
    # 本手形成冲四/活四/四三/双四/双活三时视为强攻，豁免对活三/双活三/四三等漏防惩罚
    my_strong_attack = (
        my_features.rush_four > 0
        or my_features.open_four > 0
        or my_features.four_three > 0
        or my_features.double_four > 0
        or my_features.double_three > 0
    )

    board.grid[move.row, move.col] = player
    opponent_after = _scan_existing_threat_inventory(board, opponent)
    opp_can_double_three_after = _opponent_has_move_to_double_three(board, opponent)
    opp_can_four_three_after = _opponent_has_move_to_four_three(board, opponent)
    board.grid[move.row, move.col] = 0

    attack_reward = _accumulate_shape_reward(details, board, move, my_features, config.attack_scale, config)
    block_reward = _accumulate_block_reward(details, opponent_before, opponent_after, config)
    # A direct win ends the game immediately, so opponent threats no longer matter.
    miss_penalty = 0.0
    if not is_winning_move:
        miss_penalty = _accumulate_miss_penalty(
            details,
            opponent_before,
            opponent_after,
            config,
            my_strong_attack=my_strong_attack,
            opp_has_move_to_double_three_before=opp_can_double_three_before,
            opp_has_move_to_double_three_after=opp_can_double_three_after,
            opp_has_move_to_four_three_before=opp_can_four_three_before,
            opp_has_move_to_four_three_after=opp_can_four_three_after,
        )
    missed_own_win_penalty = _accumulate_missed_own_win_penalty(details, board, move, player, config, opponent_before)

    # 错失直接获胜时，进攻奖励不该叠加——能赢不赢就不该因为"顺便形成活四"而得正分。
    if missed_own_win_penalty != 0.0 and attack_reward > 0.0:
        details[:] = [d for d in details if d.amount <= 0 or "错失" in d.reason]
        attack_reward = 0.0

    opening_position_reward = 0.0
    stones_played = int(np.count_nonzero(board.grid))
    if stones_played < config.opening_position_horizon:
        opening_position_scale = _opening_position_scale(
            details,
            opponent_before,
            miss_penalty,
            missed_own_win_penalty,
            is_winning_move,
            config,
        )
        if opening_position_scale > 0.0:
            opening_position_reward = (
                _accumulate_opening_position_reward(details, board, move, config) * opening_position_scale
            )
        # 漏防时仍对走边/走角单独扣分，使「漏防+走边」同时显示漏防与边线惩罚
        if (miss_penalty != 0.0 or missed_own_win_penalty != 0.0):
            opening_position_reward += _accumulate_opening_edge_corner_penalty_only(
                details, board, move, config
            )

    total_reward = attack_reward + block_reward + miss_penalty + missed_own_win_penalty + opening_position_reward
    clipped_reward = _clip(total_reward, -config.max_process_reward, config.max_process_reward)
    if abs(clipped_reward - total_reward) > 1e-8:
        details.append(RewardDetail(amount=clipped_reward - total_reward, reason="过程奖励裁剪"))
    return RewardResult(
        total_reward=clipped_reward,
        details=details,
        missed_own_win=(missed_own_win_penalty != 0.0),
    )


def compute_hybrid_reward_with_details(
    board: Board,
    move: Move,
    player: int,
    winner: int,
    config: RewardConfig | None = None,
) -> RewardResult:
    if config is None:
        config = RewardConfig()

    process_result = compute_process_reward_with_details(board, move, player, config)
    details = process_result.details.copy()
    total_reward = process_result.total_reward

    if winner == 0 and abs(config.draw_reward) > 1e-8:
        details.append(RewardDetail(amount=config.draw_reward, reason="终局平局奖励"))
        total_reward += config.draw_reward
    elif winner == player and _is_winning_move(board, move, player):
        details.append(RewardDetail(amount=config.final_win_reward, reason="终局获胜奖励"))
        total_reward += config.final_win_reward

    clipped_total = _clip(total_reward, -config.max_total_reward, config.max_total_reward)
    if abs(clipped_total - total_reward) > 1e-8:
        details.append(RewardDetail(amount=clipped_total - total_reward, reason="总奖励裁剪"))
    return RewardResult(
        total_reward=clipped_total,
        details=details,
        missed_own_win=process_result.missed_own_win,
    )


def compute_process_reward(
    board: Board,
    move: Move,
    player: int,
    config: RewardConfig | None = None,
) -> float:
    return compute_process_reward_with_details(board, move, player, config).total_reward


def compute_hybrid_reward(
    board: Board,
    move: Move,
    player: int,
    winner: int,
    config: RewardConfig | None = None,
) -> float:
    return compute_hybrid_reward_with_details(board, move, player, winner, config).total_reward
