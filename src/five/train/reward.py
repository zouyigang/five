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


@dataclass(slots=True)
class RewardDetail:
    amount: float
    reason: str


@dataclass(slots=True)
class RewardResult:
    total_reward: float
    details: list[RewardDetail]


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
                        # Not a valid jump, this is an open end
                        open_ends += 1
                        break
                else:
                    # Already has a jump, this is an open end
                    open_ends += 1
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
            if line.has_jump:
                if line.open_ends == 2:
                    jump_living_threes += 1
            else:
                if line.open_ends == 2:
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


def scan_existing_threats(board: Board, player: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Scan the board for existing threats including jump threats.
    Returns: (living_four_positions, blocked_four_positions, living_three_positions, 
              jump_living_four_positions, jump_blocked_four_positions, jump_living_three_positions)
    Each position is a tuple of (row, col) where a move would block the threat.
    """
    living_four_positions = []
    blocked_four_positions = []
    living_three_positions = []
    jump_living_four_positions = []
    jump_blocked_four_positions = []
    jump_living_three_positions = []
    
    for delta_row, delta_col in DIRECTIONS:
        for row in range(board.size):
            for col in range(board.size):
                if board.grid[row, col] != player:
                    continue
                    
                # Check both directions
                positions = []
                empty_positions = []
                blocked_ends = 0
                jump_positions = []  # Positions of empty cells that are jumps
                jump_count = 0  # Count of jumps in this direction
                
                for sign in [1, -1]:
                    dr, dc = sign * delta_row, sign * delta_col
                    r, c = row + dr, col + dc
                    dir_jump_count = 0  # Jump count for this direction
                    
                    while in_bounds(board.size, r, c):
                        if board.grid[r, c] == player:
                            positions.append((r, c))
                            r += dr
                            c += dc
                        elif board.grid[r, c] == 0:
                            # Check if there's a player piece after this empty cell (jump)
                            nr, nc = r + dr, c + dc
                            if in_bounds(board.size, nr, nc) and board.grid[nr, nc] == player:
                                # This is a jump - but only allow ONE jump total
                                if jump_count + dir_jump_count == 0:
                                    dir_jump_count = 1
                                    jump_positions.append((r, c))
                                    r = nr
                                    c = nc
                                else:
                                    # Already has a jump, stop here
                                    empty_positions.append((r, c))
                                    break
                            else:
                                empty_positions.append((r, c))
                                break
                        else:
                            blocked_ends += 1
                            break
                    else:
                        # Hit boundary
                        blocked_ends += 1
                    
                    jump_count += dir_jump_count
                
                total_count = 1 + len(positions)
                has_jump = len(jump_positions) > 0
                
                if total_count == 4:
                    if has_jump:
                        # Jump four - block the jump position (middle empty cell)
                        if len(empty_positions) == 2:
                            for pos in jump_positions:
                                if pos not in jump_living_four_positions:
                                    jump_living_four_positions.append(pos)
                        elif len(empty_positions) == 1 and blocked_ends == 1:
                            for pos in jump_positions:
                                if pos not in jump_blocked_four_positions:
                                    jump_blocked_four_positions.append(pos)
                    else:
                        # Normal four
                        if len(empty_positions) == 2:
                            for pos in empty_positions:
                                if pos not in living_four_positions:
                                    living_four_positions.append(pos)
                        elif len(empty_positions) == 1 and blocked_ends == 1:
                            for pos in empty_positions:
                                if pos not in blocked_four_positions:
                                    blocked_four_positions.append(pos)
                elif total_count == 3 and len(empty_positions) == 2:
                    if has_jump:
                        # Jump living three - can block either end or the jump position
                        for pos in empty_positions:
                            if pos not in jump_living_three_positions:
                                jump_living_three_positions.append(pos)
                        for pos in jump_positions:
                            if pos not in jump_living_three_positions:
                                jump_living_three_positions.append(pos)
                    else:
                        # Normal living three
                        for pos in empty_positions:
                            if pos not in living_three_positions:
                                living_three_positions.append(pos)
    
    return (living_four_positions, blocked_four_positions, living_three_positions,
            jump_living_four_positions, jump_blocked_four_positions, jump_living_three_positions)


def find_existing_living_four_moves(board: Board, player: int) -> list[Move]:
    """Find positions that would block an existing living four for the player."""
    living_fours, _, _, _, _, _ = scan_existing_threats(board, player)
    return [Move(row, col) for row, col in living_fours]


def find_existing_blocked_four_moves(board: Board, player: int) -> list[Move]:
    """Find positions that would block an existing blocked four for the player."""
    _, blocked_fours, _, _, _, _ = scan_existing_threats(board, player)
    return [Move(row, col) for row, col in blocked_fours]


def find_existing_living_three_moves(board: Board, player: int) -> list[Move]:
    """Find positions that would block an existing living three for the player."""
    _, _, living_threes, _, _, _ = scan_existing_threats(board, player)
    return [Move(row, col) for row, col in living_threes]


def find_existing_jump_living_four_moves(board: Board, player: int) -> list[Move]:
    """Find positions that would block an existing jump living four for the player."""
    _, _, _, jump_living_fours, _, _ = scan_existing_threats(board, player)
    return [Move(row, col) for row, col in jump_living_fours]


def find_existing_jump_blocked_four_moves(board: Board, player: int) -> list[Move]:
    """Find positions that would block an existing jump blocked four for the player."""
    _, _, _, _, jump_blocked_fours, _ = scan_existing_threats(board, player)
    return [Move(row, col) for row, col in jump_blocked_fours]


def find_existing_jump_living_three_moves(board: Board, player: int) -> list[Move]:
    """Find positions that would block an existing jump living three for the player."""
    _, _, _, _, _, jump_living_threes = scan_existing_threats(board, player)
    return [Move(row, col) for row, col in jump_living_threes]


@dataclass(slots=True)
class ShapeFeatures:
    five: int = 0
    open_four: int = 0
    rush_four: int = 0
    open_three: int = 0
    jump_open_three: int = 0
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
    "sleep_three",
)

BLOCK_MISS_CATEGORY_ORDER = (
    "open_four",
    "rush_four",
    "open_three",
    "jump_open_three",
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
        "sleep_three": config.sleep_three_score,
    }


def _shape_label_map() -> dict[str, str]:
    return {
        "open_four": "形成活四",
        "rush_four": "形成冲四/跳四",
        "open_three": "形成活三",
        "jump_open_three": "形成跳活三",
        "sleep_three": "形成眠三",
        "double_four": "形成双四",
        "four_three": "形成冲四活三/四三",
        "double_three": "形成双活三",
    }


def _extract_shape_features(threat: ThreatInfo) -> ShapeFeatures:
    total_fours = threat.living_fours + threat.blocked_fours + threat.jump_living_fours + threat.jump_blocked_fours
    active_threes = threat.living_threes + threat.jump_living_threes
    return ShapeFeatures(
        five=1 if threat.winning_moves > 0 else 0,
        open_four=threat.living_fours,
        rush_four=threat.blocked_fours + threat.jump_living_fours + threat.jump_blocked_fours,
        open_three=threat.living_threes,
        jump_open_three=threat.jump_living_threes,
        sleep_three=threat.blocked_threes,
        double_four=1 if total_fours >= 2 else 0,
        four_three=1 if total_fours >= 1 and active_threes >= 1 else 0,
        double_three=1 if active_threes >= 2 else 0,
    )


def _primary_category(features: ShapeFeatures) -> str | None:
    if features.five:
        return "immediate_win"
    if features.open_four:
        return "open_four"
    if features.double_four:
        return "double_four"
    if features.four_three:
        return "four_three"
    if features.double_three:
        return "double_three"
    if features.rush_four:
        return "rush_four"
    if features.open_three:
        return "open_three"
    if features.jump_open_three:
        return "jump_open_three"
    if features.sleep_three:
        return "sleep_three"
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

    inventory.immediate_win = 1 if winning_moves else 0
    inventory.open_four = 1 if living_four_moves else 0
    inventory.rush_four = 1 if (blocked_four_moves or jump_living_four_moves or jump_blocked_four_moves) else 0
    inventory.open_three = 1 if living_three_moves else 0
    inventory.jump_open_three = 1 if jump_living_three_moves else 0

    total_fours = inventory.open_four + inventory.rush_four
    total_threes = inventory.open_three + inventory.jump_open_three
    inventory.double_four = 1 if total_fours >= 2 else 0
    inventory.four_three = 1 if total_fours >= 1 and total_threes >= 1 else 0
    inventory.double_three = 1 if total_threes >= 2 else 0

    return inventory


def _accumulate_shape_reward(
    details: list[RewardDetail],
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
    amount = count * weight * scale
    details.append(RewardDetail(amount=amount, reason=f"{reason} x{count}"))
    return amount


def _accumulate_block_reward(
    details: list[RewardDetail],
    before: ThreatInventory,
    after: ThreatInventory,
    config: RewardConfig,
) -> float:
    weights = _shape_weight_map(config)
    labels = {
        "double_four": "封堵对方双四",
        "four_three": "封堵对方冲四活三/四三",
        "double_three": "封堵对方双活三",
        "rush_four": "封堵对方冲四/跳四",
        "open_three": "封堵对方活三",
        "jump_open_three": "封堵对方跳活三",
    }
    reward = 0.0
    delayed_open_four = max(0, before.open_four - after.open_four)
    if delayed_open_four > 0:
        amount = delayed_open_four * config.delay_open_four_reward
        details.append(RewardDetail(amount=amount, reason=f"延缓对方活四一手 x{delayed_open_four}"))
        reward += amount
    for category in BLOCK_MISS_CATEGORY_ORDER:
        if category == "open_four":
            continue
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
) -> float:
    penalties = (
        ("open_four", config.miss_open_four_penalty, "未阻止对方活四保持双赢点"),
        ("rush_four", config.miss_rush_four_penalty, "未化解对方冲四/跳四"),
        ("open_three", config.miss_open_three_penalty, "未压制对方活三"),
        ("jump_open_three", config.miss_jump_open_three_penalty, "未压制对方跳活三"),
    )
    total_penalty = 0.0
    for category, unit_penalty, reason in penalties:
        unresolved = min(before.get(category), after.get(category))
        if unresolved <= 0:
            continue
        amount = -unit_penalty * unresolved
        details.append(RewardDetail(amount=amount, reason=f"{reason} x{unresolved}"))
        total_penalty += amount

    # Blocking one end of an open four still leaves a one-move loss on the other end.
    downgraded_open_four = min(before.open_four, after.rush_four)
    if downgraded_open_four > 0:
        amount = -config.miss_rush_four_penalty * downgraded_open_four
        details.append(RewardDetail(amount=amount, reason=f"未化解对方冲四/跳四 x{downgraded_open_four}"))
        total_penalty += amount
    return total_penalty


def _is_winning_move(board: Board, move: Move, player: int) -> bool:
    board.grid[move.row, move.col] = player
    winner = board.check_winner(move)
    board.grid[move.row, move.col] = 0
    return winner == player


def _accumulate_missed_own_win_penalty(
    details: list[RewardDetail],
    board: Board,
    move: Move,
    player: int,
    config: RewardConfig,
) -> float:
    winning_moves = find_winning_moves(board, player)
    if not winning_moves:
        return 0.0
    if move in winning_moves:
        return 0.0

    amount = -config.miss_own_immediate_win_penalty
    details.append(RewardDetail(amount=amount, reason="错失直接获胜落点"))
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

    is_corner = (row, col) in {
        (0, 0),
        (0, last_index),
        (last_index, 0),
        (last_index, last_index),
    }
    if is_corner:
        penalty = -config.opening_corner_penalty
        details.append(RewardDetail(amount=penalty, reason="开局角落落子惩罚"))
        return center_bias + penalty

    if row in (0, last_index) or col in (0, last_index):
        penalty = -config.opening_edge_penalty
        details.append(RewardDetail(amount=penalty, reason="开局边线落子惩罚"))
        return center_bias + penalty

    radius = max(1.0, (board.size - 1) * config.opening_center_radius_ratio)
    distance_sq = (row - center) ** 2 + (col - center) ** 2
    if distance_sq <= radius ** 2:
        bonus = config.opening_center_bonus
        details.append(RewardDetail(amount=bonus, reason="开局中心落子奖励"))
        return center_bias + bonus

    return center_bias


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
    my_features = _evaluate_move_features(board, move, player)
    is_winning_move = my_features.five > 0

    board.grid[move.row, move.col] = player
    opponent_after = _scan_existing_threat_inventory(board, opponent)
    board.grid[move.row, move.col] = 0

    attack_reward = _accumulate_shape_reward(details, my_features, config.attack_scale, config)
    block_reward = _accumulate_block_reward(details, opponent_before, opponent_after, config)
    # A direct win ends the game immediately, so opponent threats no longer matter.
    miss_penalty = 0.0
    if not is_winning_move:
        miss_penalty = _accumulate_miss_penalty(details, opponent_before, opponent_after, config)
    missed_own_win_penalty = _accumulate_missed_own_win_penalty(details, board, move, player, config)
    opening_position_reward = _accumulate_opening_position_reward(details, board, move, config)

    total_reward = attack_reward + block_reward + miss_penalty + missed_own_win_penalty + opening_position_reward
    clipped_reward = _clip(total_reward, -config.max_process_reward, config.max_process_reward)
    if abs(clipped_reward - total_reward) > 1e-8:
        details.append(RewardDetail(amount=clipped_reward - total_reward, reason="过程奖励裁剪"))
    return RewardResult(total_reward=clipped_reward, details=details)


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
    return RewardResult(total_reward=clipped_total, details=details)


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
