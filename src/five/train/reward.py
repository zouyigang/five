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


# 细分棋型 -> 计分口径。跳活四与冲四同档，沿用二值化时期的归类。
_THREAT_KIND_TO_CATEGORY: dict[str, str] = {
    "living_four": "open_four",
    "blocked_four": "rush_four",
    "jump_living_four": "rush_four",
    "jump_blocked_four": "rush_four",
    "living_three": "open_three",
    "jump_living_three": "jump_open_three",
    "restricted_living_three": "restricted_open_three",
}


@dataclass(frozen=True, slots=True)
class ThreatInstance:
    """一个已成立的威胁实例。

    `stones` + `direction` 唯一标识它：同一条线上的同一组棋子，无论从其中哪颗子
    扫描到，都归并为同一个实例。`block_cells` 是能化解它的格点。
    """

    kind: str
    stones: frozenset[tuple[int, int]]
    direction: tuple[int, int]
    block_cells: tuple[tuple[int, int], ...]

    @property
    def category(self) -> str:
        return _THREAT_KIND_TO_CATEGORY[self.kind]


def _classify_threat_line(
    board: Board,
    stone_count: int,
    empty_with_dir: list[tuple[int, int, int, int]],
    jump_positions: list[tuple[int, int]],
    blocked_ends: int,
) -> tuple[str, tuple[tuple[int, int], ...]] | None:
    """把一次线扫描的结果归类为 (细分棋型, 化解格点)；不构成威胁时返回 None。"""
    has_jump = bool(jump_positions)
    empty_cells = tuple((row, col) for (row, col, _dr, _dc) in empty_with_dir)

    if stone_count == 4:
        if has_jump:
            block_cells = tuple(jump_positions)
            if len(empty_with_dir) == 2:
                return "jump_living_four", block_cells
            if len(empty_with_dir) == 1 and blocked_ends == 1:
                return "jump_blocked_four", block_cells
            return None
        if len(empty_with_dir) == 2:
            return "living_four", empty_cells
        if len(empty_with_dir) == 1 and blocked_ends == 1:
            return "blocked_four", empty_cells
        return None

    if stone_count == 3 and len(empty_with_dir) == 2:
        block_cells = empty_cells + tuple(jump_positions)
        if has_jump:
            # 跳活三：中间填跳必成活四，不区分 restricted
            return "jump_living_three", block_cells
        extendable = sum(
            1
            for (row, col, out_dr, out_dc) in empty_with_dir
            if in_bounds(board.size, row + out_dr, col + out_dc)
            and board.grid[row + out_dr, col + out_dc] == 0
        )
        if extendable == 0:
            return "restricted_living_three", block_cells
        return "living_three", block_cells

    return None


def scan_threat_instances(board: Board, player: int) -> list[ThreatInstance]:
    """全盘扫描 player 已成立的威胁，按**实例**返回（而非按化解格点）。

    同一条线上的同一组棋子会被其中每一颗子各扫描到一次，这里用 (方向, 棋子集合)
    规范化去重。因此「两个不相交的活三」会得到 2 个实例，而不是被压成一个布尔位——
    这是「堵掉两个活三之一」能拿到封堵奖励的前提。
    """
    instances: dict[tuple[tuple[int, int], frozenset[tuple[int, int]]], ThreatInstance] = {}

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

                classified = _classify_threat_line(
                    board, 1 + len(positions), empty_with_dir, jump_positions, blocked_ends
                )
                if classified is None:
                    continue
                kind, block_cells = classified
                stones = frozenset([(row, col), *positions])
                key = ((delta_row, delta_col), stones)
                if key in instances:
                    continue
                instances[key] = ThreatInstance(
                    kind=kind,
                    stones=stones,
                    direction=(delta_row, delta_col),
                    block_cells=block_cells,
                )

    return list(instances.values())


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


def _scan_existing_threat_inventory(board: Board, player: int) -> ThreatInventory:
    """统计 player 已成立的威胁，基础棋型按**实例个数**计数。

    计数化（而非 0/1）是「堵掉两个活三之一」能拿分的前提：二值化时 before/after
    都是 1，差值为 0，那一手与完全不防守的得分完全相同。

    两个例外保持二值：
    - `immediate_win`——「能不能立刻被赢下」本就是二元判断；
    - 复合项 `double_four`/`four_three`/`double_three`——描述的是「局面具备某种复合
      形态」，对其计数没有棋理意义。它们现在由实例数推导，因此「两个直活三」也能
      正确判为双活三（二值化时因两者同属 open_three 而合并成 1，判不出来）。
    """
    inventory = ThreatInventory()

    inventory.immediate_win = 1 if find_winning_moves(board, player) else 0
    for instance in scan_threat_instances(board, player):
        inventory.increment(instance.category)

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
    my_counter_threat: bool = False,
    opp_has_move_to_double_three_before: bool = False,
    opp_has_move_to_double_three_after: bool = False,
    opp_has_move_to_four_three_before: bool = False,
    opp_has_move_to_four_three_after: bool = False,
) -> tuple[float, float]:
    """返回 (实际漏防惩罚, 因己方强攻而被豁免掉的惩罚额)。

    第二项为正数量级，供调用方判断「这手强攻是否正在靠豁免免责」——是的话要折减它的
    进攻分，否则随手冲四会净赚。
    """
    # Highest priority: the opponent had at least one immediate winning move
    # before this move and still has one afterwards.  A partial block may turn
    # an open four into a rush four, but the position is still losing and must
    # therefore retain the miss penalty.  Any partial-block credit is recorded
    # independently by _accumulate_block_reward.
    if before.immediate_win > 0 and after.immediate_win > 0:
        amount = -config.miss_immediate_win_penalty
        details.append(RewardDetail(amount=amount, reason="未阻止对方制胜手"))
        return amount, 0.0

    total_penalty = 0.0
    waived_penalty = 0.0
    # 顺序：未阻止（一手成活四）；冲四/跳四未堵已由上方「未阻止对方制胜手」覆盖并 return；活四由「封堵对方活四」体现，不在此重复扣分
    penalties = (
        ("open_three", config.miss_open_three_penalty, "未阻止对方一手成活四"),
        ("jump_open_three", config.miss_jump_open_three_penalty, "未阻止对方一手成活四"),
    )
    for category, unit_penalty, reason in penalties:
        unresolved = min(before.get(category), after.get(category))
        if unresolved <= 0:
            continue
        amount = -unit_penalty * unresolved
        # 本手强攻时豁免对一手成活四（活三/跳活三）的漏防惩罚
        if my_strong_attack:
            waived_penalty += -amount
            continue
        details.append(RewardDetail(amount=amount, reason=f"{reason} x{unresolved}"))
        total_penalty += amount

    # 未阻止：对方存在一手成冲四活三/四三或一手成双活三的着法且未消除时扣分，己方强攻可豁免。
    # 本手成活三/跳活三时只是相对先手（对方可用兼具封堵与反击的一手化解），按系数部分折减。
    counter_scale = 1.0
    counter_note = ""
    if my_counter_threat and not my_strong_attack:
        counter_scale = _clip(config.counter_threat_waiver_scale, 0.0, 1.0)
        counter_note = f"（本手反击活三，惩罚 x{counter_scale:.2f}）"

    misses = (
        (
            opp_has_move_to_four_three_before and opp_has_move_to_four_three_after,
            config.miss_one_move_four_three_penalty,
            "未阻止对方一手成冲四活三/四三",
        ),
        (
            opp_has_move_to_double_three_before and opp_has_move_to_double_three_after,
            config.miss_one_move_double_three_penalty,
            "未阻止对方一手成双活三",
        ),
    )
    for unresolved, unit_penalty, reason in misses:
        if not unresolved:
            continue
        if my_strong_attack:
            waived_penalty += unit_penalty
            continue
        amount = -unit_penalty * counter_scale
        if abs(amount) < 1e-8:
            continue
        details.append(RewardDetail(amount=amount, reason=f"{reason}{counter_note}"))
        total_penalty += amount

    return total_penalty, waived_penalty


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
    # 本手形成活三/跳活三：相对先手，可部分折减「一手成双活三/四三」的漏防惩罚。
    # 仅能成冲四的活三无法成活四，不算反击先手，故不计入。
    my_counter_threat = my_features.open_three > 0 or my_features.jump_open_three > 0

    board.grid[move.row, move.col] = player
    opponent_after = _scan_existing_threat_inventory(board, opponent)
    opp_can_double_three_after = _opponent_has_move_to_double_three(board, opponent)
    opp_can_four_three_after = _opponent_has_move_to_four_three(board, opponent)
    board.grid[move.row, move.col] = 0

    attack_detail_start = len(details)
    attack_reward = _accumulate_shape_reward(details, board, move, my_features, config.attack_scale, config)
    attack_detail_end = len(details)
    block_reward = _accumulate_block_reward(details, opponent_before, opponent_after, config)
    # A direct win ends the game immediately, so opponent threats no longer matter.
    miss_penalty = 0.0
    waived_miss_penalty = 0.0
    if not is_winning_move:
        miss_penalty, waived_miss_penalty = _accumulate_miss_penalty(
            details,
            opponent_before,
            opponent_after,
            config,
            my_strong_attack=my_strong_attack,
            my_counter_threat=my_counter_threat,
            opp_has_move_to_double_three_before=opp_can_double_three_before,
            opp_has_move_to_double_three_after=opp_can_double_three_after,
            opp_has_move_to_four_three_before=opp_can_four_three_before,
            opp_has_move_to_four_three_after=opp_can_four_three_after,
        )
    # 冲四靠豁免免掉了漏防惩罚时，折减它的进攻分：豁免本身在棋理上成立（绝对先手），
    # 但「远处随手冲四 + 全额豁免 + 全额进攻分」净收益为正，会成为刷分燃料。
    # 只折减主棋形恰为冲四的情况；活四/双四/四三/双活三近乎制胜，全额保留。
    if (
        waived_miss_penalty > 0.0
        and attack_reward > 0.0
        and _primary_category(my_features) == "rush_four"
    ):
        waiver_scale = _clip(config.rush_four_waiver_attack_scale, 0.0, 1.0)
        for detail in details[attack_detail_start:attack_detail_end]:
            detail.amount *= waiver_scale
            detail.reason = f"{detail.reason}（强攻豁免漏防，进攻分 x{waiver_scale:.2f}）"
        attack_reward *= waiver_scale

    missed_own_win_penalty = _accumulate_missed_own_win_penalty(details, board, move, player, config, opponent_before)

    # 错失直接获胜时，进攻奖励不该叠加——能赢不赢就不该因为"顺便形成活四"而得正分。
    # 只删进攻项自己的明细行：封堵奖励仍计入总分，按金额正负筛会把它的明细一并删掉，
    # 让明细合计对不上总分。
    if missed_own_win_penalty != 0.0 and attack_reward > 0.0:
        del details[attack_detail_start:attack_detail_end]
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
            opening_detail_start = len(details)
            opening_position_reward = (
                _accumulate_opening_position_reward(details, board, move, config) * opening_position_scale
            )
            # 缩放要同步写回明细，否则明细行是原值、总分是缩放值，两边对不上账。
            if opening_position_scale != 1.0:
                for detail in details[opening_detail_start:]:
                    detail.amount *= opening_position_scale
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
