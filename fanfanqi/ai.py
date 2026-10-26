# ai.js - AI 接口与内置简单 AI

import math, random, time
from .game import ROWS, COLS, DIRS, PIECE_CONFIG


class AIPlayer:
    """
    AI 玩家基类 — 所有 AI 实现需继承此类
    返回格式:
      翻棋: { type: 'flip', row, col, poolIndex? }
      炮吃未翻: { type: 'capture', fromRow, fromCol, toRow, toCol, poolIndex? }
      移动/吃子: { type: 'move'|'capture', fromRow, fromCol, toRow, toCol }
    """

    def __init__(self, side):
        self.side = side

    def get_move(self, game):
        raise Exception('getMove not implemented')


STRATEGIC_PIECE_VALUE = {
    '将': 100,
    '仕': 40,
    '象': 20,
    '车': 10,
    '马': 5,
    '炮': 3,
    '兵': 1,
}

HP_VALUE_WEIGHT = 2.4


# ===== 内置简单 AI =====
class SimpleAI(AIPlayer):
    def get_move(self, game):
        actions = game.get_all_actions()
        if len(actions) == 0:
            return None

        best_action = None
        best_score = -math.inf

        for action in actions:
            score = self.evaluate(game, action) + self.evaluate_opponent_response(game, action)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def choose_setup_piece(self, game, side):
        pool = game.remaining_pool
        best_idx = -1
        best_score = -math.inf

        for index, piece in enumerate(pool):
            if piece['side'] != side:
                continue
            score = piece['hp'] * 3 + piece['rank'] * 2
            if piece['type'] == '仕':
                score += 6
            if piece['type'] == '炮':
                score += 4
            if piece['type'] == '兵':
                score -= 3
            if score > best_score:
                best_score = score
                best_idx = index

        return best_idx

    def evaluate(self, game, action):
        rand = random.random() * 0.15

        if action['type'] == 'flip':
            return self.evaluate_flip(game, action) + rand

        if action['type'] == 'capture':
            return self.evaluate_capture(game, action) + rand

        if action['type'] == 'move':
            return self.evaluate_move(game, action) + rand

        return rand

    def evaluate_flip(self, game, action):
        row = action.get('row')
        col = action.get('col')
        score = 6

        cannon_mouth_score = self.score_cannon_mouth(game, row, col)
        reverse_cannon_score = self.score_reverse_cannon_mouth(game, row, col)

        score += cannon_mouth_score
        score += reverse_cannon_score
        score += self.score_secondary_cannon_mouth(game, row, col) * 0.6
        score += self.score_invincible_advisor_flip(game, row, col)
        score += self.score_around_enemy_king(game, row, col)
        score += self.score_around_enemy_pieces(game, row, col)
        score += self.score_empty_flip(game, row, col)
        score += self.score_probabilistic_flip(game, row, col)
        score += self.score_hp_race_flip_bias(game, row, col)
        score -= self.score_enemy_big_piece_flip_trap(game, row, col)
        score -= self.score_around_own_pieces(game, row, col)

        if self.is_behind(game):
            score += cannon_mouth_score * 0.5
            score += reverse_cannon_score * 0.3
            score += self.score_desperation_flip(game, row, col)

        return score

    def evaluate_capture(self, game, action):
        attacker = game.get_piece(action.get('from_row'), action.get('from_col'))
        target = game.get_piece(action.get('to_row'), action.get('to_col'))
        if not attacker or not target:
            return -999

        score = 18
        attacker_value = self.material_value(attacker['type'])

        if target['face_up']:
            score += target['hp'] * 8
            score += self.material_value(target['type']) * 0.35
            score += self.score_hp_swing(game, target['hp'], 0)
            score += (target['rank'] - attacker['rank']) * 2
            if self.is_mutual_capture_pair(attacker, target):
                score += target['hp'] * 6 + self.material_value(target['type']) * 0.25 + 10
            if target['type'] == '将':
                score += 24
            if attacker['type'] == '兵' and target['type'] == '将':
                score += 45
            if attacker['type'] == '将' and target['type'] == '仕':
                score += 18
            if target['type'] == '仕':
                score += 10
            if attacker['type'] == '将' and target['type'] == '仕':
                score += 10 if self.count_living_pieces(game, self.side, '仕') > 0 else 4
        else:
            score += self.score_cannon_mouth(game, action.get('to_row'), action.get('to_col')) * 0.5
            score += self.score_secondary_cannon_mouth(game, action.get('to_row'), action.get('to_col')) * 0.4
            score -= 20

        score += self.score_cannon_emergency_capture(game, action, attacker, target)

        if attacker['type'] == '炮' and (action.get('to_row') == 0 or action.get('to_row') == ROWS - 1):
            score += 10

        if self.is_in_danger(game, action.get('from_row'), action.get('from_col'), attacker):
            score += attacker['hp'] * 2

        if self.would_be_in_danger_after_move(game, action, attacker):
            score -= attacker['hp'] * 5
            score -= self.score_hp_swing(game, 0, attacker['hp']) * 0.45
            score -= attacker_value * 0.12
        else:
            score += 8

        score += self.score_support(action.get('to_row'), action.get('to_col'), game, self.side) * 4
        score += self.score_block_enemy(action.get('to_row'), action.get('to_col'), game) * 3
        score += self.score_protect_soldiers(action.get('to_row'), action.get('to_col'), game, attacker)
        score += self.score_protect_important_pieces(game, action, attacker, target)
        score += self.score_advisor_fortress_plan(game, action, attacker, target)
        score += self.score_endgame_material_plan(game, attacker, target)

        return score

    def evaluate_move(self, game, action):
        piece = game.get_piece(action.get('from_row'), action.get('from_col'))
        if not piece:
            return -999

        score = 4
        current_mobility = len(game.get_valid_moves(action.get('from_row'), action.get('from_col')))
        next_mobility = self.estimate_mobility_at(action.get('to_row'), action.get('to_col'), piece, game)

        score += (next_mobility - current_mobility) * 4

        if self.is_in_danger(game, action.get('from_row'), action.get('from_col'), piece):
            score += piece['hp'] * 2.5
            score += self.score_hp_swing(game, 0, piece['hp']) * -0.15

        if self.would_be_in_danger_after_move(game, action, piece):
            score -= piece['hp'] * 5
        else:
            score += 6

        score += self.score_support(action.get('to_row'), action.get('to_col'), game, self.side) * 4
        score -= 5 if self.count_adjacent_friends(game, action.get('to_row'), action.get('to_col'), self.side) > 2 else 0
        score += self.score_block_enemy(action.get('to_row'), action.get('to_col'), game) * 3
        score += self.score_protect_soldiers(action.get('to_row'), action.get('to_col'), game, piece)
        score += self.score_protect_important_pieces(game, action, piece, None)
        score += self.score_advisor_fortress_plan(game, action, piece, None)
        score += self.score_cannon_structure(action.get('to_row'), action.get('to_col'), piece, game)
        score += self.score_flexibility_plan(action.get('to_row'), action.get('to_col'), piece, game)
        score += self.score_hp_race_mobility(game, piece, action)

        if piece['type'] == '兵' and self.enemy_king_alive(game):
            if self.would_be_in_danger_after_move(game, action, piece):
                score -= 12
            else:
                score += 5

        return score

    def evaluate_opponent_response(self, game, action):
        if action['type'] == 'flip':
            return 0

        target = game.get_piece(action.get('to_row'), action.get('to_col'))
        if action['type'] == 'capture' and target and not target['face_up']:
            return 0

        snapshot = game.create_snapshot()
        history = list(game.history)

        try:
            applied = game.move_piece(action.get('from_row'), action.get('from_col'), action.get('to_row'), action.get('to_col'))
            if not applied:
                return 0
            if game.phase == 'over':
                return 120

            enemy_ai = SimpleAI(self.enemy_side())
            enemy_actions = game.get_all_actions()
            if len(enemy_actions) == 0:
                return 20

            enemy_best = -math.inf
            for enemy_action in enemy_actions:
                score = enemy_ai.evaluate(game, enemy_action)
                if score > enemy_best:
                    enemy_best = score

            return 0 if enemy_best == -math.inf else -enemy_best * 0.45
        finally:
            game.restore_snapshot(snapshot)
            game.history = history

    def score_cannon_mouth(self, game, row, col):
        score = 0
        for cannon in self.get_face_up_pieces(game, self.side, '炮'):
            moves = game.get_valid_moves(cannon['row'], cannon['col'])
            if any(move['row'] == row and move['col'] == col and move['type'] == 'capture' for move in moves):
                score += 24
                if self.has_no_perpendicular_escape(game, cannon['row'], cannon['col'], row, col):
                    score += 10
        return score

    def score_secondary_cannon_mouth(self, game, row, col):
        score = 0
        for cannon in self.get_face_up_pieces(game, self.side, '炮'):
            for dr, dc in DIRS:
                r = cannon['row'] + dr
                c = cannon['col'] + dc
                occupied_seen = 0
                first_hidden_target = None

                while r >= 0 and r < ROWS and c >= 0 and c < COLS:
                    cell = game.get_piece(r, c)
                    if cell:
                        occupied_seen += 1
                        if occupied_seen == 2:
                            first_hidden_target = {'row': r, 'col': c} if not cell['face_up'] else None
                        elif occupied_seen == 4 and first_hidden_target is not None and not cell['face_up']:
                            if r == row and c == col:
                                score += 10
                            break
                    r += dr
                    c += dc
        return score

    def score_reverse_cannon_mouth(self, game, row, col):
        score = 0
        for enemy_piece in self.get_face_up_pieces(game, self.enemy_side(), None):
            if not self.is_reverse_cannon_mouth_for_enemy(game, row, col, enemy_piece['row'], enemy_piece['col']):
                continue
            score += 16
            fake_cannon = {'side': self.side, 'type': '炮', 'rank': PIECE_CONFIG['炮']['rank'], 'hp': PIECE_CONFIG['炮']['hp'], 'face_up': True}
            if not self.is_in_danger(game, row, col, fake_cannon):
                score += 8
        return score

    def score_invincible_advisor_flip(self, game, row, col):
        score = 0
        for advisor in self.get_face_up_pieces(game, self.side, '仕'):
            if not self.is_adjacent(row, col, advisor['row'], advisor['col']):
                continue
            if not self.has_invincible_advisor_anchor(game, advisor['row'], advisor['col']):
                continue
            score += 14
            if self.is_line_open_after_flip(game, row, col, advisor['row'], advisor['col']):
                score += 6
        return score

    def score_around_enemy_king(self, game, row, col):
        enemy_kings = self.get_face_up_pieces(game, self.enemy_side(), '将')
        enemy_king = enemy_kings[0] if len(enemy_kings) > 0 else None
        if not enemy_king:
            return 0
        if not self.is_adjacent(row, col, enemy_king['row'], enemy_king['col']):
            return 0

        hidden_own_soldiers = len([piece for piece in game.remaining_pool if piece['side'] == self.side and piece['type'] == '兵'])
        probability = hidden_own_soldiers / max(len(game.remaining_pool), 1)
        score = probability * 36 - 8

        if self.is_behind(game):
            score += probability * 8

        return score

    def score_around_enemy_pieces(self, game, row, col):
        score = 0
        for nr, nc in self.neighbors(row, col):
            piece = game.get_piece(nr, nc)
            if not piece or not piece['face_up']:
                continue
            if piece['side'] == self.side:
                if piece['type'] == '仕' or piece['type'] == '兵':
                    score -= 2
                else:
                    score -= 12 if piece['hp'] >= 10 else 7
                continue

            if piece['type'] == '将':
                score += 10
            elif piece['type'] == '仕':
                score -= 14
            elif piece['type'] == '象':
                score += 4
            elif piece['type'] == '车' or piece['type'] == '马':
                score += 10
            elif piece['type'] == '炮':
                score += 8
            elif piece['type'] == '兵':
                score -= 6
        return score

    def score_around_own_pieces(self, game, row, col):
        score = 0
        for nr, nc in self.neighbors(row, col):
            piece = game.get_piece(nr, nc)
            if not piece or not piece['face_up'] or piece['side'] != self.side:
                continue

            if piece['type'] == '仕' and self.has_invincible_advisor_anchor(game, nr, nc):
                score -= 1
            elif piece['type'] == '兵':
                score += 2
            else:
                score += 12 if piece['hp'] >= 10 else 6
        return score

    def score_empty_flip(self, game, row, col):
        touching_face_up = 0
        for nr, nc in self.neighbors(row, col):
            piece = game.get_piece(nr, nc)
            if piece and piece['face_up']:
                touching_face_up += 1
        return 8 if touching_face_up == 0 else (3 if touching_face_up == 1 else 0)

    def score_support(self, row, col, game, side):
        score = 0
        for nr, nc in self.neighbors(row, col):
            piece = game.get_piece(nr, nc)
            if piece and piece['face_up'] and piece['side'] == side:
                score += 1
        return score

    def score_block_enemy(self, row, col, game):
        score = 0
        for nr, nc in self.neighbors(row, col):
            piece = game.get_piece(nr, nc)
            if not piece or not piece['face_up'] or piece['side'] != self.enemy_side():
                continue
            if piece['type'] == '炮':
                score += 3
            elif piece['hp'] >= 10:
                score += 4
            else:
                score += 2
        return score

    def score_protect_soldiers(self, row, col, game, moving_piece):
        if not self.enemy_king_alive(game):
            return 0
        score = 0

        for nr, nc in self.neighbors(row, col):
            piece = game.get_piece(nr, nc)
            if not piece or not piece['face_up'] or piece['side'] != self.side or piece['type'] != '兵':
                continue
            score += 6
            if moving_piece['type'] == '仕':
                score += 4

        return score

    def score_protect_important_pieces(self, game, action, moving_piece, target):
        score = 0

        if action['type'] == 'capture' and target and target['face_up']:
            score += self.score_capture_threat_removal(game, action.get('to_row'), action.get('to_col'), target)

        baseline_threat = self.compute_important_threat_score(game, self.side)
        baseline_support = self.count_supported_important_pieces(game, self.side)

        snapshot = game.create_snapshot()
        history = list(game.history)

        try:
            applied = game.move_piece(action.get('from_row'), action.get('from_col'), action.get('to_row'), action.get('to_col'))
            if not applied:
                return score

            next_threat = self.compute_important_threat_score(game, self.side)
            next_support = self.count_supported_important_pieces(game, self.side)
            score += (baseline_threat - next_threat) * 0.9
            score += (next_support - baseline_support) * 4.5

            for nr, nc in self.neighbors(action.get('to_row'), action.get('to_col')):
                ally = game.get_piece(nr, nc)
                if not ally or not ally['face_up'] or ally['side'] != self.side or not self.is_priority_protection_target(ally):
                    continue
                if ally['type'] == '将':
                    score += 10
                elif self.is_in_danger(game, nr, nc, ally):
                    score += 4 + self.protection_priority(game, {**ally, 'row': nr, 'col': nc}) * 2.5

            return score
        finally:
            game.restore_snapshot(snapshot)
            game.history = history

    def score_advisor_fortress_plan(self, game, action, moving_piece, target):
        if not self.is_advisor_fortress_phase(game, self.side):
            return 0

        score = 0
        before = self.compute_advisor_fortress_score(game, self.side)

        if moving_piece['type'] == '仕' and action['type'] == 'capture' and target and target['face_up']:
            removes_critical_threat = self.score_capture_threat_removal(game, action.get('to_row'), action.get('to_col'), target) >= 20
            if not removes_critical_threat and target['type'] != '将' and target['type'] != '炮':
                score -= 18

        snapshot = game.create_snapshot()
        history = list(game.history)

        try:
            applied = game.move_piece(action.get('from_row'), action.get('from_col'), action.get('to_row'), action.get('to_col'))
            if not applied:
                return score

            after = self.compute_advisor_fortress_score(game, self.side)
            score += (after - before) * 1.2

            if moving_piece['type'] == '仕':
                moved_advisor = game.get_piece(action.get('to_row'), action.get('to_col'))
                if moved_advisor and moved_advisor['type'] == '仕':
                    leaves_king_ring = not self.is_adjacent_to_own_king(game, action.get('to_row'), action.get('to_col'), self.side)
                    loses_anchor = not self.has_invincible_advisor_anchor(game, action.get('to_row'), action.get('to_col'))

                    if leaves_king_ring:
                        score -= 16
                    if loses_anchor:
                        score -= 14

                    if action['type'] == 'capture' and target and target['face_up']:
                        removes_critical_threat = self.score_capture_threat_removal(game, action.get('to_row'), action.get('to_col'), target) >= 20
                        if not removes_critical_threat and target['type'] != '将' and target['type'] != '炮':
                            score -= 26
                            if leaves_king_ring or loses_anchor:
                                score -= 18

            return score
        finally:
            game.restore_snapshot(snapshot)
            game.history = history

    def score_cannon_structure(self, row, col, piece, game):
        if piece['type'] != '炮':
            return 0
        score = 0

        if row == 0 or row == ROWS - 1:
            score += 12
        if col == 0 or col == COLS - 1:
            score += 4

        for dr, dc in DIRS:
            r = row + dr
            c = col + dc
            mount_found = False
            while r >= 0 and r < ROWS and c >= 0 and c < COLS:
                cell = game.get_piece(r, c)
                if cell:
                    if not mount_found:
                        mount_found = True
                    else:
                        if cell['face_up'] and cell['side'] == self.enemy_side():
                            score += 6
                        break
                r += dr
                c += dc

        return score

    def score_probabilistic_flip(self, game, row, col):
        pool = game.remaining_pool
        if not pool:
            return 0

        total = 0
        for piece in pool:
            total += self.score_reveal_as_piece(game, row, col, piece)

        return total / len(pool)

    def score_reveal_as_piece(self, game, row, col, piece):
        revealed = {**piece, 'face_up': True}
        value = self.material_value(piece['type'])
        support = self.score_support(row, col, game, piece['side'])
        pressure = self.score_support(row, col, game, 'black' if piece['side'] == 'red' else 'red')
        danger = self.is_in_danger(game, row, col, revealed)
        immediate_loss_risk = self.estimate_immediate_loss_risk(game, row, col, revealed)
        mobility = self.estimate_mobility_at(row, col, revealed, game)
        cannon_threats = self.count_cannon_attackers(game, row, col, 'black' if piece['side'] == 'red' else 'red')
        direct_threats = self.count_direct_attackers(game, row, col, revealed, 'black' if piece['side'] == 'red' else 'red')
        direct_pressure = direct_threats + cannon_threats * 2
        score = 0

        if piece['side'] == self.side:
            score += support * 4
            score += mobility * 3
            score -= pressure * 4

            if danger:
                score -= value * 0.14
            score -= immediate_loss_risk
            score -= self.score_hp_swing(game, 0, self.piece_hp(piece['type'])) * 0.18

            if immediate_loss_risk > 0 and support == 0:
                score -= 6

            if piece['type'] == '兵' and self.is_adjacent_to_enemy_king(game, row, col):
                score += 32 + self.count_living_pieces(game, self.side, '兵') * 2

            if piece['type'] == '仕' and self.has_invincible_advisor_anchor(game, row, col):
                score += 22

            if piece['type'] == '将':
                score -= 8
        else:
            score += self.count_direct_attackers(game, row, col, revealed, self.side) * (8 + value * 0.04)
            score += self.count_cannon_attackers(game, row, col, self.side) * (10 + value * 0.05)
            score += self.score_hp_swing(game, self.piece_hp(piece['type']), 0) * 0.22
            score -= mobility * 2.5
            score -= support * 2
            score -= direct_pressure * 2

            if piece['type'] == '将' and self.count_living_pieces(game, self.side, '兵') > 0:
                score += 20

            if piece['type'] == '仕':
                score -= 10
            if piece['type'] == '兵':
                score -= 6

        return score

    def score_desperation_flip(self, game, row, col):
        pool = game.remaining_pool
        if not pool:
            return 0

        comeback_weight = 0
        for piece in pool:
            if piece['side'] != self.side:
                continue
            if piece['type'] == '兵' and self.is_adjacent_to_enemy_king(game, row, col):
                comeback_weight += 10
            elif piece['type'] == '炮':
                comeback_weight += 5 if self.score_cannon_mouth(game, row, col) > 0 else 2
            elif piece['type'] == '车' or piece['type'] == '马':
                comeback_weight += 3

        return comeback_weight / len(pool)

    def score_hp_race_flip_bias(self, game, row, col):
        score = 0
        own_hp = game.hp[self.side]
        enemy_hp = game.hp[self.enemy_side()]

        if own_hp <= 20:
            score += self.score_empty_flip(game, row, col) * 1.2
            score -= self.score_enemy_big_piece_flip_trap(game, row, col) * 0.15

        if enemy_hp <= 20 and self.is_adjacent_to_enemy_king(game, row, col):
            score += 8

        if own_hp + 10 < enemy_hp:
            score += self.score_cannon_mouth(game, row, col) * 0.12

        return score

    def score_enemy_big_piece_flip_trap(self, game, row, col):
        own_pool = [piece for piece in game.remaining_pool if piece['side'] == self.side]
        if not own_pool:
            return 0

        penalty = 0
        for nr, nc in self.neighbors(row, col):
            piece = game.get_piece(nr, nc)
            if not piece or not piece['face_up'] or piece['side'] != self.enemy_side():
                continue
            if not self.is_major_threat_for_flip(piece):
                continue

            expected_loss = 0
            for own_piece in own_pool:
                revealed = {**own_piece, 'face_up': True}
                if piece['type'] == '炮':
                    if self.can_cannon_attack_square(game, nr, nc, row, col):
                        expected_loss += self.material_value(own_piece['type']) * 0.7
                    continue

                if self.can_capture_like(piece, revealed):
                    expected_loss += self.material_value(own_piece['type'])

            average_loss = expected_loss / len(own_pool)
            penalty += average_loss * (1.35 if piece['type'] == '仕' else 1.05)

        return penalty


    def score_cannon_emergency_capture(self, game, action, attacker, target):
        if not attacker or attacker['type'] != '炮':
            return 0

        immediate_threats = self.count_adjacent_capturers(game, action['from_row'], action['from_col'], attacker, self.enemy_side())
        if immediate_threats == 0:
            return 0

        score = immediate_threats * 18
        score += 8 if target['face_up'] else 26

        if not self.would_be_in_danger_after_move(game, action, attacker):
            score += 18

        if action['to_row'] == 0 or action['to_row'] == ROWS - 1 or action['to_col'] == 0 or action['to_col'] == COLS - 1:
            score += 6

        return score

    def score_capture_threat_removal(self, game, row, col, target):
        score = 0
        for ally in self.get_face_up_pieces(game, self.side, None):
            if not self.is_priority_protection_target(ally):
                continue
            if not self.enemy_piece_threatens_square(game, target, row, col, ally['row'], ally['col'], ally):
                continue

            score += 8 + self.protection_priority(game, ally) * 10 + self.material_value(ally['type']) * 0.05
            score += self.score_hp_swing(game, (target.get('hp') if target.get('hp') is not None else self.piece_hp(target.get('type'))), 0) * 0.2
            if target['type'] == '兵' and ally['type'] == '将':
                score += 20
            if target['type'] == '炮':
                score += 8

        return score

    def score_endgame_material_plan(self, game, attacker, target):
        score = 0
        score += self.score_hp_swing(game, (target.get('hp') if target.get('hp') is not None else self.piece_hp(target.get('type'))), 0) * 0.35

        if self.enemy_king_alive(game):
            if target['type'] == '仕' and attacker['type'] == '将':
                score += 8
            if target['type'] == '兵':
                score -= 4

        if attacker['type'] == '兵' and target['type'] != '将':
            score -= 3

        return score

    def score_hp_race_mobility(self, game, piece, action):
        score = 0
        own_hp = game.hp[self.side]
        enemy_hp = game.hp[self.enemy_side()]

        if own_hp <= 20 and not self.would_be_in_danger_after_move(game, action, piece):
            score += piece['hp'] * 0.8

        if enemy_hp <= 15 and piece['type'] == '兵' and self.is_adjacent_to_enemy_king(game, action['to_row'], action['to_col']):
            score += 10

        return score

    def score_flexibility_plan(self, row, col, piece, game):
        score = 0
        mobility = self.estimate_mobility_at(row, col, piece, game)

        if piece['type'] == '将' or piece['type'] == '仕':
            score += mobility * 2

        friend_count = self.count_adjacent_friends(game, row, col, self.side)
        if friend_count == 0:
            score += 2
        if friend_count >= 3:
            score -= 4

        for dr, dc in DIRS:
            r = row + dr
            c = col + dc
            blockers = 0
            while r >= 0 and r < ROWS and c >= 0 and c < COLS:
                cell = game.get_piece(r, c)
                if cell:
                    blockers += 1
                    if blockers == 2:
                        if cell['face_up'] and cell['side'] == self.enemy_side() and cell['type'] == '炮':
                            score -= 5
                        break
                r += dr
                c += dc

        return score

    def estimate_immediate_loss_risk(self, game, row, col, piece):
        enemy = 'black' if piece['side'] == 'red' else 'red'
        value = self.material_value(piece['type'])
        risk = 0

        for enemy_piece in self.get_face_up_pieces(game, enemy, None):
            if enemy_piece['type'] == '炮':
                if self.can_cannon_attack_square(game, enemy_piece['row'], enemy_piece['col'], row, col):
                    risk += value * 0.18 + self.piece_hp(piece['type']) * 1.4 + 6
                continue

            if not self.is_adjacent(enemy_piece['row'], enemy_piece['col'], row, col):
                continue
            if not self.can_capture_like(enemy_piece, piece):
                continue

            risk += value * (0.2 if self.is_major_threat_for_flip(enemy_piece) else 0.12)
            risk += self.piece_hp(piece['type']) * 1.1
            risk += self.material_value(enemy_piece['type']) * 0.12

        return risk

    def compute_important_threat_score(self, game, side):
        score = 0

        for piece in self.get_face_up_pieces(game, side, None):
            if not self.is_priority_protection_target(piece):
                continue
            threat_count = self.count_threats_to_piece(game, piece['row'], piece['col'], piece)
            if threat_count == 0:
                continue

            priority = self.protection_priority(game, piece)
            piece_threat = threat_count * (5 + priority * 4 + self.material_value(piece['type']) * 0.04 + self.piece_hp(piece['type']) * 0.5)
            if piece['type'] == '将':
                piece_threat += 34
                piece_threat += self.count_adjacent_enemy_soldiers(game, piece['row'], piece['col'], piece['side']) * 20
            elif piece['type'] == '仕':
                piece_threat += 6 + self.score_advisor_structure_value(game, piece)
            elif piece['type'] == '炮':
                piece_threat += 14
            elif piece['type'] == '车':
                piece_threat += 8
            elif piece['type'] == '马':
                piece_threat += 6

            if self.is_in_danger(game, piece['row'], piece['col'], piece):
                piece_threat += 8 + priority * 3

            score += piece_threat

        return score

    def compute_advisor_fortress_score(self, game, side):
        advisors = self.get_face_up_pieces(game, side, '仕')
        if len(advisors) == 0:
            return 0

        score = 0
        _kings = self.get_face_up_pieces(game, side, '将')
        own_king = _kings[0] if _kings else None

        for advisor in advisors:
            score += 12
            if self.has_invincible_advisor_anchor(game, advisor['row'], advisor['col']):
                score += 14
            if own_king and self.is_adjacent(advisor['row'], advisor['col'], own_king['row'], own_king['col']):
                score += 10
            if not self.is_in_danger(game, advisor['row'], advisor['col'], advisor):
                score += 8

        if len(advisors) >= 2:
            score += 18
            if any(self.has_invincible_advisor_anchor(game, advisor['row'], advisor['col']) for advisor in advisors):
                score += 10
            if own_king and all(self.manhattan_distance(advisor['row'], advisor['col'], own_king['row'], own_king['col']) <= 2 for advisor in advisors):
                score += 10

        return score

    def count_supported_important_pieces(self, game, side):
        count = 0
        for piece in self.get_face_up_pieces(game, side, None):
            if not self.is_priority_protection_target(piece):
                continue
            if self.score_support(piece['row'], piece['col'], game, side) > 0:
                count += self.protection_priority(game, piece)
        return count

    def estimate_mobility_at(self, row, col, piece, game):
        if piece['type'] == '炮':
            return self.score_cannon_structure(row, col, piece, game) / 3

        mobility = 0
        for dr, dc in DIRS:
            nr = row + dr
            nc = col + dc
            if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
                continue
            cell = game.get_piece(nr, nc)
            if not cell:
                mobility += 1
            elif cell['face_up'] and cell['side'] != piece['side'] and self.can_capture_like(piece, cell):
                mobility += 1.5
        return mobility

    def can_capture_like(self, attacker, defender):
        if not defender.get('face_up') or attacker.get('side') == defender.get('side'):
            return False
        if attacker.get('type') == '兵' and defender.get('type') == '将':
            return True
        if attacker.get('type') == '将' and defender.get('type') == '兵':
            return False
        # JS: undefined/null rank 的比较结果为 false（暗子 rank 为 undefined）
        a_rank = attacker.get('rank')
        d_rank = defender.get('rank')
        if a_rank is None or d_rank is None:
            return False
        return a_rank >= d_rank

    def is_mutual_capture_pair(self, attacker, defender):
        if not attacker or not defender or attacker.get('side') == defender.get('side'):
            return False
        visible_attacker = attacker if attacker.get('face_up') else {**attacker, 'face_up': True}
        visible_defender = defender if defender.get('face_up') else {**defender, 'face_up': True}
        return (self.can_capture_like(visible_attacker, visible_defender)
                and self.can_capture_like(visible_defender, visible_attacker))

    def material_value(self, type):
        _v = STRATEGIC_PIECE_VALUE.get(type)
        return (_v if _v is not None else 0) + self.piece_hp(type) * HP_VALUE_WEIGHT

    def piece_hp(self, type):
        _cfg = PIECE_CONFIG.get(type)
        _hp = _cfg.get('hp') if _cfg is not None else None
        return _hp if _hp is not None else 0

    def score_hp_swing(self, game, enemy_damage, self_damage):
        enemy_remaining = game.hp[self.enemy_side()]
        own_remaining = game.hp[self.side]
        next_enemy_hp = enemy_remaining - enemy_damage
        next_own_hp = own_remaining - self_damage
        score = enemy_damage * 4.2 - self_damage * 5.6

        if enemy_damage > 0:
            if next_enemy_hp <= 0:
                score += 220
            elif next_enemy_hp <= 10:
                score += (12 - next_enemy_hp) * 8
            elif next_enemy_hp <= 20:
                score += (22 - next_enemy_hp) * 3

        if self_damage > 0:
            if next_own_hp <= 0:
                score -= 260
            elif next_own_hp <= 10:
                score -= (12 - next_own_hp) * 10
            elif next_own_hp <= 20:
                score -= (22 - next_own_hp) * 4

        if own_remaining <= 20:
            score -= self_damage * 2.5
        if enemy_remaining <= 20:
            score += enemy_damage * 2.2

        return score

    def is_priority_protection_target(self, piece):
        return piece['type'] == '将' or piece['type'] == '仕' or piece['type'] == '炮' or piece['type'] == '车' or piece['type'] == '马'

    def is_advisor_fortress_phase(self, game, side):
        if self.count_face_up_pieces(game, side, '仕') < 2:
            return False
        return self.count_all_face_up_pieces(game) <= 10 or not self.enemy_king_alive(game) or game.hp[side] >= game.hp[self.enemy_side()] - 8

    def protection_priority(self, game, piece):
        if piece['type'] == '将':
            return 3.6
        if piece['type'] == '炮':
            return 2.4
        if piece['type'] == '车':
            return 1.9
        if piece['type'] == '马':
            return 1.6
        if piece['type'] == '仕':
            priority = 1.7
            if self.count_face_up_pieces(game, piece['side'], '仕') >= 2:
                priority += 0.9
            if self.has_invincible_advisor_anchor(game, piece['row'], piece['col']):
                priority += 0.7
            return priority
        return 1

    def score_advisor_structure_value(self, game, piece):
        if piece['type'] != '仕':
            return 0

        score = 0
        if self.count_face_up_pieces(game, piece['side'], '仕') >= 2:
            score += 12
        if self.has_invincible_advisor_anchor(game, piece['row'], piece['col']):
            score += 10
        if self.is_adjacent_to_own_king(game, piece['row'], piece['col'], piece['side']):
            score += 8
        return score

    def is_major_threat_for_flip(self, piece):
        return piece['type'] == '将' or piece['type'] == '仕' or piece['type'] == '象' or piece['type'] == '车' or piece['type'] == '马' or piece['type'] == '炮'

    def is_adjacent_to_enemy_king(self, game, row, col):
        _kings = self.get_face_up_pieces(game, self.enemy_side(), '将')
        enemy_king = _kings[0] if _kings else None
        return bool(enemy_king) and self.is_adjacent(row, col, enemy_king['row'], enemy_king['col'])

    def is_adjacent_to_own_king(self, game, row, col, side):
        _kings = self.get_face_up_pieces(game, side, '将')
        own_king = _kings[0] if _kings else None
        return bool(own_king) and self.is_adjacent(row, col, own_king['row'], own_king['col'])

    def manhattan_distance(self, r1, c1, r2, c2):
        return abs(r1 - r2) + abs(c1 - c2)

    def count_direct_attackers(self, game, row, col, defender, attacker_side):
        count = 0
        for attacker in self.get_face_up_pieces(game, attacker_side, None):
            if attacker['type'] == '炮':
                continue
            if not self.is_adjacent(attacker['row'], attacker['col'], row, col):
                continue
            if self.can_capture_like(attacker, defender):
                count += 1
        return count

    def count_threats_to_piece(self, game, row, col, defender):
        enemy = 'black' if defender['side'] == 'red' else 'red'
        return self.count_direct_attackers(game, row, col, defender, enemy) + self.count_cannon_attackers(game, row, col, enemy)

    def count_face_up_pieces(self, game, side, type):
        return len(game.face_up_pieces(side, type or None))

    def count_all_face_up_pieces(self, game):
        count = 0
        for r in range(ROWS):
            for c in range(COLS):
                piece = game.get_piece(r, c)
                if piece and piece['face_up']:
                    count += 1
        return count

    def count_adjacent_enemy_soldiers(self, game, row, col, side):
        enemy = 'black' if side == 'red' else 'red'
        count = 0
        for nr, nc in self.neighbors(row, col):
            piece = game.get_piece(nr, nc)
            if not piece or not piece['face_up'] or piece['side'] != enemy or piece['type'] != '兵':
                continue
            count += 1
        return count

    def enemy_piece_threatens_square(self, game, enemy_piece, enemy_row, enemy_col, target_row, target_col, defender):
        if enemy_piece['type'] == '炮':
            return self.can_cannon_attack_square(game, enemy_row, enemy_col, target_row, target_col)

        return self.is_adjacent(enemy_row, enemy_col, target_row, target_col) and self.can_capture_like(enemy_piece, defender)

    def count_adjacent_capturers(self, game, row, col, defender, attacker_side):
        count = 0
        for nr, nc in self.neighbors(row, col):
            attacker = game.get_piece(nr, nc)
            if not attacker or not attacker['face_up'] or attacker['side'] != attacker_side or attacker['type'] == '炮':
                continue
            if self.can_capture_like(attacker, defender):
                count += 1
        return count

    def count_cannon_attackers(self, game, row, col, attacker_side):
        count = 0
        for cannon in self.get_face_up_pieces(game, attacker_side, '炮'):
            if self.can_cannon_attack_square(game, cannon['row'], cannon['col'], row, col):
                count += 1
        return count

    def can_cannon_attack_square(self, game, from_row, from_col, to_row, to_col):
        if from_row != to_row and from_col != to_col:
            return False

        dr = (to_row > from_row) - (to_row < from_row)
        dc = (to_col > from_col) - (to_col < from_col)
        r = from_row + dr
        c = from_col + dc
        blockers = 0

        while r >= 0 and r < ROWS and c >= 0 and c < COLS:
            if r == to_row and c == to_col:
                return blockers == 1
            if game.get_piece(r, c):
                blockers += 1
            r += dr
            c += dc

        return False

    def would_be_in_danger_after_move(self, game, action, piece):
        enemy = self.enemy_side()

        for dr, dc in DIRS:
            adj = game.get_piece(action['to_row'] + dr, action['to_col'] + dc)
            if not adj or not adj['face_up'] or adj['side'] != enemy or adj['type'] == '炮':
                continue
            if adj['type'] == '兵' and piece['type'] == '将':
                return True
            if piece['type'] == '兵' and adj['type'] == '将':
                continue
            if adj['rank'] >= piece['rank']:
                return True

        for dr, dc in DIRS:
            r = action['to_row'] + dr
            c = action['to_col'] + dc
            screen_count = 0

            while r >= 0 and r < ROWS and c >= 0 and c < COLS:
                is_origin = r == action.get('from_row') and c == action.get('from_col')
                cell = None if is_origin else game.get_piece(r, c)
                if cell:
                    screen_count += 1
                    if screen_count == 2:
                        if cell['face_up'] and cell['side'] == enemy and cell['type'] == '炮':
                            return True
                        break
                r += dr
                c += dc

        return False

    def is_in_danger(self, game, row, col, piece):
        enemy = 'black' if piece['side'] == 'red' else 'red'

        for dr, dc in DIRS:
            adj = game.get_piece(row + dr, col + dc)
            if not adj or not adj['face_up'] or adj['side'] != enemy or adj['type'] == '炮':
                continue
            if adj['type'] == '兵' and piece['type'] == '将':
                return True
            if piece['type'] == '兵' and adj['type'] == '将':
                continue
            if adj['rank'] >= piece['rank']:
                return True

        for dr, dc in DIRS:
            r = row + dr
            c = col + dc
            found_mount = False
            while r >= 0 and r < ROWS and c >= 0 and c < COLS:
                p = game.get_piece(r, c)
                if p:
                    if not found_mount:
                        found_mount = True
                    else:
                        if p['face_up'] and p['side'] == enemy and p['type'] == '炮':
                            return True
                        break
                r += dr
                c += dc

        return False

    def enemy_side(self):
        return 'black' if self.side == 'red' else 'red'

    def is_behind(self, game):
        return game.hp[self.side] + 8 < game.hp[self.enemy_side()]

    def enemy_king_alive(self, game):
        return self.count_living_pieces(game, self.enemy_side(), '将') > 0

    def count_living_pieces(self, game, side, type):
        # [Python 版性能优化] 棋盘部分使用按局面缓存的明子列表，结果与逐格扫描一致
        count = len(game.face_up_pieces(side, type or None))

        for piece in game.remaining_pool:
            if piece['side'] != side:
                continue
            if not type or piece['type'] == type:
                count += 1

        return count

    def get_face_up_pieces(self, game, side, type):
        # [Python 版性能优化] 按局面缓存，避免每次逐格扫描；返回新列表，元素 dict 只读
        return list(game.face_up_pieces(side, type or None))

    def neighbors(self, row, col):
        cells = []
        for dr, dc in DIRS:
            nr = row + dr
            nc = col + dc
            if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
                continue
            cells.append([nr, nc])
        return cells

    def is_adjacent(self, r1, c1, r2, c2):
        return abs(r1 - r2) + abs(c1 - c2) == 1

    def has_invincible_advisor_anchor(self, game, row, col):
        if not self.enemy_king_alive(game):
            return True

        def _is_anchor(nr, nc):
            piece = game.get_piece(nr, nc)
            return bool(piece and piece['face_up'] and piece['side'] == self.side and piece['type'] == '兵')

        return any(_is_anchor(nr, nc) for nr, nc in self.neighbors(row, col))

    def is_line_open_after_flip(self, game, flip_row, flip_col, piece_row, piece_col):
        dr = (flip_row > piece_row) - (flip_row < piece_row)
        dc = (flip_col > piece_col) - (flip_col < piece_col)
        if abs(flip_row - piece_row) + abs(flip_col - piece_col) != 1:
            return False
        next_row = flip_row + dr
        next_col = flip_col + dc
        if next_row < 0 or next_row >= ROWS or next_col < 0 or next_col >= COLS:
            return False
        return not game.get_piece(next_row, next_col)

    def has_no_perpendicular_escape(self, game, cannon_row, cannon_col, row, col):
        same_row = cannon_row == row
        escape_dirs = [[-1, 0], [1, 0]] if same_row else [[0, -1], [0, 1]]
        for dr, dc in escape_dirs:
            nr = row + dr
            nc = col + dc
            if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
                continue
            if not game.get_piece(nr, nc):
                return False
        return True

    def is_reverse_cannon_mouth_for_enemy(self, game, row, col, enemy_row, enemy_col):
        if row != enemy_row and col != enemy_col:
            return False

        dr = (enemy_row > row) - (enemy_row < row)
        dc = (enemy_col > col) - (enemy_col < col)
        r = row + dr
        c = col + dc
        blockers = 0

        while r >= 0 and r < ROWS and c >= 0 and c < COLS:
            if r == enemy_row and c == enemy_col:
                return blockers == 1
            if game.get_piece(r, c):
                blockers += 1
            r += dr
            c += dc

        return False

    def count_adjacent_friends(self, game, row, col, side):
        count = 0
        for nr, nc in self.neighbors(row, col):
            piece = game.get_piece(nr, nc)
            if piece and piece['face_up'] and piece['side'] == side:
                count += 1
        return count


class StrategicV2AI(SimpleAI):
    def choose_setup_piece(self, game, side):
        pos = game.get_current_setup_pos()
        best_idx = -1
        best_score = -math.inf

        for index, piece in enumerate(game.remaining_pool):
            if piece['side'] != side:
                continue

            score = piece['hp'] * 2 + piece['rank'] * 2.5
            if piece['type'] == '仕':
                score += 12
            if piece['type'] == '炮':
                score += 10
            if piece['type'] == '车':
                score += 7
            if piece['type'] == '马':
                score += 6
            if piece['type'] == '兵':
                score -= 8

            # 将帅重要但不急着过早亮出，避免一开始就承担波动与围攻风险
            if piece['type'] == '将':
                score += 4 if game.setup_index >= 2 else -6

            # 边线位置更适合炮/车，内线更适合仕/将构型
            if pos:
                is_edge_file = pos['col'] == 0 or pos['col'] == COLS - 1
                if is_edge_file:
                    if piece['type'] == '炮':
                        score += 5
                    if piece['type'] == '车':
                        score += 3
                    if piece['type'] == '将':
                        score -= 4
                else:
                    if piece['type'] == '仕':
                        score += 4
                    if piece['type'] == '将':
                        score += 2

            if score > best_score:
                best_score = score
                best_idx = index

        return best_idx

    def evaluate_flip(self, game, action):
        score = super().evaluate_flip(game, action)
        row = action.get('row')
        col = action.get('col')

        # 防范未然 + 安全意识：优先翻相对安全暗子，避免最坏情况突变
        score += self.score_safe_hidden_flip(game, row, col)
        score -= self.score_worst_case_flip_risk(game, row, col)

        # 注意打串：翻子后尽量不暴露我方大子于敌炮连打路线
        score -= self.score_enemy_cannon_chain_exposure_if_flipped(game, row, col)

        # 逆风时允许在可接受范围内搏杀（尤其贴近敌将区域）
        if self.is_losing_state(game):
            score += self.score_comeback_flip_gamble(game, row, col)

        # 优势求稳
        if self.is_winning_state(game):
            score -= self.score_aggressive_flip_tax(game, row, col)

        score -= self.score_flip_volatility_control(game, action)

        return score

    def evaluate_capture(self, game, action):
        attacker = game.get_piece(action.get('from_row'), action.get('from_col'))
        target = game.get_piece(action.get('to_row'), action.get('to_col'))
        score = super().evaluate_capture(game, action)
        if not attacker or not target:
            return score

        # 叼住不吃：对已经“死型”的低价值目标不过早兑现；多兵时优先先吃一个
        score += self.score_restrain_capture(game, action, attacker, target)

        # 注意打串：规划己方炮线并切断敌方炮线
        score += self.score_cannon_line_safety(game, action)

        # 有利地形：压缩敌方阵型到边角，己方往开阔处展开
        score += self.score_terrain_direct(game, action, attacker)

        # 优势求稳：大优时降低冒险硬打；劣势时允许更激进的反扑
        score += self.score_advantage_stability(game, action, attacker, target)
        score += self.score_follow_up_pressure(game, action, attacker)

        return score

    def evaluate_move(self, game, action):
        piece = game.get_piece(action.get('from_row'), action.get('from_col'))
        score = super().evaluate_move(game, action)
        if not piece:
            return score

        score += self.score_cannon_line_safety(game, action)
        score += self.score_terrain_direct(game, action, piece)
        score += self.score_advantage_stability(game, action, piece, None)
        score += self.score_follow_up_pressure(game, action, piece)

        return score

    def is_winning_state(self, game):
        return game.hp[self.side] - game.hp[self.enemy_side()] >= 15

    def is_losing_state(self, game):
        return game.hp[self.side] + 12 < game.hp[self.enemy_side()]

    def own_king_face_up(self, game):
        return self.count_face_up_pieces(game, self.side, '将') > 0

    def score_safe_hidden_flip(self, game, row, col):
        face_up_around = 0
        enemy_pressure = 0
        own_support = 0

        for nr, nc in self.neighbors(row, col):
            p = game.get_piece(nr, nc)
            if not p or not p['face_up']:
                continue
            face_up_around += 1
            if p['side'] == self.side:
                own_support += 1
            else:
                enemy_pressure += 2.2 if p['type'] == '炮' else 1

        enemy_cannon_pressure = self.count_cannon_attackers(game, row, col, self.enemy_side())
        own_cannon_support = self.count_cannon_attackers(game, row, col, self.side)

        # 相对安全暗子：附近显性子少、敌方炮压小、我方可控性高
        return 3 - face_up_around * 0.8 - enemy_pressure * 1.6 - enemy_cannon_pressure * 2 + own_support * 1.2 + own_cannon_support * 0.8

    def score_worst_case_flip_risk(self, game, row, col):
        pool = game.remaining_pool
        if not pool:
            return 0

        worst = math.inf
        for piece in pool:
            reveal_score = self.score_reveal_as_piece(game, row, col, piece)
            if reveal_score < worst:
                worst = reveal_score

        if worst >= 0:
            return 0
        return abs(worst) * 0.08

    def score_enemy_cannon_chain_exposure_if_flipped(self, game, row, col):
        enemy_cannons = self.get_face_up_pieces(game, self.enemy_side(), '炮')
        if len(enemy_cannons) == 0:
            return 0

        penalty = 0
        for enemy_cannon in enemy_cannons:
            if self.can_cannon_attack_square(game, enemy_cannon['row'], enemy_cannon['col'], row, col):
                penalty += 2.5
        return penalty

    def score_comeback_flip_gamble(self, game, row, col):
        _kings = self.get_face_up_pieces(game, self.enemy_side(), '将')
        enemy_king = _kings[0] if _kings else None
        if not enemy_king:
            return 0
        dist = self.manhattan_distance(row, col, enemy_king['row'], enemy_king['col'])
        if dist > 1:
            return 0

        own_soldiers = self.count_living_pieces(game, self.side, '兵')
        own_cannons = self.count_living_pieces(game, self.side, '炮')
        return 8 + own_soldiers * 2.0 + own_cannons * 1.2

    def score_aggressive_flip_tax(self, game, row, col):
        enemy_neighbors = 0
        for nr, nc in self.neighbors(row, col):
            p = game.get_piece(nr, nc)
            if p and p['face_up'] and p['side'] == self.enemy_side():
                enemy_neighbors += 1

        return enemy_neighbors * 1.2

    def score_flip_volatility_control(self, game, action):
        actions = game.get_all_actions()
        safe_captures = 0
        safe_moves = 0

        for candidate in actions:
            if candidate['type'] == 'flip':
                continue
            piece = game.get_piece(candidate.get('from_row'), candidate.get('from_col'))
            if not piece:
                continue
            is_safe = not self.would_be_in_danger_after_move(game, candidate, piece)

            if candidate['type'] == 'capture':
                target = game.get_piece(candidate.get('to_row'), candidate.get('to_col'))
                if target and target['face_up'] and is_safe:
                    safe_captures += 1
            elif candidate['type'] == 'move' and is_safe:
                safe_moves += 1

        penalty = 0
        if safe_captures > 0:
            penalty += 8 if self.is_winning_state(game) else (2 if self.is_losing_state(game) else 5)
        if safe_moves >= 3:
            penalty += 4 if self.is_winning_state(game) else 2

        if self.is_losing_state(game):
            penalty *= 0.45

        if len(game.remaining_pool) <= 8:
            penalty *= 0.7

        return penalty

    def score_restrain_capture(self, game, action, attacker, target):
        if not target['face_up']:
            return 0

        score = 0
        enemy_soldiers = self.count_living_pieces(game, self.enemy_side(), '兵')

        if target['type'] == '兵' and enemy_soldiers >= 2:
            # 双兵/多兵时建议先吃一个，防将帅被围
            score += 6

        return score

    def is_likely_dead_target(self, game, row, col, target):
        threat = self.count_direct_attackers(game, row, col, target, self.side) + self.count_cannon_attackers(game, row, col, self.side)
        support = self.count_adjacent_friends(game, row, col, target['side'])
        mobility = self.estimate_mobility_at(row, col, target, game)
        return threat >= 2 and support == 0 and mobility <= 1

    def score_cannon_line_safety(self, game, action):
        to_row = action.get('to_row')
        to_col = action.get('to_col')
        score = 0

        # 奖励：己方炮从 action 目标位获得新的攻击线
        for cannon in self.get_face_up_pieces(game, self.side, '炮'):
            if cannon['row'] == action.get('from_row') and cannon['col'] == action.get('from_col'):
                continue
            if self.can_cannon_attack_square(game, cannon['row'], cannon['col'], to_row, to_col):
                score += 3

        return score

    def score_terrain_direct(self, game, action, piece):
        score = 0

        # 将帅特殊地形评估：不宜缩角、不宜在走廊狭道
        if piece['type'] == '将':
            to_row = action.get('to_row')
            to_col = action.get('to_col')
            open_ = self.count_open_neighbors(game, to_row, to_col)
            corner_dist = self.min_corner_distance(to_row, to_col)
            score += open_ * 1.8
            if corner_dist == 0:
                score -= 8
            if open_ <= 1:
                score -= 6

        return score

    def score_follow_up_pressure(self, game, action, piece):
        if not piece or (piece['type'] != '马' and piece['type'] != '车' and piece['type'] != '兵' and piece['type'] != '炮'):
            return 0

        snapshot = game.create_snapshot()
        history = list(game.history)

        try:
            applied = game.move_piece(action.get('from_row'), action.get('from_col'), action.get('to_row'), action.get('to_col'))
            if not applied:
                return 0

            original_turn = game.current_turn
            game.current_turn = self.side
            follow_ups = game.get_valid_moves(action.get('to_row'), action.get('to_col'))
            game.current_turn = original_turn

            capture_count = len([move for move in follow_ups if move['type'] == 'capture'])
            if capture_count == 0:
                return 0

            score = 7 if capture_count >= 2 else 3
            if piece['type'] == '兵' and capture_count >= 1:
                score += 2
            if piece['type'] == '车' or piece['type'] == '马':
                score += capture_count

            return score * 0.25 if self.would_be_in_danger_after_move(game, action, piece) else score
        finally:
            game.restore_snapshot(snapshot)
            game.history = history

    def count_open_neighbors(self, game, row, col):
        open_ = 0
        for nr, nc in self.neighbors(row, col):
            if not game.get_piece(nr, nc):
                open_ += 1
        return open_

    def min_corner_distance(self, row, col):
        corners = [
            [0, 0],
            [0, COLS - 1],
            [ROWS - 1, 0],
            [ROWS - 1, COLS - 1],
        ]
        best = math.inf
        for cr, cc in corners:
            best = min(best, self.manhattan_distance(row, col, cr, cc))
        return best

    def score_advantage_stability(self, game, action, attacker, target):
        score = 0
        winning = self.is_winning_state(game)
        losing = self.is_losing_state(game)

        if winning:
            # 大优求稳：避免无必要的高风险交换
            if self.would_be_in_danger_after_move(game, action, attacker):
                score -= attacker['hp'] * 0.6

            if attacker['type'] == '炮' and game.hp[self.side] <= 30 and not self.own_king_face_up(game):
                # 低血且将帅未出时，避免炮硬打导致波动
                score -= 7

            if target and target['face_up'] and target['type'] == '兵' and attacker['type'] != '兵':
                score -= 2

        if losing:
            # 劣势时允许适度冒险，尤其贴敌将区域
            if action['type'] == 'flip':
                score += self.score_comeback_flip_gamble(game, action.get('row'), action.get('col'))
            else:
                _kings = self.get_face_up_pieces(game, self.enemy_side(), '将')
                enemy_king = _kings[0] if _kings else None
                if enemy_king:
                    d = self.manhattan_distance(action.get('to_row'), action.get('to_col'), enemy_king['row'], enemy_king['col'])
                    if d <= 1:
                        score += 12
                    elif d <= 2:
                        score += 5
                # 劣势时积极吃子减小HP差距
                if action['type'] == 'capture' and target and target['face_up']:
                    score += target['hp'] * 0.8

        return score


class StrategicV3AI(StrategicV2AI):
    def choose_setup_piece(self, game, side):
        pos = game.get_current_setup_pos()
        own_setup_pieces = self.get_face_up_pieces(game, side, None)
        own_types = [piece['type'] for piece in own_setup_pieces]
        best_idx = -1
        best_score = -math.inf

        for index, piece in enumerate(game.remaining_pool):
            if piece['side'] != side:
                continue

            score = piece['hp'] * 2 + piece['rank'] * 2.5
            if piece['type'] == '仕':
                score += 12
            if piece['type'] == '炮':
                score += 10
            if piece['type'] == '车':
                score += 7
            if piece['type'] == '马':
                score += 6
            if piece['type'] == '兵':
                score -= 8

            if piece['type'] == '将':
                score += 4 if game.setup_index >= 2 else -8

            if pos:
                is_edge_file = pos['col'] == 0 or pos['col'] == COLS - 1
                if is_edge_file:
                    if piece['type'] == '炮':
                        score += 5
                    if piece['type'] == '车':
                        score += 3
                    if piece['type'] == '将':
                        score -= 5
                else:
                    if piece['type'] == '仕':
                        score += 4
                    if piece['type'] == '将':
                        score += 2

            score += self.score_setup_pair_synergy(piece['type'], own_setup_pieces, pos, game.setup_index)
            score += self.score_weak_side_setup_tuning(side, piece['type'], own_types, pos, game.setup_index)

            if score > best_score:
                best_score = score
                best_idx = index

        return best_idx

    def evaluate_flip(self, game, action):
        score = super().evaluate_flip(game, action)
        score += self.score_weak_side_flip_tuning(game, action)
        return score

    def evaluate_capture(self, game, action):
        attacker = game.get_piece(action.get('from_row'), action.get('from_col'))
        target = game.get_piece(action.get('to_row'), action.get('to_col'))
        score = super().evaluate_capture(game, action)
        if not attacker or not target:
            return score

        score += self.score_weak_side_action_tuning(game, action, attacker, target)
        return score

    def evaluate_move(self, game, action):
        piece = game.get_piece(action.get('from_row'), action.get('from_col'))
        score = super().evaluate_move(game, action)
        if not piece:
            return score

        score += self.score_weak_side_action_tuning(game, action, piece, None)
        return score

    def score_setup_pair_synergy(self, type, own_setup_pieces, pos, setup_index):
        if not own_setup_pieces:
            return 4 if type == '炮' or type == '仕' else 0

        own_types = [piece['type'] for piece in own_setup_pieces]
        has = lambda piece_type: piece_type in own_types
        score = 0

        if has('仕') and type == '炮':
            score += 10
        if has('炮') and type == '仕':
            score += 8
        if has('炮') and type == '车':
            score += 10
        if has('车') and type == '炮':
            score += 8
        if has('马') and type == '车':
            score += 6
        if has('车') and type == '马':
            score += 5
        if has('仕') and type == '仕':
            score += 7
        if has('炮') and type == '炮':
            score += 4

        if type == '将':
            structure_ready = has('仕') or (has('炮') and has('车'))
            score += 6 if structure_ready else -10

        if type == '兵' and setup_index <= 2:
            score -= 4

        if pos:
            is_edge_file = pos['col'] == 0 or pos['col'] == COLS - 1
            if is_edge_file and has('炮') and type == '车':
                score += 3
            if not is_edge_file and has('仕') and type == '将':
                score += 4

        return score

    def score_weak_side_flip_tuning(self, game, action):
        if self.side != 'black':
            return 0

        score = self.score_safe_hidden_flip(game, action.get('row'), action.get('col')) * 0.2
        if self.count_all_face_up_pieces(game) <= 8:
            score += 1.5
        if action.get('col') == 1 or action.get('col') == 2:
            score += 0.8
        if action.get('row') >= 2 and action.get('row') <= 5:
            score += 0.7
        return score

    def score_weak_side_action_tuning(self, game, action, piece, target):
        if self.side != 'black':
            return 0

        score = 0
        early_phase = self.count_all_face_up_pieces(game) <= 10
        safe_action = not self.would_be_in_danger_after_move(game, action, piece)

        if safe_action and early_phase:
            if action['type'] == 'capture' and target and target['face_up']:
                score += 2 + target['hp'] * 0.35
            elif action['type'] == 'move':
                score += 1.5

        if safe_action and (piece['type'] == '车' or piece['type'] == '马' or piece['type'] == '炮'):
            score += 1.5

        central_file = action.get('to_col') == 1 or action.get('to_col') == 2
        central_band = action.get('to_row') >= 2 and action.get('to_row') <= 5
        if safe_action and central_file and central_band:
            score += 1.2

        if safe_action and self.count_enemy_neighbors(game, action.get('to_row'), action.get('to_col')) >= 1:
            score += 1.4 if action['type'] == 'capture' else 0.8

        return score

    def count_enemy_neighbors(self, game, row, col):
        count = 0
        for nr, nc in self.neighbors(row, col):
            piece = game.get_piece(nr, nc)
            if piece and piece['face_up'] and piece['side'] == self.enemy_side():
                count += 1
        return count

    def score_weak_side_setup_tuning(self, side, type, own_types, pos, setup_index):
        if side != 'black':
            return 0

        score = 0
        has = lambda piece_type: piece_type in own_types

        if setup_index == 1:
            if type == '炮':
                score += 4
            if type == '车' or type == '马':
                score += 2
            if type == '将':
                score -= 4

        if setup_index == 3:
            if has('炮') and type == '车':
                score += 4
            if has('车') and type == '炮':
                score += 3
            if has('仕') and type == '将':
                score += 3

        if pos and pos['col'] == 0 and type == '炮':
            score += 2

        return score


class StrategicV4AI(StrategicV3AI):
    def choose_setup_piece(self, game, side):
        return super().choose_setup_piece(game, side)

    def evaluate(self, game, action):
        rand = random.random() * 0.03

        if action['type'] == 'flip':
            return self.evaluate_flip(game, action) + rand

        if action['type'] == 'capture':
            return self.evaluate_capture(game, action) + rand

        if action['type'] == 'move':
            return self.evaluate_move(game, action) + rand

        return rand

    def evaluate_flip(self, game, action):
        score = super().evaluate_flip(game, action)
        score += self.score_v4_flip_discipline(game, action)
        return score

    def evaluate_capture(self, game, action):
        attacker = game.get_piece(action.get('from_row'), action.get('from_col'))
        target = game.get_piece(action.get('to_row'), action.get('to_col'))
        score = super().evaluate_capture(game, action)
        if not attacker or not target:
            return score

        score += self.score_v4_conversion_capture(game, action, attacker, target)
        return score

    def evaluate_move(self, game, action):
        piece = game.get_piece(action.get('from_row'), action.get('from_col'))
        score = super().evaluate_move(game, action)
        if not piece:
            return score

        score += self.score_v4_formation_move(game, action, piece)
        return score

    def score_v4_flip_discipline(self, game, action):
        score = 0
        visible = self.count_all_face_up_pieces(game)
        actions = game.get_all_actions()
        safe_captures = 0
        safe_moves = 0

        for candidate in actions:
            if candidate['type'] == 'flip':
                continue
            piece = game.get_piece(candidate.get('from_row'), candidate.get('from_col'))
            if not piece:
                continue
            if self.would_be_in_danger_after_move(game, candidate, piece):
                continue

            if candidate['type'] == 'capture':
                target = game.get_piece(candidate.get('to_row'), candidate.get('to_col'))
                if target and target['face_up']:
                    safe_captures += 1
            elif candidate['type'] == 'move':
                safe_moves += 1

        if visible >= 10 and safe_captures >= 1:
            score -= 5 if self.is_winning_state(game) else 2.5
        if visible >= 12 and safe_moves >= 4:
            score -= 2.5 if self.is_winning_state(game) else 1
        if self.is_winning_state(game) and len(game.remaining_pool) <= 12:
            score -= 1.5

        if self.is_losing_state(game):
            _kings = self.get_face_up_pieces(game, self.enemy_side(), '将')
            enemy_king = _kings[0] if _kings else None
            if enemy_king:
                dist = self.manhattan_distance(action.get('row'), action.get('col'), enemy_king['row'], enemy_king['col'])
                if dist <= 2:
                    score += 5
            if action.get('col') == 1 or action.get('col') == 2:
                score += 1.2
            if action.get('row') >= 2 and action.get('row') <= 5:
                score += 1

        return score

    def score_v4_conversion_capture(self, game, action, attacker, target):
        score = 0
        safe_action = not self.would_be_in_danger_after_move(game, action, attacker)
        next_enemy_hp = game.hp[self.enemy_side()] - (target['hp'] if target['face_up'] else 0)

        if target['face_up']:
            if next_enemy_hp <= 20:
                score += (22 - next_enemy_hp) * 1.1
            if next_enemy_hp <= 10:
                score += 8
            if next_enemy_hp <= 5:
                score += 10
            if target['type'] == '仕':
                score += 4

        if safe_action:
            _kings = self.get_face_up_pieces(game, self.enemy_side(), '将')
            enemy_king = _kings[0] if _kings else None
            if enemy_king:
                dist = self.manhattan_distance(action.get('to_row'), action.get('to_col'), enemy_king['row'], enemy_king['col'])
                if dist <= 1:
                    score += 12 if attacker['type'] == '兵' else 7
                elif dist == 2 and (attacker['type'] == '车' or attacker['type'] == '马' or attacker['type'] == '炮'):
                    score += 3

            if self.score_follow_up_pressure(game, action, attacker) > 0:
                score += 3

        if self.is_winning_state(game) and attacker['type'] == '将' and target['type'] != '兵':
            score -= 5

        if not safe_action and attacker['type'] != '兵' and attacker['type'] != '炮':
            score -= attacker['hp'] * 0.5

        return score

    def score_v4_formation_move(self, game, action, piece):
        score = 0
        safe_action = not self.would_be_in_danger_after_move(game, action, piece)
        central_file = action.get('to_col') == 1 or action.get('to_col') == 2
        central_band = action.get('to_row') >= 2 and action.get('to_row') <= 5

        if safe_action and central_file and central_band and (piece['type'] == '车' or piece['type'] == '马' or piece['type'] == '炮'):
            score += 1.5

        if safe_action and self.score_follow_up_pressure(game, action, piece) > 0:
            score += 2.2

        _kings = self.get_face_up_pieces(game, self.enemy_side(), '将')
        enemy_king = _kings[0] if _kings else None
        if enemy_king:
            dist = self.manhattan_distance(action.get('to_row'), action.get('to_col'), enemy_king['row'], enemy_king['col'])
            if safe_action and dist <= 2 and (piece['type'] == '车' or piece['type'] == '马' or piece['type'] == '兵'):
                score += 4 if dist == 1 else 1.5

        if piece['type'] == '将':
            open_ = self.count_open_neighbors(game, action.get('to_row'), action.get('to_col'))
            if self.is_winning_state(game) and open_ <= 1:
                score -= 6

        if self.is_winning_state(game) and not safe_action and piece['type'] != '兵':
            score -= piece['hp'] * 0.65

        return score

    def score_king_zone_control(self, game, attacker_side):
        defender_side = 'black' if attacker_side == 'red' else 'red'
        _kings = self.get_face_up_pieces(game, defender_side, '将')
        enemy_king = _kings[0] if _kings else None
        if not enemy_king:
            return 0

        score = 0
        for piece in self.get_face_up_pieces(game, attacker_side, None):
            dist = self.manhattan_distance(piece['row'], piece['col'], enemy_king['row'], enemy_king['col'])
            if piece['type'] == '兵' and dist <= 1:
                score += 14
            elif (piece['type'] == '车' or piece['type'] == '马') and dist <= 2:
                score += 10 if dist == 1 else 6
            elif piece['type'] == '炮' and self.can_cannon_attack_square(game, piece['row'], piece['col'], enemy_king['row'], enemy_king['col']):
                score += 9
            elif piece['type'] == '将' and dist <= 2:
                score += 5

        return score

    def score_major_piece_activity(self, game, side):
        score = 0
        for piece in self.get_face_up_pieces(game, side, None):
            if piece['type'] != '车' and piece['type'] != '马' and piece['type'] != '炮' and piece['type'] != '将':
                continue
            mobility = self.estimate_mobility_at(piece['row'], piece['col'], piece, game)
            central_file = piece['col'] == 1 or piece['col'] == 2
            central_band = piece['row'] >= 2 and piece['row'] <= 5
            score += mobility * 2.2
            if central_file and central_band:
                score += 3
            if not self.is_in_danger(game, piece['row'], piece['col'], piece):
                score += 2
        return score


class StrategicV5AI(StrategicV4AI):
    def evaluate(self, game, action):
        rand = random.random() * 0.02

        if action['type'] == 'flip':
            return self.evaluate_flip(game, action) + rand

        if action['type'] == 'capture':
            return self.evaluate_capture(game, action) + rand

        if action['type'] == 'move':
            return self.evaluate_move(game, action) + rand

        return rand

    def evaluate_flip(self, game, action):
        score = super().evaluate_flip(game, action)
        score -= self.score_v5_enemy_cannon_target_flip(game, action.get('row'), action.get('col'))
        score -= self.score_v5_adjacent_enemy_flip_penalty(game, action.get('row'), action.get('col'))
        score += self.score_v5_cannon_flip_opportunity(game, action.get('row'), action.get('col'))
        return score

    def evaluate_capture(self, game, action):
        attacker = game.get_piece(action.get('from_row'), action.get('from_col'))
        target = game.get_piece(action.get('to_row'), action.get('to_col'))
        score = super().evaluate_capture(game, action)
        if not attacker or not target:
            return score

        score += self.score_v5_king_threat_defense(game, action, attacker)
        score += self.score_v5_cannon_activation_action(game, action, attacker, target)
        return score

    def evaluate_move(self, game, action):
        piece = game.get_piece(action.get('from_row'), action.get('from_col'))
        score = super().evaluate_move(game, action)
        if not piece:
            return score

        score += self.score_v5_king_threat_defense(game, action, piece)
        score += self.score_v5_cannon_activation_action(game, action, piece, None)
        return score

    def score_v5_enemy_cannon_target_flip(self, game, row, col):
        enemy_cannons = self.get_face_up_pieces(game, self.enemy_side(), '炮')
        if not enemy_cannons or not game.remaining_pool:
            return 0

        attack_count = 0
        for cannon in enemy_cannons:
            if self.can_cannon_attack_square(game, cannon['row'], cannon['col'], row, col):
                attack_count += 1
        if not attack_count:
            return 0

        pool = game.remaining_pool
        own_pool = [piece for piece in pool if piece['side'] == self.side]
        own_major_pool = [piece for piece in own_pool if piece['type'] != '兵']
        own_critical_pool = [piece for piece in own_pool if piece['type'] == '将' or piece['type'] == '仕' or piece['type'] == '车' or piece['type'] == '马']
        enemy_pool = [piece for piece in pool if piece['side'] == self.enemy_side()]

        penalty = attack_count * 8
        penalty += attack_count * (len(own_pool) / len(pool)) * 14
        penalty += attack_count * (len(own_major_pool) / len(pool)) * 16
        penalty += attack_count * (len(own_critical_pool) / len(pool)) * 20
        penalty += attack_count * (len(enemy_pool) / len(pool)) * 5

        if self.is_adjacent_to_own_king(game, row, col, self.side):
            penalty += 10

        return penalty

    def score_v5_adjacent_enemy_flip_penalty(self, game, row, col):
        penalty = 0
        own_support = 0

        for nr, nc in self.neighbors(row, col):
            piece = game.get_piece(nr, nc)
            if not piece or not piece['face_up']:
                continue

            if piece['side'] == self.side:
                own_support += 1
                continue

            if piece['type'] == '炮':
                penalty += 8
            elif piece['type'] == '将' or piece['type'] == '仕':
                penalty += 7
            elif piece['type'] == '车' or piece['type'] == '马':
                penalty += 5
            else:
                penalty += 3

        penalty -= own_support * 1.4

        enemy_cannon_pressure = self.count_cannon_attackers(game, row, col, self.enemy_side())
        penalty += enemy_cannon_pressure * 4
        return max(0, penalty)

    def score_v5_cannon_flip_opportunity(self, game, row, col):
        own_cannons = len([piece for piece in game.remaining_pool if piece['side'] == self.side and piece['type'] == '炮'])
        if not own_cannons or not game.remaining_pool:
            return 0

        own_cannon_prob = own_cannons / len(game.remaining_pool)
        immediate_shots = self.count_potential_cannon_shots_from_square(game, row, col, self.side)
        hp_diff = game.hp[self.side] - game.hp[self.enemy_side()]

        if self.is_losing_state(game):
            return own_cannon_prob * immediate_shots * 1.2 if immediate_shots > 0 else -own_cannon_prob * 2

        if abs(hp_diff) <= 6 or (game.hp[self.side] == 60 and game.hp[self.enemy_side()] == 60):
            return own_cannon_prob * immediate_shots * 8

        return own_cannon_prob * immediate_shots * 4

    def score_v5_king_threat_defense(self, game, action, piece):
        if action['type'] != 'move' and action['type'] != 'capture':
            return 0
        if not piece:
            return 0

        before = self.estimate_v5_own_king_threat(game, self.side)
        snapshot = game.create_snapshot()
        history = list(game.history)

        try:
            target = game.get_piece(action.get('to_row'), action.get('to_col'))
            applied = (False if target and not target['face_up']
                       else game.move_piece(action.get('from_row'), action.get('from_col'), action.get('to_row'), action.get('to_col')))
            if not applied:
                return 0

            after = self.estimate_v5_own_king_threat(game, self.side)
            score = (before - after) * 0.9

            if piece['type'] == '将':
                score += (before - after) * 0.35

            return score
        finally:
            game.restore_snapshot(snapshot)
            game.history = history

    def estimate_v5_own_king_threat(self, game, side):
        _kings = self.get_face_up_pieces(game, side, '将')
        own_king = _kings[0] if _kings else None
        if not own_king:
            return 0

        enemy = 'black' if side == 'red' else 'red'
        threat = 0
        threat += self.count_threats_to_piece(game, own_king['row'], own_king['col'], own_king) * 18
        threat += self.count_adjacent_enemy_soldiers(game, own_king['row'], own_king['col'], side) * 20

        for cannon in self.get_face_up_pieces(game, enemy, '炮'):
            if cannon['row'] != own_king['row'] and cannon['col'] != own_king['col']:
                continue
            blockers = self.count_pieces_between(game, cannon['row'], cannon['col'], own_king['row'], own_king['col'])
            if blockers == 1:
                threat += 56
            elif blockers == 0:
                threat += 20
            elif blockers == 2:
                threat += 16
            elif blockers == 3:
                threat += 6

        own_advisors = self.get_face_up_pieces(game, side, '仕')
        for advisor in own_advisors:
            if self.is_adjacent(advisor['row'], advisor['col'], own_king['row'], own_king['col']):
                threat -= 6

        return max(0, threat)

    def count_pieces_between(self, game, from_row, from_col, to_row, to_col):
        if from_row != to_row and from_col != to_col:
            return math.inf

        dr = (to_row > from_row) - (to_row < from_row)
        dc = (to_col > from_col) - (to_col < from_col)
        r = from_row + dr
        c = from_col + dc
        count = 0

        while r != to_row or c != to_col:
            if game.get_piece(r, c):
                count += 1
            r += dr
            c += dc

        return count

    def count_potential_cannon_shots_from_square(self, game, row, col, side):
        shot_count = 0

        for dr, dc in DIRS:
            r = row + dr
            c = col + dc
            blockers = 0

            while r >= 0 and r < ROWS and c >= 0 and c < COLS:
                piece = game.get_piece(r, c)
                if piece:
                    blockers += 1
                    if blockers == 2:
                        if not piece['face_up'] or piece['side'] != side:
                            shot_count += 1
                        break
                r += dr
                c += dc

        return shot_count

    def score_v5_cannon_activation_action(self, game, action, piece, target):
        if not piece or piece['type'] != '炮':
            return 0

        shots = self.count_potential_cannon_shots_from_square(game, action.get('to_row'), action.get('to_col'), self.side)
        if not shots:
            return -2 if self.is_losing_state(game) else 0

        score = shots * 4
        if target and target['face_up']:
            score += 2
        if game.hp[self.side] == 60 and game.hp[self.enemy_side()] == 60:
            score += 2
        if self.is_losing_state(game):
            score *= 0.5
        return score


class StrategicV6AI(StrategicV5AI):
    def evaluate(self, game, action):
        rand = random.random() * 0.015

        if action['type'] == 'flip':
            return self.evaluate_flip(game, action) + rand

        if action['type'] == 'capture':
            return self.evaluate_capture(game, action) + rand

        if action['type'] == 'move':
            return self.evaluate_move(game, action) + rand

        return rand

    def evaluate_flip(self, game, action):
        score = super().evaluate_flip(game, action)
        score -= self.score_v6_controlled_no_flip(game, action)
        score -= self.score_v6_hidden_king_soldier_flip_risk(game, action.get('row'), action.get('col'))
        score += self.score_v6_late_clarity_probabilistic_flip(game, action.get('row'), action.get('col'))
        return score

    def evaluate_capture(self, game, action):
        attacker = game.get_piece(action.get('from_row'), action.get('from_col'))
        score = super().evaluate_capture(game, action)
        if not attacker:
            return score

        score += self.score_v6_endgame_clamp_action(game, action, attacker)
        score += self.score_v6_board_control_action(game, action, attacker)
        return score

    def evaluate_move(self, game, action):
        piece = game.get_piece(action.get('from_row'), action.get('from_col'))
        score = super().evaluate_move(game, action)
        if not piece:
            return score

        score += self.score_v6_endgame_clamp_action(game, action, piece)
        score += self.score_v6_board_control_action(game, action, piece)
        return score

    def score_v6_controlled_no_flip(self, game, action):
        own_hidden = len([piece for piece in game.remaining_pool if piece['side'] == self.side])
        enemy_hidden = len(game.remaining_pool) - own_hidden
        non_flip_actions = [candidate for candidate in game.get_all_actions() if candidate['type'] != 'flip']

        def _is_safe_active(candidate):
            piece = game.get_piece(candidate.get('from_row'), candidate.get('from_col'))
            if not piece:
                return False
            return not self.would_be_in_danger_after_move(game, candidate, piece)

        safe_active_actions = [candidate for candidate in non_flip_actions if _is_safe_active(candidate)]
        safe_captures = len([candidate for candidate in safe_active_actions if candidate['type'] == 'capture'])
        visible_lead = self.count_all_face_up_pieces(game) >= 12
        own_info_advantage = own_hidden <= 1 and enemy_hidden >= 4
        almost_all_enemy_hidden = own_hidden == 0 and enemy_hidden >= 3
        winning = game.hp[self.side] >= game.hp[self.enemy_side()] - 2

        penalty = 0
        if visible_lead and winning and len(safe_active_actions) >= 2:
            penalty += 5
        if own_info_advantage and len(safe_active_actions) >= 1:
            penalty += 16
        if almost_all_enemy_hidden and len(safe_active_actions) >= 2:
            penalty += 14
        if safe_captures >= 1:
            penalty += 10 if own_info_advantage else 4
        if self.is_winning_state(game) and own_hidden <= 1 and enemy_hidden >= 2:
            penalty += 6

        if self.is_losing_state(game):
            penalty *= 0.4

        return penalty

    def score_v6_hidden_king_soldier_flip_risk(self, game, row, col):
        enemy_king = next(iter(self.get_face_up_pieces(game, self.enemy_side(), '将')), None)
        own_king_face_up = self.count_face_up_pieces(game, self.side, '将') > 0
        if not enemy_king or own_king_face_up:
            return 0

        penalty = 0
        for nr, nc in self.neighbors(row, col):
            piece = game.get_piece(nr, nc)
            if not piece or not piece['face_up'] or piece['side'] != self.enemy_side():
                continue
            if piece['type'] == '兵':
                penalty += 12
                if self.is_adjacent(piece.get('row') if piece.get('row') is not None else nr,
                                    piece.get('col') if piece.get('col') is not None else nc,
                                    enemy_king['row'], enemy_king['col']):
                    penalty += 6
            elif piece['type'] == '将':
                penalty += 2

        enemy_neighbors = self.count_enemy_neighbors(game, row, col)
        penalty += enemy_neighbors * 1.8

        if len(game.remaining_pool) <= 6 and game.hp[self.side] + 10 < game.hp[self.enemy_side()]:
            penalty *= 0.45

        return penalty

    def score_v6_late_clarity_probabilistic_flip(self, game, row, col):
        if len(game.remaining_pool) > 8:
            return 0

        own_hidden = len([piece for piece in game.remaining_pool if piece['side'] == self.side])
        enemy_hidden = len(game.remaining_pool) - own_hidden
        probability_score = self.score_probabilistic_flip(game, row, col)

        def _is_safe(candidate):
            if candidate['type'] == 'flip':
                return False
            piece = game.get_piece(candidate.get('from_row'), candidate.get('from_col'))
            return bool(piece) and not self.would_be_in_danger_after_move(game, candidate, piece)

        safe_action_count = len([candidate for candidate in game.get_all_actions() if _is_safe(candidate)])

        score = 0
        if probability_score > 0:
            score += probability_score * (0.5 if len(game.remaining_pool) <= 4 else 0.3)
        if enemy_hidden <= own_hidden:
            score += 2
        if safe_action_count == 0:
            score += 4
        if self.is_losing_state(game):
            score += 3

        return score

    def score_v6_endgame_clamp_action(self, game, action, piece):
        if not piece or (action['type'] != 'move' and action['type'] != 'capture'):
            return 0

        enemy_king = next(iter(self.get_face_up_pieces(game, self.enemy_side(), '将')), None)
        if not enemy_king:
            return 0

        is_endgame = self.count_all_face_up_pieces(game) >= 10 or len(game.remaining_pool) <= 8
        if not is_endgame:
            return 0

        before = self.evaluate_v6_enemy_king_trap(game, self.enemy_side())
        snapshot = game.create_snapshot()
        history = list(game.history)

        try:
            target = game.get_piece(action['to_row'], action['to_col'])
            applied = (False if (target and not target['face_up'])
                       else game.move_piece(action['from_row'], action['from_col'], action['to_row'], action['to_col']))
            if not applied:
                return 0

            after = self.evaluate_v6_enemy_king_trap(game, self.enemy_side())
            score = (after - before) * 0.95

            if piece['type'] == '兵':
                distance_after = self.manhattan_distance(action['to_row'], action['to_col'], enemy_king['row'], enemy_king['col'])
                if distance_after <= 1:
                    score += 12
                elif distance_after == 2:
                    score += 4

            if (piece['type'] == '仕' or piece['type'] == '将') and self.is_winning_state(game):
                score += max(0, after - before) * 0.18

            return score
        finally:
            game.restore_snapshot(snapshot)
            game.history = history

    def score_v6_board_control_action(self, game, action, piece):
        if not piece or (action['type'] != 'move' and action['type'] != 'capture'):
            return 0

        own_hidden = len([candidate for candidate in game.remaining_pool if candidate['side'] == self.side])
        enemy_hidden = len(game.remaining_pool) - own_hidden
        control_phase = own_hidden <= 1 and enemy_hidden >= 2
        if not control_phase:
            return 0

        snapshot = game.create_snapshot()
        history = list(game.history)

        try:
            target = game.get_piece(action['to_row'], action['to_col'])
            applied = (False if (target and not target['face_up'])
                       else game.move_piece(action['from_row'], action['from_col'], action['to_row'], action['to_col']))
            if not applied:
                return 0

            own_control = self.score_v6_spatial_control(game, self.side)
            enemy_control = self.score_v6_spatial_control(game, self.enemy_side())
            score = (own_control - enemy_control) * 0.22

            if not self.would_be_in_danger_after_move(game, action, piece):
                score += 2

            if piece['type'] == '兵' or piece['type'] == '仕':
                score += 1.5

            return score
        finally:
            game.restore_snapshot(snapshot)
            game.history = history

    def evaluate_v6_enemy_king_trap(self, game, king_side):
        king = next(iter(self.get_face_up_pieces(game, king_side, '将')), None)
        if not king:
            return 0

        attacker_side = 'black' if king_side == 'red' else 'red'
        corner_distance = self.min_corner_distance(king['row'], king['col'])
        mobility = self.count_king_mobility(game, king_side)
        score = 0

        score += (3 - min(3, corner_distance)) * 10
        score += (4 - min(4, mobility)) * 8

        for piece in self.get_face_up_pieces(game, attacker_side, None):
            dist = self.manhattan_distance(piece['row'], piece['col'], king['row'], king['col'])
            if piece['type'] == '兵':
                if dist <= 1:
                    score += 16
                elif dist == 2:
                    score += 6
            elif piece['type'] == '仕':
                if dist <= 2:
                    score += 7
            elif piece['type'] == '将':
                if dist <= 2:
                    score += 6
            elif piece['type'] == '炮' and self.can_cannon_attack_square(game, piece['row'], piece['col'], king['row'], king['col']):
                score += 10

        return score

    def count_king_mobility(self, game, side):
        king = next(iter(self.get_face_up_pieces(game, side, '将')), None)
        if not king:
            return 0

        snapshot_turn = game.current_turn
        try:
            game.current_turn = side
            return len(game.get_valid_moves(king['row'], king['col']))
        finally:
            game.current_turn = snapshot_turn

    def score_v6_spatial_control(self, game, side):
        score = 0
        for piece in self.get_face_up_pieces(game, side, None):
            central_file = piece['col'] == 1 or piece['col'] == 2
            central_band = piece['row'] >= 2 and piece['row'] <= 5
            score += self.estimate_mobility_at(piece['row'], piece['col'], piece, game) * 1.6
            if central_file and central_band:
                score += 2
            if not self.is_in_danger(game, piece['row'], piece['col'], piece):
                score += 1.5
            if piece['type'] == '兵' or piece['type'] == '仕':
                score += 0.8
        return score


class StrategicV7AI(StrategicV6AI):
    def evaluate(self, game, action):
        rand = random.random() * 0.012

        if action['type'] == 'flip':
            return self.evaluate_flip(game, action) + rand

        if action['type'] == 'capture':
            return self.evaluate_capture(game, action) + rand

        if action['type'] == 'move':
            return self.evaluate_move(game, action) + rand

        return rand

    def evaluate_flip(self, game, action):
        score = super().evaluate_flip(game, action)
        opening_plan = self.get_v7_opening_plan(game)

        score += self.score_v7_relative_safe_flip(game, action.get('row'), action.get('col'))
        score += self.score_v7_favorable_flip_gate(game, action.get('row'), action.get('col'))
        score += self.score_v7_opening_plan_flip(game, action, opening_plan)
        return score

    def evaluate_capture(self, game, action):
        attacker = game.get_piece(action.get('from_row'), action.get('from_col'))
        target = game.get_piece(action.get('to_row'), action.get('to_col'))
        score = super().evaluate_capture(game, action)
        if not attacker or not target:
            return score

        score += self.score_v7_opening_plan_action(game, action, attacker, target, self.get_v7_opening_plan(game))
        return score

    def evaluate_move(self, game, action):
        piece = game.get_piece(action.get('from_row'), action.get('from_col'))
        score = super().evaluate_move(game, action)
        if not piece:
            return score

        score += self.score_v7_opening_plan_action(game, action, piece, None, self.get_v7_opening_plan(game))
        return score

    def get_v7_opening_plan(self, game):
        if game.setup_index < 4 or self.count_all_face_up_pieces(game) > 10:
            return 'neutral'

        own_visible = self.get_face_up_pieces(game, self.side, None)
        enemy_visible = self.get_face_up_pieces(game, self.enemy_side(), None)
        own_types = [piece['type'] for piece in own_visible]
        enemy_types = [piece['type'] for piece in enemy_visible]

        def has_own(type_):
            return type_ in own_types

        def has_enemy(type_):
            return type_ in enemy_types

        if has_own('仕') and has_own('将'):
            return 'fortress'
        if has_own('炮') and (has_own('车') or has_own('马')):
            return 'attack'
        if has_enemy('将') and self.count_living_pieces(game, self.side, '兵') >= 3:
            return 'king-hunt'
        if has_enemy('炮') and not has_own('将'):
            return 'safe-control'
        return 'control'

    def score_v7_relative_safe_flip(self, game, row, col):
        if not game.remaining_pool:
            return 0

        best = -math.inf
        worst = math.inf
        total = 0
        favorable = 0
        severe_loss = 0

        for piece in game.remaining_pool:
            reveal_score = self.score_reveal_as_piece(game, row, col, piece)
            best = max(best, reveal_score)
            worst = min(worst, reveal_score)
            total += reveal_score
            if reveal_score >= 0:
                favorable += 1
            if reveal_score <= -12:
                severe_loss += 1

        average = total / len(game.remaining_pool)
        span = best - worst
        enemy_pressure = self.count_enemy_neighbors(game, row, col) + self.count_cannon_attackers(game, row, col, self.enemy_side()) * 1.5
        score = 0

        if worst > -8 and span <= 20:
            score += 9
        elif worst > -12 and span <= 28:
            score += 5

        if average > 0:
            score += min(6, average * 0.25)
        score += (favorable / len(game.remaining_pool)) * 4
        score -= severe_loss * 1.7
        score -= enemy_pressure * 1.4

        return score

    def score_v7_favorable_flip_gate(self, game, row, col):
        if not game.remaining_pool:
            return 0

        favorable = 0
        unfavorable = 0
        expected = 0

        for piece in game.remaining_pool:
            reveal_score = self.score_reveal_as_piece(game, row, col, piece)
            expected += reveal_score
            if reveal_score >= 4:
                favorable += 1
            if reveal_score <= -4:
                unfavorable += 1

        favorable_ratio = favorable / len(game.remaining_pool)
        unfavorable_ratio = unfavorable / len(game.remaining_pool)
        expected_score = expected / len(game.remaining_pool)

        def _is_safe(candidate):
            if candidate['type'] == 'flip':
                return False
            piece = game.get_piece(candidate.get('from_row'), candidate.get('from_col'))
            return bool(piece) and not self.would_be_in_danger_after_move(game, candidate, piece)

        safe_non_flip_actions = len([candidate for candidate in game.get_all_actions() if _is_safe(candidate)])

        score = 0
        if favorable_ratio >= 0.58 and expected_score > 0:
            score += 7 + expected_score * 0.22
        if favorable_ratio <= unfavorable_ratio or expected_score < 0:
            score -= 14 if safe_non_flip_actions >= 1 else 7
        if favorable_ratio < 0.45 and unfavorable_ratio >= 0.35:
            score -= 10 if safe_non_flip_actions >= 2 else 4

        if self.is_losing_state(game):
            score *= 0.55

        return score

    def score_v7_opening_plan_flip(self, game, action, plan):
        if plan == 'neutral':
            return 0

        score = 0
        enemy_neighbors = self.count_enemy_neighbors(game, action['row'], action['col'])
        own_support = self.score_support(action['row'], action['col'], game, self.side)

        if plan == 'fortress' or plan == 'safe-control':
            score += self.score_safe_hidden_flip(game, action['row'], action['col']) * 0.8
            score -= enemy_neighbors * 2.2
            score -= self.score_worst_case_flip_risk(game, action['row'], action['col']) * 0.55
        elif plan == 'attack' or plan == 'king-hunt':
            enemy_king = next(iter(self.get_face_up_pieces(game, self.enemy_side(), '将')), None)
            if enemy_king:
                dist = self.manhattan_distance(action['row'], action['col'], enemy_king['row'], enemy_king['col'])
                if dist <= 1:
                    score += 8 if plan == 'king-hunt' else 5
                elif dist == 2:
                    score += 2
            score += self.score_cannon_mouth(game, action['row'], action['col']) * 0.18
            score -= self.score_worst_case_flip_risk(game, action['row'], action['col']) * 0.25
            score -= enemy_neighbors * 1.2
        elif plan == 'control':
            score += own_support * 1.8
            score += self.score_v7_relative_safe_flip(game, action['row'], action['col']) * 0.5
            score -= enemy_neighbors * 1.6

        return score

    def score_v7_opening_plan_action(self, game, action, piece, target, plan):
        if plan == 'neutral':
            return 0

        safe_action = not self.would_be_in_danger_after_move(game, action, piece)
        score = 0

        if (plan == 'fortress' or plan == 'safe-control') and safe_action:
            score += self.score_support(action['to_row'], action['to_col'], game, self.side) * 3
            if piece['type'] == '仕' or piece['type'] == '将':
                score += 3
            if piece['type'] == '炮' and target and not target['face_up']:
                score -= 4

        if (plan == 'attack' or plan == 'king-hunt') and safe_action:
            enemy_king = next(iter(self.get_face_up_pieces(game, self.enemy_side(), '将')), None)
            if enemy_king:
                dist = self.manhattan_distance(action['to_row'], action['to_col'], enemy_king['row'], enemy_king['col'])
                if dist <= 1 and (piece['type'] == '兵' or piece['type'] == '车' or piece['type'] == '马'):
                    score += 7
                elif dist == 2 and piece['type'] != '将':
                    score += 2.5
            if piece['type'] == '炮':
                score += self.count_potential_cannon_shots_from_square(game, action['to_row'], action['to_col'], self.side) * 1.5

        if plan == 'control' and safe_action:
            central_file = action['to_col'] == 1 or action['to_col'] == 2
            central_band = action['to_row'] >= 2 and action['to_row'] <= 5
            if central_file and central_band:
                score += 2
            if piece['type'] == '兵' or piece['type'] == '仕':
                score += 1.8

        return score


class StrategicV8AI(StrategicV7AI):
    def evaluate(self, game, action):
        rand = random.random() * 0.01

        if action['type'] == 'flip':
            return self.evaluate_flip(game, action) + rand

        if action['type'] == 'capture':
            return self.evaluate_capture(game, action) + rand

        if action['type'] == 'move':
            return self.evaluate_move(game, action) + rand

        return rand

    def evaluate_flip(self, game, action):
        score = super().evaluate_flip(game, action)
        score -= self.score_v8_lead_closeout_flip_suppression(game, action)
        return score

    def evaluate_capture(self, game, action):
        attacker = game.get_piece(action.get('from_row'), action.get('from_col'))
        target = game.get_piece(action.get('to_row'), action.get('to_col'))
        score = super().evaluate_capture(game, action)
        if not attacker or not target:
            return score

        score += self.score_v8_lead_closeout_action(game, action, attacker, target)
        score += self.score_v8_conversion_priority(game, action, attacker, target)
        return score

    def evaluate_move(self, game, action):
        piece = game.get_piece(action.get('from_row'), action.get('from_col'))
        score = super().evaluate_move(game, action)
        if not piece:
            return score

        score += self.score_v8_lead_closeout_action(game, action, piece, None)
        return score

    def is_v8_closeout_phase(self, game):
        hp_lead = game.hp[self.side] - game.hp[self.enemy_side()]
        if hp_lead < 8:
            return False
        return len(game.remaining_pool) <= 8 or self.count_all_face_up_pieces(game) >= 12 or game.hp[self.enemy_side()] <= 20

    def score_v8_lead_closeout_flip_suppression(self, game, action):
        if not self.is_v8_closeout_phase(game):
            return 0

        own_hidden = len([piece for piece in game.remaining_pool if piece['side'] == self.side])
        enemy_hidden = len(game.remaining_pool) - own_hidden

        def _is_safe(candidate):
            if candidate['type'] == 'flip':
                return False
            piece = game.get_piece(candidate.get('from_row'), candidate.get('from_col'))
            return bool(piece) and not self.would_be_in_danger_after_move(game, candidate, piece)

        safe_non_flip_actions = len([candidate for candidate in game.get_all_actions() if _is_safe(candidate)])

        penalty = 0
        if safe_non_flip_actions >= 1:
            penalty += 10
        if safe_non_flip_actions >= 3:
            penalty += 8
        if own_hidden <= 1 and enemy_hidden >= 2:
            penalty += 10
        if self.score_probabilistic_flip(game, action['row'], action['col']) < 2:
            penalty += 6
        return penalty

    def score_v8_lead_closeout_action(self, game, action, piece, target):
        if not piece or not self.is_v8_closeout_phase(game):
            return 0
        if action['type'] != 'move' and action['type'] != 'capture':
            return 0

        safe_action = not self.would_be_in_danger_after_move(game, action, piece)
        before = self.compute_v8_closeout_score(game)
        enemy_king_before = next(iter(self.get_face_up_pieces(game, self.enemy_side(), '将')), None)
        before_king_mobility = self.count_king_mobility(game, self.enemy_side())
        before_escape_control = self.count_v8_enemy_king_escape_pressure(game, self.side)
        snapshot = game.create_snapshot()
        history = list(game.history)

        try:
            board_target = game.get_piece(action['to_row'], action['to_col'])
            applied = (False if (board_target and not board_target['face_up'])
                       else game.move_piece(action['from_row'], action['from_col'], action['to_row'], action['to_col']))
            if not applied:
                return 0

            after = self.compute_v8_closeout_score(game)
            after_king_mobility = self.count_king_mobility(game, self.enemy_side())
            after_escape_control = self.count_v8_enemy_king_escape_pressure(game, self.side)
            score = (after - before) * 0.95

            if safe_action:
                score += 2.5
            if piece['type'] == '兵' or piece['type'] == '仕':
                score += 1.5

            score += (before_king_mobility - after_king_mobility) * 8
            score += (after_escape_control - before_escape_control) * 5

            enemy_king = next(iter(self.get_face_up_pieces(game, self.enemy_side(), '将')), None)
            if enemy_king:
                dist = self.manhattan_distance(action['to_row'], action['to_col'], enemy_king['row'], enemy_king['col'])
                if piece['type'] == '兵' and dist <= 1:
                    score += 10
                elif (piece['type'] == '将' or piece['type'] == '仕') and dist <= 2:
                    score += 4

            if enemy_king_before:
                before_dist = self.manhattan_distance(action['from_row'], action['from_col'], enemy_king_before['row'], enemy_king_before['col'])
                after_dist = self.manhattan_distance(action['to_row'], action['to_col'], enemy_king_before['row'], enemy_king_before['col'])
                if (piece['type'] == '兵' or piece['type'] == '仕' or piece['type'] == '将') and after_dist < before_dist:
                    score += 3.5

            if after <= before and action['type'] == 'move':
                score -= 6 if safe_action else 10

            return score
        finally:
            game.restore_snapshot(snapshot)
            game.history = history

    def score_v8_conversion_priority(self, game, action, attacker, target):
        if not self.is_v8_closeout_phase(game) or not target or not target['face_up']:
            return 0

        safe_action = not self.would_be_in_danger_after_move(game, action, attacker)
        score = target['hp'] * 0.75 + self.material_value(target['type']) * 0.08

        if safe_action:
            score += 4
        if target['type'] == '仕':
            score += 5
        if target['type'] == '兵' and self.enemy_king_alive(game):
            score += 4
        if target['type'] == '炮':
            score += 3
        if attacker['type'] == '兵' and target['type'] != '将':
            score += 2
        if safe_action and game.hp[self.side] - game.hp[self.enemy_side()] >= 12:
            score += 4

        return score

    def compute_v8_closeout_score(self, game):
        enemy_king_trap = self.evaluate_v6_enemy_king_trap(game, self.enemy_side())
        enemy_mobility = self.score_v8_side_mobility(game, self.enemy_side())
        own_control = self.score_v6_spatial_control(game, self.side)
        enemy_control = self.score_v6_spatial_control(game, self.enemy_side())
        own_hidden = len([piece for piece in game.remaining_pool if piece['side'] == self.side])
        enemy_hidden = len(game.remaining_pool) - own_hidden
        hp_lead = game.hp[self.side] - game.hp[self.enemy_side()]

        score = 0
        score += enemy_king_trap * 1.1
        score += (own_control - enemy_control) * 0.45
        score -= enemy_mobility * 1.35
        score += max(0, hp_lead) * 0.6

        if own_hidden <= 1 and enemy_hidden >= 2:
            score += enemy_hidden * 3

        return score

    def score_v8_side_mobility(self, game, side):
        mobility = 0
        for piece in self.get_face_up_pieces(game, side, None):
            mobility += self.estimate_mobility_at(piece['row'], piece['col'], piece, game)
            if piece['type'] == '将':
                mobility += self.count_king_mobility(game, side) * 2
        return mobility

    def count_v8_enemy_king_escape_pressure(self, game, attacker_side):
        defender_side = 'black' if attacker_side == 'red' else 'red'
        enemy_king = next(iter(self.get_face_up_pieces(game, defender_side, '将')), None)
        if not enemy_king:
            return 0

        score = 0
        for nr, nc in self.neighbors(enemy_king['row'], enemy_king['col']):
            occupant = game.get_piece(nr, nc)
            if occupant and occupant['face_up'] and occupant['side'] == attacker_side:
                score += 3 if occupant['type'] == '兵' else 2
                continue

            for piece in self.get_face_up_pieces(game, attacker_side, None):
                if piece['type'] == '炮':
                    if self.can_cannon_attack_square(game, piece['row'], piece['col'], nr, nc):
                        score += 1.5
                        break
                    continue

                if self.is_adjacent(piece['row'], piece['col'], nr, nc):
                    score += 2.5 if piece['type'] == '兵' else 1.2
                    break

        return score


class StrategicV9AI(StrategicV8AI):
    def evaluate(self, game, action):
        rand = random.random() * 0.009

        if action['type'] == 'flip':
            return self.evaluate_flip(game, action) + rand

        if action['type'] == 'capture':
            return self.evaluate_capture(game, action) + rand

        if action['type'] == 'move':
            return self.evaluate_move(game, action) + rand

        return rand

    def evaluate_flip(self, game, action):
        score = super().evaluate_flip(game, action)
        score -= self.score_v9_closeout_flip_penalty(game, action)
        return score

    def evaluate_capture(self, game, action):
        attacker = game.get_piece(action.get('from_row'), action.get('from_col'))
        target = game.get_piece(action.get('to_row'), action.get('to_col'))
        score = super().evaluate_capture(game, action)
        if not attacker or not target:
            return score

        score += self.score_v9_edge_seal_action(game, action, attacker, target)
        score += self.score_v9_black_closeout_tuning(game, action, attacker, target)
        return score

    def evaluate_move(self, game, action):
        piece = game.get_piece(action.get('from_row'), action.get('from_col'))
        score = super().evaluate_move(game, action)
        if not piece:
            return score

        score += self.score_v9_edge_seal_action(game, action, piece, None)
        score += self.score_v9_black_closeout_tuning(game, action, piece, None)
        return score

    def is_v9_closeout_phase(self, game):
        hp_lead = game.hp[self.side] - game.hp[self.enemy_side()]
        threshold = 6 if self.side == 'black' else 8
        if hp_lead < threshold:
            return False
        return len(game.remaining_pool) <= 10 or self.count_all_face_up_pieces(game) >= 12 or game.hp[self.enemy_side()] <= 22

    def score_v9_closeout_flip_penalty(self, game, action):
        if not self.is_v9_closeout_phase(game):
            return 0

        penalty = self.score_v8_lead_closeout_flip_suppression(game, action) * 0.65
        enemy_king = next(iter(self.get_face_up_pieces(game, self.enemy_side(), '将')), None)
        if enemy_king:
            dist = self.manhattan_distance(action['row'], action['col'], enemy_king['row'], enemy_king['col'])
            if dist > 2 and game.hp[self.side] - game.hp[self.enemy_side()] >= 12:
                penalty += 3
            if self.min_corner_distance(enemy_king['row'], enemy_king['col']) <= 1 and not self.is_edge_square(action['row'], action['col']) and game.hp[self.side] - game.hp[self.enemy_side()] >= 12:
                penalty += 3
        if self.side == 'black':
            penalty += 1.5
        return penalty

    def score_v9_edge_seal_action(self, game, action, piece, target):
        if not piece or not self.is_v9_closeout_phase(game):
            return 0
        if action['type'] != 'move' and action['type'] != 'capture':
            return 0

        enemy_king = next(iter(self.get_face_up_pieces(game, self.enemy_side(), '将')), None)
        if not enemy_king:
            return 0
        strong_lead = game.hp[self.side] - game.hp[self.enemy_side()] >= 12
        edge_seal_active = ((self.is_edge_square(enemy_king['row'], enemy_king['col']) or self.min_corner_distance(enemy_king['row'], enemy_king['col']) <= 1)
                            if (self.side == 'black' or strong_lead)
                            else False)

        before = self.compute_v9_edge_seal_score(game)
        snapshot = game.create_snapshot()
        history = list(game.history)

        try:
            board_target = game.get_piece(action['to_row'], action['to_col'])
            applied = (False if (board_target and not board_target['face_up'])
                       else game.move_piece(action['from_row'], action['from_col'], action['to_row'], action['to_col']))
            if not applied:
                return 0

            after = self.compute_v9_edge_seal_score(game)
            score = (after - before) * (1.0 if edge_seal_active else 0.2)

            after_enemy_king = next(iter(self.get_face_up_pieces(game, self.enemy_side(), '将')), None)
            if after_enemy_king:
                before_corner = self.min_corner_distance(enemy_king['row'], enemy_king['col'])
                after_corner = self.min_corner_distance(after_enemy_king['row'], after_enemy_king['col'])
                if after_corner < before_corner:
                    score += 8 if edge_seal_active else 2

                to_edge = self.is_edge_square(action['to_row'], action['to_col'])
                to_center = action['to_col'] == 1 or action['to_col'] == 2
                if edge_seal_active and (piece['type'] == '兵' or piece['type'] == '仕' or piece['type'] == '将') and to_edge:
                    score += 3
                if edge_seal_active and (piece['type'] == '兵' or piece['type'] == '仕' or piece['type'] == '将') and to_center and after_corner <= 1:
                    score -= 4

            if edge_seal_active and target and target['face_up'] and self.is_edge_square(action['to_row'], action['to_col']):
                score += 3

            return score
        finally:
            game.restore_snapshot(snapshot)
            game.history = history

    def score_v9_black_closeout_tuning(self, game, action, piece, target):
        if self.side != 'black' or not piece or not self.is_v9_closeout_phase(game):
            return 0
        if action['type'] != 'move' and action['type'] != 'capture':
            return 0

        safe_action = not self.would_be_in_danger_after_move(game, action, piece)
        score = 0

        if safe_action:
            score += 3
        if piece['type'] == '兵' or piece['type'] == '仕':
            score += 2
        if target and target['face_up']:
            score += target['hp'] * 0.35

        enemy_king = next(iter(self.get_face_up_pieces(game, self.enemy_side(), '将')), None)
        if enemy_king:
            dist = self.manhattan_distance(action['to_row'], action['to_col'], enemy_king['row'], enemy_king['col'])
            if dist <= 1 and (piece['type'] == '兵' or piece['type'] == '仕' or piece['type'] == '将'):
                score += 5
            if self.is_edge_square(enemy_king['row'], enemy_king['col']) and self.is_edge_square(action['to_row'], action['to_col']):
                score += 2.5

        return score

    def compute_v9_edge_seal_score(self, game):
        enemy_king = next(iter(self.get_face_up_pieces(game, self.enemy_side(), '将')), None)
        if not enemy_king:
            return 0

        score = 0
        corner_distance = self.min_corner_distance(enemy_king['row'], enemy_king['col'])
        edge_neighbors = self.count_v9_edge_blocks_near_king(game, self.side)
        strong_lead = game.hp[self.side] - game.hp[self.enemy_side()] >= 12
        edge_seal_active = ((self.is_edge_square(enemy_king['row'], enemy_king['col']) or corner_distance <= 1)
                            if (self.side == 'black' or strong_lead)
                            else False)
        center_penalty = 6 if (edge_seal_active and (enemy_king['col'] == 1 or enemy_king['col'] == 2)) else 0

        score += (3 - min(3, corner_distance)) * (14 if edge_seal_active else 6)
        score += edge_neighbors * (5 if edge_seal_active else 2)
        score -= center_penalty
        score -= self.count_king_mobility(game, self.enemy_side()) * (4 if edge_seal_active else 1.5)

        return score

    def count_v9_edge_blocks_near_king(self, game, attacker_side):
        defender_side = 'black' if attacker_side == 'red' else 'red'
        enemy_king = next(iter(self.get_face_up_pieces(game, defender_side, '将')), None)
        if not enemy_king:
            return 0

        count = 0
        for nr, nc in self.neighbors(enemy_king['row'], enemy_king['col']):
            if not self.is_edge_square(nr, nc):
                continue
            occupant = game.get_piece(nr, nc)
            if occupant and occupant['face_up'] and occupant['side'] == attacker_side:
                count += 2 if occupant['type'] == '兵' else 1
                continue

            for piece in self.get_face_up_pieces(game, attacker_side, None):
                if piece['type'] == '炮':
                    if self.can_cannon_attack_square(game, piece['row'], piece['col'], nr, nc):
                        count += 1
                        break
                    continue

                if self.is_adjacent(piece['row'], piece['col'], nr, nc):
                    count += 1.5 if piece['type'] == '兵' else 1
                    break

        return count

    def is_edge_square(self, row, col):
        return row == 0 or row == ROWS - 1 or col == 0 or col == COLS - 1


class StrategicV10AI(StrategicV9AI):
    def evaluate(self, game, action):
        rand = random.random() * 0.01

        if action['type'] == 'flip':
            return self.evaluate_flip(game, action) + rand

        if action['type'] == 'capture':
            return self.evaluate_capture(game, action) + rand

        if action['type'] == 'move':
            return self.evaluate_move(game, action) + rand

        return rand

    # De-stack flip evaluation: rebuild from V6 base, skipping V7's flip scorers
    # and V9's closeout flip penalty (keep only V8 suppression with higher threshold)
    def evaluate_flip(self, game, action):
        # Call V6's evaluateFlip (not V7/V8/V9 layers)
        score = self.evaluate_flip_v6_base(game, action)
        # Keep V8 closeout flip suppression but with higher threshold
        score -= self.score_v10_closeout_flip_suppression(game, action)
        return score

    # Replicate the V6 flip evaluation chain: SimpleAI → V2 → V3 → V4 → V5 → V6
    def evaluate_flip_v6_base(self, game, action):
        # Call StrategicV6AI.prototype.evaluateFlip via the chain
        return StrategicV6AI.evaluate_flip(self, game, action)

    # Closeout flip suppression with higher threshold (hp >= 16 instead of 8)
    def score_v10_closeout_flip_suppression(self, game, action):
        if not self.is_v10_closeout_phase(game):
            return 0
        # Use V8's suppression logic directly but under V10's stricter gate
        return self.score_v8_lead_closeout_flip_suppression(game, action) * 0.5

    def is_v10_closeout_phase(self, game):
        hp_lead = game.hp[self.side] - game.hp[self.enemy_side()]
        if hp_lead < 16:
            return False
        return len(game.remaining_pool) <= 8 or self.count_all_face_up_pieces(game) >= 14 or game.hp[self.enemy_side()] <= 15

    # Override V9's closeout gate for edge seal and black tuning
    def is_v9_closeout_phase(self, game):
        hp_lead = game.hp[self.side] - game.hp[self.enemy_side()]
        threshold = 6 if self.side == 'black' else 8
        if hp_lead < threshold:
            return False
        return len(game.remaining_pool) <= 10 or self.count_all_face_up_pieces(game) >= 12 or game.hp[self.enemy_side()] <= 22

    # De-stack capture: skip V7 plan action, keep V8 conversion priority
    def evaluate_capture(self, game, action):
        attacker = game.get_piece(action.get('from_row'), action.get('from_col'))
        target = game.get_piece(action.get('to_row'), action.get('to_col'))
        # Build from V6 base
        score = StrategicV6AI.evaluate_capture(self, game, action)
        if not attacker or not target:
            return score

        # Keep V8 closeout action (with V10 threshold), conversion priority, V9 edge seal + black tuning
        score += self.score_v10_closeout_action(game, action, attacker, target)
        score += self.score_v8_conversion_priority(game, action, attacker, target)
        score += self.score_v9_edge_seal_action(game, action, attacker, target)
        score += self.score_v9_black_closeout_tuning(game, action, attacker, target)
        return score

    # De-stack move: skip V7 plan action
    def evaluate_move(self, game, action):
        piece = game.get_piece(action.get('from_row'), action.get('from_col'))
        # Build from V6 base
        score = StrategicV6AI.evaluate_move(self, game, action)
        if not piece:
            return score

        # Keep V8 closeout action (with V10 threshold) and V9 edge seal + black tuning
        score += self.score_v10_closeout_action(game, action, piece, None)
        score += self.score_v9_edge_seal_action(game, action, piece, None)
        score += self.score_v9_black_closeout_tuning(game, action, piece, None)
        return score

    # V8 closeout action but with V10's stricter threshold
    def score_v10_closeout_action(self, game, action, piece, target):
        if not self.is_v10_closeout_phase(game):
            return 0
        return self.score_v8_lead_closeout_action(game, action, piece, target)


class StrategicV11AI(StrategicV10AI):
    def evaluate(self, game, action):
        rand = random.random() * 0.01
        if action['type'] == 'flip':
            return self.evaluate_flip(game, action) + rand
        if action['type'] == 'capture':
            return self.evaluate_capture(game, action) + rand
        if action['type'] == 'move':
            return self.evaluate_move(game, action) + rand
        return rand

    # V11: Better flip quality via pool-wide variance analysis (from V7)
    def evaluate_flip(self, game, action):
        score = super().evaluate_flip(game, action)
        score += self.score_v7_relative_safe_flip(game, action.get('row'), action.get('col')) * 0.4
        return score


class StrategicV12AI(StrategicV11AI):
    def evaluate_flip(self, game, action):
        score = super().evaluate_flip(game, action)
        score -= self.score_v12_hidden_king_flip_trap(game, action['row'], action['col'])
        score -= self.score_v12_urgent_defense_flip_penalty(game)
        # [改进1] 当有可用吃子/移动时，大幅压制翻棋
        score -= self.score_v12_flip_suppression_when_actions_exist(game)
        # [改进2] 在己方重要棋子旁翻棋的额外惩罚
        score -= self.score_v12_flip_near_own_high_value_penalty(game, action['row'], action['col'])
        # [改进3] 处于劣势时翻棋的额外惩罚
        score -= self.score_v12_losing_flip_penalty(game)
        return score

    def evaluate_capture(self, game, action):
        score = super().evaluate_capture(game, action)
        score += self.score_v12_own_king_safety_delta(game, action)
        # [改进4] 炮吃未翻开棋子时的自吃风险惩罚
        score -= self.score_v12_cannon_blind_capture_penalty(game, action)
        return score

    def evaluate_move(self, game, action):
        score = super().evaluate_move(game, action)
        score += self.score_v12_own_king_safety_delta(game, action)
        # [改进5] 反复移动（来回走）的惩罚
        score -= self.score_v12_anti_oscillation(game, action)
        # [改进6] 优势时主动追击敌方棋子
        score += self.score_v12_active_pursuit(game, action)
        return score

    def get_move(self, game):
        actions = game.get_all_actions()
        if len(actions) == 0:
            return None

        scored_actions = []
        for action in actions:
            base_score = self.evaluate(game, action)
            response_score = self.evaluate_opponent_response(game, action)
            scored_actions.append({
                'action': action,
                'total_score': base_score + response_score,
                'base_score': base_score,
                'response_score': response_score,
            })

        soldier_mate_defense_actions = self.collect_v12_soldier_mate_defense_actions(game, scored_actions)
        defense_actions = self.collect_v12_king_defense_actions(game, scored_actions)
        forcing_actions = self.collect_v12_forcing_actions(game, scored_actions)
        candidates = (
            soldier_mate_defense_actions if len(soldier_mate_defense_actions) > 0
            else defense_actions if len(defense_actions) > 0
            else forcing_actions if len(forcing_actions) > 0
            else [{**entry, 'tactical_bonus': 0} for entry in scored_actions]
        )

        best_action = None
        best_score = -math.inf

        for candidate in candidates:
            tactical_bonus = candidate.get('tactical_bonus')
            final_score = candidate['total_score'] + (tactical_bonus if tactical_bonus is not None else 0)
            if final_score > best_score:
                best_score = final_score
                best_action = candidate['action']

        return best_action

    def collect_v12_soldier_mate_defense_actions(self, game, scored_actions):
        kings = self.get_face_up_pieces(game, self.side, '将')
        king = kings[0] if kings else None
        if not king:
            return []

        adjacent_enemy_soldiers = self.get_v12_adjacent_enemy_soldiers(game, king['row'], king['col'], self.side)
        if len(adjacent_enemy_soldiers) == 0:
            return []

        baseline_danger = self.get_v12_own_king_danger(game)
        defenses = []

        for entry in scored_actions:
            if entry['action']['type'] == 'flip':
                continue
            plan = self.analyze_v12_soldier_mate_defense(game, entry['action'], baseline_danger, len(adjacent_enemy_soldiers))
            if not plan['compelling']:
                continue
            defenses.append({
                **entry,
                'tactical_bonus': plan['bonus'],
            })

        return defenses

    def analyze_v12_soldier_mate_defense(self, game, action, baseline_danger, baseline_soldiers):
        if not self.can_simulate_v12_action(game, action):
            return {'compelling': False, 'bonus': 0}

        snapshot = game.create_snapshot()
        history = list(game.history)

        try:
            applied = game.move_piece(action['from_row'], action['from_col'], action['to_row'], action['to_col'])
            if not applied:
                return {'compelling': False, 'bonus': 0}

            kings = self.get_face_up_pieces(game, self.side, '将')
            king = kings[0] if kings else None
            if not king:
                return {'compelling': False, 'bonus': 0}

            after_soldiers = len(self.get_v12_adjacent_enemy_soldiers(game, king['row'], king['col'], self.side))
            after_danger = self.get_v12_own_king_danger(game)
            bonus = (baseline_soldiers - after_soldiers) * 160
            bonus += (baseline_danger - after_danger) * 1.2

            original_target = snapshot['board'][action['to_row']][action['to_col']] if action['type'] == 'capture' else None
            if original_target and original_target['face_up'] and original_target['side'] == self.enemy_side() and original_target['type'] == '兵':
                bonus += 140

            moving_piece = snapshot['board'][action['from_row']][action['from_col']]
            king_moved = bool(moving_piece) and moving_piece['face_up'] and moving_piece['side'] == self.side and moving_piece['type'] == '将'
            if king_moved:
                bonus += 24

            compelling = after_soldiers < baseline_soldiers or after_danger + 80 < baseline_danger
            return {'compelling': compelling, 'bonus': bonus}
        finally:
            game.restore_snapshot(snapshot)
            game.history = history

    def collect_v12_king_defense_actions(self, game, scored_actions):
        baseline_danger = self.get_v12_own_king_danger(game)
        if baseline_danger < 90:
            return []

        defenses = []
        for entry in scored_actions:
            if entry['action']['type'] == 'flip':
                continue
            plan = self.analyze_v12_king_defense(game, entry['action'], baseline_danger)
            if not plan['compelling']:
                continue
            defenses.append({
                **entry,
                'tactical_bonus': plan['bonus'],
            })

        return defenses

    def analyze_v12_king_defense(self, game, action, baseline_danger):
        if not self.can_simulate_v12_action(game, action):
            return {'compelling': False, 'bonus': 0}

        before_escapes = self.get_v12_own_king_escape_count(game)
        snapshot = game.create_snapshot()
        history = list(game.history)

        try:
            applied = game.move_piece(action['from_row'], action['from_col'], action['to_row'], action['to_col'])
            if not applied:
                return {'compelling': False, 'bonus': 0}

            after_danger = self.get_v12_own_king_danger(game)
            after_escapes = self.get_v12_own_king_escape_count(game)
            bonus = (baseline_danger - after_danger) * 1.7
            bonus += (after_escapes - before_escapes) * 20

            target = game.get_piece(action['to_row'], action['to_col']) if action['type'] == 'capture' else None
            if target and target['side'] == self.side:
                bonus += 0

            original_target = snapshot['board'][action['to_row']][action['to_col']] if action['type'] == 'capture' else None
            if original_target and original_target['face_up'] and original_target['side'] == self.enemy_side() and original_target['type'] == '兵':
                bonus += 96

            if after_danger <= 30:
                bonus += 28

            if after_danger >= baseline_danger:
                bonus -= 42

            compelling = after_danger + 35 < baseline_danger or after_danger <= 35 or after_escapes > before_escapes
            return {'compelling': compelling, 'bonus': bonus}
        finally:
            game.restore_snapshot(snapshot)
            game.history = history

    def collect_v12_forcing_actions(self, game, scored_actions):
        capture_plans = []
        setup_plans = []

        for entry in scored_actions:
            if entry['action']['type'] == 'capture':
                plan = self.analyze_v12_capture(game, entry['action'])
                if plan['compelling']:
                    capture_plans.append({
                        **entry,
                        'tactical_bonus': plan['bonus'],
                    })
                continue

            if entry['action']['type'] == 'move':
                plan = self.analyze_v12_attack_setup(game, entry['action'])
                if plan['compelling']:
                    setup_plans.append({
                        **entry,
                        'tactical_bonus': plan['bonus'],
                    })

        if len(capture_plans) > 0:
            return capture_plans

        if len(setup_plans) > 0:
            return setup_plans

        return []

    def analyze_v12_capture(self, game, action):
        attacker = game.get_piece(action['from_row'], action['from_col'])
        target = game.get_piece(action['to_row'], action['to_col'])
        if not attacker or not target:
            return {'compelling': False, 'bonus': 0}

        is_safe = not self.would_be_in_danger_after_move(game, action, attacker)
        bonus = 0

        if target['face_up']:
            target_value = self.material_value(target['type'])
            bonus += target_value * 1.2
            bonus += target['hp'] * 10

            if target['type'] == '将':
                bonus += 260
            elif target['type'] == '仕':
                bonus += 110
            elif target['type'] == '象':
                bonus += 75
            elif target['type'] == '车' or target['type'] == '马':
                bonus += 65
            elif target['type'] == '炮':
                bonus += 48
            else:
                bonus += 22

            if self.is_in_danger(game, action['from_row'], action['from_col'], attacker):
                bonus += attacker['hp'] * 10 + self.material_value(attacker['type']) * 0.16

            bonus += self.score_hp_swing(game, target['hp'], 0 if is_safe else attacker['hp']) * 0.8
            bonus += self.score_capture_threat_removal(game, action['to_row'], action['to_col'], target) * 0.7

            if is_safe:
                bonus += 36
            else:
                bonus -= attacker['hp'] * 12
                bonus -= self.material_value(attacker['type']) * 0.3

            compelling = (
                (bonus >= 85 or self.is_v12_priority_target(target))
                if is_safe
                else (target['type'] == '将' or bonus >= 145)
            )

            return {'compelling': compelling, 'bonus': bonus}

        bonus += 8
        bonus += self.score_cannon_mouth(game, action['to_row'], action['to_col']) * 0.25
        if is_safe:
            bonus += 6

        return {
            'compelling': bonus >= 42,
            'bonus': bonus,
        }

    def analyze_v12_attack_setup(self, game, action):
        piece = game.get_piece(action['from_row'], action['from_col'])
        if not piece:
            return {'compelling': False, 'bonus': 0}

        baseline_pressure = self.get_v12_best_visible_capture_pressure(game)
        snapshot = game.create_snapshot()
        history = list(game.history)

        try:
            applied = game.move_piece(action['from_row'], action['from_col'], action['to_row'], action['to_col'])
            if not applied:
                return {'compelling': False, 'bonus': 0}

            moved_piece = game.get_piece(action['to_row'], action['to_col'])
            if not moved_piece:
                return {'compelling': False, 'bonus': 0}

            own_visible_pressure = self.get_v12_best_visible_capture_pressure(game, self.side)
            moved_piece_pressure = self.get_v12_best_visible_capture_pressure_from_square(game, action['to_row'], action['to_col'], moved_piece)
            created_pressure = max(0, own_visible_pressure['score'] - baseline_pressure['score'])
            is_safe = not self.is_in_danger(game, action['to_row'], action['to_col'], moved_piece)
            pressure_target = moved_piece_pressure['target'] or own_visible_pressure['target']
            bonus = 0

            bonus += created_pressure * 0.95
            bonus += moved_piece_pressure['score'] * 0.75

            if pressure_target:
                bonus += self.material_value(pressure_target['type']) * 0.35
                if self.is_v12_priority_target(pressure_target):
                    bonus += 28

            if self.is_in_danger(game, action['from_row'], action['from_col'], piece):
                bonus += piece['hp'] * 6

            if is_safe:
                bonus += 18
            else:
                bonus -= piece['hp'] * 10
                bonus -= self.material_value(piece['type']) * 0.2

            compelling = is_safe and (
                moved_piece_pressure['score'] >= 70 or
                own_visible_pressure['score'] >= 92 or
                bool(pressure_target and self.is_v12_priority_target(pressure_target) and moved_piece_pressure['score'] >= 48) or
                created_pressure >= 42
            )

            return {'compelling': compelling, 'bonus': bonus}
        finally:
            game.restore_snapshot(snapshot)
            game.history = history

    def get_v12_best_visible_capture_pressure(self, game, side=None):
        if side is None:
            side = self.side
        best = {'score': 0, 'target': None}
        original_turn = game.current_turn

        try:
            game.current_turn = side

            for piece in self.get_face_up_pieces(game, side, None):
                moves = game.get_valid_moves(piece['row'], piece['col'])
                for move in moves:
                    if move['type'] != 'capture':
                        continue
                    target = game.get_piece(move['row'], move['col'])
                    if not target or not target['face_up']:
                        continue

                    score = self.score_v12_future_capture(game, piece, target, {
                        'from_row': piece['row'],
                        'from_col': piece['col'],
                        'to_row': move['row'],
                        'to_col': move['col'],
                    })

                    if score > best['score']:
                        best = {'score': score, 'target': target}
        finally:
            game.current_turn = original_turn

        return best

    def get_v12_best_visible_capture_pressure_from_square(self, game, row, col, piece):
        original_turn = game.current_turn
        best = {'score': 0, 'target': None}

        try:
            game.current_turn = piece['side']
            moves = game.get_valid_moves(row, col)
            for move in moves:
                if move['type'] != 'capture':
                    continue
                target = game.get_piece(move['row'], move['col'])
                if not target or not target['face_up']:
                    continue

                score = self.score_v12_future_capture(game, piece, target, {
                    'from_row': row,
                    'from_col': col,
                    'to_row': move['row'],
                    'to_col': move['col'],
                })

                if score > best['score']:
                    best = {'score': score, 'target': target}
        finally:
            game.current_turn = original_turn

        return best

    def score_v12_future_capture(self, game, attacker, target, action):
        score = self.material_value(target['type']) * 1.15 + target['hp'] * 9

        if target['type'] == '将':
            score += 220
        elif target['type'] == '仕':
            score += 95
        elif target['type'] == '象':
            score += 64
        elif target['type'] == '车' or target['type'] == '马':
            score += 52
        elif target['type'] == '炮':
            score += 38

        if not self.is_in_danger(game, action['to_row'], action['to_col'], attacker):
            score += 18

        return score

    def score_v12_own_king_safety_delta(self, game, action):
        if not self.can_simulate_v12_action(game, action):
            return 0

        before_danger = self.get_v12_own_king_danger(game)
        before_escapes = self.get_v12_own_king_escape_count(game)
        before_adjacent_soldiers = self.get_v12_own_king_adjacent_soldier_count(game)
        if before_danger == 0 and before_escapes == 0:
            return 0

        snapshot = game.create_snapshot()
        history = list(game.history)

        try:
            applied = game.move_piece(action['from_row'], action['from_col'], action['to_row'], action['to_col'])
            if not applied:
                return 0

            after_danger = self.get_v12_own_king_danger(game)
            after_escapes = self.get_v12_own_king_escape_count(game)
            after_adjacent_soldiers = self.get_v12_own_king_adjacent_soldier_count(game)
            score = (before_danger - after_danger) * 0.9
            score += (after_escapes - before_escapes) * 12
            score += (before_adjacent_soldiers - after_adjacent_soldiers) * 90

            if before_danger >= 90 and after_danger >= before_danger:
                score -= 34
            if before_adjacent_soldiers > 0 and after_adjacent_soldiers >= before_adjacent_soldiers:
                score -= 46
            if after_danger <= 35:
                score += 18

            return score
        finally:
            game.restore_snapshot(snapshot)
            game.history = history

    def score_v12_urgent_defense_flip_penalty(self, game):
        danger = self.get_v12_own_king_danger(game)
        if danger < 90:
            return 0

        actions = game.get_all_actions()
        for action in actions:
            if action['type'] == 'flip':
                continue
            plan = self.analyze_v12_king_defense(game, action, danger)
            if plan['compelling']:
                return 180 + min(120, plan['bonus'] * 0.3)

        return 0

    def score_v12_hidden_king_flip_trap(self, game, row, col):
        if not self.is_v12_own_king_hidden(game):
            return 0

        penalty = self.score_v12_square_danger_for_king(game, row, col, self.side) * 1.2

        for piece in self.get_face_up_pieces(game, self.enemy_side(), None):
            if piece['type'] == '兵':
                dist = self.manhattan_distance(piece['row'], piece['col'], row, col)
                if dist == 1:
                    penalty += 320
                elif dist == 2 and self.has_v12_soldier_approach_path(game, piece['row'], piece['col'], row, col, self.side):
                    penalty += 120

            if piece['type'] == '炮' and self.can_cannon_attack_square(game, piece['row'], piece['col'], row, col):
                penalty += 150

        return penalty

    def get_v12_own_king_danger(self, game, side=None):
        if side is None:
            side = self.side
        kings = self.get_face_up_pieces(game, side, '将')
        king = kings[0] if kings else None
        if not king:
            return 0
        return self.score_v12_square_danger_for_king(game, king['row'], king['col'], side)

    def get_v12_own_king_adjacent_soldier_count(self, game, side=None):
        if side is None:
            side = self.side
        kings = self.get_face_up_pieces(game, side, '将')
        king = kings[0] if kings else None
        if not king:
            return 0
        return len(self.get_v12_adjacent_enemy_soldiers(game, king['row'], king['col'], side))

    def get_v12_own_king_escape_count(self, game, side=None):
        if side is None:
            side = self.side
        kings = self.get_face_up_pieces(game, side, '将')
        king = kings[0] if kings else None
        if not king:
            return 0

        original_turn = game.current_turn
        try:
            game.current_turn = side
            moves = game.get_valid_moves(king['row'], king['col'])

            def _filter(move):
                destination_piece = game.get_piece(move['row'], move['col'])
                if move['type'] == 'capture' and destination_piece and destination_piece['type'] == '兵':
                    return False

                candidate = {
                    'type': move['type'],
                    'from_row': king['row'],
                    'from_col': king['col'],
                    'to_row': move['row'],
                    'to_col': move['col'],
                }
                if self.would_be_in_danger_after_move(game, candidate, king):
                    return False

                return self.score_v12_square_danger_for_king(game, move['row'], move['col'], side) < 90

            return len([move for move in moves if _filter(move)])
        finally:
            game.current_turn = original_turn

    def score_v12_square_danger_for_king(self, game, row, col, side):
        enemy = 'black' if side == 'red' else 'red'
        danger = self.count_cannon_attackers(game, row, col, enemy) * 120

        for piece in self.get_face_up_pieces(game, enemy, None):
            dist = self.manhattan_distance(piece['row'], piece['col'], row, col)
            if piece['type'] == '兵':
                if dist == 1:
                    danger += 280
                elif dist == 2 and self.has_v12_soldier_approach_path(game, piece['row'], piece['col'], row, col, side):
                    danger += 95
                continue

            if piece['type'] == '将' and dist == 1:
                danger += 110

        return danger

    def has_v12_soldier_approach_path(self, game, soldier_row, soldier_col, king_row, king_col, defender_side):
        for nr, nc in self.neighbors(king_row, king_col):
            if not self.is_adjacent(soldier_row, soldier_col, nr, nc):
                continue
            cell = game.get_piece(nr, nc)
            if not cell:
                return True
            if cell['face_up'] and cell['side'] == defender_side and cell['type'] == '将':
                return True
        return False

    def get_v12_adjacent_enemy_soldiers(self, game, row, col, side):
        enemy = 'black' if side == 'red' else 'red'
        soldiers = []
        for nr, nc in self.neighbors(row, col):
            piece = game.get_piece(nr, nc)
            if not piece or not piece['face_up'] or piece['side'] != enemy or piece['type'] != '兵':
                continue
            soldiers.append({**piece, 'row': nr, 'col': nc})
        return soldiers

    def is_v12_own_king_hidden(self, game):
        return self.count_face_up_pieces(game, self.side, '将') == 0 and self.count_living_pieces(game, self.side, '将') > 0

    def can_simulate_v12_action(self, game, action):
        if action['type'] == 'move':
            return True
        if action['type'] != 'capture':
            return False
        target = game.get_piece(action['to_row'], action['to_col'])
        return bool(target) and target['face_up']

    def is_v12_priority_target(self, piece):
        return piece['type'] == '将' or piece['type'] == '仕' or piece['type'] == '象' or piece['type'] == '车' or piece['type'] == '马' or piece['type'] == '炮'

    # =========================================================
    # 以下为基于棋谱分析新增的改进策略
    # =========================================================

    def score_v12_flip_suppression_when_actions_exist(self, game):
        """
        [改进1] 当存在可用的吃子或安全移动时，大幅压制翻棋得分
        分析发现：AI翻棋比例平均65%，最差91%，导致浪费大量回合。
        当有高价值吃子或安全移动可选时，翻棋应被强力抑制。
        """
        actions = game.get_all_actions()
        best_capture_score = -math.inf
        safe_moves_count = 0
        has_capture_action = False

        for action in actions:
            if action['type'] == 'flip':
                continue

            if action['type'] == 'capture':
                has_capture_action = True
                attacker = game.get_piece(action['from_row'], action['from_col'])
                target = game.get_piece(action['to_row'], action['to_col'])
                if attacker and target and target['face_up']:
                    is_safe = not self.would_be_in_danger_after_move(game, action, attacker)
                    capture_value = target['hp'] * 6 + self.material_value(target['type']) * 0.3
                    if is_safe:
                        capture_value += 20
                    if target['type'] == '将':
                        capture_value += 200
                    if capture_value > best_capture_score:
                        best_capture_score = capture_value

            if action['type'] == 'move':
                piece = game.get_piece(action['from_row'], action['from_col'])
                if piece and not self.would_be_in_danger_after_move(game, action, piece):
                    safe_moves_count += 1

        penalty = 0

        # 有高价值吃子目标时，强力抑制翻棋
        if best_capture_score > 30:
            penalty += min(80, best_capture_score * 0.6)

        # 有吃子动作存在时，基础抑制
        if has_capture_action:
            penalty += 15

        # 有多个安全移动可选时，翻棋不应是首选
        if safe_moves_count >= 2:
            penalty += 8 + safe_moves_count * 2

        return penalty

    def score_v12_flip_near_own_high_value_penalty(self, game, row, col):
        """
        [改进2] 在己方重要棋子旁翻棋的额外危险惩罚
        分析发现：翻出的敌方棋子经常紧邻己方重要棋子，导致下回合被吃。
        翻棋位置与己方将/仕/炮相邻时，给予高额惩罚。
        """
        penalty = 0

        for nr, nc in self.neighbors(row, col):
            piece = game.get_piece(nr, nc)
            if not piece or not piece['face_up'] or piece['side'] != self.side:
                continue

            if piece['type'] == '将':
                penalty += 45
            elif piece['type'] == '仕':
                penalty += 20
            elif piece['type'] == '炮':
                penalty += 15
            elif piece['type'] == '车':
                penalty += 12

        # 如果池中敌方棋子比例高，翻出敌方棋子概率更大，惩罚加重
        pool = game.remaining_pool
        if len(pool) > 0:
            enemy_in_pool = len([p for p in pool if p['side'] == self.enemy_side()])
            enemy_ratio = enemy_in_pool / len(pool)
            if enemy_ratio > 0.5:
                penalty *= 1 + (enemy_ratio - 0.5) * 1.5

        return penalty

    def score_v12_losing_flip_penalty(self, game):
        """
        [改进3] 处于劣势时翻棋的额外惩罚
        分析发现：AI在HP落后时仍大量翻棋，浪费宝贵回合。
        落后时应优先执行吃子、防守等实质性操作。
        """
        hp_diff = game.hp[self.side] - game.hp[self.enemy_side()]
        if hp_diff >= 0:
            return 0

        penalty = 0
        deficit = abs(hp_diff)

        # 越落后，翻棋惩罚越大
        if deficit >= 20:
            penalty += 35
        elif deficit >= 10:
            penalty += 20
        else:
            penalty += 10

        # 如果有可执行的吃子动作，进一步加重翻棋惩罚
        actions = game.get_all_actions()

        def _has_capturable(a):
            if a['type'] != 'capture':
                return False
            target = game.get_piece(a['to_row'], a['to_col'])
            return bool(target and target['face_up'])

        has_capturable = any(_has_capturable(a) for a in actions)
        if has_capturable:
            penalty += 25

        return penalty

    def score_v12_cannon_blind_capture_penalty(self, game, action):
        """
        [改进4] 炮吃未翻开棋子时的自吃风险惩罚
        分析发现：游戏8中黑炮吃了自己的黑象（隐藏子来自池中，随机为己方）。
        当炮吃隐藏子时，必须评估池中己方棋子比例，避免自吃。
        """
        if action['type'] != 'capture':
            return 0

        attacker = game.get_piece(action['from_row'], action['from_col'])
        target = game.get_piece(action['to_row'], action['to_col'])

        # 仅处理炮吃未翻开棋子的情况
        if not attacker or attacker['type'] != '炮':
            return 0
        if not target or target['face_up']:
            return 0

        # 计算池中己方棋子比例
        pool = game.remaining_pool
        if len(pool) == 0:
            return 0

        own_in_pool = len([p for p in pool if p['side'] == self.side])
        own_ratio = own_in_pool / len(pool)

        penalty = 0

        # 己方棋子在池中占多数时，自吃风险很高
        if own_ratio >= 0.6:
            penalty += 80 + own_ratio * 60
        elif own_ratio >= 0.4:
            penalty += 40 + own_ratio * 30
        elif own_ratio >= 0.2:
            penalty += 15

        # 计算期望HP损失：自吃时会扣己方HP
        expected_own_damage = 0
        for p in pool:
            if p['side'] == self.side:
                expected_own_damage += p['hp']
        expected_own_damage /= len(pool)
        penalty += expected_own_damage * own_ratio * 4

        # 如果HP很低，自吃可能直接输掉比赛
        if game.hp[self.side] <= expected_own_damage * own_ratio + 5:
            penalty += 150

        return penalty

    def score_v12_anti_oscillation(self, game, action):
        """
        [改进5] 反复移动（来回走）的惩罚
        分析发现：游戏5/6/11中，AI的帅/车在两个位置来回移动20+回合，
        完全浪费行动力。检测并惩罚这种重复移动模式。
        """
        history = game.history
        if not history or len(history) < 3:
            return 0

        # 检查是否在移回上上个位置（即来回走）
        prev_snap = history[len(history) - 1]
        if not prev_snap or not prev_snap.get('last_action'):
            return 0

        last_action = prev_snap['last_action']

        # 自己上一步的目标位置
        penalty = 0

        # 检查最近的历史记录，找到当前棋子的移动模式
        piece = game.get_piece(action['from_row'], action['from_col'])
        if not piece:
            return 0

        # 检测：如果之前相同棋子从目标位置移到当前位置，现在又要移回去
        if len(history) >= 2:
            for i in range(len(history) - 1, max(0, len(history) - 6) - 1, -1):
                snap = history[i]
                if not snap.get('last_action'):
                    continue
                prev_action = snap['last_action']

                if prev_action['type'] == 'move' or prev_action['type'] == 'capture':
                    from_ = prev_action.get('from')
                    to = prev_action.get('to')
                    if not from_ or not to:
                        continue

                    # 检测来回走：之前从A到B，现在从B到A
                    if (to['row'] == action['from_row'] and to['col'] == action['from_col'] and
                            from_['row'] == action['to_row'] and from_['col'] == action['to_col']):
                        penalty += 35
                        break

        # 额外检测：如果过去多次在相同两点之间移动
        oscillation_count = 0
        for i in range(len(history) - 1, max(0, len(history) - 10) - 1, -1):
            snap = history[i]
            if not snap.get('last_action'):
                continue
            prev_action = snap['last_action']
            if prev_action['type'] != 'move':
                continue
            from_ = prev_action.get('from')
            to = prev_action.get('to')
            if not from_ or not to:
                continue

            is_oscillating = (
                (from_['row'] == action['from_row'] and from_['col'] == action['from_col'] and to['row'] == action['to_row'] and to['col'] == action['to_col']) or
                (to['row'] == action['from_row'] and to['col'] == action['from_col'] and from_['row'] == action['to_row'] and from_['col'] == action['to_col'])
            )

            if is_oscillating:
                oscillation_count += 1

        if oscillation_count >= 2:
            penalty += 25 + oscillation_count * 15

        return penalty

    def score_v12_active_pursuit(self, game, action):
        """
        [改进6] 优势时主动追击敌方棋子
        分析发现：AI在HP领先时消极移动，不接近敌方棋子。
        当领先时，应主动靠近敌方棋子以制造吃子机会。
        """
        hp_lead = game.hp[self.side] - game.hp[self.enemy_side()]
        if hp_lead < 5:
            return 0

        piece = game.get_piece(action['from_row'], action['from_col'])
        if not piece:
            return 0

        # 如果移动后会陷入危险，不追击
        if self.would_be_in_danger_after_move(game, action, piece):
            return 0

        score = 0
        enemy_pieces = self.get_face_up_pieces(game, self.enemy_side(), None)
        if len(enemy_pieces) == 0:
            return 0

        for enemy in enemy_pieces:
            before_dist = self.manhattan_distance(action['from_row'], action['from_col'], enemy['row'], enemy['col'])
            after_dist = self.manhattan_distance(action['to_row'], action['to_col'], enemy['row'], enemy['col'])

            # 拉近距离得分
            dist_delta = before_dist - after_dist
            if dist_delta <= 0:
                continue

            approach_bonus = dist_delta * 4

            # 根据目标价值加权
            if enemy['type'] == '将':
                approach_bonus *= 3
            elif enemy['type'] == '仕':
                approach_bonus *= 2
            elif enemy['type'] == '象' or enemy['type'] == '车':
                approach_bonus *= 1.5

            # 兵接近将特别重要
            if piece['type'] == '兵' and enemy['type'] == '将':
                approach_bonus += 15

            # 仕接近敌方棋子也重要
            if piece['type'] == '仕':
                approach_bonus += 5

            # 移动后与敌方相邻，可以构成吃子威胁
            if after_dist == 1 and piece['type'] != '炮':
                if self.can_capture_like(piece, enemy):
                    approach_bonus += 12

            score += approach_bonus

        # 根据优势程度缩放
        lead_multiplier = min(2.0, 1.0 + hp_lead / 30)
        return score * lead_multiplier



# ===== StrategicV13AI =====
# 基于V12AI的败局棋谱分析，聚焦两点改进:
# 核心改进1: 大幅增加翻棋邻近己方将的惩罚（败局67.6%有翻棋致将被吃）
# 核心改进2: 提高安全明吃的优先级，减少被动空走
class StrategicV13AI(StrategicV12AI):

    def score_v12_flip_suppression_when_actions_exist(self, game):
        penalty = super().score_v12_flip_suppression_when_actions_exist(game)
        actions = game.get_all_actions()
        safe_visible_captures = 0
        best_safe_capture_score = 0

        for action in actions:
            if action['type'] != 'capture':
                continue
            attacker = game.get_piece(action.get('from_row'), action.get('from_col'))
            target = game.get_piece(action.get('to_row'), action.get('to_col'))
            if not attacker or not target or not target['face_up']:
                continue
            if self.would_be_in_danger_after_move(game, action, attacker):
                continue

            safe_visible_captures += 1
            capture_score = target['hp'] * 6 + self.material_value(target['type']) * 0.25
            if capture_score > best_safe_capture_score:
                best_safe_capture_score = capture_score

        if safe_visible_captures > 0:
            penalty += 6
            penalty += min(14, best_safe_capture_score * 0.08)
            if safe_visible_captures >= 2:
                penalty += 4

        return penalty

    # 仅增强将的邻近翻棋惩罚，其余保持V12水平
    def score_v12_flip_near_own_high_value_penalty(self, game, row, col):
        penalty = 0

        for nr, nc in self.neighbors(row, col):
            piece = game.get_piece(nr, nc)
            if not piece or not piece['face_up'] or piece['side'] != self.side:
                continue

            if piece['type'] == '将':
                penalty += 90   # V12=45, 翻棋邻近将是最大败因
            elif piece['type'] == '仕':
                penalty += 25   # V12=20
            elif piece['type'] == '炮':
                penalty += 15   # 保持V12
            elif piece['type'] == '车':
                penalty += 12   # 保持V12

        pool = game.remaining_pool
        if len(pool) > 0 and penalty > 0:
            enemy_in_pool = len([p for p in pool if p['side'] == self.enemy_side()])
            enemy_ratio = enemy_in_pool / len(pool)
            if enemy_ratio > 0.5:
                penalty *= 1 + (enemy_ratio - 0.5) * 1.5

        return penalty

    def score_v12_cannon_blind_capture_penalty(self, game, action):
        penalty = super().score_v12_cannon_blind_capture_penalty(game, action)
        if action['type'] != 'capture':
            return penalty

        attacker = game.get_piece(action.get('from_row'), action.get('from_col'))
        target = game.get_piece(action.get('to_row'), action.get('to_col'))
        if not attacker or attacker['type'] != '炮' or not target or target['face_up']:
            return penalty

        pool = game.remaining_pool
        if len(pool) == 0:
            return penalty

        own_in_pool = len([piece for piece in pool if piece['side'] == self.side])
        own_ratio = own_in_pool / len(pool)
        if own_ratio >= 0.5:
            penalty += 12
        elif own_ratio >= 0.35:
            penalty += 5

        return penalty

    def evaluate_move(self, game, action):
        score = super().evaluate_move(game, action)
        score += self.score_v13_safe_pressure(game, action)
        return score

    def score_v13_safe_pressure(self, game, action):
        piece = game.get_piece(action.get('from_row'), action.get('from_col'))
        if not piece:
            return 0
        if self.would_be_in_danger_after_move(game, action, piece):
            return 0

        def _is_visible_capture(candidate):
            if candidate['type'] != 'capture':
                return False
            target = game.get_piece(candidate.get('to_row'), candidate.get('to_col'))
            return bool(target and target['face_up'])

        has_visible_capture = any(_is_visible_capture(candidate) for candidate in game.get_all_actions())
        if has_visible_capture:
            return 0

        bonus = 0
        enemy_pieces = self.get_face_up_pieces(game, self.enemy_side(), None)
        for enemy in enemy_pieces:
            before_dist = self.manhattan_distance(action.get('from_row'), action.get('from_col'), enemy['row'], enemy['col'])
            after_dist = self.manhattan_distance(action.get('to_row'), action.get('to_col'), enemy['row'], enemy['col'])
            dist_delta = before_dist - after_dist
            if dist_delta <= 0:
                continue

            pressure = dist_delta * 2
            if enemy['type'] == '将':
                pressure += 4
            elif enemy['type'] == '仕':
                pressure += 2

            if after_dist == 1 and piece['type'] != '炮' and self.can_capture_like(piece, enemy):
                pressure += 6

            bonus += pressure

        if self.is_in_danger(game, action.get('from_row'), action.get('from_col'), piece):
            bonus += piece['hp'] * 0.5

        return bonus

    def evaluate_capture(self, game, action):
        attacker = game.get_piece(action.get('from_row'), action.get('from_col'))
        target = game.get_piece(action.get('to_row'), action.get('to_col'))
        score = super().evaluate_capture(game, action)

        if not attacker or not target or not target['face_up']:
            return score

        safe_action = not self.would_be_in_danger_after_move(game, action, attacker)
        if not safe_action:
            return score

        bonus = target['hp'] * 0.8 + self.material_value(target['type']) * 0.06

        if self.is_in_danger(game, action.get('from_row'), action.get('from_col'), attacker):
            bonus += attacker['hp'] * 0.8

        if target['type'] == '仕':
            bonus += 4
        if target['type'] == '炮':
            bonus += 3
        if target['type'] == '兵' and self.enemy_king_alive(game):
            bonus += 3
        if attacker['type'] == '兵' and target['type'] != '将':
            bonus += 2
        if game.hp[self.side] <= game.hp[self.enemy_side()]:
            bonus += 4

        return score + bonus


# ===== StrategicV14AI =====
# 从“单步动作打分”升级到“详细动作 + alpha-beta 搜索”。
# 关键变化:
# 1. flip / 炮吃暗子会把 poolIndex 作为动作的一部分参与搜索。
# 2. 使用 deterministic negamax + alpha-beta + transposition table。
# 3. 叶子节点评估改为局面评估，而不是沿用随机单步 evaluate()。
# 4. 集成V12战术覆盖（将防御/兵杀防御）作为搜索前过滤。
# 5. 轻量化叶节点评估以提升搜索速度。
class StrategicV14AI(StrategicV13AI):
    def __init__(self, side):
        super().__init__(side)
        self.tt_max_size = 50000
        self.max_pool_choices = 2
        self.max_flip_squares = 3
        self.quiescence_depth = 1
        self.max_root_actions = 12
        self.max_node_actions = 6
        self.max_noisy_actions = 4
        self.mutual_capture_search_extension = 1
        self.mutual_capture_max_ply = 4
        self.transposition_table = {}
        self.evaluator_cache = {}

    def get_move(self, game):
        actions = self.get_detailed_actions(game, game.current_turn)
        if not actions:
            return None

        # V12 tactical overrides: king defense takes priority
        tactical_action = self.get_tactical_override(game)
        if tactical_action:
            return tactical_action

        self.transposition_table.clear()
        depth = self.choose_search_depth(game, len(actions))
        ordered_actions = self.order_actions(game, actions, game.current_turn, 0, None, self.max_root_actions)

        alpha = -math.inf
        beta = math.inf
        best_score = -math.inf
        best_action = ordered_actions[0]

        for action in ordered_actions:
            score = self.search_after_action(game, action, depth - 1, -beta, -alpha, 1)
            if score > best_score:
                best_score = score
                best_action = action
            if score > alpha:
                alpha = score

        return best_action

    # Use V12/V13's tactical systems for urgent situations
    def get_tactical_override(self, game):
        evaluator = self.get_side_evaluator(game.current_turn)
        actions = game.get_all_actions()
        if not actions:
            return None
        hidden_cells = self.count_hidden_board_cells(game)

        def _is_visible_capture(action):
            if action['type'] != 'capture':
                return False
            target = game.get_piece(action.get('to_row'), action.get('to_col'))
            return bool(target) and target['face_up']

        visible_capture_actions = [action for action in actions if _is_visible_capture(action)]

        # Quick check: only compute scored actions if we actually need defense
        king_danger = evaluator.get_v12_own_king_danger(game)

        def _soldier_threat(s):
            king_list = evaluator.get_face_up_pieces(game, evaluator.side, '将')
            king = king_list[0] if king_list else None
            return bool(king) and evaluator.manhattan_distance(s['row'], s['col'], king['row'], king['col']) <= 2

        has_soldier_mate_threat = any(
            _soldier_threat(s) for s in evaluator.get_face_up_pieces(game, evaluator.enemy_side(), '兵')
        )
        forcing_window = hidden_cells <= 8 and len(actions) <= 14

        if king_danger < 90 and not has_soldier_mate_threat and not forcing_window and len(visible_capture_actions) == 0:
            return None

        scored_actions = []
        for action in actions:
            base_score = evaluator.evaluate(game, action)
            response_score = evaluator.evaluate_opponent_response(game, action)
            scored_actions.append({
                'action': action,
                'total_score': base_score + response_score,
                'base_score': base_score,
                'response_score': response_score,
            })

        # Check soldier mate defense (highest priority)
        if has_soldier_mate_threat:
            soldier_mate_defense = evaluator.collect_v12_soldier_mate_defense_actions(game, scored_actions)
            if len(soldier_mate_defense) > 0:
                best_action = None
                best_score = -math.inf
                for candidate in soldier_mate_defense:
                    tb = candidate.get('tactical_bonus')
                    final_score = candidate['total_score'] + (tb if tb is not None else 0)
                    if final_score > best_score:
                        best_score = final_score
                        best_action = candidate['action']
                return best_action

        # Check king defense (high priority)
        if king_danger >= 90:
            king_defense = evaluator.collect_v12_king_defense_actions(game, scored_actions)
            if len(king_defense) > 0:
                best_action = None
                best_score = -math.inf
                for candidate in king_defense:
                    tb = candidate.get('tactical_bonus')
                    final_score = candidate['total_score'] + (tb if tb is not None else 0)
                    if final_score > best_score:
                        best_score = final_score
                        best_action = candidate['action']
                return best_action

        immediate_capture_action = self.get_immediate_capture_override(game, evaluator, visible_capture_actions)
        if immediate_capture_action:
            return immediate_capture_action

        # Reuse V12's strong forcing heuristics only in low-hidden, moderate-branching root positions.
        if forcing_window:
            forcing_actions = evaluator.collect_v12_forcing_actions(game, scored_actions)
            if len(forcing_actions) > 0:
                best_forcing_action = None
                best_forcing_score = -math.inf
                best_forcing_bonus = -math.inf
                for candidate in forcing_actions:
                    tb = candidate.get('tactical_bonus')
                    tactical_bonus = tb if tb is not None else 0
                    final_score = candidate['total_score'] + tactical_bonus
                    if final_score > best_forcing_score:
                        best_forcing_score = final_score
                        best_forcing_bonus = tactical_bonus
                        best_forcing_action = candidate['action']

                baseline_best_score = -math.inf
                for candidate in scored_actions:
                    if candidate['total_score'] > baseline_best_score:
                        baseline_best_score = candidate['total_score']

                if best_forcing_action:
                    forcing_margin = best_forcing_score - baseline_best_score
                    is_capture = best_forcing_action['type'] == 'capture'
                    if is_capture:
                        if best_forcing_bonus >= 80 or forcing_margin >= 24:
                            return best_forcing_action
                    elif best_forcing_bonus >= 120 and forcing_margin >= 48:
                        return best_forcing_action

        return None

    def get_immediate_capture_override(self, game, evaluator, visible_capture_actions):
        if not visible_capture_actions:
            return None

        best_safe_action = None
        best_safe_score = -math.inf
        best_trade_action = None
        best_trade_score = -math.inf

        for action in visible_capture_actions:
            verdict = self.evaluate_immediate_capture_override(game, evaluator, action)
            if not verdict:
                continue

            if verdict['is_safe'] and verdict['score'] > best_safe_score:
                best_safe_score = verdict['score']
                best_safe_action = action

            if verdict['is_compelling_trade'] and verdict['score'] > best_trade_score:
                best_trade_score = verdict['score']
                best_trade_action = action

        if best_safe_action:
            return best_safe_action
        if best_trade_action:
            return best_trade_action
        return None

    def evaluate_immediate_capture_override(self, game, evaluator, action):
        attacker = game.get_piece(action.get('from_row'), action.get('from_col'))
        target = game.get_piece(action.get('to_row'), action.get('to_col'))
        if not attacker or not target or not target['face_up']:
            return None

        is_safe = not evaluator.would_be_in_danger_after_move(game, action, attacker)
        attacker_threatened = evaluator.is_in_danger(game, action.get('from_row'), action.get('from_col'), attacker)
        mutual_capture = evaluator.is_mutual_capture_pair(attacker, target)
        priority_target = evaluator.is_v12_priority_target(target)
        attacker_value = evaluator.material_value(attacker['type'])
        target_value = evaluator.material_value(target['type'])
        trade_margin = target['hp'] * 9 + target_value * 0.7

        if not is_safe:
            trade_margin -= attacker['hp'] * 9 + attacker_value * 0.7
        if attacker_threatened:
            trade_margin += attacker['hp'] * 4 + attacker_value * 0.12

        score = evaluator.evaluate_capture(game, action) + evaluator.evaluate_opponent_response(game, action)
        if is_safe:
            score += 60
        if attacker_threatened:
            score += attacker['hp'] * 14 + attacker_value * 0.22
        if mutual_capture:
            score += 24
        if priority_target:
            score += 18
        if target['hp'] >= attacker['hp']:
            score += 12
        score += trade_margin * 0.35

        return {
            'is_safe': is_safe,
            'is_compelling_trade': bool(attacker_threatened) and bool(
                is_safe or
                priority_target or
                mutual_capture or
                trade_margin >= -12 or
                target['hp'] >= attacker['hp']
            ),
            'score': score,
        }

    def choose_setup_piece(self, game, side):
        evaluator = self.get_side_evaluator(side)
        best_index = -1
        best_score = -math.inf

        for index in range(len(game.remaining_pool)):
            piece = game.remaining_pool[index]
            if piece['side'] != side:
                continue

            score = piece['hp'] * 4 + evaluator.material_value(piece['type']) * 0.6
            if piece['type'] == '将':
                score += 24
            if piece['type'] == '仕':
                score += 16
            if piece['type'] == '炮':
                score += 10
            if piece['type'] == '兵':
                score -= 8

            if score > best_score:
                best_score = score
                best_index = index

        return best_index

    def choose_search_depth(self, game, action_count):
        hidden_cells = self.count_hidden_board_cells(game)
        side = game.current_turn
        enemy = self.other_side(side)
        enemy_evaluator = self.get_side_evaluator(enemy)
        enemy_piece_count = len(enemy_evaluator.get_face_up_pieces(game, enemy, None))

        # Endgame with few options: deeper search
        if len(game.remaining_pool) <= 4 and hidden_cells <= 3 and action_count <= 4:
            return 3
        if action_count <= 8 and self.has_visible_mutual_capture(game, game.current_turn):
            return 3
        # Endgame chase: enemy has few pieces, search deeper to find the kill
        if enemy_piece_count <= 2 and hidden_cells <= 4 and action_count <= 10:
            return 3
        # Early game with many flips: only depth 1 to stay fast
        if hidden_cells >= 12 or action_count >= 20:
            return 1
        return 2

    def search_after_action(self, game, action, depth, alpha, beta, ply):
        snapshot = game.create_snapshot()
        history = list(game.history)

        try:
            if not self.apply_detailed_action(game, action):
                return -999999 + ply
            extension = self.get_search_extension(game, action, depth, ply)
            return -self.negamax(game, depth + extension, -beta, -alpha, ply)
        finally:
            game.restore_snapshot(snapshot)
            game.history = history

    def negamax(self, game, depth, alpha, beta, ply):
        if game.phase == 'over':
            return self.evaluate_terminal(game, ply)

        if depth <= 0:
            return self.quiescence(game, alpha, beta, ply, self.quiescence_depth)

        key = self.get_transposition_key(game)
        entry = self.transposition_table.get(key)
        if entry and entry['depth'] >= depth:
            if entry['flag'] == 'exact':
                return entry['value']
            if entry['flag'] == 'lower':
                alpha = max(alpha, entry['value'])
            if entry['flag'] == 'upper':
                beta = min(beta, entry['value'])
            if alpha >= beta:
                return entry['value']

        actions = self.get_detailed_actions(game, game.current_turn)
        if not actions:
            return self.evaluate_position(game, game.current_turn)

        ordered_actions = self.order_actions(
            game,
            actions,
            game.current_turn,
            ply,
            entry['best_action_key'] if entry else None,
            self.max_node_actions,
        )
        original_alpha = alpha
        best_value = -math.inf
        best_action_key = None

        for action in ordered_actions:
            value = self.search_after_action(game, action, depth - 1, -beta, -alpha, ply + 1)
            if value > best_value:
                best_value = value
                best_action_key = self.action_key(action)
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break

        self.store_transposition(key, depth, best_value, original_alpha, beta, best_action_key)
        return best_value

    def quiescence(self, game, alpha, beta, ply, depth_left):
        stand_pat = self.evaluate_position(game, game.current_turn)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat
        if depth_left <= 0:
            return stand_pat

        actions = self.get_noisy_actions(game, game.current_turn)
        if not actions:
            return stand_pat

        ordered_actions = self.order_actions(game, actions, game.current_turn, ply, None, self.max_noisy_actions)
        for action in ordered_actions:
            value = self.search_after_action(game, action, 0, -beta, -alpha, ply + 1)
            if value >= beta:
                return value
            if value > alpha:
                alpha = value

        return alpha

    def evaluate_terminal(self, game, ply):
        if game.phase != 'over' or not game.winner:
            return self.evaluate_position(game, game.current_turn)

        return 200000 - ply * 100 if game.winner == game.current_turn else -200000 + ply * 100

    def get_detailed_actions(self, game, side=None):
        if side is None:
            side = game.current_turn
        evaluator = self.get_side_evaluator(side)
        actions = self.get_actions_for_side(game, side)
        visible_actions = []
        flip_actions = []
        hidden_capture_actions = []

        for action in actions:
            if action['type'] == 'flip':
                flip_actions.append(action)
                continue

            target = game.get_piece(action.get('to_row'), action.get('to_col'))
            if action['type'] == 'capture' and target and not target['face_up']:
                hidden_capture_actions.append(action)
                continue

            visible_actions.append({**action})

        detailed_actions = [*visible_actions]
        ordered_flips = [{'action': action, 'score': evaluator.evaluate_flip(game, action)} for action in flip_actions]
        ordered_flips.sort(key=lambda e: e['score'], reverse=True)
        ordered_flips = ordered_flips[0:min(self.max_flip_squares, max(2, len(flip_actions)))]

        for entry in ordered_flips:
            for pool_index in self.pick_pool_candidates(game, entry['action'], side, self.max_pool_choices):
                detailed_actions.append({**entry['action'], 'pool_index': pool_index})

        for action in hidden_capture_actions:
            for pool_index in self.pick_pool_candidates(game, action, side, self.max_pool_choices):
                detailed_actions.append({**action, 'pool_index': pool_index})

        return detailed_actions

    def get_noisy_actions(self, game, side=None):
        if side is None:
            side = game.current_turn
        actions = self.get_actions_for_side(game, side)
        noisy_actions = []

        for action in actions:
            if action['type'] != 'capture':
                continue

            target = game.get_piece(action.get('to_row'), action.get('to_col'))
            if target and target['face_up']:
                noisy_actions.append({**action})
                continue

            if not target or target['face_up']:
                continue

            for pool_index in self.pick_pool_candidates(game, action, side, 2):
                piece = game.remaining_pool[pool_index]
                if piece and piece['side'] != side and piece['hp'] >= 5:
                    noisy_actions.append({**action, 'pool_index': pool_index})

        return noisy_actions

    def get_search_extension(self, game, action, depth, ply):
        if game.phase == 'over' or depth > 1 or ply > self.mutual_capture_max_ply:
            return 0

        # Only extend for the action that just happened, not broadly
        if self.is_visible_mutual_capture_action(game, action):
            return self.mutual_capture_search_extension

        return 0

    def is_visible_mutual_capture_action(self, game, action):
        if not action or action['type'] != 'capture':
            return False

        attacker = game.get_piece(action.get('from_row'), action.get('from_col'))
        target = game.get_piece(action.get('to_row'), action.get('to_col'))
        if not attacker or not target or not target['face_up']:
            return False

        return self.get_side_evaluator(attacker['side']).is_mutual_capture_pair(attacker, target)

    def has_visible_mutual_capture(self, game, side):
        evaluator = self.get_side_evaluator(side)
        actions = self.get_actions_for_side(game, side)

        for action in actions:
            if action['type'] != 'capture':
                continue

            attacker = game.get_piece(action.get('from_row'), action.get('from_col'))
            target = game.get_piece(action.get('to_row'), action.get('to_col'))
            if not attacker or not target or not target['face_up']:
                continue
            if evaluator.is_mutual_capture_pair(attacker, target):
                return True

        return False

    def pick_pool_candidates(self, game, action, side, max_choices):
        unique_pool_indices = self.get_unique_pool_indices(game.remaining_pool)
        if len(unique_pool_indices) <= max_choices:
            return unique_pool_indices

        scored = [
            {
                'pool_index': pool_index,
                'score': self.score_pool_choice_for_action(game, action, side, game.remaining_pool[pool_index]),
            }
            for pool_index in unique_pool_indices
        ]
        scored.sort(key=lambda e: e['score'], reverse=True)
        return [entry['pool_index'] for entry in scored[0:max_choices]]

    def get_unique_pool_indices(self, pool):
        signature_to_index = {}
        for index in range(len(pool)):
            piece = pool[index]
            signature = f"{piece['side']}:{piece['type']}"
            if signature not in signature_to_index:
                signature_to_index[signature] = index
        return [*signature_to_index.values()]

    def score_pool_choice_for_action(self, game, action, side, piece):
        if not piece:
            return -math.inf

        if action['type'] == 'flip':
            return self.score_flip_reveal_choice(game, action.get('row'), action.get('col'), side, piece)

        if action['type'] == 'capture':
            return self.score_blind_capture_choice(game, action, side, piece)

        return 0

    def score_flip_reveal_choice(self, game, row, col, side, piece):
        evaluator = self.get_side_evaluator(side)
        enemy = self.other_side(side)
        enemy_neighbors = evaluator.count_enemy_neighbors(game, row, col)
        support = evaluator.score_support(row, col, game, side)
        enemy_cannons = evaluator.count_cannon_attackers(game, row, col, enemy)
        score = piece['hp'] * 6 + evaluator.material_value(piece['type']) * 1.6

        if piece['side'] == side:
            score += 22
            score += support * 5
            score -= enemy_neighbors * (20 if piece['type'] == '将' else 7)
            score -= enemy_cannons * (26 if piece['type'] == '将' else 9)
            if piece['type'] == '将':
                score -= evaluator.count_adjacent_enemy_soldiers(game, row, col, side) * 55
            if piece['type'] == '炮':
                score += evaluator.score_cannon_mouth(game, row, col) * 0.35
        else:
            score = -score
            score -= enemy_neighbors * 3
            score -= enemy_cannons * 4
            score -= support * 2
            if piece['type'] == '将':
                score -= 32

        return score

    def score_blind_capture_choice(self, game, action, side, piece):
        evaluator = self.get_side_evaluator(side)
        attacker = game.get_piece(action.get('from_row'), action.get('from_col'))
        if not attacker or not piece:
            return -math.inf

        score = piece['hp'] * 14 + evaluator.material_value(piece['type']) * 2.8
        if piece['side'] == side:
            score = -score
            if game.hp[side] <= piece['hp'] + 4:
                score -= 150
        else:
            score += 24
            if evaluator.is_mutual_capture_pair(attacker, piece):
                score += piece['hp'] * 6 + evaluator.material_value(piece['type']) * 0.35 + 10
            if attacker['type'] == '兵' and piece['type'] == '将':
                score += 50
            if piece['type'] == '将':
                score += 30
            if piece['type'] == '仕':
                score += 12

        return score

    def order_actions(self, game, actions, side, ply, preferred_action_key=None, limit=None):
        evaluator = self.get_side_evaluator(side)
        is_root = ply == 0
        scored = [
            {
                'action': action,
                'score': (
                    self.score_action_ordering(game, action, evaluator, side, ply, preferred_action_key)
                    if is_root
                    else self.score_action_ordering_fast(game, action, evaluator, side, preferred_action_key)
                ),
            }
            for action in actions
        ]
        scored.sort(key=lambda e: e['score'], reverse=True)
        ordered = [entry['action'] for entry in scored]

        return ordered[0:limit] if (isinstance(limit, int) and not isinstance(limit, bool)) and limit > 0 else ordered

    # Fast ordering for inner nodes - avoids expensive V13 evaluate chain
    def score_action_ordering_fast(self, game, action, evaluator, side, preferred_action_key):
        if preferred_action_key:
            action_key = self.action_key(action)
            if action_key == preferred_action_key:
                return 1_000_000

        pool_index = action.get('pool_index')
        pool_index_is_int = isinstance(pool_index, int) and not isinstance(pool_index, bool)

        if action['type'] == 'flip':
            if pool_index_is_int:
                return self.score_flip_reveal_choice(
                    game, action.get('row'), action.get('col'), side, game.remaining_pool[pool_index])
            return -100

        piece = game.get_piece(action.get('from_row'), action.get('from_col'))
        target = game.get_piece(action.get('to_row'), action.get('to_col'))
        score = 0

        if action['type'] == 'capture':
            score += 300
            if target and target['face_up']:
                score += target['hp'] * 25 + evaluator.material_value(target['type']) * 3
                if piece and evaluator.is_mutual_capture_pair(piece, target):
                    score += target['hp'] * 18 + 20
            elif pool_index_is_int:
                score += self.score_blind_capture_choice(game, action, side, game.remaining_pool[pool_index])

        if piece and evaluator.is_in_danger(game, action.get('from_row'), action.get('from_col'), piece):
            score += piece['hp'] * 12

        return score

    def score_action_ordering(self, game, action, evaluator, side, ply, preferred_action_key):
        action_key = self.action_key(action)
        if preferred_action_key and action_key == preferred_action_key:
            return 1_000_000

        pool_index = action.get('pool_index')
        pool_index_is_int = isinstance(pool_index, int) and not isinstance(pool_index, bool)

        score = 0
        if action['type'] == 'flip':
            score += evaluator.evaluate_flip(game, action) * 2
            if pool_index_is_int:
                piece = game.remaining_pool[pool_index]
                score += self.score_flip_reveal_choice(game, action.get('row'), action.get('col'), side, piece)
            score -= ply
            return score

        piece = game.get_piece(action.get('from_row'), action.get('from_col'))
        target = game.get_piece(action.get('to_row'), action.get('to_col'))
        if action['type'] == 'capture':
            score += 300
            if target and target['face_up']:
                score += evaluator.evaluate_capture(game, action) * 2
                score += target['hp'] * 25 + evaluator.material_value(target['type']) * 3
                if piece and evaluator.is_mutual_capture_pair(piece, target):
                    score += target['hp'] * 18 + evaluator.material_value(target['type']) * 1.2 + 20
            elif pool_index_is_int:
                revealed = game.remaining_pool[pool_index]
                score += self.score_blind_capture_choice(game, action, side, revealed)
        else:
            score += evaluator.evaluate_move(game, action)

        if piece and evaluator.is_in_danger(game, action.get('from_row'), action.get('from_col'), piece):
            score += piece['hp'] * 12
        if piece and not evaluator.would_be_in_danger_after_move(game, action, piece):
            score += 14

        return score - ply

    def apply_detailed_action(self, game, action):
        pool_index = action.get('pool_index')
        pool_index_is_int = isinstance(pool_index, int) and not isinstance(pool_index, bool)

        if action['type'] == 'flip':
            return (game.flip_piece(action.get('row'), action.get('col'), pool_index)
                    if pool_index_is_int
                    else False)

        if action['type'] == 'move':
            return game.move_piece(action.get('from_row'), action.get('from_col'), action.get('to_row'), action.get('to_col'))

        target = game.get_piece(action.get('to_row'), action.get('to_col'))
        if target and not target['face_up']:
            return (game.move_piece(action.get('from_row'), action.get('from_col'),
                                    action.get('to_row'), action.get('to_col'), pool_index)
                    if pool_index_is_int
                    else False)

        return game.move_piece(action.get('from_row'), action.get('from_col'), action.get('to_row'), action.get('to_col'))

    def evaluate_position(self, game, side=None):
        if side is None:
            side = game.current_turn
        enemy = self.other_side(side)
        tempo = 8 if game.current_turn == side else 0
        return self.evaluate_side_position(game, side) - self.evaluate_side_position(game, enemy) + tempo

    # Lightweight position evaluation for search leaf nodes.
    # Avoids expensive scoreV6SpatialControl and getV12OwnKingEscapeCount.
    def evaluate_side_position(self, game, side):
        evaluator = self.get_side_evaluator(side)
        enemy = self.other_side(side)
        face_up_pieces = evaluator.get_face_up_pieces(game, side, None)
        enemy_pieces = evaluator.get_face_up_pieces(game, enemy, None)
        enemy_king = next((p for p in enemy_pieces if p['type'] == '将'), None)
        score = game.hp[side] * 16

        for piece in face_up_pieces:
            piece_score = piece['hp'] * 8 + evaluator.material_value(piece['type']) * 2.4
            piece_score += evaluator.estimate_mobility_at(piece['row'], piece['col'], piece, game) * (4 if piece['type'] == '炮' else 3)

            if enemy_king:
                dist_to_enemy_king = evaluator.manhattan_distance(piece['row'], piece['col'], enemy_king['row'], enemy_king['col'])
                if piece['type'] == '兵':
                    # Soldier proximity to enemy king is critical for closing out games
                    if dist_to_enemy_king <= 1:
                        piece_score += 55
                    elif dist_to_enemy_king <= 2:
                        piece_score += 30
                    else:
                        piece_score += max(0, 6 - dist_to_enemy_king) * 6
                else:
                    # Other pieces help box in the king
                    piece_score += max(0, 5 - min(5, dist_to_enemy_king)) * 2

            if evaluator.is_in_danger(game, piece['row'], piece['col'], piece):
                piece_score -= piece['hp'] * 7 + evaluator.material_value(piece['type']) * 0.4
                if piece['type'] == '将':
                    piece_score -= 60

            # V13-style safe pressure: bonus for threatening adjacent enemies
            for dr, dc in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                nr = piece['row'] + dr
                nc = piece['col'] + dc
                if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
                    continue
                adj = game.get_piece(nr, nc)
                if adj and adj['face_up'] and adj['side'] == enemy and evaluator.can_capture_like(piece, adj):
                    piece_score += adj['hp'] * 2 + 4

            score += piece_score

        # King danger (fast check only)
        king_danger = evaluator.get_v12_own_king_danger(game, side)
        score -= king_danger * 0.38

        # Enemy king trap (only if visible) - scale up in endgame
        if enemy_king:
            trap_score = evaluator.evaluate_v6_enemy_king_trap(game, enemy)
            pieces_total = len(face_up_pieces) + len(enemy_pieces)
            endgame_scale = 0.8 if pieces_total <= 6 else (0.6 if pieces_total <= 10 else 0.45)
            score += trap_score * endgame_scale

        # Pool leverage (simplified)
        score += self.score_pool_leverage(game, side)

        # Hidden own king penalty
        if evaluator.count_living_pieces(game, side, '将') > 0 and evaluator.count_face_up_pieces(game, side, '将') == 0:
            score -= 28

        return score

    def score_pool_leverage(self, game, side):
        evaluator = self.get_side_evaluator(side)
        enemy = self.other_side(side)

        own_pool = 0
        enemy_pool = 0

        for piece in game.remaining_pool:
            value = piece['hp'] * 3 + evaluator.material_value(piece['type']) * 0.8
            if piece['side'] == side:
                own_pool += value
            elif piece['side'] == enemy:
                enemy_pool += value

        return (own_pool - enemy_pool) * 0.22

    def count_hidden_board_cells(self, game):
        count = 0
        for row in range(ROWS):
            for col in range(COLS):
                piece = game.get_piece(row, col)
                if piece and not piece['face_up']:
                    count += 1
        return count

    def get_actions_for_side(self, game, side):
        actions = []

        def _collect():
            nonlocal actions
            actions = [{**action} for action in game.get_all_actions()]

        self.with_turn(game, side, _collect)
        return actions

    def with_turn(self, game, side, fn):
        original_turn = game.current_turn
        try:
            game.current_turn = side
            return fn()
        finally:
            game.current_turn = original_turn

    def get_side_evaluator(self, side):
        if side not in self.evaluator_cache:
            self.evaluator_cache[side] = StrategicV13AI(side)
        return self.evaluator_cache[side]

    def other_side(self, side):
        return 'black' if side == 'red' else 'red'

    def action_key(self, action):
        pool_index = action.get('pool_index')
        parts = [
            action.get('type'),
            action.get('row'),
            action.get('col'),
            action.get('from_row'),
            action.get('from_col'),
            action.get('to_row'),
            action.get('to_col'),
            pool_index if (isinstance(pool_index, int) and not isinstance(pool_index, bool)) else '',
        ]
        return ':'.join('' if v is None else str(v) for v in parts)

    def get_transposition_key(self, game):
        board_key = []
        for row in range(ROWS):
            for col in range(COLS):
                piece = game.get_piece(row, col)
                if not piece:
                    board_key.append('.')
                elif not piece['face_up']:
                    board_key.append('?')
                else:
                    board_key.append(f"{piece['side'][0]}{piece['type']}")

        pool_summary = {}
        for piece in game.remaining_pool:
            key = f"{piece['side'][0]}{piece['type']}"
            pool_summary[key] = (pool_summary.get(key) or 0) + 1
        pool_key = ','.join(
            f"{key}{count}" for key, count in sorted(pool_summary.items(), key=lambda kv: kv[0])
        )

        return '|'.join(str(v) for v in [
            game.current_turn,
            game.phase,
            game.hp['red'],
            game.hp['black'],
            ''.join(board_key),
            pool_key,
        ])

    def store_transposition(self, key, depth, value, original_alpha, beta, best_action_key):
        flag = 'exact'
        if value <= original_alpha:
            flag = 'upper'
        elif value >= beta:
            flag = 'lower'

        if len(self.transposition_table) >= self.tt_max_size:
            first_key = next(iter(self.transposition_table))
            del self.transposition_table[first_key]

        self.transposition_table[key] = {
            'depth': depth,
            'value': value,
            'flag': flag,
            'best_action_key': best_action_key,
        }


# ===== StrategicV15AI =====
# 在 V14 的 alpha-beta 搜索之上引入"迭代加深 + 时间预算 + PV/历史启发"，
# 在不依赖显卡训练的前提下显著提升实战搜索深度与战术准确度：
# 1. 迭代加深：从浅到深逐层搜索，随时可在时间预算耗尽时返回当前最优着法。
# 2. 时间预算：默认每步约 900ms（测试/自对弈可调小），低分支局面可搜到很深。
# 3. PV 优先 + 置换表：上一层的最优着法优先搜索，提升 alpha-beta 剪枝效率。
# 4. 更宽的根/节点分支上限：在时间允许时考虑更多候选着法，减少漏算。
# 5. 复用 V14 的全部局面评估、战术覆盖与着法生成，保证行为稳健。
class StrategicV15AI(StrategicV14AI):
    def __init__(self, side):
        super().__init__(side)
        # 时间预算（毫秒）。浏览器中默认 900ms；Node 自对弈/测试可在外部调小。
        self.time_budget_ms = 900
        self.max_iterative_depth = 7
        # 比 V14 更宽的搜索分支，配合迭代加深与剪枝
        self.max_root_actions = 16
        self.max_node_actions = 8
        self.max_noisy_actions = 6
        self.max_pool_choices = 2
        self.max_flip_squares = 4
        self.quiescence_depth = 2
        # 时间检测节流：每搜索若干节点检查一次时钟
        self.time_check_interval = 1024
        self.node_counter = 0
        self.search_deadline = math.inf
        self.timed_out = False

    def now_ms(self):
        return time.perf_counter() * 1000

    def get_move(self, game):
        root_actions = self.get_detailed_actions(game, game.current_turn)
        if not root_actions:
            return None

        # 紧急战术（将防御、兵杀防御、强制吃子）优先，与 V14 一致
        tactical_action = self.get_tactical_override(game)
        if tactical_action:
            # [Python 版] 记录决策依据，供提示界面展示
            self.last_decision_info = {'source': 'tactical'}
            return tactical_action

        self.transposition_table.clear()
        self.node_counter = 0
        self.timed_out = False
        self.search_deadline = self.now_ms() + max(1, self.time_budget_ms)

        max_depth = self.choose_max_depth(game, len(root_actions))
        ordered = self.order_actions(game, root_actions, game.current_turn, 0, None, self.max_root_actions)
        best_action = ordered[0]
        best_score = -math.inf
        completed_depth = 0

        for depth in range(1, max_depth + 1):
            pv_key = self.action_key(best_action) if best_action else None
            iteration_actions = self.order_root_with_pv(ordered, pv_key)

            alpha = -math.inf
            beta = math.inf
            iteration_best = None
            iteration_best_score = -math.inf
            aborted = False

            for action in iteration_actions:
                score = self.search_after_action(game, action, depth - 1, -beta, -alpha, 1)
                if self.timed_out:
                    aborted = True
                    break
                if score > iteration_best_score:
                    iteration_best_score = score
                    iteration_best = action
                if score > alpha:
                    alpha = score

            if iteration_best and not aborted:
                best_action = iteration_best
                best_score = iteration_best_score
                ordered = self.order_root_with_pv(ordered, self.action_key(best_action))
                completed_depth = depth

            if aborted:
                break
            if self.now_ms() >= self.search_deadline:
                break
            # 已找到必胜/必败，无需继续加深
            if best_score >= 150000 or best_score <= -150000:
                break

        self.last_decision_info = {
            'source': 'search',
            'depth': completed_depth,
            'score': best_score if completed_depth else None,
        }
        return best_action or ordered[0]

    # 在置换表辅助下，将主变着法放到搜索序列最前
    def order_root_with_pv(self, actions, pv_key):
        if not pv_key:
            return list(actions)
        pv = []
        rest = []
        for action in actions:
            if self.action_key(action) == pv_key:
                pv.append(action)
            else:
                rest.append(action)
        return [*pv, *rest]

    # 根据局面复杂度选择迭代加深上限：低分支局面（残局/对杀）允许搜很深。
    def choose_max_depth(self, game, action_count):
        hidden = self.count_hidden_board_cells(game)
        if hidden >= 12 or action_count >= 24:
            return 2
        if hidden >= 8:
            return 3
        if action_count <= 8:
            return self.max_iterative_depth
        if action_count <= 12:
            return 5
        return 4

    # 覆盖 negamax，加入时间预算检测（节流），超时即返回当前评估并标记超时。
    def negamax(self, game, depth, alpha, beta, ply):
        if self.timed_out:
            return self.evaluate_position(game, game.current_turn)
        self.node_counter += 1
        if (self.node_counter % self.time_check_interval) == 0 and self.now_ms() >= self.search_deadline:
            self.timed_out = True
            return self.evaluate_position(game, game.current_turn)
        return super().negamax(game, depth, alpha, beta, ply)
