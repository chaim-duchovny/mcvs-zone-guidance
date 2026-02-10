import numpy as np
import time
import random
import math
from datetime import datetime
from typing import Dict, Tuple, Optional

from draughts_mcvs import (
    Draughts,
    MCTSNode,
    UCTSearcher,
    HilbertOrderedZoneDatabase,
    ABCModelDynamic,
    WeightedMatrixABC
)

# ============================================================================
# ZONE-GUIDED PUCT SEARCHER (no neural nets, zone bonus on rollouts)
# ============================================================================

class ZoneMCVSSearcher:
    def __init__(self, zone_db, cpuct=1.41, lambda_zone=1.0, k_zone=5):
        self.zone_db = zone_db
        self.cpuct = cpuct
        self.lambda_zone = lambda_zone
        self.k_zone = k_zone

    def _rollout(self, game: Draughts) -> float:
        current = game.copy()
        while not current.is_terminal():
            moves = current.get_legal_moves()
            if not moves:
                break
            current.apply_move(random.choice(moves))
        winner = current.check_winner()
        if winner is None:
            return 0.0
        # Value from perspective of the player who started the rollout (current player at leaf)
        return 1.0 if winner == current.current_player() else -1.0

    def _zone_bonus(self, W: np.ndarray) -> float:
        if len(self.zone_db.winning_matrices) == 0:
            return 0.0
        
        # Dummy ABC model - parameters match those used in game matrix computation
        dummy_abc = ABCModelDynamic(n=2, t=1.0, T=1.41)
        
        current = WeightedMatrixABC(dummy_abc)
        current.W = W
        
        min_dist = float('inf')
        for win_W in self.zone_db.winning_matrices[-self.k_zone:]:
            comparator = WeightedMatrixABC(dummy_abc)
            comparator.W = win_W
            dist = current.compute_manhattan_distance(comparator)
            min_dist = min(min_dist, dist)
        
        scale = 5000.0
        normalized_similarity = 1.0 - (min_dist / scale)
        bonus = max(0.0, normalized_similarity)
        
        return bonus

    def search_with_time_budget(self, game: Draughts, time_budget: float, add_noise: bool = True) -> Dict[Tuple, int]:
        root = MCTSNode(game.copy())

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return {}

        num_moves = len(legal_moves)

        # Uniform priors + Dirichlet noise at root
        priors = {m: 1.0 / num_moves for m in legal_moves}
        if add_noise and num_moves > 1:
            alpha = 0.03
            noise = np.random.dirichlet([alpha] * num_moves)
            legal_list = list(legal_moves)
            for i, m in enumerate(legal_list):
                priors[m] = 0.75 * priors[m] + 0.25 * noise[i]
            total = sum(priors.values())
            if total > 0:
                for m in priors:
                    priors[m] /= total

        # Root children
        for m in legal_moves:
            child_game = game.copy()
            child_game.apply_move(m)
            root.children[m] = MCTSNode(child_game, prior=priors[m], parent=root, move=m)

        start_time = time.time()
        while time.time() - start_time < time_budget:
            node = root

            # Selection (PUCT)
            while node.is_expanded() and not node.game.is_terminal():
                node = max(
                    node.children.values(),
                    key=lambda c: c.value() + self.cpuct * c.prior * math.sqrt(node.visit_count) / (1 + c.visit_count)
                )

            # Evaluation / Expansion
            if node.game.is_terminal():
                winner = node.game.check_winner()
                value = 0.0 if winner is None else (1.0 if winner == node.game.current_player() else -1.0)
            else:
                # Expand with uniform child priors
                child_legal = node.game.get_legal_moves()
                if child_legal:
                    child_num = len(child_legal)
                    child_priors = {m: 1.0 / child_num for m in child_legal}
                    for m in child_legal:
                        child_game = node.game.copy()
                        child_game.apply_move(m)
                        node.children[m] = MCTSNode(child_game, prior=child_priors[m], parent=node, move=m)

                # Rollout + zone bonus
                value = self._rollout(node.game)
                if self.lambda_zone > 0:
                    zone_bonus = self._zone_bonus(node.game.get_weighted_adjacency_matrix())
                    value += self.lambda_zone * zone_bonus

            # Backpropagation
            v = value
            current = node
            while current is not None:
                current.visit_count += 1
                current.value_sum += v
                v = -v
                current = current.parent

        return {move: child.visit_count for move, child in root.children.items()}

# ============================================================================
# TOURNAMENT IMPLEMENTATION
# ============================================================================

class DetailedMoveLogger:
    def __init__(self, filepath="draughts_zone_only_move_log.txt"):
        self.filepath = filepath
        self.file_handle = None
        
    def open(self):
        self.file_handle = open(self.filepath, 'w', encoding='utf-8')
        self._write_header()
    
    def _write_header(self):
        self.file_handle.write("=" * 100 + "\n")
        self.file_handle.write("DRAUGHTS ABLATION (ZONE ONLY) - MOVE LOG\n")
        self.file_handle.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file_handle.write("=" * 100 + "\n\n")
        self.file_handle.flush()
    
    def log_move(self, game_num, move_num, player_name, agent_type, 
                 simulations, time_taken, move):
        msg = (
            f"[Game {game_num:3d}] [Move {move_num:3d}] "
            f"Player: {player_name:4s} ({agent_type:4s}) | "
            f"Sims: {simulations:5d} | "
            f"Time: {time_taken:6.3f}s | "
            f"Move: {move}\n"
        )
        self.file_handle.write(msg)
        self.file_handle.flush()
    
    def log_game_summary(self, game_num, winner, total_moves, mcvs_stats, uct_stats):
        self.file_handle.write("\n" + "-" * 100 + "\n")
        self.file_handle.write(f"GAME {game_num} SUMMARY - Winner: {winner}\n")
        self.file_handle.write(f"Total moves: {total_moves}\n\n")
        
        self.file_handle.write(f"MCVS (Zone-Guided) Statistics:\n")
        self.file_handle.write(f"  Total simulations: {mcvs_stats['total_sims']:,}\n")
        self.file_handle.write(f"  Avg sims/move: {mcvs_stats['avg_sims']:.1f}\n")
        self.file_handle.write(f"  Total time: {mcvs_stats['total_time']:.2f}s\n")
        self.file_handle.write(f"  Avg time/move: {mcvs_stats['avg_time']:.3f}s\n\n")
        
        self.file_handle.write(f"UCT Statistics:\n")
        self.file_handle.write(f"  Total simulations: {uct_stats['total_sims']:,}\n")
        self.file_handle.write(f"  Avg sims/move: {uct_stats['avg_sims']:.1f}\n")
        self.file_handle.write(f"  Total time: {uct_stats['total_time']:.2f}s\n")
        self.file_handle.write(f"  Avg time/move: {uct_stats['avg_time']:.3f}s\n")
        self.file_handle.write("-" * 100 + "\n\n")
        self.file_handle.flush()
    
    def close(self):
        if self.file_handle:
            self.file_handle.write("\n" + "=" * 100 + "\n")
            self.file_handle.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file_handle.write("=" * 100 + "\n")
            self.file_handle.close()

class DraughtsZoneAblationTournament:
    def __init__(self, num_games=200, time_per_move=1.0, max_moves_per_game=200, output_file="draughts_zone_only_results.txt"):
        self.num_games = num_games
        self.time_per_move = time_per_move
        self.max_moves_per_game = max_moves_per_game
        self.output_file = output_file

        self.move_logger = DetailedMoveLogger()
        self.report_lines = []

        self.move_logger.open()

        self.zone_db = HilbertOrderedZoneDatabase("draughts_zone_db.npz")

        self.agent_mcvs = ZoneMCVSSearcher(self.zone_db, cpuct=1.41, lambda_zone=1.0, k_zone=5)
        self.agent_uct = UCTSearcher(cpuct=np.sqrt(2))

        self.results = {
            "MCVS": {"wins": 0, "as_white": 0, "as_black": 0, "moves": []},
            "UCT": {"wins": 0, "as_white": 0, "as_black": 0, "moves": []},
            "Draw": {"count": 0, "moves": []}
        }

    def log_line(self, text):
        print(text)
        self.report_lines.append(text)

    def play_game(self, agent_white, agent_black, game_idx):
        game = Draughts()
        moves_made = []
        moves = 0

        mcvs_stats = {"total_sims": 0, "total_time": 0.0, "moves": 0}
        uct_stats = {"total_sims": 0, "total_time": 0.0, "moves": 0}

        while not game.is_terminal() and moves < self.max_moves_per_game:
            moves += 1
            is_white = (moves % 2 == 1)

            agent = agent_white if is_white else agent_black
            agent_type = "MCVS" if agent is self.agent_mcvs else "UCT"

            start_time = time.time()
            visits = agent.search_with_time_budget(game.copy(), self.time_per_move)
            t_taken = time.time() - start_time

            sims = sum(visits.values()) if visits else 0

            if agent_type == "MCVS":
                mcvs_stats["total_sims"] += sims
                mcvs_stats["total_time"] += t_taken
                mcvs_stats["moves"] += 1
            else:
                uct_stats["total_sims"] += sims
                uct_stats["total_time"] += t_taken
                uct_stats["moves"] += 1

            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            if visits:
                chosen_move = max(visits, key=visits.get)
            else:
                chosen_move = random.choice(legal_moves)

            player_name = "P1" if is_white else "P2"
            self.move_logger.log_move(
                game_idx, moves, player_name, agent_type,
                sims, t_taken, str(chosen_move)
            )

            game.apply_move(chosen_move)
            moves_made.append(chosen_move)

        # Statistics
        if mcvs_stats["moves"] > 0:
            mcvs_stats["avg_sims"] = mcvs_stats["total_sims"] / mcvs_stats["moves"]
            mcvs_stats["avg_time"] = mcvs_stats["total_time"] / mcvs_stats["moves"]
        if uct_stats["moves"] > 0:
            uct_stats["avg_sims"] = uct_stats["total_sims"] / uct_stats["moves"]
            uct_stats["avg_time"] = uct_stats["total_time"] / uct_stats["moves"]

        # Winner determination (absolute player 1/2 + string for logging)
        explicit_winner: Optional[int] = game.check_winner()
        if explicit_winner is not None:
            winner_player = explicit_winner
            winner_str = "MCVS" if (explicit_winner == 1 and agent_white is self.agent_mcvs) or (explicit_winner == 2 and agent_black is self.agent_mcvs) else "UCT"
        elif moves >= self.max_moves_per_game:
            winner_player = None
            winner_str = "Draw"
        else:  # Stalemate (no legal moves)
            stalled_player = game.current_player()
            winner_player = 3 - stalled_player
            winner_str = "MCVS" if (winner_player == 1 and agent_white is self.agent_mcvs) or (winner_player == 2 and agent_black is self.agent_mcvs) else "UCT"

        self.move_logger.log_game_summary(game_idx, winner_str, moves, mcvs_stats, uct_stats)

        return winner_str, moves, moves_made, winner_player

    def add_zone_data(self, moves_made: list, winner_player: Optional[int]):
        if not moves_made:
            return

        W_states = []
        temp_game = Draughts()
        for move in moves_made:
            W_states.append(temp_game.get_weighted_adjacency_matrix().copy())
            temp_game.apply_move(move)

        # Sample positions (start, end, ~30% random middle)
        selected = [0]
        if len(W_states) > 1:
            selected.append(len(W_states) - 1)
            for i in range(1, len(W_states) - 1):
                if random.random() < 0.3:
                    selected.append(i)

        for i in selected:
            W = W_states[i]
            player_to_move = 1 if i % 2 == 0 else 2
            if winner_player is None:
                v = 0.0
            elif winner_player == player_to_move:
                v = 1.0
            else:
                v = -1.0

            if v > 0:
                self.zone_db.add_winning_matrix(W)
            elif v < 0:
                self.zone_db.add_losing_matrix(W)
            else:
                self.zone_db.add_draw_matrix(W)

    def run_tournament(self):
        start_time = time.time()
        self.log_line("Starting 200-game Draughts ablation tournament (Zone-Guided MCVS vs standard UCT)")

        for game_idx in range(1, self.num_games + 1):
            if game_idx % 2 == 1:
                winner_str, moves, moves_made, winner_player = self.play_game(self.agent_mcvs, self.agent_uct, game_idx)
                white_name = "MCVS"
            else:
                winner_str, moves, moves_made, winner_player = self.play_game(self.agent_uct, self.agent_mcvs, game_idx)
                white_name = "UCT"

            if winner_str == "Draw":
                self.results["Draw"]["count"] += 1
                self.results["Draw"]["moves"].append(moves)
            else:
                self.results[winner_str]["wins"] += 1
                self.results[winner_str]["moves"].append(moves)
                if winner_str == white_name:
                    self.results[winner_str]["as_white"] += 1
                else:
                    self.results[winner_str]["as_black"] += 1

            # Add zone data and save DB
            self.add_zone_data(moves_made, winner_player)
            self.zone_db.save()

            db_size = len(self.zone_db.winning_matrices) + len(self.zone_db.losing_matrices) + len(self.zone_db.draw_matrices)

            elapsed = time.time() - start_time
            remaining = elapsed / game_idx * (self.num_games - game_idx)
            self.log_line(
                f"Game {game_idx:3d}/{self.num_games} | {winner_str:10s} (moves={moves:3d}) | "
                f"MCVS wins: {self.results['MCVS']['wins']:3d} | Draws: {self.results['Draw']['count']:3d} | "
                f"Zone DB size: {db_size:,} | ETA: {remaining/60:.1f} min"
            )

        # Final summary
        self.log_line("\n" + "=" * 80)
        self.log_line("DRAUGHTS ABLATION TOURNAMENT SUMMARY (ZONE GUIDANCE ONLY - NO NEURAL NETS)")
        self.log_line("=" * 80)
        
        mcvs_wins = self.results["MCVS"]["wins"]
        uct_wins = self.results["UCT"]["wins"]
        draws = self.results["Draw"]["count"]
        
        self.log_line(f"\nMCVS (Zone-Guided PUCT): {mcvs_wins} wins")
        self.log_line(f"UCT (standard UCB): {uct_wins} wins")
        self.log_line(f"Draws: {draws}")
        self.log_line(f"Total: {self.num_games}")
        
        if mcvs_wins + uct_wins > 0:
            win_rate = (mcvs_wins / (mcvs_wins + uct_wins)) * 100
            self.log_line(f"\nMCVS Win Rate (excluding draws): {win_rate:.1f}%")
        
        if self.results["MCVS"]["moves"]:
            self.log_line(f"Avg moves (MCVS wins): {np.mean(self.results['MCVS']['moves']):.1f}")
        if self.results["UCT"]["moves"]:
            self.log_line(f"Avg moves (UCT wins): {np.mean(self.results['UCT']['moves']):.1f}")
        
        self.log_line(f"\nMCVS White: {self.results['MCVS']['as_white']}, Black: {self.results['MCVS']['as_black']}")
        self.log_line(f"UCT White: {self.results['UCT']['as_white']}, Black: {self.results['UCT']['as_black']}")
        
        total_db = len(self.zone_db.winning_matrices) + len(self.zone_db.losing_matrices) + len(self.zone_db.draw_matrices)
        self.log_line(f"\nFinal Zone Database: {total_db:,} positions")
        
        total_time = time.time() - start_time
        self.log_line("\n" + "=" * 80)
        self.log_line(f"Tournament Duration: {total_time/60:.1f} minutes")
        self.log_line(f"Avg time per game: {total_time/self.num_games:.1f}s")
        self.log_line("=" * 80 + "\n")

        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_lines))

        self.move_logger.close()

if __name__ == "__main__":
    tournament = DraughtsZoneAblationTournament(num_games=200, time_per_move=1.5, max_moves_per_game=150)
    tournament.run_tournament()
