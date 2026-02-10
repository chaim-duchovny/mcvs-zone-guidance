# filename: mcvs_vs_uct.py
"""
DRAUGHTS MCVS vs UCT TOURNAMENT - 200 GAMES WITH DETAILED LOGGING

Exact analogue of the Breakthrough tournament script, fully adapted for Draughts.
- Zone database saved AFTER EVERY GAME
- Detailed per-game training logging
- Progressive learning shown in console and logs
"""

import torch
import numpy as np
import os
import time
import threading
import queue
import random
from datetime import datetime
from draughts_mcvs import (
    Draughts,
    PolicyNetworkCNN,
    ValueNetworkCNN,
    HilbertOrderedZoneDatabase,
    MCVSSearcher,
    UCTSearcher,
    train_networks
)

class DetailedMoveLogger:
    """Logs detailed move generation statistics to move_log.txt"""
    
    def __init__(self, filepath="draughts_move_log.txt"):
        self.filepath = filepath
        self.file_handle = None
        
    def open(self):
        self.file_handle = open(self.filepath, 'w', encoding='utf-8')
        self._write_header()
    
    def _write_header(self):
        self.file_handle.write("=" * 100 + "\n")
        self.file_handle.write("DRAUGHTS TOURNAMENT - MOVE GENERATION LOG\n")
        self.file_handle.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file_handle.write("=" * 100 + "\n\n")
        self.file_handle.flush()
    
    def log_move(self, game_num, move_num, player_name, agent_type, 
                 simulations, time_taken, move, board_state=""):
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
        
        self.file_handle.write(f"MCVS Statistics:\n")
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

class LearningLogger:
    """Logs training phase timing and statistics to learning_log.txt"""
    
    def __init__(self, filepath="draughts_learning_log.txt"):
        self.filepath = filepath
        self.file_handle = None
        
    def open(self):
        self.file_handle = open(self.filepath, 'w', encoding='utf-8')
        self._write_header()
    
    def _write_header(self):
        self.file_handle.write("=" * 100 + "\n")
        self.file_handle.write("DRAUGHTS TOURNAMENT - LEARNING PHASE LOG\n")
        self.file_handle.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file_handle.write("=" * 100 + "\n\n")
        self.file_handle.flush()
    
    def log_training_queued(self, game_num, num_examples):
        msg = f"→ Queued training data from game {game_num} ({num_examples} examples)\n"
        self.file_handle.write(msg)
        self.file_handle.flush()

    def log_training_completed(self, game_num, num_examples, time_taken):
        msg = f"✓ Completed training on game {game_num} data ({num_examples} examples) in {time_taken:.2f}s\n"
        self.file_handle.write(msg)
        self.file_handle.flush()
    
    def log_initial_training(self, game_num, num_examples, time_taken, epochs):
        msg = (
            f"\n{'=' * 100}\n"
            f"INITIAL TRAINING (Game {game_num})\n"
            f"{'=' * 100}\n"
            f"Training examples: {num_examples}\n"
            f"Epochs: {epochs}\n"
            f"Training time: {time_taken:.2f}s\n"
            f"Zone guidance enabled: λ = 1.0\n"
            f"{'=' * 100}\n\n"
        )
        self.file_handle.write(msg)
        self.file_handle.flush()
    
    def log_tournament_summary(self, total_games_learned, total_training_time, avg_time_per_game):
        msg = (
            f"\n{'=' * 100}\n"
            f"FINAL TRAINING SUMMARY\n"
            f"{'=' * 100}\n"
            f"Games learned from: {total_games_learned}\n"
            f"Total training time: {total_training_time/60:.1f} minutes\n"
            f"Avg time per game: {avg_time_per_game:.1f}s\n"
            f"{'=' * 100}\n"
        )
        self.file_handle.write(msg)
        self.file_handle.flush()
    
    def close(self):
        if self.file_handle:
            self.file_handle.close()

class TrainingManager:
    def __init__(self, policy_net, value_net, device, learning_logger):
        self.policy_net = policy_net
        self.value_net = value_net
        self.device = device
        self.learning_logger = learning_logger
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.stop_event = threading.Event()
        self.total_training_time = 0.0
        self.games_processed = 0
        self.thread.start()

    def queue_training(self, W_states, policies, values, game_num):
        self.queue.put((W_states, policies, values, game_num))
        self.learning_logger.log_training_queued(game_num, len(W_states))

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                W_states, policies, values, game_num = self.queue.get(timeout=1.0)
                start = time.time()
                train_networks(self.policy_net, self.value_net, W_states, policies, values, epochs=10, device=self.device)
                elapsed = time.time() - start
                self.total_training_time += elapsed
                self.games_processed += 1
                self.learning_logger.log_training_completed(game_num, len(W_states), elapsed)
                self.queue.task_done()
            except queue.Empty:
                pass

    def stop(self):
        self.stop_event.set()
        self.thread.join()

class DraughtsTournament:
    def __init__(self, num_games=200, time_per_move=3.0, max_moves_per_game=400, device='cpu', output_file="draughts_tournament_200_results.txt"):
        self.num_games = num_games
        self.time_per_move = time_per_move
        self.max_moves_per_game = max_moves_per_game
        self.device = device
        self.output_file = output_file

        self.move_logger = DetailedMoveLogger()
        self.learning_logger = LearningLogger()
        self.report_lines = []

        self.move_logger.open()
        self.learning_logger.open()

        self.zone_db = HilbertOrderedZoneDatabase("draughts_zone_db.npz")
        self.policy_net = PolicyNetworkCNN().to(device)
        self.value_net = ValueNetworkCNN().to(device)

        self.agent_mcvs = MCVSSearcher(self.policy_net, self.value_net, self.zone_db, device=device, lambda_zone=0.0)
        self.agent_uct = UCTSearcher()

        self.training_manager = None
        self.networks_initialized = False

        self.results = {
            "MCVS": {"wins": 0, "as_white": 0, "as_black": 0, "moves": []},
            "UCT": {"wins": 0, "as_white": 0, "as_black": 0, "moves": []},
            "Draw": {"count": 0, "moves": []}
        }

    def log_line(self, text):
        print(text)
        self.report_lines.append(text)

    def extract_training_and_zone_data(self, moves_made, pi_records, outcome):
        game = Draughts()
        W_states = []
        policies = []
        values = []

        for step, move in enumerate(moves_made):
            W = game.get_weighted_adjacency_matrix().copy()
            W_states.append(W)

            pi = np.zeros(4096, dtype=np.float32)

            visits = pi_records[step]

            legal_moves = game.get_legal_moves()

            if visits is not None and sum(visits.values()) > 0:
                total = sum(visits.values())
                for m, count in visits.items():
                    fr, fc, tr, tc = m
                    from_sq = fr * 8 + fc
                    to_sq = tr * 8 + tc
                    idx = from_sq * 64 + to_sq
                    pi[idx] = count / total
            else:
                if legal_moves:
                    p = 1.0 / len(legal_moves)
                    for m in legal_moves:
                        fr, fc, tr, tc = m
                        from_sq = fr * 8 + fc
                        to_sq = tr * 8 + tc
                        idx = from_sq * 64 + to_sq
                        pi[idx] = p

            policies.append(pi)

            # Value from perspective of player to move
            if outcome is None:  # Draw or stalemate treated as draw
                v = 0.0
            elif outcome == 1:  # P1 win
                v = 1.0 if (step % 2 == 0) else -1.0
            else:  # P2 win
                v = 1.0 if (step % 2 == 1) else -1.0

            values.append(v)

            game.apply_move(move)

        return W_states, policies, values

    def play_game(self, agent_white, agent_black, game_idx):
        game = Draughts()
        moves_made = []
        pi_records = []
        moves = 0

        mcvs_stats = {"total_sims": 0, "total_time": 0.0, "moves": 0}
        uct_stats = {"total_sims": 0, "total_time": 0.0, "moves": 0}

        while not game.is_terminal() and moves < self.max_moves_per_game:
            moves += 1
            is_white = (moves % 2 == 1)

            if is_white:
                agent = agent_white
                agent_type = "MCVS" if agent is self.agent_mcvs else "UCT"
            else:
                agent = agent_black
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

            temperature = 1.0
            chosen_move = None

            if visits:
                moves_list = list(visits.keys())
                if moves_list:
                    counts = np.array([visits[m] for m in moves_list], dtype=np.float32)
                    total_visits = counts.sum()

                    if total_visits > 0:
                        if temperature < 0.0001:
                            chosen_idx = int(np.argmax(counts))
                        else:
                            probs = counts ** (1.0 / temperature)
                            probs /= probs.sum()
                            chosen_idx = int(np.random.choice(len(moves_list), p=probs))
                        chosen_move = moves_list[chosen_idx]

            if chosen_move is None:
                chosen_move = random.choice(legal_moves)

            player_name = "P1" if is_white else "P2"
            self.move_logger.log_move(
                game_idx, moves, player_name, agent_type,
                sims, t_taken, str(chosen_move)
            )

            game.apply_move(chosen_move)
            moves_made.append(chosen_move)
            pi_records.append(visits if agent_type == "MCVS" else None)

        # Averages
        if mcvs_stats["moves"] > 0:
            mcvs_stats["avg_sims"] = mcvs_stats["total_sims"] / mcvs_stats["moves"]
            mcvs_stats["avg_time"] = mcvs_stats["total_time"] / mcvs_stats["moves"]
        else:
            mcvs_stats["avg_sims"] = mcvs_stats["avg_time"] = 0.0

        if uct_stats["moves"] > 0:
            uct_stats["avg_sims"] = uct_stats["total_sims"] / uct_stats["moves"]
            uct_stats["avg_time"] = uct_stats["total_time"] / uct_stats["moves"]
        else:
            uct_stats["avg_sims"] = uct_stats["avg_time"] = 0.0

        # Winner determination
        if moves >= self.max_moves_per_game:
            winner = "Draw"
            outcome = None
            self.move_logger.log_game_summary(game_idx, "Draw (move limit)", moves, mcvs_stats, uct_stats)
        else:
            explicit_winner = game.check_winner()  # 1 or 2 if all opponent pieces captured
            if explicit_winner is not None:
                winner = "MCVS" if (explicit_winner == 1 and agent_white is self.agent_mcvs) or (explicit_winner == 2 and agent_black is self.agent_mcvs) else "UCT"
                outcome = explicit_winner
            else:
                # Stalemate: player to move has no legal moves → loses
                next_player = 1 if (moves % 2 == 0) else 2
                winner = "MCVS" if (next_player == 2 and agent_white is self.agent_mcvs) or (next_player == 1 and agent_black is self.agent_mcvs) else "UCT"
                outcome = None  # treated as draw for training values

            self.move_logger.log_game_summary(game_idx, winner, moves, mcvs_stats, uct_stats)

        return winner, moves, moves_made, pi_records, outcome

    def run_tournament(self):
        start_time = time.time()
        self.log_line("Starting 200-game Draughts MCVS vs UCT tournament")

        for game_idx in range(1, self.num_games + 1):
            if game_idx % 2 == 1:
                winner, moves, moves_made, pi_records, outcome = self.play_game(self.agent_mcvs, self.agent_uct, game_idx)
                white_name = "MCVS"
            else:
                winner, moves, moves_made, pi_records, outcome = self.play_game(self.agent_uct, self.agent_mcvs, game_idx)
                white_name = "UCT"

            if winner == "Draw":
                self.results["Draw"]["count"] += 1
                self.results["Draw"]["moves"].append(moves)
            else:
                self.results[winner]["wins"] += 1
                self.results[winner]["moves"].append(moves)
                if winner == white_name:
                    self.results[winner]["as_white"] += 1
                else:
                    self.results[winner]["as_black"] += 1

            W_states, policies, values = self.extract_training_and_zone_data(moves_made, pi_records, outcome)

            # Sample positions for zone DB
            selected = [0]
            if len(W_states) > 1:
                selected.append(len(W_states)-1)
                for i in range(1, len(W_states)-1):
                    if random.random() < 0.3:
                        selected.append(i)

            for i in selected:
                v = values[i]
                W = W_states[i]
                if v > 0:
                    self.zone_db.add_winning_matrix(W)
                elif v < 0:
                    self.zone_db.add_losing_matrix(W)
                else:
                    self.zone_db.add_draw_matrix(W)

            self.zone_db.save()

            if game_idx == 1:
                train_start = time.time()
                train_networks(self.policy_net, self.value_net, W_states, policies, values, epochs=10, device=self.device)
                train_time = time.time() - train_start
                self.learning_logger.log_initial_training(game_idx, len(W_states), train_time, 10)
                self.agent_mcvs.lambda_zone = 1.0
                self.networks_initialized = True
                self.training_manager = TrainingManager(self.policy_net, self.value_net, self.device, self.learning_logger)
                print("Initial training completed. Neural networks now active with zone guidance.")
            else:
                self.training_manager.queue_training(W_states, policies, values, game_idx)

            elapsed = time.time() - start_time
            remaining = elapsed / game_idx * (self.num_games - game_idx)
            queue_size = self.training_manager.queue.qsize() if self.networks_initialized else 0
            learned_games = self.training_manager.games_processed if self.networks_initialized else 0
            db_size = len(self.zone_db.winning_matrices) + len(self.zone_db.losing_matrices) + len(self.zone_db.draw_matrices)
            self.log_line(
                f"Game {game_idx:3d}/{self.num_games} | {winner:10s} (moves={moves:3d}) | "
                f"MCVS wins: {self.results['MCVS']['wins']:3d} | Draws: {self.results['Draw']['count']:3d} | "
                f"Queue: {queue_size} | Learned: {learned_games} | DB: {db_size} | ETA: {remaining/60:.1f} min"
            )

        if self.networks_initialized:
            self.log_line("\nWaiting for remaining training to finish...")
            self.training_manager.stop()

            avg_training_time = (self.training_manager.total_training_time / 
                                 self.training_manager.games_processed 
                                 if self.training_manager.games_processed > 0 else 0)
            self.learning_logger.log_tournament_summary(
                self.training_manager.games_processed,
                self.training_manager.total_training_time,
                avg_training_time
            )
            self.log_line("All training completed.")

        # Final summary (identical structure, just Draughts titles)
        self.log_line("\n" + "=" * 80)
        self.log_line("DRAUGHTS TOURNAMENT SUMMARY")
        self.log_line("=" * 80)
        
        mcvs_wins = self.results["MCVS"]["wins"]
        uct_wins = self.results["UCT"]["wins"]
        draws = self.results["Draw"]["count"]
        
        self.log_line(f"\nMCVS (λ-PUCT): {mcvs_wins} wins")
        self.log_line(f"UCT (Standard): {uct_wins} wins")
        self.log_line(f"Draws: {draws}")
        self.log_line(f"Total: {self.num_games}")
        
        if mcvs_wins + uct_wins > 0:
            win_rate = (mcvs_wins / (mcvs_wins + uct_wins)) * 100
            self.log_line(f"\nMCVS Win Rate (excluding draws): {win_rate:.1f}%")
        
        if self.results["MCVS"]["moves"]:
            avg_moves_mcvs = np.mean(self.results["MCVS"]["moves"])
            self.log_line(f"Avg moves (MCVS wins): {avg_moves_mcvs:.1f}")
        
        if self.results["UCT"]["moves"]:
            avg_moves_uct = np.mean(self.results["UCT"]["moves"])
            self.log_line(f"Avg moves (UCT wins): {avg_moves_uct:.1f}")
        
        self.log_line(f"\nMCVS White: {self.results['MCVS']['as_white']}, Black: {self.results['MCVS']['as_black']}")
        self.log_line(f"UCT White: {self.results['UCT']['as_white']}, Black: {self.results['UCT']['as_black']}")
        
        if self.networks_initialized:
            total_db = len(self.zone_db.winning_matrices) + len(self.zone_db.losing_matrices) + len(self.zone_db.draw_matrices)
            self.log_line(f"\nZone Database Statistics:")
            self.log_line(f"  Winning positions: {len(self.zone_db.winning_matrices)}")
            self.log_line(f"  Losing positions: {len(self.zone_db.losing_matrices)}")
            self.log_line(f"  Draw positions: {len(self.zone_db.draw_matrices)}")
            self.log_line(f"  Total positions: {total_db:,}")
            self.log_line(f"  Games learned from: {self.training_manager.games_processed}")
            self.log_line(f"  Total training time: {self.training_manager.total_training_time/60:.1f} min")

        total_time = time.time() - start_time
        self.log_line("\n" + "=" * 80)
        self.log_line(f"Tournament Duration: {total_time/60:.1f} minutes")
        self.log_line(f"Avg time per game: {total_time/self.num_games:.1f}s")
        self.log_line("=" * 80 + "\n")

        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_lines))

        self.move_logger.close()
        self.learning_logger.close()

if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    tournament = DraughtsTournament(num_games=200, time_per_move=1.5, max_moves_per_game=150, device=DEVICE)
    tournament.run_tournament()
