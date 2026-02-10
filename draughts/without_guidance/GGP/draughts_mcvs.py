import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional
import random
import time
import os
import bisect
import math

from abc_model import ABCModelDynamic
from matrix_model import WeightedMatrixABC

# ============================================================================
# HILBERT CURVE UTILITIES
# ============================================================================

def xy2d(n: int, x: int, y: int) -> int:
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        s //= 2
    return d

def matrix_to_hilbert_index(W: np.ndarray) -> int:
    if W.size == 0:
        return 0
    W_flat = W.flatten()
    max_idx = np.argmax(W_flat)
    x = max_idx % 64
    y = max_idx // 64
    return xy2d(64, x, y)

# ============================================================================
# DRAUGHTS GAME LOGIC
# ============================================================================

class Draughts:
    ROWS, COLS = 8, 8
    EMPTY, P1_MAN, P1_KING, P2_MAN, P2_KING = 0, 1, 3, 2, 4

    def __init__(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int32)
        # Standard setup: 12 pieces each on dark squares ((r+c) % 2 == 1 usually dark)
        # Player 1 (white, first mover) at bottom (rows 5-7), Player 2 at top (rows 0-2)
        self.board[0, 1::2] = self.P2_MAN   # row 0, odd columns
        self.board[1, ::2] = self.P2_MAN   # row 1, even columns
        self.board[2, 1::2] = self.P2_MAN   # row 2, odd columns
        self.board[5, ::2] = self.P1_MAN   # row 5, even columns
        self.board[6, 1::2] = self.P1_MAN  # row 6, odd columns
        self.board[7, ::2] = self.P1_MAN   # row 7, even columns
        self.move_count = 0
        self._cached_matrix = None

    def copy(self):
        new = Draughts()
        new.board = self.board.copy()
        new.move_count = self.move_count
        new._cached_matrix = self._cached_matrix.copy() if self._cached_matrix is not None else None
        return new

    def current_player(self) -> int:
        return 1 if self.move_count % 2 == 0 else 2

    def get_legal_moves(self) -> List[Tuple[int, int, int, int]]:
        capture_moves = []
        quiet_moves = []
        player = self.current_player()
        direction = -1 if player == 1 else 1  # P1 moves up (decrease row), P2 down
        man = self.P1_MAN if player == 1 else self.P2_MAN
        king = self.P1_KING if player == 1 else self.P2_KING
        opponent = (self.P2_MAN, self.P2_KING) if player == 1 else (self.P1_MAN, self.P1_KING)

        for r in range(self.ROWS):
            for c in range(self.COLS):
                piece = self.board[r, c]
                if piece not in (man, king):
                    continue
                is_king = piece == king
                dirs = [direction, -direction] if is_king else [direction]
                for dr in dirs:
                    for dc in [-1, 1]:
                        # Quiet move
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.ROWS and 0 <= nc < self.COLS and self.board[nr, nc] == self.EMPTY:
                            quiet_moves.append((r, c, nr, nc))
                        # Capture (single jump)
                        jr, jc = r + 2 * dr, c + 2 * dc
                        if 0 <= jr < self.ROWS and 0 <= jc < self.COLS and self.board[jr, jc] == self.EMPTY:
                            mr, mc = r + dr, c + dc
                            if self.board[mr, mc] in opponent:
                                capture_moves.append((r, c, jr, jc))

        return capture_moves if capture_moves else quiet_moves

    def apply_move(self, move: Tuple[int, int, int, int]):
        fr, fc, tr, tc = move
        piece = self.board[fr, fc]
        self.board[tr, tc] = piece
        self.board[fr, fc] = self.EMPTY
        if abs(tr - fr) == 2:  # capture
            mr, mc = (fr + tr) // 2, (fc + tc) // 2
            self.board[mr, mc] = self.EMPTY
        # Promotion
        if piece == self.P1_MAN and tr == 0:
            self.board[tr, tc] = self.P1_KING
        elif piece == self.P2_MAN and tr == 7:
            self.board[tr, tc] = self.P2_KING
        self.move_count += 1
        self._cached_matrix = None

    def check_winner(self) -> Optional[int]:
        p1_count = np.sum((self.board == self.P1_MAN) | (self.board == self.P1_KING))
        p2_count = np.sum((self.board == self.P2_MAN) | (self.board == self.P2_KING))
        if p1_count == 0:
            return 2
        if p2_count == 0:
            return 1
        return None

    def is_terminal(self) -> bool:
        return self.check_winner() is not None or len(self.get_legal_moves()) == 0

    def get_weighted_adjacency_matrix(self) -> np.ndarray:
        if self._cached_matrix is not None:
            return self._cached_matrix
        abc = ABCModelDynamic(n=2, t=1.0, T=1.41)
        abc.piece_positions = []
        abc.kappa = {}
        for idx in range(64):
            r = idx // 8
            c = idx % 8
            x = c - 3.5
            y = 3.5 - r
            abc.piece_positions.append((x, y))
            piece = self.board[r, c]
            if piece == self.EMPTY:
                delta = (0.0, 0.0, 0.0)
            elif piece == self.P1_MAN:
                delta = (1.0, 1.0, 1.0)
            elif piece == self.P1_KING:
                delta = (1.0, 1.2, 1.0)
            elif piece == self.P2_MAN:
                delta = (1.0, 1.1, 1.0)
            elif piece == self.P2_KING:
                delta = (1.0, 1.3, 1.0)
            else:
                delta = (0.0, 0.0, 0.0)
            abc.kappa[idx] = np.array(delta)
        abc.stage = 64
        abc.history = [{} for _ in range(64)]
        try:
            builder = WeightedMatrixABC(abc, sigma=1.0)
            W = builder.compute_weighted_matrix().astype(np.float32)
            if W.shape != (64, 64):
                W_fixed = np.zeros((64, 64), dtype=np.float32)
                n = min(64, W.shape[0])
                W_fixed[:n, :n] = W[:n, :n]
                W = W_fixed
            self._cached_matrix = W
        except Exception as e:
            print(f"Matrix error: {e}")
            W = np.zeros((64, 64), dtype=np.float32)
            self._cached_matrix = W
        return W

# ============================================================================
# NEURAL NETWORKS
# ============================================================================

class PolicyNetworkCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(8)
        self.fc = nn.Linear(64 * 8 * 8, 4096)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ValueNetworkCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.tanh(self.fc2(x))

# ============================================================================
# MCTS NODE AND SEARCHERS
# ============================================================================

class MCTSNode:
    def __init__(self, game, prior: float = 0.0, parent=None, move=None):
        self.game = game
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: Dict[Tuple, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

class UCTSearcher:
    def __init__(self, cpuct=np.sqrt(2)):
        self.cpuct = cpuct

    def search_with_time_budget(self, game: Draughts, time_budget: float) -> Dict[Tuple, int]:
        root = MCTSNode(game.copy())
        legal = game.get_legal_moves()
        for m in legal:
            root.children[m] = MCTSNode(game.copy(), prior=1.0, parent=root, move=m)  # uniform prior

        start_time = time.time()
        while time.time() - start_time < time_budget:
            node = root
            # Selection
            while node.is_expanded() and not node.game.is_terminal():
                node = max(node.children.values(), key=lambda c: c.value() + self.cpuct * math.sqrt(node.visit_count) / (1 + c.visit_count))
            # Expansion & evaluation
            if not node.game.is_terminal():
                node.visit_count += 1
                value = self._rollout(node.game)
            else:
                winner = node.game.check_winner()
                value = 0.0 if winner is None else (1.0 if winner == game.current_player() else -1.0)
            # Backpropagation
            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                value = -value
                node = node.parent

        return {move: child.visit_count for move, child in root.children.items()}

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
        return 1.0 if winner == game.current_player() else -1.0

class MCVSSearcher:
    def __init__(self, policy_net, value_net, zone_db, device='cpu', cpuct=1.41, lambda_zone=1.0, k_zone=5):
        self.policy_net = policy_net
        self.value_net = value_net
        self.zone_db = zone_db
        self.device = device
        self.cpuct = cpuct
        self.lambda_zone = lambda_zone
        self.k_zone = k_zone

    def _move_to_index(self, move: Tuple[int, int, int, int]) -> int:
        fr, fc, tr, tc = move
        from_sq = fr * 8 + fc
        to_sq = tr * 8 + tc
        return from_sq * 64 + to_sq

    def search_with_time_budget(self, game: Draughts, time_budget: float, add_noise: bool = True) -> Dict[Tuple, int]:
        root = MCTSNode(game.copy())

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return {}

        # === Root expansion with policy priors (+ optional Dirichlet noise for self-play) ===
        W = torch.tensor(game.get_weighted_adjacency_matrix()[np.newaxis, np.newaxis, :, :],
                         dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.policy_net(W)[0].cpu().numpy()

        # Stable softmax over full 4096
        logits = logits - np.max(logits)
        policy_full = np.exp(logits)
        policy_full /= policy_full.sum() + 1e-10

        # Mask to legal moves only
        priors = {}
        sum_p = 0.0
        for m in legal_moves:
            idx = self._move_to_index(m)
            p = policy_full[idx] if idx < 4096 else 0.0
            priors[m] = p
            sum_p += p

        if sum_p == 0:
            p = 1.0 / len(legal_moves)
            for m in legal_moves:
                priors[m] = p
        else:
            for m in priors:
                priors[m] /= sum_p

        # Dirichlet noise for exploration/variety in self-play
        if add_noise and len(legal_moves) > 1:
            alpha = 0.03  # standard for chess-like games
            noise = np.random.dirichlet([alpha] * len(legal_moves))
            i = 0
            for m in legal_moves:
                priors[m] = 0.75 * priors[m] + 0.25 * noise[i]
                i += 1
            # Re-normalize
            sum_p = sum(priors.values())
            for m in priors:
                priors[m] /= sum_p

        # Create root children
        for m in legal_moves:
            child_game = game.copy()
            child_game.apply_move(m)
            root.children[m] = MCTSNode(child_game, prior=priors[m], parent=root, move=m)

        start_time = time.time()
        while time.time() - start_time < time_budget:
            node = root

            # === Selection ===
            while node.is_expanded() and not node.game.is_terminal():
                node = max(
                    node.children.values(),
                    key=lambda c: c.value() +
                    self.cpuct * c.prior * math.sqrt(node.visit_count) / (1 + c.visit_count)
                )

            # === Evaluation / Expansion ===
            if node.game.is_terminal():
                winner = node.game.check_winner()
                value = 0.0 if winner is None else (1.0 if winner == node.game.current_player() else -1.0)
            else:
                # Neural evaluation on current (leaf) state
                W_leaf = torch.tensor(node.game.get_weighted_adjacency_matrix()[np.newaxis, np.newaxis, :, :],
                                      dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    leaf_logits = self.policy_net(W_leaf)[0].cpu().numpy()
                    value = self.value_net(W_leaf)[0].item()

                # Zone bonus
                if self.lambda_zone > 0:
                    zone_bonus = self._zone_bonus(node.game.get_weighted_adjacency_matrix())
                    value += self.lambda_zone * zone_bonus

                # Policy for child priors (masked to legal)
                leaf_logits = leaf_logits - np.max(leaf_logits)
                policy_full = np.exp(leaf_logits)
                policy_full /= policy_full.sum() + 1e-10

                child_legal = node.game.get_legal_moves()
                child_priors = {}
                sum_p = 0.0
                for m in child_legal:
                    idx = self._move_to_index(m)
                    p = policy_full[idx] if idx < 4096 else 0.0
                    child_priors[m] = p
                    sum_p += p
                if sum_p == 0:
                    p = 1.0 / len(child_legal)
                    for m in child_legal:
                        child_priors[m] = p
                else:
                    for m in child_legal:
                        child_priors[m] /= sum_p

                # Expand children
                for m in child_legal:
                    child_game = node.game.copy()
                    child_game.apply_move(m)
                    node.children[m] = MCTSNode(child_game, prior=child_priors.get(m, 0.0),
                                                parent=node, move=m)

            # === Backpropagation ===
            current = node
            while current is not None:
                current.visit_count += 1
                current.value_sum += value
                value = -value
                current = current.parent

        return {move: child.visit_count for move, child in root.children.items()}

    def _zone_bonus(self, W: np.ndarray) -> float:
        """Compute a simple zone guidance bonus based on Manhattan distance to known winning matrices."""
        if len(self.zone_db.winning_matrices) == 0:
            return 0.0
        
        # Create dummy ABC model (not used for distance computation)
        dummy_abc = ABCModelDynamic()
        
        # Current position "object"
        current = WeightedMatrixABC(dummy_abc)
        current.W = W  # Set directly - skips expensive recomputation
        
        min_dist = float('inf')
        for win_W in self.zone_db.winning_matrices[-self.k_zone:]:  # Use k most recent winning positions
            comparator = WeightedMatrixABC(dummy_abc)
            comparator.W = win_W
            dist = current.compute_manhattan_distance(comparator)
            min_dist = min(min_dist, dist)
        
        # Rough normalization: W entries are typically small (0-1 range), 64x64 matrix
        # Max possible L1 distance ~4096 if all differ by 1, but actual values much smaller
        scale = 5000.0  # Tunable - adjust based on observed distances
        normalized_similarity = 1.0 - (min_dist / scale)
        bonus = max(0.0, normalized_similarity)
        
        return bonus

# ============================================================================
# ZONE DATABASE
# ============================================================================

class HilbertOrderedZoneDatabase:
    def __init__(self, filepath: str = "draughts_zone_db.npz", max_size: int = 100000):
        self.filepath = filepath
        self.max_size = max_size
        self.winning_matrices = []
        self.winning_indices = []   # For sorted order / locality only
        self.losing_matrices = []
        self.losing_indices = []
        self.draw_matrices = []
        self.draw_indices = []
        self.load()

    def add_winning_matrix(self, W: np.ndarray):
        if W.shape != (64, 64) or len(self.winning_matrices) >= self.max_size:
            return
        hilbert_idx = matrix_to_hilbert_index(W)
        pos = bisect.bisect_left(self.winning_indices, hilbert_idx)
        self.winning_indices.insert(pos, hilbert_idx)
        self.winning_matrices.insert(pos, W.copy())  

    def add_losing_matrix(self, W: np.ndarray):
        if W.shape != (64, 64) or len(self.losing_matrices) >= self.max_size:
            return
        hilbert_idx = matrix_to_hilbert_index(W)
        pos = bisect.bisect_left(self.losing_indices, hilbert_idx)
        self.losing_indices.insert(pos, hilbert_idx)
        self.losing_matrices.insert(pos, W.copy()) 

    def add_draw_matrix(self, W: np.ndarray):
        if W.shape != (64, 64) or len(self.draw_matrices) >= self.max_size:
            return
        hilbert_idx = matrix_to_hilbert_index(W)
        pos = bisect.bisect_left(self.draw_indices, hilbert_idx)
        self.draw_indices.insert(pos, hilbert_idx)
        self.draw_matrices.insert(pos, W.copy()) 

    def load(self):
        if os.path.exists(self.filepath):
            try:
                data = np.load(self.filepath, allow_pickle=True)
                self.winning_matrices = list(data['winning'])
                self.winning_indices = list(data['winning_indices'])
                self.losing_matrices = list(data['losing'])
                self.losing_indices = list(data['losing_indices'])
                self.draw_matrices = list(data['draw'])
                self.draw_indices = list(data['draw_indices'])
            except Exception as e:
                print(f"Zone DB load failed: {e}")

    def save(self):
        try:
            np.savez_compressed(
                self.filepath,
                winning=np.array(self.winning_matrices, dtype=object),
                winning_indices=np.array(self.winning_indices),
                losing=np.array(self.losing_matrices, dtype=object),
                losing_indices=np.array(self.losing_indices),
                draw=np.array(self.draw_matrices, dtype=object),
                draw_indices=np.array(self.draw_indices),
            )
        except Exception as e:
            print(f"Zone DB save failed: {e}")

# ============================================================================
# TRAINING
# ============================================================================

def train_networks(policy_net, value_net, W_states, policies, values, epochs=5, batch_size=32, device='cpu'):
    policy_net.train()
    value_net.train()
    opt_p = optim.Adam(policy_net.parameters(), lr=0.001)
    opt_v = optim.Adam(value_net.parameters(), lr=0.001)
    W_tensor = torch.tensor(np.array(W_states)[:, np.newaxis, :, :], dtype=torch.float32).to(device)
    pi_tensor = torch.tensor(np.array(policies), dtype=torch.float32).to(device)
    v_tensor = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1).to(device)
    dataset = TensorDataset(W_tensor, pi_tensor, v_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_W, batch_pi, batch_v in loader:
            logits = policy_net(batch_W)
            policy_loss = F.kl_div(F.log_softmax(logits, dim=1), batch_pi, reduction='batchmean')
            pred_v = value_net(batch_W)
            value_loss = F.mse_loss(pred_v, batch_v)
            loss = policy_loss + value_loss
            total_loss += loss.item()
            opt_p.zero_grad()
            opt_v.zero_grad()
            loss.backward()
            opt_p.step()
            opt_v.step()
        print(f"Training Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

# ============================================================================
# MAIN (self-play training loop)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DRAUGHTS MCVS - INCREMENTAL SELF-PLAY TRAINING")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    policy_net = PolicyNetworkCNN().to(device)
    value_net = ValueNetworkCNN().to(device)

    zone_db = HilbertOrderedZoneDatabase("draughts_zone_db.npz", max_size=10000)

    checkpoint_path = "draughts_checkpoint.pt"
    iteration = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy_net.load_state_dict(checkpoint['policy'])
        value_net.load_state_dict(checkpoint['value'])
        iteration = checkpoint.get('iteration', 0)
        print(f"Resumed from iteration {iteration}")

    searcher = MCVSSearcher(policy_net, value_net, zone_db, device=device, lambda_zone=1.0)

    while True:
        print(f"\n=== ITERATION {iteration} ===")
        # Simple self-play loop (replace with your full generate_self_play_data if preferred)
        W_states, policies, values = [], [], []
        for game_idx in range(20):  # games per iteration
            game = Draughts()
            trajectory = []
            while not game.is_terminal():
                visits = searcher.search_with_time_budget(game.copy(), time_budget=1.0)
                if not visits:
                    break
                move = max(visits, key=visits.get)
                W = game.get_weighted_adjacency_matrix().copy()
                pi = np.zeros(4096)
                total = sum(visits.values())
                if total == 0:
                    total = 1
                for m, cnt in visits.items():
                    fr, fc, tr, tc = m
                    from_sq = fr * 8 + fc
                    to_sq = tr * 8 + tc
                    idx = from_sq * 64 + to_sq
                    pi[idx] = cnt / total
                trajectory.append((W, pi))
                game.apply_move(move)
            winner = game.check_winner()
            for i, (W, pi) in enumerate(trajectory):
                v = 0.0 if winner is None else (1.0 if winner == (1 if i%2==0 else 2) else -1.0)
                W_states.append(W)
                policies.append(pi)
                values.append(v)
            # Add to zone DB
            for W, v in zip([t[0] for t in trajectory], values):
                if v > 0:
                    zone_db.add_winning_matrix(W)
                elif v < 0:
                    zone_db.add_losing_matrix(W)
                else:
                    zone_db.add_draw_matrix(W)

        if W_states:
            train_networks(policy_net, value_net, W_states, policies, values, epochs=10, device=device)

        torch.save({
            'policy': policy_net.state_dict(),
            'value': value_net.state_dict(),
            'iteration': iteration
        }, checkpoint_path)
        zone_db.save()
        iteration += 1
