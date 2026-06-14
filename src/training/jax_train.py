# src/training/jax_train.py
import os
import sys
import json
import pickle
import random
import time
# Enable persistent JAX compilation caching to eliminate startup delays on subsequent runs
import os
os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "true"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/home/vishal/AI-Monopoly-JAX/.jax_cache"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "2.0"
os.makedirs("/home/vishal/AI-Monopoly-JAX/.jax_cache", exist_ok=True)
import jax
import jax.numpy as jnp
import numpy as np
from src.simulator.jax_game import (
    BOARD_TYPE, BOARD_COST, BOARD_HOUSE_COST, BOARD_MORTGAGE,
    BOARD_COLOR_GROUP, COLOR_GROUP_SIZE, BOARD_RENT_TABLE,
    get_mlp_inputs, mlp_forward, play_game_scan, get_net_worths
)

ROLE_NAMES = ["Player0", "Player1", "Player2", "Player3"]

MAX_TURNS = 600  # Must match play_game_scan default
NUM_GAMES = 32

# ==============================================================================
# 1. JAX Fitness Evaluation Function
# ==============================================================================

def compute_single_game_fitness(final_state):
    """Fitness function:
    1. Dynamic Rank-Based Scoring (Winner reward scales with total bankruptcies)
    2. Linear Time Penalty (-0.001 per turn up to game completion/cap)
    3. Timeout Penalty (-1.5 flat penalty for survivors if turn 600 cap reached)
    """
    net_worths = get_net_worths(final_state)
    is_bankrupt = final_state["players_is_bankrupt"]
    bankruptcy_order = final_state["players_bankruptcy_order"]
    
    # Active players: score is 4.0 + (net_worth / 1e7)
    # Bankrupt players: score is their bankruptcy order (1, 2, or 3)
    scores = jnp.where(
        is_bankrupt,
        bankruptcy_order.astype(jnp.float32),
        4.0 + net_worths / 1e7
    )
    
    # Rank players from 0 (worst/4th) to 3 (best/1st)
    ranks = jnp.argsort(jnp.argsort(scores))
    
    # Count total opponent bankruptcies (0 to 3) to scale the winner's reward
    total_bankruptcies = jnp.sum(is_bankrupt).astype(jnp.float32)
    winner_reward = 1.33 * total_bankruptcies  # 3 bankrupts = 4.0, 2 = 2.66, 1 = 1.33, 0 = 0.0
    
    rewards = jnp.stack([-2.0, -1.0, 0.0, winner_reward])
    base_fitness = rewards[ranks]
    
    # Linear Time Penalty: -0.001 per turn
    end_turn = final_state["game_end_turn"]
    effective_turns = jnp.where(end_turn > 0, end_turn.astype(jnp.float32), 600.0)
    time_penalty = -0.001 * effective_turns
    
    # Timeout Penalty: -1.5 flat penalty to survivors if limit reached
    limit_reached = (end_turn == 0)
    timeout_penalty = jnp.where(limit_reached & (~is_bankrupt), -3.0, 0.0)
    
    player_fitnesses = base_fitness + time_penalty + timeout_penalty
    return player_fitnesses[0]

# ==============================================================================
# 2. Parallel Evaluation via double vmap
# ==============================================================================

def init_random_weights(key):
    """Initializes a random MLP weights dictionary. Input size is 30 (no personality)."""
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(k1, (64, 30)) * jnp.sqrt(2.0 / 30.0),
        "b1": jnp.zeros((64,)),
        "w2": jax.random.normal(k2, (32, 64)) * jnp.sqrt(2.0 / 64.0),
        "b2": jnp.zeros((32,)),
        "w3": jax.random.normal(k3, (8, 32)) * jnp.sqrt(2.0 / 32.0),
        "b3": jnp.zeros((8,))
    }

# Vectorized simulation runner: runs 1 game using weights
def single_game_run(rng_key, weights):
    final_state = play_game_scan(rng_key, weights, max_turns=MAX_TURNS)
    return compute_single_game_fitness(final_state)

# Double vmap: maps over NUM_GAMES games AND 128 genomes
evaluate_population_games = jax.jit(
    jax.vmap(
        jax.vmap(single_game_run, in_axes=(0, 0)),  # inner vmap: map over game keys AND weights
        in_axes=(None, 0)                           # outer vmap: map over genomes
    )
)

# ==============================================================================
# 3. Evolution Loop Implementation
# ==============================================================================


def save_champion_pickle(filepath, weights):
    """Saves JAX champion weights dict converted to standard NumPy arrays."""
    np_weights = {k: np.array(v) for k, v in weights.items()}
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    temp_filepath = filepath + ".tmp"
    with open(temp_filepath, "wb") as f:
        pickle.dump(np_weights, f)
    os.replace(temp_filepath, filepath)

# ==============================================================================
# 4. NumPy Showcase Game Generator for Replay
# ==============================================================================

def simulate_showcase_game_py(weights):
    """CPU-based interpreter to run a showcase game and compile detailed JSON replay logs."""
    # Convert JAX weights to NumPy for CPU fast execution
    np_weights = {k: np.array(v) for k, v in weights.items()}
    
    # Initialize CPU Game State
    cash = np.array([1500, 1500, 1500, 1500], dtype=np.int32)
    position = np.array([0, 0, 0, 0], dtype=np.int32)
    in_jail = np.array([False, False, False, False], dtype=bool)
    get_out_cards = np.array([0, 0, 0, 0], dtype=np.int32)
    is_bankrupt = np.array([False, False, False, False], dtype=bool)
    
    prop_owner = np.full(40, -1, dtype=np.int32)
    prop_houses = np.zeros(40, dtype=np.int32)
    prop_mortgaged = np.zeros(40, dtype=bool)
    
    logs = []
    doubles_count = np.zeros(4, dtype=np.int32)  # consecutive doubles per player
    
    # Board Space names list helper
    from src.simulator.board import BOARD_SPACES
    space_names = [s["name"] for s in BOARD_SPACES]
    
    # Static board parameters locally in numpy
    np_type = np.array(BOARD_TYPE)
    np_cost = np.array(BOARD_COST)
    np_house_cost = np.array(BOARD_HOUSE_COST)
    np_mortgage = np.array(BOARD_MORTGAGE)
    np_color_grp = np.array(BOARD_COLOR_GROUP)
    np_color_size = np.array(COLOR_GROUP_SIZE)
    np_rent_table = np.array(BOARD_RENT_TABLE)
    turn_num = 0
    curr_player = 0
    
    def get_py_net_worths():
        nw = cash.copy()
        for i in range(4):
            mask = (prop_owner == i)
            nw[i] += np.sum(np_cost[mask]) + np.sum(prop_houses[mask] * np_house_cost[mask])
        return nw

    def get_py_mlp_inputs(player_idx, prop_id, trade_cash_diff=0.0, trade_prop_diff=0.0, mask_blocker=False):
        nw = get_py_net_worths()
        order = (np.arange(4) + player_idx) % 4
        
        cash_feat = cash[order] / 1500.0
        nw_feat = nw[order] / 3000.0
        pos_feat = position[order] / 40.0
        jail_feat = np.where(in_jail[order], 1.0, 0.0)
        
        cost_f = np_cost[prop_id] / 400.0
        h_cost_f = np_house_cost[prop_id] / 200.0
        houses_f = prop_houses[prop_id] / 5.0
        mort_f = 1.0 if prop_mortgaged[prop_id] else 0.0
        
        is_st = 1.0 if np_type[prop_id] == 2 else 0.0
        is_ut = 1.0 if np_type[prop_id] == 3 else 0.0
        is_pr = 1.0 if np_type[prop_id] == 1 else 0.0
        
        grp = np_color_grp[prop_id]
        if grp >= 0:
            mask = (np_color_grp == grp) & (np_type == 1)
            g_size = max(1, np_color_size[grp])
            owned = np.sum((prop_owner == player_idx) & mask)
            own_frac = owned / g_size
            opp_counts = [np.sum((prop_owner == o) & mask) for o in [(player_idx+1)%4, (player_idx+2)%4, (player_idx+3)%4]]
            opp_max = max(opp_counts)
            opp_frac = opp_max / g_size
            blocker = 1.0 if (owned == 1 and opp_max == g_size - 1) else 0.0
        else:
            own_frac = 0.0
            opp_frac = 0.0
            blocker = 0.0
        
        # Strip blocking signals when evaluating trades to break the blocking equilibrium
        if mask_blocker:
            opp_frac = 0.0
            blocker = 0.0
            
        return np.concatenate([
            cash_feat, nw_feat, pos_feat, jail_feat,
            [cost_f, h_cost_f, houses_f, mort_f, is_st, is_ut, is_pr, own_frac],
            [trade_cash_diff / 1000.0, trade_prop_diff / 1000.0],
            [opp_frac, blocker, 0.0, 0.0]
        ])

    def np_mlp_forward(player_idx, x):
        # Activation helper
        w1 = np_weights["w1"][player_idx]
        b1 = np_weights["b1"][player_idx]
        w2 = np_weights["w2"][player_idx]
        b2 = np_weights["b2"][player_idx]
        w3 = np_weights["w3"][player_idx]
        b3 = np_weights["b3"][player_idx]
        
        h1 = np.dot(w1, x) + b1
        h1 = np.maximum(0, h1)
        h2 = np.dot(w2, h1) + b2
        h2 = np.maximum(0, h2)
        out = np.dot(w3, h2) + b3
        return 1.0 / (1.0 + np.exp(-np.clip(out, -500, 500)))

    # Run game step-by-step
    while turn_num < MAX_TURNS:
        if np.sum(~is_bankrupt) <= 1:  # 1 or fewer players remain — game over
            break

        p_name = ROLE_NAMES[curr_player]
        if is_bankrupt[curr_player]:
            curr_player = (curr_player + 1) % 4
            continue
            
        turn_num += 1
        d1, d2 = random.randint(1, 6), random.randint(1, 6)
        dice_sum = d1 + d2
        rolled_doubles = (d1 == d2)
        
        actions = []
        turn_context = {"creditor": -1}
        dialogue = None
        explain_ai = None
        
        # Jail Resolution
        if in_jail[curr_player]:
            feats = get_py_mlp_inputs(curr_player, position[curr_player])
            outputs = np_mlp_forward(curr_player, feats)
            
            escaped = False
            if rolled_doubles:
                in_jail[curr_player] = False
                escaped = True
                # Doubles jail escape: player moves but does NOT get to re-roll
                rolled_doubles = False
                doubles_count[curr_player] = 0
                actions.append(f"{p_name} rolled doubles ({d1}, {d2}) and escaped from Jail.")
            elif get_out_cards[curr_player] > 0 and outputs[3] > 0.5:
                get_out_cards[curr_player] -= 1
                in_jail[curr_player] = False
                escaped = True
                actions.append(f"{p_name} used a Get Out of Jail Free card.")
            elif cash[curr_player] >= 50 and outputs[2] > 0.5:
                cash[curr_player] -= 50
                in_jail[curr_player] = False
                escaped = True
                actions.append(f"{p_name} paid £50 fine to escape Jail.")
            else:
                actions.append(f"{p_name} rolled ({d1}, {d2}) and remains in Jail.")
                
            if not escaped:
                board_state_snap = {str(k): {"owner": ROLE_NAMES[prop_owner[k]] if prop_owner[k] >= 0 else None, "houses": int(prop_houses[k]), "mortgaged": bool(prop_mortgaged[k])} for k in range(40)}
                players_snap = [{"name": ROLE_NAMES[i], "cash": int(cash[i]), "position": int(position[i]), "in_jail": bool(in_jail[i]), "is_bankrupt": bool(is_bankrupt[i])} for i in range(4)]
                logs.append({
                    "turn": turn_num, "player": p_name, "personality": ROLE_NAMES[curr_player],
                    "dice": [d1, d2], "actions": actions, "dialogue": None, "explain_ai": None,
                    "board_state": board_state_snap, "players_state": players_snap
                })
                doubles_count[curr_player] = 0
                curr_player = (curr_player + 1) % 4
                continue
        
        # 3 consecutive doubles → go to jail immediately, no move
        if rolled_doubles:
            doubles_count[curr_player] += 1
            if doubles_count[curr_player] >= 3:
                in_jail[curr_player] = True
                position[curr_player] = 10
                doubles_count[curr_player] = 0
                actions.append(f"{p_name} rolled doubles ({d1}, {d2}) for the 3rd time — sent to Jail!")
                board_state_snap = {str(k): {"owner": ROLE_NAMES[prop_owner[k]] if prop_owner[k] >= 0 else None, "houses": int(prop_houses[k]), "mortgaged": bool(prop_mortgaged[k])} for k in range(40)}
                players_snap = [{"name": ROLE_NAMES[i], "cash": int(cash[i]), "position": int(position[i]), "in_jail": bool(in_jail[i]), "is_bankrupt": bool(is_bankrupt[i])} for i in range(4)]
                logs.append({
                    "turn": turn_num, "player": p_name, "personality": ROLE_NAMES[curr_player],
                    "dice": [d1, d2], "actions": actions, "dialogue": None, "explain_ai": None,
                    "board_state": board_state_snap, "players_state": players_snap
                })
                curr_player = (curr_player + 1) % 4
                continue
            else:
                actions.append(f"{p_name} rolled doubles ({d1}, {d2})! (streak: {int(doubles_count[curr_player])})")
        else:
            doubles_count[curr_player] = 0

        # Move
        old_p = position[curr_player]
        new_p = (old_p + dice_sum) % 40
        position[curr_player] = new_p
        
        actions.append(f"{p_name} rolled {d1} and {d2}, moving from {space_names[old_p]} to {space_names[new_p]}.")
        if new_p < old_p:
            cash[curr_player] += 200
            actions.append(f"{p_name} passed GO and collected £200.")
            
        # ── Helper: resolve whatever square `pid` the player is now standing on ──
        def resolve_landing(pid):
            ltype = np_type[pid]
            lowner = prop_owner[pid]

            if ltype in [1, 2, 3]:  # Property / Station / Utility
                if lowner == -1:
                    feats = get_py_mlp_inputs(curr_player, pid)
                    outputs = np_mlp_forward(curr_player, feats)
                    if outputs[0] > 0.5 and cash[curr_player] >= np_cost[pid]:
                        prop_owner[pid] = curr_player
                        cash[curr_player] -= np_cost[pid]
                        actions.append(f"{p_name} bought {space_names[pid]} for £{np_cost[pid]}.")
                    else:
                        bid_factors = []
                        for i in range(4):
                            if not is_bankrupt[i]:
                                f_inputs = get_py_mlp_inputs(i, pid)
                                f_out = np_mlp_forward(i, f_inputs)
                                factor = 0.1 + 1.4 * f_out[6]
                                bid_limit = min(cash[i], int(np_cost[pid] * factor))
                                bid_factors.append((bid_limit + random.random() * 0.01, i))
                            else:
                                bid_factors.append((0.0, i))
                        bid_factors.sort(key=lambda x: x[0], reverse=True)
                        win_bid_raw, win_p = bid_factors[0]
                        win_bid = int(win_bid_raw)
                        if win_bid > 0:
                            prop_owner[pid] = win_p
                            cash[win_p] -= win_bid
                            actions.append(f"[AUCTION] {ROLE_NAMES[win_p]} won {space_names[pid]} with a bid of £{win_bid}.")
                elif lowner != curr_player:
                    has_mon = False
                    color = np_color_grp[pid]
                    if color >= 0:
                        color_mask = (np_color_grp == color) & (np_type == 1)
                        has_mon = np.sum(prop_owner[color_mask] == lowner) == np_color_size[color]
                    if ltype == 1:
                        rent_idx = (prop_houses[pid] + 1) if prop_houses[pid] > 0 else (1 if has_mon else 0)
                        rent = np_rent_table[pid, rent_idx]
                    elif ltype == 2:
                        station_count = np.sum((np_type == 2) & (prop_owner == lowner))
                        rent = 25 * (2 ** (station_count - 1))
                    else:
                        util_count = np.sum((np_type == 3) & (prop_owner == lowner))
                        rent = dice_sum * (10 if util_count >= 2 else 4)
                    if not prop_mortgaged[pid]:
                        cash[curr_player] -= rent
                        cash[lowner] += rent
                        turn_context["creditor"] = lowner
                        actions.append(f"{p_name} paid £{rent} rent to {ROLE_NAMES[lowner]} at {space_names[pid]}.")

            elif ltype == 4:  # Tax
                tax = 200 if pid == 4 else 100
                cash[curr_player] -= tax
                actions.append(f"{p_name} paid £{tax} tax to the Bank.")

            elif ltype == 5:  # Go to Jail
                in_jail[curr_player] = True
                position[curr_player] = 10
                actions.append(f"{p_name} was sent directly to Jail.")

        # Resolve the dice-roll landing square
        resolve_landing(new_p)

        # ── Chance / Community Chest card draw ───────────────────────────────────
        ptype = np_type[new_p]
        if ptype in [6, 7]:
            card_id = 1 + (turn_num % 6)
            if card_id == 1:
                position[curr_player] = 0
                cash[curr_player] += 200
                actions.append(f"Card Drawn: Advance to GO. Collected £200.")
            elif card_id == 2:
                card_dest = 24  # Trafalgar Square
                cash[curr_player] += (200 if new_p > card_dest else 0)
                position[curr_player] = card_dest
                actions.append(f"Card Drawn: Advance to Trafalgar Square.")
                resolve_landing(card_dest)   # ← resolve the destination square
            elif card_id == 3:
                cash[curr_player] -= 50
                actions.append(f"Card Drawn: Doctor's fees. Paid £50.")
            elif card_id == 4:
                in_jail[curr_player] = True
                position[curr_player] = 10
                actions.append(f"Card Drawn: Go directly to Jail.")
            elif card_id == 5:
                get_out_cards[curr_player] += 1
                actions.append(f"Card Drawn: Received Get Out of Jail Free card.")
            else:
                cash[curr_player] += 100
                actions.append(f"Card Drawn: Collect dividend. Received £100.")

        # Resolving Bankruptcy / Liquidity Crisis
        if cash[curr_player] < 0:
            # Sell houses
            for pid in range(40):
                if prop_owner[pid] == curr_player and prop_houses[pid] > 0 and cash[curr_player] < 0:
                    refund = np_house_cost[pid] // 2
                    houses_to_sell = min(prop_houses[pid], int(np.ceil(-cash[curr_player] / refund)))
                    prop_houses[pid] -= houses_to_sell
                    cash[curr_player] += houses_to_sell * refund
                    actions.append(f"{p_name} sold {houses_to_sell} houses on {space_names[pid]} for liquidity.")
                    
            # Mortgage properties
            for pid in range(40):
                if prop_owner[pid] == curr_player and not prop_mortgaged[pid] and prop_houses[pid] == 0 and cash[curr_player] < 0:
                    prop_mortgaged[pid] = True
                    cash[curr_player] += np_mortgage[pid]
                    actions.append(f"{p_name} mortgaged {space_names[pid]} for £{np_mortgage[pid]}.")
                    
            # Still bankrupt?
            if cash[curr_player] < 0:
                is_bankrupt[curr_player] = True
                cash[curr_player] = 0
                creditor = turn_context["creditor"]
                prop_mask = (prop_owner == curr_player)
                prop_owner[prop_mask] = creditor
                prop_houses[prop_mask] = 0
                if creditor == -1:
                    prop_mortgaged[prop_mask] = False
                
                if creditor >= 0:
                    actions.append(f"🚨 {p_name} declared bankruptcy! All properties transferred to {ROLE_NAMES[creditor]}.")
                else:
                    actions.append(f"🚨 {p_name} declared bankruptcy! All properties returned to the Bank.")

        # Building Management (House construction & unmortgaging)
        if not is_bankrupt[curr_player]:
            # Unmortgage
            for pid in range(40):
                if prop_owner[pid] == curr_player and prop_mortgaged[pid]:
                    feats = get_py_mlp_inputs(curr_player, pid)
                    outputs = np_mlp_forward(curr_player, feats)
                    cost_unm = int(np_mortgage[pid] * 1.1)
                    if outputs[3] > 0.5 and cash[curr_player] >= cost_unm:
                        prop_mortgaged[pid] = False
                        cash[curr_player] -= cost_unm
                        actions.append(f"{p_name} unmortgaged {space_names[pid]} for £{cost_unm}.")
            
            # Build houses
            for pid in range(40):
                color = np_color_grp[pid]
                if color >= 0:
                    color_mask = (np_color_grp == color) & (np_type == 1)
                    has_mon = np.sum(prop_owner[color_mask] == curr_player) == np_color_size[color]
                    if has_mon and not prop_mortgaged[pid] and prop_houses[pid] < 5:
                        feats = get_py_mlp_inputs(curr_player, pid)
                        outputs = np_mlp_forward(curr_player, feats)
                        h_cost = np_house_cost[pid]
                        
                        # Even building check
                        mask_color = (np_color_grp == color) & (np_type == 1)
                        min_houses = np.min(np.where(mask_color, prop_houses, 5))
                        is_even = (prop_houses[pid] == min_houses)
                        
                        if outputs[1] > 0.5 and cash[curr_player] >= h_cost and is_even:
                            prop_houses[pid] += 1
                            cash[curr_player] -= h_cost
                            actions.append(f"{p_name} built a house on {space_names[pid]} for £{h_cost}.")
            
            # Property Trading Phase (Strategy-Free matching JAX implementation)
            ownable_pids = [pid for pid in range(40) if np_type[pid] in [1, 2, 3]]
            
            # 1. Proposer A evaluates target properties owned by active opponents
            best_target_pid = -1
            best_target_score = -1.0
            for pid in ownable_pids:
                owner = prop_owner[pid]
                if owner >= 0 and owner != curr_player and not is_bankrupt[owner]:
                    if prop_houses[pid] == 0 and not prop_mortgaged[pid]:
                        # Run Proposer MLP on target (blocker-masked)
                        feats = get_py_mlp_inputs(curr_player, pid, mask_blocker=True)
                        outputs = np_mlp_forward(curr_player, feats)
                        score = outputs[4] # Output 4: target desirability
                        if score > best_target_score:
                            best_target_score = score
                            best_target_pid = pid
                            
            # 2. Proposer A evaluates offer properties they own
            best_offer_pid = -1
            best_offer_score = -1.0
            for spid in ownable_pids:
                if prop_owner[spid] == curr_player and prop_houses[spid] == 0 and not prop_mortgaged[spid]:
                    # Run Proposer MLP on offer (blocker-masked)
                    feats = get_py_mlp_inputs(curr_player, spid, mask_blocker=True)
                    outputs = np_mlp_forward(curr_player, feats)
                    score = outputs[5] # Output 5: proposer willingness
                    if score > best_offer_score:
                        best_offer_score = score
                        best_offer_pid = spid
                        
            # Propose and evaluate if scores are valid
            if best_target_pid >= 0 and best_offer_pid >= 0 and best_target_score > 0.5:
                # Proposer calculates cash offer using Output 6 (blocker-masked)
                feats_propose = get_py_mlp_inputs(curr_player, best_target_pid, mask_blocker=True)
                proposer_outputs = np_mlp_forward(curr_player, feats_propose)
                
                cost_target = np_cost[best_target_pid]
                cost_swap = np_cost[best_offer_pid]
                cost_diff = cost_target - cost_swap
                
                cash_offer = int(cost_diff + cost_target * (proposer_outputs[6] - 0.5) * 3.0)
                owner_b = prop_owner[best_target_pid]
                p_name_target = space_names[best_target_pid]
                p_name_swap = space_names[best_offer_pid]
                
                # Financial validity
                proposer_has_cash = cash[curr_player] >= cash_offer
                owner_has_cash = cash[owner_b] >= -cash_offer
                cash_ok = proposer_has_cash if cash_offer >= 0 else owner_has_cash
                
                if cash_ok:
                    # Receiver B evaluates with dedicated Output 7 (blocker-masked)
                    owner_feats = get_py_mlp_inputs(owner_b, best_target_pid, cash_offer, cost_swap - cost_target, mask_blocker=True)
                    owner_outputs = np_mlp_forward(owner_b, owner_feats)
                    
                    if owner_outputs[7] > 0.5:
                        # Execute Trade!
                        prop_owner[best_target_pid] = curr_player
                        prop_owner[best_offer_pid] = owner_b
                        
                        cash[curr_player] -= cash_offer
                        cash[owner_b] += cash_offer
                        
                        if cash_offer > 0:
                            actions.append(f"🤝 [TRADE] {p_name} swapped {p_name_swap} + £{cash_offer} with {ROLE_NAMES[owner_b]} for {p_name_target}.")
                        elif cash_offer < 0:
                            actions.append(f"🤝 [TRADE] {p_name} swapped {p_name_swap} with {ROLE_NAMES[owner_b]} for {p_name_target} + £{-cash_offer}.")
                        else:
                            actions.append(f"🤝 [TRADE] {p_name} swapped {p_name_swap} with {ROLE_NAMES[owner_b]} for {p_name_target}.")
                    else:
                        # Receiver declined
                        cash_str = f" + £{cash_offer}" if cash_offer > 0 else (f" + £{-cash_offer} back" if cash_offer < 0 else "")
                        actions.append(f"❌ [TRADE REJECTED] {ROLE_NAMES[owner_b]} declined {p_name}'s offer of {p_name_swap}{cash_str} for {p_name_target}.")
                else:
                    # Insufficient funds
                    if cash_offer >= 0:
                        actions.append(f"❌ [TRADE REJECTED] {p_name} couldn't afford £{cash_offer} cash sweetener for {p_name_target} (has £{cash[curr_player]}).")
                    else:
                        actions.append(f"❌ [TRADE REJECTED] {ROLE_NAMES[owner_b]} couldn't afford £{-cash_offer} cash sweetener to accept {p_name_target} trade (has £{cash[owner_b]}).")
            elif best_target_pid >= 0 and best_offer_pid >= 0:
                # Proposer's desire score was below threshold
                p_name_target = space_names[best_target_pid]
                p_name_swap = space_names[best_offer_pid]
                actions.append(f"❌ [TRADE REJECTED] {p_name} considered trading {p_name_swap} for {p_name_target} but decided against proposing (score too low).") 

        # Log frame
        board_state_snap = {str(k): {"owner": ROLE_NAMES[prop_owner[k]] if prop_owner[k] >= 0 else None, "houses": int(prop_houses[k]), "mortgaged": bool(prop_mortgaged[k])} for k in range(40)}
        players_snap = [{"name": ROLE_NAMES[i], "cash": int(cash[i]), "position": int(position[i]), "in_jail": bool(in_jail[i]), "is_bankrupt": bool(is_bankrupt[i])} for i in range(4)]
        
        logs.append({
            "turn": turn_num,
            "player": p_name,
            "personality": p_name,
            "dice": [d1, d2],
            "actions": actions,
            "dialogue": dialogue,
            "explain_ai": explain_ai,
            "board_state": board_state_snap,
            "players_state": players_snap
        })
        
        # Check if winner remains
        active_counts = np.sum(~is_bankrupt)
        if active_counts <= 1:
            break
        
        # Doubles: same player rolls again (unless they went to jail this turn)
        if rolled_doubles and not in_jail[curr_player]:
            pass  # curr_player stays the same — they roll again next iteration
        else:
            doubles_count[curr_player] = 0
            curr_player = (curr_player + 1) % 4
        
    # Find winner name
    final_nw = get_py_net_worths()
    final_nw = np.where(is_bankrupt, 0, final_nw)
    winner_idx = np.argmax(final_nw)
    winner_name = ROLE_NAMES[winner_idx]
    
    return {
        "winner": winner_name,
        "log": logs
    }

# ==============================================================================
# 5. GPU-Vectorized Breeding, Crossover, Mutation, Selection & Stacking
# ==============================================================================

def init_population_weights(key, pop_size=128):
    """Initializes a population of random weights as a stacked PyTree."""
    keys = jax.random.split(key, pop_size)
    return jax.vmap(init_random_weights)(keys)

@jax.jit
def build_asymmetrical_weights(key, population, elite_pool):
    """Constructs a weights dictionary of shape (pop_size, NUM_GAMES, 4, layer_shape) on GPU.
    Player 0 is the candidate genome.
    Players 1, 2, 3 are randomly sampled from the elite pool.
    """
    pop_size = jax.tree_util.tree_leaves(population)[0].shape[0]
    elite_size = jax.tree_util.tree_leaves(elite_pool)[0].shape[0]
    
    elite_indices = jax.random.randint(key, (pop_size, NUM_GAMES, 3), 0, elite_size)
    
    pop_weights_stacked = {}
    for k in population.keys():
        cand = population[k]
        cand_expanded = jnp.expand_dims(jnp.expand_dims(cand, axis=1), axis=2)
        param_shape = cand.shape[1:]
        cand_tiled = jnp.tile(cand_expanded, (1, NUM_GAMES, 1) + (1,) * len(param_shape))
        
        elite_taken = jnp.take(elite_pool[k], elite_indices, axis=0)
        pop_weights_stacked[k] = jnp.concatenate([cand_tiled, elite_taken], axis=2)
        
    return pop_weights_stacked

@jax.jit(static_argnums=(3, 4))
def breed_next_generation(key, population, fitnesses, elite_size=12, pop_size=128):
    """Performs selection, crossover, and mutation entirely on the GPU."""
    # Sort population by fitnesses descending
    sorted_idxs = jnp.argsort(-fitnesses)
    
    # Extract elites
    elites = jax.tree_util.tree_map(lambda x: x[sorted_idxs[:elite_size]], population)
    
    # Parents pool is top 50%
    parent_pool_size = pop_size // 2
    parent_pool = jax.tree_util.tree_map(lambda x: x[sorted_idxs[:parent_pool_size]], population)
    
    # Generate keys
    k_p1, k_p2, k_cross, k_mut = jax.random.split(key, 4)
    
    child_size = pop_size - elite_size
    
    # Select parent indices from top parent_pool_size
    p1_idx = jax.random.randint(k_p1, (child_size,), 0, parent_pool_size)
    p2_idx = jax.random.randint(k_p2, (child_size,), 0, parent_pool_size)
    
    p1 = jax.tree_util.tree_map(lambda x: x[p1_idx], parent_pool)
    p2 = jax.tree_util.tree_map(lambda x: x[p2_idx], parent_pool)
    
    k_cross_split = jax.random.split(k_cross, 6)
    k_mut_split = jax.random.split(k_mut, 6)
    
    new_children = {}
    for i, name in enumerate(["w1", "b1", "w2", "b2", "w3", "b3"]):
        param_shape = population[name].shape[1:]
        
        # Bernoulli mask for uniform crossover
        mask = jax.random.bernoulli(k_cross_split[i], 0.5, (child_size,) + param_shape)
        child = jnp.where(mask, p1[name], p2[name])
        
        # Mutation noise
        noise = jax.random.normal(k_mut_split[i], (child_size,) + param_shape) * 0.03
        new_children[name] = child + noise
        
    # Combine elites and children
    new_population = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate([x, y], axis=0),
        elites,
        new_children
    )
    return new_population, elites

# ==============================================================================
# 6. Main Evolutionary Training Run Method
# ==============================================================================

def run_jax_training(generations=500, showcase_interval=1):
    print("="*60)
    print("JAX CO-EVOLUTION TRAINING ARENA (GPU ACCELERATED)")
    print("="*60)
    print("Initializing population of 128 dense MLP networks...")
    
    rng_key = jax.random.PRNGKey(42)
    pop_size = 128
    
    # Initialize population of weights entirely on GPU
    rng_key, pop_key = jax.random.split(rng_key)
    population = init_population_weights(pop_key, pop_size=pop_size)
    elite_pool = jax.tree_util.tree_map(lambda x: x[:12], population)  # Start with top 12 random genomes
    
    training_stats = []
    best_fitness_history = []
    
    print("Compilation phase: JIT compiling parallel evaluation pipeline...")
    comp_start = time.time()
    
    # Fast evaluation check
    test_game_keys = jax.random.split(rng_key, NUM_GAMES)
    rng_key, subkey_stack = jax.random.split(rng_key)
    pop_weights_stacked = build_asymmetrical_weights(subkey_stack, population, elite_pool)
    _ = evaluate_population_games(test_game_keys, pop_weights_stacked)
    
    comp_time = time.time() - comp_start
    print(f"JIT Compilation complete! Time: {comp_time:.2f} seconds.")
    print(f"Evolving {pop_size} genomes for {generations} generations...")
    print(f"Each genome is evaluated on {NUM_GAMES} parallel games ({pop_size * NUM_GAMES} total game simulations/gen).")
    print("="*60)
    
    best_weights = None
    
    for gen in range(generations):
        gen_start = time.time()
        
        # Split key for game seeds
        rng_key, subkey_game, subkey_stack, subkey_breed = jax.random.split(rng_key, 4)
        game_keys = jax.random.split(subkey_game, NUM_GAMES)
        
        # Stack population weights asymmetrically on GPU
        pop_weights_stacked = build_asymmetrical_weights(subkey_stack, population, elite_pool)
        
        # Evaluate population fitnesses in ONE vectorized call
        fitness_matrix = evaluate_population_games(game_keys, pop_weights_stacked)
        fitnesses = jnp.mean(fitness_matrix, axis=1)
        
        best_idx = int(jnp.argmax(fitnesses))
        best_fit = float(fitnesses[best_idx])
        avg_fit = float(jnp.mean(fitnesses))
        
        # Extract best weights (slice out of population PyTree)
        best_weights = jax.tree_util.tree_map(lambda x: x[best_idx], population)
        
        gen_time = time.time() - gen_start
        print(f"Generation {gen+1:03d} | Best Fitness: {best_fit:7.2f} | Avg Fitness: {avg_fit:7.2f} | Time: {gen_time:.2f}s")
        
        # Log training statistics
        training_stats.append({
            "generation": gen + 1,
            "best_fitness": best_fit,
            "avg_fitness": avg_fit
        })
        
        # Save statistics
        with open("data/jax_stats.json", "w") as f:
            json.dump(training_stats, f, indent=2)
            
        # Determine if we run showcase logs and save champion on this generation
        is_showcase_gen = ((gen + 1) % showcase_interval == 0) or (gen == generations - 1)
        
        if is_showcase_gen:
            import threading
            
            def bg_logging_and_save(best_weights_cp, elite_pool_cp, stats_cp):
                try:
                    # Convert to numpy for CPU showcase
                    np_best_weights = {k: np.array(v) for k, v in best_weights_cp.items()}
                    np_elite_pool = {k: np.array(v) for k, v in elite_pool_cp.items()}
                    
                    showcase_weights = {}
                    for k in np_best_weights.keys():
                        param_shape = np_best_weights[k].shape
                        stacked_arr = np.zeros((4,) + param_shape, dtype=np.float32)
                        stacked_arr[0] = np_best_weights[k]
                        for p in range(1, 4):
                            elite_idx = random.randint(0, np_elite_pool[k].shape[0] - 1)
                            stacked_arr[p] = np_elite_pool[k][elite_idx]
                        showcase_weights[k] = jnp.array(stacked_arr)
                        
                    showcase_log = simulate_showcase_game_py(showcase_weights)
                    with open("data/latest_game_log.json", "w") as f_log:
                        json.dump(showcase_log, f_log)
                except Exception as e:
                    print(f"\n[Background Thread] Showcase logging failed: {e}")
                    
                try:
                    save_champion_pickle("data/best_jax_agent.pkl", best_weights_cp)
                except Exception as e:
                    print(f"\n[Background Thread] Saving champion failed: {e}")
            
            # Run synchronously on the final generation to ensure files are fully written before exiting
            if gen == generations - 1:
                bg_logging_and_save(best_weights, elite_pool, training_stats.copy())
            else:
                t = threading.Thread(
                    target=bg_logging_and_save,
                    args=(best_weights, elite_pool, training_stats.copy())
                )
                t.start()
        
        # Breed next generation entirely on the GPU
        population, elite_pool = breed_next_generation(subkey_breed, population, fitnesses, elite_size=12, pop_size=pop_size)

    print("="*60)
    print("Evolution complete!")
    print("Champion weights saved to: data/best_jax_agent.pkl")
    print("Training metrics saved to: data/jax_stats.json")
    print("Latest game logs saved to: data/latest_game_log.json")
    print("="*60)


if __name__ == "__main__":
    if "--run" in sys.argv:
        os.makedirs("data", exist_ok=True)
        gens = 500
        if "--generations" in sys.argv:
            try:
                g_idx = sys.argv.index("--generations")
                gens = int(sys.argv[g_idx + 1])
            except (ValueError, IndexError):
                pass
                
        showcase_interval = 1
        # Support both --si / -si and --showcase-interval
        si_arg = None
        for flag in ["--si", "-si", "--showcase-interval"]:
            if flag in sys.argv:
                si_arg = flag
                break
        if si_arg:
            try:
                s_idx = sys.argv.index(si_arg)
                showcase_interval = int(sys.argv[s_idx + 1])
            except (ValueError, IndexError):
                pass
                
        run_jax_training(generations=gens, showcase_interval=showcase_interval)
    else:
        print("="*60)
        print("JAX MONOPOLY TRAINING ENGINE READY")
        print("="*60)
        print("To start training the JAX population, execute:")
        print("  python -m src.training.jax_train --run")
        print("  Options:")
        print("    --generations <N>         (default: 500)")
        print("    --si <N>                  (default: 5) (Showcase Interval)")
        print("="*60)
