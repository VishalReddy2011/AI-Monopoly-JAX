# src/simulator/jax_game.py
import jax
import jax.numpy as jnp

# ==============================================================================
# 1. Static Board Definition Arrays
# ==============================================================================

# Space Types: 0=Go/Visiting/Parking/Jail, 1=Property, 2=Station, 3=Utility, 4=Tax, 5=Go to Jail, 6=Chance, 7=Chest
BOARD_TYPE = jnp.array([
    0, 1, 7, 1, 4, 2, 1, 6, 1, 1,  # 0-9
    0, 1, 3, 1, 1, 2, 1, 7, 1, 1,  # 10-19
    0, 1, 6, 1, 1, 2, 1, 1, 3, 1,  # 20-29
    5, 1, 1, 7, 1, 2, 6, 1, 4, 1   # 30-39
], dtype=jnp.int32)

BOARD_COST = jnp.array([
    0, 60, 0, 60, 200, 200, 100, 0, 100, 120,
    0, 140, 150, 140, 160, 200, 180, 0, 180, 200,
    0, 220, 0, 220, 240, 200, 260, 260, 150, 280,
    0, 300, 300, 0, 320, 200, 0, 350, 100, 400
], dtype=jnp.int32)

BOARD_HOUSE_COST = jnp.array([
    0, 50, 0, 50, 0, 0, 50, 0, 50, 50,
    0, 100, 0, 100, 100, 0, 100, 0, 100, 100,
    0, 150, 0, 150, 150, 0, 150, 150, 0, 150,
    0, 200, 200, 0, 200, 0, 0, 200, 0, 200
], dtype=jnp.int32)

BOARD_MORTGAGE = jnp.array([
    0, 30, 0, 30, 0, 100, 50, 0, 50, 60,
    0, 70, 75, 70, 80, 100, 90, 0, 90, 100,
    0, 110, 0, 110, 120, 100, 130, 130, 75, 140,
    0, 150, 150, 0, 160, 100, 0, 175, 0, 200
], dtype=jnp.int32)

# Color groups: -1 for non-property, 0=Brown, 1=Light Blue, 2=Pink, 3=Orange, 4=Red, 5=Yellow, 6=Green, 7=Dark Blue
BOARD_COLOR_GROUP = jnp.array([
    -1, 0, -1, 0, -1, -1, 1, -1, 1, 1,
    -1, 2, -1, 2, 2, -1, 3, -1, 3, 3,
    -1, 4, -1, 4, 4, -1, 5, 5, -1, 5,
    -1, 6, 6, -1, 6, -1, -1, 7, -1, 7
], dtype=jnp.int32)

COLOR_GROUP_SIZE = jnp.array([2, 3, 3, 3, 3, 3, 3, 2], dtype=jnp.int32)

# Ownable and property spaces indices to optimize loop sizes
PROPERTY_INDICES = jnp.array([
    1, 3, 6, 8, 9, 11, 13, 14, 16, 18, 19, 21, 23, 24, 26, 27, 29, 31, 32, 34, 37, 39
], dtype=jnp.int32)

OWNABLE_INDICES = jnp.array([
    1, 3, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 21, 23, 24, 25, 26, 27, 28, 29, 31, 32, 34, 35, 37, 39
], dtype=jnp.int32)

# Rent matrix grid (40 spaces, 7 columns: base, monopoly, 1h, 2h, 3h, 4h, hotel)
BOARD_RENT_TABLE = jnp.array([
    [0, 0, 0, 0, 0, 0, 0],
    [2, 4, 10, 30, 90, 160, 250],
    [0, 0, 0, 0, 0, 0, 0],
    [4, 8, 20, 60, 180, 320, 450],
    [0, 0, 0, 0, 0, 0, 0],
    [25, 50, 100, 200, 0, 0, 0],
    [6, 12, 30, 90, 270, 400, 550],
    [0, 0, 0, 0, 0, 0, 0],
    [6, 12, 30, 90, 270, 400, 550],
    [8, 16, 40, 100, 300, 450, 600],
    [0, 0, 0, 0, 0, 0, 0],
    [10, 20, 50, 150, 450, 625, 750],
    [0, 0, 0, 0, 0, 0, 0],
    [10, 20, 50, 150, 450, 625, 750],
    [12, 24, 60, 180, 500, 700, 900],
    [25, 50, 100, 200, 0, 0, 0],
    [14, 28, 70, 200, 550, 750, 950],
    [0, 0, 0, 0, 0, 0, 0],
    [14, 28, 70, 200, 550, 750, 950],
    [16, 32, 80, 220, 600, 800, 1000],
    [0, 0, 0, 0, 0, 0, 0],
    [18, 36, 90, 250, 700, 875, 1050],
    [0, 0, 0, 0, 0, 0, 0],
    [18, 36, 90, 250, 700, 875, 1050],
    [20, 40, 100, 300, 750, 925, 1100],
    [25, 50, 100, 200, 0, 0, 0],
    [22, 44, 110, 330, 800, 975, 1150],
    [22, 44, 110, 330, 800, 975, 1150],
    [0, 0, 0, 0, 0, 0, 0],
    [24, 48, 120, 360, 850, 1025, 1200],
    [0, 0, 0, 0, 0, 0, 0],
    [26, 52, 130, 390, 900, 1100, 1275],
    [26, 52, 130, 390, 900, 1100, 1275],
    [0, 0, 0, 0, 0, 0, 0],
    [28, 56, 150, 450, 1000, 1200, 1400],
    [25, 50, 100, 200, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [35, 70, 175, 500, 1100, 1300, 1500],
    [0, 0, 0, 0, 0, 0, 0],
    [50, 100, 200, 600, 1400, 1700, 2000]
], dtype=jnp.int32)

# ==============================================================================
# 2. MLP Forward Pass Implementation
# ==============================================================================

def mlp_forward(weights, x):
    """Computes dense forward pass with two hidden layers (64, 32) and 8 output decisions."""
    x = jnp.dot(weights["w1"], x) + weights["b1"]
    x = jax.nn.relu(x)
    x = jnp.dot(weights["w2"], x) + weights["b2"]
    x = jax.nn.relu(x)
    x = jnp.dot(weights["w3"], x) + weights["b3"]
    return jax.nn.sigmoid(x)

# ==============================================================================
# 3. Dynamic Inputs Extraction (37 Features)
# ==============================================================================

def get_net_worths(state):
    """Computes current net worths for all 4 players in JAX."""
    net_worths = state["players_cash"]
    for i in range(4):
        owner_mask = (state["properties_owner"] == i)
        prop_val = jnp.sum(jnp.where(owner_mask, BOARD_COST, 0))
        house_val = jnp.sum(jnp.where(owner_mask, state["properties_houses"] * BOARD_HOUSE_COST, 0))
        net_worths = net_worths.at[i].add(prop_val + house_val)
    return net_worths

def get_mlp_inputs(state, player_idx, prop_id, trade_info, net_worths=None):
    """Extracts the 30-element feature vector for the given player."""
    if net_worths is None:
        net_worths = get_net_worths(state)
    order_idxs = (jnp.arange(4) + player_idx) % 4
    
    cash_feats = state["players_cash"][order_idxs] / 1500.0
    nw_feats = net_worths[order_idxs] / 3000.0
    pos_feats = state["players_position"][order_idxs] / 40.0
    jail_feats = jnp.where(state["players_in_jail"][order_idxs], 1.0, 0.0)
    
    prop_cost = BOARD_COST[prop_id] / 400.0
    prop_house_cost = BOARD_HOUSE_COST[prop_id] / 200.0
    prop_houses = state["properties_houses"][prop_id] / 5.0
    prop_mortgaged = jnp.where(state["properties_mortgaged"][prop_id], 1.0, 0.0)
    
    ptype = BOARD_TYPE[prop_id]
    is_station = jnp.where(ptype == 2, 1.0, 0.0)
    is_utility = jnp.where(ptype == 3, 1.0, 0.0)
    is_property = jnp.where(ptype == 1, 1.0, 0.0)
    
    color_grp = BOARD_COLOR_GROUP[prop_id]
    in_group_mask = (BOARD_COLOR_GROUP == color_grp) & (BOARD_TYPE == 1)
    group_size = jnp.maximum(1, COLOR_GROUP_SIZE[jnp.maximum(0, color_grp)])
    owned_by_self = jnp.sum(jnp.where((state["properties_owner"] == player_idx) & in_group_mask, 1, 0))
    ownership_fraction = jnp.where(color_grp >= 0, owned_by_self / group_size, 0.0)
    
    trade_cash_diff = trade_info["cash_diff"] / 1000.0
    trade_prop_diff = trade_info["prop_val_diff"] / 1000.0
    
    opp_mask = (state["properties_owner"] != player_idx) & (state["properties_owner"] >= 0) & in_group_mask
    opp1 = jnp.sum(jnp.where((state["properties_owner"] == (player_idx+1)%4) & in_group_mask, 1, 0))
    opp2 = jnp.sum(jnp.where((state["properties_owner"] == (player_idx+2)%4) & in_group_mask, 1, 0))
    opp3 = jnp.sum(jnp.where((state["properties_owner"] == (player_idx+3)%4) & in_group_mask, 1, 0))
    opp_max_count = jnp.maximum(opp1, jnp.maximum(opp2, opp3))
    opp_set_progress = jnp.where(color_grp >= 0, opp_max_count / group_size, 0.0)
    
    is_blocker = jnp.where((color_grp >= 0) & (owned_by_self == 1) & (opp_max_count == group_size - 1), 1.0, 0.0)
    hotel_danger = 0.0
    landing_threat = 0.0
    
    scalars = jnp.array([
        prop_cost, prop_house_cost, prop_houses, prop_mortgaged,
        is_station, is_utility, is_property, ownership_fraction,
        trade_cash_diff, trade_prop_diff,
        opp_set_progress, is_blocker, hotel_danger, landing_threat
    ])
    
    return jnp.concatenate([
        cash_feats, nw_feats, pos_feats, jail_feats, scalars
    ])

def get_trade_mlp_inputs(state, player_idx, prop_id, trade_info, net_worths=None):
    """Like get_mlp_inputs but zeroes opp_set_progress (idx 26) and is_blocker (idx 27).
    Used for all trade evaluation calls to prevent the blocking Nash equilibrium."""
    feats = get_mlp_inputs(state, player_idx, prop_id, trade_info, net_worths)
    feats = feats.at[26].set(0.0)  # zero opp_set_progress
    feats = feats.at[27].set(0.0)  # zero is_blocker
    return feats

# ==============================================================================
# 4. Game Simulator Helpers
# ==============================================================================

def check_has_monopoly(state, player_idx, color_group):
    """Returns True if the player owns all properties of the given color group."""
    in_group_mask = (BOARD_COLOR_GROUP == color_group) & (BOARD_TYPE == 1)
    group_size = COLOR_GROUP_SIZE[jnp.maximum(0, color_group)]
    owned_count = jnp.sum(jnp.where((state["properties_owner"] == player_idx) & in_group_mask, 1, 0))
    return owned_count == group_size

def calculate_rent(state, prop_id, dice_sum):
    """Calculates rent due for a landing space."""
    owner = state["properties_owner"][prop_id]
    rent_is_zero = (owner == -1) | state["properties_mortgaged"][prop_id]
    ptype = BOARD_TYPE[prop_id]
    
    houses = state["properties_houses"][prop_id]
    color_group = BOARD_COLOR_GROUP[prop_id]
    has_monopoly = check_has_monopoly(state, owner, color_group)
    
    rent_idx = jnp.where(houses > 0, houses + 1, jnp.where(has_monopoly, 1, 0))
    prop_rent = BOARD_RENT_TABLE[prop_id, rent_idx]
    
    station_mask = (BOARD_TYPE == 2) & (state["properties_owner"] == owner)
    station_count = jnp.maximum(1, jnp.sum(jnp.where(station_mask, 1, 0)))
    station_rent = 25 * jnp.power(2, station_count - 1)
    
    utility_mask = (BOARD_TYPE == 3) & (state["properties_owner"] == owner)
    utility_count = jnp.sum(jnp.where(utility_mask, 1, 0))
    utility_mult = jnp.where(utility_count >= 2, 10, 4)
    utility_rent = dice_sum * utility_mult
    
    computed_rent = jnp.where(ptype == 1, prop_rent,
                              jnp.where(ptype == 2, station_rent,
                                        jnp.where(ptype == 3, utility_rent, 0)))
    return jnp.where(rent_is_zero, 0, computed_rent)

def pay_money(state, from_idx, to_idx, amount):
    """Transfers cash between players, or to the Bank (to_idx = -1)."""
    new_cash = state["players_cash"]
    new_cash = new_cash.at[from_idx].add(-amount)
    
    state_to_idx = to_idx >= 0
    to_idx_safe = jnp.maximum(0, to_idx)
    new_cash = jnp.where(state_to_idx, new_cash.at[to_idx_safe].add(amount), new_cash)
    return {**state, "players_cash": new_cash}

# ==============================================================================
# 5. JAX Card Draw Resolution
# ==============================================================================

def resolve_card_draw(state, player_idx, card_id, is_chance):
    """Vectorized resolution of Chance or Community Chest cards in JAX."""
    old_pos = state["players_position"][player_idx]
    new_cash = state["players_cash"]
    new_pos = state["players_position"]
    new_in_jail = state["players_in_jail"]
    new_cards = state["players_get_out_of_jail_cards"]
    
    c1_pos = 0
    c1_cash = new_cash.at[player_idx].add(200)
    
    c2_pos = 24
    c2_cash = jnp.where(old_pos > 24, new_cash.at[player_idx].add(200), new_cash)
    
    c3_cash = new_cash.at[player_idx].add(-50)
    
    c4_pos = 10
    c4_in_jail = new_in_jail.at[player_idx].set(True)
    
    c5_cards = new_cards.at[player_idx].add(1)
    
    c6_cash = new_cash.at[player_idx].add(100)
    c7_cash = new_cash.at[player_idx].add(50)
    
    pos_res = jnp.where(card_id == 1, c1_pos,
              jnp.where(card_id == 2, c2_pos,
              jnp.where(card_id == 4, c4_pos, old_pos)))
              
    cash_res = jnp.where(card_id == 1, c1_cash,
               jnp.where(card_id == 2, c2_cash,
               jnp.where(card_id == 3, c3_cash,
               jnp.where(card_id == 6, c6_cash,
               jnp.where(card_id == 7, c7_cash, new_cash)))))
               
    injail_res = jnp.where(card_id == 4, c4_in_jail, new_in_jail)
    cards_res = jnp.where(card_id == 5, c5_cards, new_cards)
    
    return {
        **state,
        "players_cash": cash_res,
        "players_position": new_pos.at[player_idx].set(pos_res),
        "players_in_jail": injail_res,
        "players_get_out_of_jail_cards": cards_res
    }

# ==============================================================================
# 6. Auction Resolution
# ==============================================================================

def resolve_auction(state, prop_id, mlp_weights):
    """Resolves an auction when a player declines to buy a landed property."""
    cost = BOARD_COST[prop_id]
    trade_info = {"cash_diff": 0.0, "prop_val_diff": 0.0}
    def get_player_bid(i):
        active = ~state["players_is_bankrupt"][i]
        inputs = get_mlp_inputs(state, i, prop_id, trade_info)
        player_weights = jax.tree_util.tree_map(lambda x: x[i], mlp_weights)
        outputs = mlp_forward(player_weights, inputs)
        bid_factor = 0.1 + 1.4 * outputs[6]
        max_bid = (cost * bid_factor).astype(jnp.int32)
        max_bid = jnp.minimum(state["players_cash"][i], max_bid)
        return jnp.where(active & (max_bid > 0), max_bid, 0)
        
    bids = jax.vmap(get_player_bid)(jnp.arange(4))
        
    highest_bid = jnp.max(bids)
    
    # Break ties randomly by adding tiny random noise to the bids
    rng_key, subkey = jax.random.split(state["rng_key"])
    noise = jax.random.uniform(subkey, (4,), minval=0.0, maxval=0.01)
    bids_with_noise = bids.astype(jnp.float32) + noise
    winner_idx = jnp.argmax(bids_with_noise)
    has_bidder = (highest_bid > 0)
    
    new_owner = jnp.where(has_bidder, state["properties_owner"].at[prop_id].set(winner_idx), state["properties_owner"])
    new_cash = jnp.where(has_bidder, state["players_cash"].at[winner_idx].add(-highest_bid), state["players_cash"])
    
    return {
        **state,
        "properties_owner": new_owner,
        "players_cash": new_cash,
        "rng_key": rng_key
    }

# ==============================================================================
# 7. Optimized JAX-Native Liquidity Crisis Resolver
# ==============================================================================

def resolve_liquidity_crisis(state, player_idx, needed_cash):
    """Sells houses and mortgages properties using fast JAX loop constructs."""
    
    # 1. Sell houses loop
    def sell_houses_body(idx, loop_state):
        pid = PROPERTY_INDICES[idx]
        houses, cash, mortgaged = loop_state
        own_and_has_houses = (state["properties_owner"][pid] == player_idx) & (houses[pid] > 0) & (cash[player_idx] < needed_cash)
        refund_value = BOARD_HOUSE_COST[pid] // 2
        
        h_to_sell = jnp.minimum(houses[pid], jnp.maximum(0, (needed_cash - cash[player_idx] + refund_value - 1) // refund_value))
        h_to_sell = jnp.where(own_and_has_houses, h_to_sell, 0)
        
        new_houses = houses.at[pid].add(-h_to_sell)
        new_cash = cash.at[player_idx].add(h_to_sell * refund_value)
        return (new_houses, new_cash, mortgaged)
 
    init_sell = (state["properties_houses"], state["players_cash"], state["properties_mortgaged"])
    houses, new_cash, mortgaged = jax.lax.fori_loop(0, 22, sell_houses_body, init_sell)
 
    # 2. Mortgage properties loop
    def mortgage_body(idx, loop_state):
        pid = OWNABLE_INDICES[idx]
        houses_val, cash_val, mortgaged_val = loop_state
        can_mortgage = (state["properties_owner"][pid] == player_idx) & (~mortgaged_val[pid]) & (houses_val[pid] == 0) & (cash_val[player_idx] < needed_cash)
        
        new_mortgaged = jnp.where(can_mortgage, mortgaged_val.at[pid].set(True), mortgaged_val)
        new_cash = jnp.where(can_mortgage, cash_val.at[player_idx].add(BOARD_MORTGAGE[pid]), cash_val)
        return (houses_val, new_cash, new_mortgaged)
 
    init_mort = (houses, new_cash, mortgaged)
    houses, new_cash, mortgaged = jax.lax.fori_loop(0, 28, mortgage_body, init_mort)
 
    return {
        **state,
        "players_cash": new_cash,
        "properties_houses": houses,
        "properties_mortgaged": mortgaged
    }

# ==============================================================================
# 8. Single Turn Step (State Machine)
# ==============================================================================

def resolve_trading_phase(state, mlp_weights, p_idx):
    """Allows Player A (p_idx) to evaluate target properties and offer one of their own properties with cash adjustment."""
    
    # 1. Evaluate Target Properties (properties owned by opponents)
    def evaluate_target_prop(pid):
        owner = state["properties_owner"][pid]
        is_target = (owner >= 0) & (owner != p_idx) & (~state["players_is_bankrupt"][owner])
        no_houses = (state["properties_houses"][pid] == 0)
        not_mortgaged = ~state["properties_mortgaged"][pid]
        valid_target = is_target & no_houses & not_mortgaged
        
        weights_a = jax.tree_util.tree_map(lambda x: x[p_idx], mlp_weights)
        trade_info = {"cash_diff": 0.0, "prop_val_diff": 0.0}
        inputs = get_trade_mlp_inputs(state, p_idx, pid, trade_info)  # blocker-masked
        outputs = mlp_forward(weights_a, inputs)
        return jnp.where(valid_target, outputs[4], -1.0)
        
    target_desirabilities = jax.vmap(evaluate_target_prop)(OWNABLE_INDICES)
    best_target_idx = jnp.argmax(target_desirabilities)
    best_target_score = target_desirabilities[best_target_idx]
    target_pid = OWNABLE_INDICES[best_target_idx]
    target_pid_safe = jnp.maximum(0, target_pid)
    
    # 2. Evaluate Offer Properties (properties owned by proposer)
    def evaluate_offer_prop(pid):
        is_owned_by_self = (state["properties_owner"][pid] == p_idx)
        no_houses = (state["properties_houses"][pid] == 0)
        not_mortgaged = ~state["properties_mortgaged"][pid]
        valid_offer = is_owned_by_self & no_houses & not_mortgaged
        
        weights_a = jax.tree_util.tree_map(lambda x: x[p_idx], mlp_weights)
        trade_info = {"cash_diff": 0.0, "prop_val_diff": 0.0}
        inputs = get_trade_mlp_inputs(state, p_idx, pid, trade_info)  # blocker-masked
        outputs = mlp_forward(weights_a, inputs)
        return jnp.where(valid_offer, outputs[5], -1.0)
        
    offer_desirabilities = jax.vmap(evaluate_offer_prop)(OWNABLE_INDICES)
    best_offer_idx = jnp.argmax(offer_desirabilities)
    best_offer_score = offer_desirabilities[best_offer_idx]
    swap_pid = OWNABLE_INDICES[best_offer_idx]
    swap_pid_safe = jnp.maximum(0, swap_pid)
    
    # 3. Calculate Cash Offer (proposer's network Output 6 determines multiplier)
    proposer_weights = jax.tree_util.tree_map(lambda x: x[p_idx], mlp_weights)
    trade_info_propose = {"cash_diff": 0.0, "prop_val_diff": 0.0}
    inputs_propose = get_trade_mlp_inputs(state, p_idx, target_pid_safe, trade_info_propose)  # blocker-masked
    proposer_outputs = mlp_forward(proposer_weights, inputs_propose)
    
    cost_target = BOARD_COST[target_pid_safe]
    
    has_valid_offer = (best_offer_score > -0.5)
    cost_swap = jnp.where(has_valid_offer, BOARD_COST[swap_pid_safe], 0)
    cost_diff = cost_target - cost_swap
    
    cash_offer = (cost_diff + cost_target * (proposer_outputs[6] - 0.5) * 3.0).astype(jnp.int32)
    
    owner_b = state["properties_owner"][target_pid_safe]
    owner_b_safe = jnp.maximum(0, owner_b)
    
    # 4. Financial Validity
    proposer_has_cash = state["players_cash"][p_idx] >= cash_offer
    owner_has_cash = state["players_cash"][owner_b_safe] >= -cash_offer
    cash_ok = jnp.where(cash_offer >= 0, proposer_has_cash, owner_has_cash)
    
    # 5. Receiver evaluates with Output 7 (dedicated acceptance output, blocker-masked inputs)
    owner_weights = jax.tree_util.tree_map(lambda x: x[owner_b_safe], mlp_weights)
    trade_info_evaluate = {
        "cash_diff": cash_offer.astype(jnp.float32), 
        "prop_val_diff": (cost_swap - cost_target).astype(jnp.float32)
    }
    inputs_evaluate = get_trade_mlp_inputs(state, owner_b_safe, target_pid_safe, trade_info_evaluate)
    owner_outputs = mlp_forward(owner_weights, inputs_evaluate)
    
    # Stochastic acceptance: receiver accepts with probability = output[7]
    # This prevents deterministic blocking and allows trade exploration during training
    rng_key, subkey = jax.random.split(state["rng_key"])
    accepts = (jax.random.uniform(subkey) < owner_outputs[7])
    
    # 6. Execute Trade
    has_valid_target = (best_target_score > 0.5)
    execute_trade = has_valid_target & has_valid_offer & cash_ok & accepts
    
    new_owners = state["properties_owner"]
    new_owners = jnp.where(execute_trade, new_owners.at[target_pid_safe].set(p_idx), new_owners)
    new_owners = jnp.where(execute_trade, new_owners.at[swap_pid_safe].set(owner_b_safe), new_owners)
    
    new_cash = state["players_cash"]
    new_cash = jnp.where(execute_trade, new_cash.at[p_idx].add(-cash_offer), new_cash)
    new_cash = jnp.where(execute_trade, new_cash.at[owner_b_safe].add(cash_offer), new_cash)
    
    return {
        **state,
        "properties_owner": new_owners,
        "players_cash": new_cash,
        "rng_key": rng_key
    }


def game_step(state, mlp_weights):
    """Simulates a single player's turn in JAX, using fast JAX loops."""
    p_idx = state["current_player_idx"]
    is_bankrupt = state["players_is_bankrupt"][p_idx]
    
    # Generate dice roll keys upfront
    rng_key, subkey1, subkey2 = jax.random.split(state["rng_key"], 3)
    
    def bankrupt_turn(s):
        # Just advance turn index
        next_player = (p_idx + 1) % 4
        return {
            **s,
            "current_player_idx": next_player,
            "turn_number": s["turn_number"] + 1,
            "rng_key": rng_key
        }
        
    def active_turn(s):
        # Slice weights for current active player
        curr_player_weights = jax.tree_util.tree_map(lambda x: x[p_idx], mlp_weights)
        
        d1 = jax.random.randint(subkey1, (), 1, 7)
        d2 = jax.random.randint(subkey2, (), 1, 7)
        dice_sum = d1 + d2
        
        # Check Jail Escape (MLP Gating)
        in_jail = s["players_in_jail"][p_idx]
        has_card = s["players_get_out_of_jail_cards"][p_idx] > 0
        
        def handle_jail_escape(state_arg):
            trade_info = {"cash_diff": 0.0, "prop_val_diff": 0.0}
            inputs = get_mlp_inputs(state_arg, p_idx, state_arg["players_position"][p_idx], trade_info)
            outputs = mlp_forward(curr_player_weights, inputs)
            
            escaped_by_roll = (d1 == d2)
            escaped_by_card = has_card & (outputs[3] > 0.5)
            escaped_by_pay = (~escaped_by_card) & (state_arg["players_cash"][p_idx] >= 50) & (outputs[2] > 0.5)
            
            escaped = escaped_by_roll | escaped_by_card | escaped_by_pay
            new_in_jail = ~escaped
            new_cards = jnp.where(escaped_by_card, state_arg["players_get_out_of_jail_cards"].at[p_idx].add(-1), state_arg["players_get_out_of_jail_cards"])
            new_cash = jnp.where(escaped_by_pay, state_arg["players_cash"].at[p_idx].add(-50), state_arg["players_cash"])
            
            return {
                **state_arg,
                "players_in_jail": state_arg["players_in_jail"].at[p_idx].set(new_in_jail),
                "players_get_out_of_jail_cards": new_cards,
                "players_cash": new_cash
            }
            
        temp_state = jax.lax.cond(
            in_jail,
            handle_jail_escape,
            lambda state_arg: state_arg,
            s
        )
        
        can_move = ~temp_state["players_in_jail"][p_idx]
        old_pos = temp_state["players_position"][p_idx]
        new_pos = jnp.where(can_move, (old_pos + dice_sum) % 40, old_pos)
        
        passed_go = can_move & (new_pos < old_pos)
        temp_state["players_cash"] = jnp.where(passed_go, temp_state["players_cash"].at[p_idx].add(200), temp_state["players_cash"])
        temp_state["players_position"] = temp_state["players_position"].at[p_idx].set(new_pos)
        
        # LANDING RESOLUTION
        landing_pos = temp_state["players_position"][p_idx]
        ptype = BOARD_TYPE[landing_pos]
        owner = temp_state["properties_owner"][landing_pos]
        cost = BOARD_COST[landing_pos]
        
        # Unowned Property Buy (MLP Gating)
        is_unowned_purchasable = (owner == -1) & ((ptype == 1) | (ptype == 2) | (ptype == 3))
        
        def decide_buy(state_arg):
            trade_info = {"cash_diff": 0.0, "prop_val_diff": 0.0}
            buy_inputs = get_mlp_inputs(state_arg, p_idx, landing_pos, trade_info)
            buy_outputs = mlp_forward(curr_player_weights, buy_inputs)
            wants_to_buy = (buy_outputs[0] > 0.5) & (state_arg["players_cash"][p_idx] >= cost)
            
            new_owner = jnp.where(wants_to_buy, state_arg["properties_owner"].at[landing_pos].set(p_idx), state_arg["properties_owner"])
            new_cash = jnp.where(wants_to_buy, state_arg["players_cash"].at[p_idx].add(-cost), state_arg["players_cash"])
            
            state_after_buy = {
                **state_arg,
                "properties_owner": new_owner,
                "players_cash": new_cash
            }
            
            # Run Auction
            return jax.lax.cond(
                wants_to_buy,
                lambda state_s: state_s,
                lambda state_s: resolve_auction(state_s, landing_pos, mlp_weights),
                state_after_buy
            )
            
        temp_state = jax.lax.cond(
            is_unowned_purchasable,
            decide_buy,
            lambda state_arg: state_arg,
            temp_state
        )
        
        # Pay Rent
        is_rent_due = (temp_state["properties_owner"][landing_pos] >= 0) & (temp_state["properties_owner"][landing_pos] != p_idx) & ((ptype == 1) | (ptype == 2) | (ptype == 3))
        rent_amount = calculate_rent(temp_state, landing_pos, dice_sum)
        rent_amount = jnp.where(is_rent_due, rent_amount, 0)
        temp_state = pay_money(temp_state, p_idx, temp_state["properties_owner"][landing_pos], rent_amount)
        
        # Tax Space
        is_tax = (ptype == 4)
        tax_amount = jnp.where(landing_pos == 4, 200, jnp.where(landing_pos == 38, 100, 0))
        tax_amount = jnp.where(is_tax, tax_amount, 0)
        temp_state["players_cash"] = temp_state["players_cash"].at[p_idx].add(-tax_amount)
        
        # Go to jail space
        is_go_to_jail = (ptype == 5)
        temp_state["players_in_jail"] = jnp.where(is_go_to_jail, temp_state["players_in_jail"].at[p_idx].set(True), temp_state["players_in_jail"])
        temp_state["players_position"] = jnp.where(is_go_to_jail, temp_state["players_position"].at[p_idx].set(10), temp_state["players_position"])
        
        # Draw Cards
        is_chance = (ptype == 6)
        is_chest = (ptype == 7)
        card_id = (1 + (s["turn_number"] % 8)).astype(jnp.int32)
        temp_state = jax.lax.cond(
            is_chance | is_chest,
            lambda state_s: resolve_card_draw(state_s, p_idx, card_id, is_chance),
            lambda state_s: state_s,
            temp_state
        )
        
        # Liquidity Crisis
        insolvent = temp_state["players_cash"][p_idx] < 0
        temp_state = jax.lax.cond(
            insolvent,
            lambda state_s: resolve_liquidity_crisis(state_s, p_idx, -state_s["players_cash"][p_idx]),
            lambda state_s: state_s,
            temp_state
        )
        
        # Bankruptcy
        still_bankrupt = temp_state["players_cash"][p_idx] < 0
        became_bankrupt = still_bankrupt & (~s["players_is_bankrupt"][p_idx])
        temp_state["players_is_bankrupt"] = jnp.where(still_bankrupt, temp_state["players_is_bankrupt"].at[p_idx].set(True), temp_state["players_is_bankrupt"])
        
        num_already_bankrupt = jnp.sum(s["players_is_bankrupt"])
        new_order = jnp.where(became_bankrupt, num_already_bankrupt + 1, s["players_bankruptcy_order"][p_idx])
        temp_state["players_bankruptcy_order"] = s["players_bankruptcy_order"].at[p_idx].set(new_order)
        
        num_bankrupt = jnp.sum(temp_state["players_is_bankrupt"])
        is_over = num_bankrupt >= 3
        first_time_over = is_over & (s["game_end_turn"] == 0)
        temp_state["game_end_turn"] = jnp.where(first_time_over, s["turn_number"], s["game_end_turn"])
        
        # Record the turn when the FIRST bankruptcy occurs (bankruptcy_turn stays 0 until first hit)
        first_bankruptcy = still_bankrupt & (temp_state["bankruptcy_turn"] == 0)
        temp_state["bankruptcy_turn"] = jnp.where(first_bankruptcy, s["turn_number"], temp_state["bankruptcy_turn"])
        
        prop_mask_self = (temp_state["properties_owner"] == p_idx)
        creditor = jnp.where(is_rent_due, temp_state["properties_owner"][landing_pos], -1)
        temp_state["properties_owner"] = jnp.where(still_bankrupt & prop_mask_self, creditor, temp_state["properties_owner"])
        temp_state["properties_houses"] = jnp.where(still_bankrupt & prop_mask_self, 0, temp_state["properties_houses"])
        temp_state["properties_mortgaged"] = jnp.where(still_bankrupt & prop_mask_self & (creditor == -1), False, temp_state["properties_mortgaged"])
        temp_state["players_cash"] = jnp.where(still_bankrupt, temp_state["players_cash"].at[p_idx].set(0), temp_state["players_cash"])
        
        # Precompute Net Worths once before building/unmortgaging loops
        net_worths = get_net_worths(temp_state)
        trade_info = {"cash_diff": 0.0, "prop_val_diff": 0.0}
        
        # Vectorized Building Decisions (evaluate all 22 properties in parallel)
        def eval_build_mlp(pid):
            grp = BOARD_COLOR_GROUP[pid]
            owns_monopoly = check_has_monopoly(temp_state, p_idx, grp)
            not_mortgaged = ~temp_state["properties_mortgaged"][pid]
            h_cost = BOARD_HOUSE_COST[pid]
            
            group_pids = (BOARD_COLOR_GROUP == grp) & (BOARD_TYPE == 1)
            min_houses_in_group = jnp.min(jnp.where(group_pids, temp_state["properties_houses"], 5))
            is_even = (temp_state["properties_houses"][pid] == min_houses_in_group)
            
            can_build_potential = owns_monopoly & not_mortgaged & is_even & (temp_state["properties_houses"][pid] < 5) & (temp_state["players_cash"][p_idx] >= h_cost)
            
            return jax.lax.cond(
                can_build_potential,
                lambda _: mlp_forward(curr_player_weights, get_mlp_inputs(temp_state, p_idx, pid, trade_info, net_worths))[1] > 0.5,
                lambda _: False,
                None
            )
            
        wants_build_all = jax.vmap(eval_build_mlp)(PROPERTY_INDICES)
        
        # Apply house building sequentially without any MLP calls (very fast)
        def build_apply_body(idx, state_arg):
            pid = PROPERTY_INDICES[idx]
            wants_build = wants_build_all[idx]
            grp = BOARD_COLOR_GROUP[pid]
            owns_monopoly = check_has_monopoly(state_arg, p_idx, grp)
            not_mortgaged = ~state_arg["properties_mortgaged"][pid]
            h_cost = BOARD_HOUSE_COST[pid]
            
            group_pids = (BOARD_COLOR_GROUP == grp) & (BOARD_TYPE == 1)
            min_houses_in_group = jnp.min(jnp.where(group_pids, state_arg["properties_houses"], 5))
            is_even = (state_arg["properties_houses"][pid] == min_houses_in_group)
            
            can_build = owns_monopoly & not_mortgaged & wants_build & is_even & (state_arg["properties_houses"][pid] < 5) & (state_arg["players_cash"][p_idx] >= h_cost)
            
            new_houses = jnp.where(can_build, state_arg["properties_houses"].at[pid].add(1), state_arg["properties_houses"])
            new_cash = jnp.where(can_build, state_arg["players_cash"].at[p_idx].add(-h_cost), state_arg["players_cash"])
            return {
                **state_arg,
                "properties_houses": new_houses,
                "players_cash": new_cash
            }
            
        # Vectorized Unmortgaging Decisions (evaluate all 28 ownables in parallel)
        def eval_unmortgage_mlp(pid):
            is_mortgaged = temp_state["properties_mortgaged"][pid]
            owns_prop = (temp_state["properties_owner"][pid] == p_idx)
            cost_unmortgage = (BOARD_MORTGAGE[pid] * 1.1).astype(jnp.int32)
            
            can_unmortgage_potential = is_mortgaged & owns_prop & (temp_state["players_cash"][p_idx] >= cost_unmortgage)
            
            return jax.lax.cond(
                can_unmortgage_potential,
                lambda _: mlp_forward(curr_player_weights, get_mlp_inputs(temp_state, p_idx, pid, trade_info, net_worths))[3] > 0.5,
                lambda _: False,
                None
            )
            
        wants_unmortgage_all = jax.vmap(eval_unmortgage_mlp)(OWNABLE_INDICES)
        
        # Apply unmortgage sequentially without any MLP calls (very fast)
        def unmortgage_apply_body(idx, state_arg):
            pid = OWNABLE_INDICES[idx]
            wants_unmortgage = wants_unmortgage_all[idx]
            is_mortgaged = state_arg["properties_mortgaged"][pid]
            owns_prop = (state_arg["properties_owner"][pid] == p_idx)
            cost_unmortgage = (BOARD_MORTGAGE[pid] * 1.1).astype(jnp.int32)
            
            can_unmortgage = is_mortgaged & owns_prop & wants_unmortgage & (state_arg["players_cash"][p_idx] >= cost_unmortgage)
            
            new_mortgaged = jnp.where(can_unmortgage, state_arg["properties_mortgaged"].at[pid].set(False), state_arg["properties_mortgaged"])
            new_cash = jnp.where(can_unmortgage, state_arg["players_cash"].at[p_idx].add(-cost_unmortgage), state_arg["players_cash"])
            return {
                **state_arg,
                "properties_mortgaged": new_mortgaged,
                "players_cash": new_cash
            }
            
        # Conditionally run building loop if not bankrupt
        temp_state = jax.lax.cond(
            ~temp_state["players_is_bankrupt"][p_idx],
            lambda state_arg: jax.lax.fori_loop(0, 22, build_apply_body, state_arg),
            lambda state_arg: state_arg,
            temp_state
        )
        
        # Conditionally run unmortgage loop if not bankrupt
        temp_state = jax.lax.cond(
            ~temp_state["players_is_bankrupt"][p_idx],
            lambda state_arg: jax.lax.fori_loop(0, 28, unmortgage_apply_body, state_arg),
            lambda state_arg: state_arg,
            temp_state
        )
        
        # Conditionally run trading phase if not bankrupt
        temp_state = jax.lax.cond(
            ~temp_state["players_is_bankrupt"][p_idx],
            lambda state_arg: resolve_trading_phase(state_arg, mlp_weights, p_idx),
            lambda state_arg: state_arg,
            temp_state
        )
        
        next_player = (p_idx + 1) % 4
        return {
            **temp_state,
            "current_player_idx": next_player,
            "turn_number": temp_state["turn_number"] + 1,
            "rng_key": rng_key
        }
        
    return jax.lax.cond(
        is_bankrupt,
        bankrupt_turn,
        active_turn,
        state
    )

# ==============================================================================
# 9. Play Full Game Scan Loop
# ==============================================================================

def play_game_scan(rng_key, mlp_weights, max_turns=600):
    """Runs a full simulation loop for max_turns using jax.lax.scan.
    Only final_state is returned — trajectory is not accumulated to save memory.
    """
    key_deck1, key_deck2, key_game = jax.random.split(rng_key, 3)
    
    init_state = {
        "players_cash": jnp.array([1500, 1500, 1500, 1500], dtype=jnp.int32),
        "players_position": jnp.array([0, 0, 0, 0], dtype=jnp.int32),
        "players_in_jail": jnp.array([False, False, False, False], dtype=jnp.bool_),
        "players_jail_turns": jnp.array([0, 0, 0, 0], dtype=jnp.int32),
        "players_get_out_of_jail_cards": jnp.array([0, 0, 0, 0], dtype=jnp.int32),
        "players_is_bankrupt": jnp.array([False, False, False, False], dtype=jnp.bool_),
        "properties_owner": jnp.full(40, -1, dtype=jnp.int32),
        "properties_houses": jnp.zeros(40, dtype=jnp.int32),
        "properties_mortgaged": jnp.zeros(40, dtype=jnp.bool_),
        "chance_deck": jax.random.permutation(key_deck1, jnp.arange(1, 17, dtype=jnp.int32)),
        "chance_deck_ptr": jnp.zeros((), dtype=jnp.int32),
        "community_chest_deck": jax.random.permutation(key_deck2, jnp.arange(1, 17, dtype=jnp.int32)),
        "community_chest_deck_ptr": jnp.zeros((), dtype=jnp.int32),
        "turn_number": jnp.zeros((), dtype=jnp.int32),
        "bankruptcy_turn": jnp.zeros((), dtype=jnp.int32),  # 0 = no bankruptcy yet
        "players_bankruptcy_order": jnp.zeros(4, dtype=jnp.int32),
        "game_end_turn": jnp.zeros((), dtype=jnp.int32),
        "current_player_idx": jnp.zeros((), dtype=jnp.int32),
        "rng_key": key_game
    }
    
    def scan_step(state, _):
        next_state = game_step(state, mlp_weights)
        return next_state, None  # Don't accumulate trajectory — saves huge memory
        
    final_state, _ = jax.lax.scan(scan_step, init_state, None, length=max_turns)
    return final_state
