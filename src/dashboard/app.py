# src/dashboard/app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.simulator.board import BOARD_SPACES

# Configure Page
st.set_page_config(page_title="🎩 AI Monopoly JAX Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom Dark glassmorphic styling
st.markdown("""
<style>
    .reportview-container {
        background: #0d1117;
    }
    .stApp {
        background-color: #0b0f19;
        color: #f0f3f8;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    .badge {
        background-color: #10b981;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Path definitions
GAME_LOG_PATH = "data/latest_game_log.json"
JAX_STATS_PATH = "data/jax_stats.json"

def load_data():
    game_log = {}
    jax_stats = []
    if os.path.exists(GAME_LOG_PATH):
        try:
            with open(GAME_LOG_PATH, "r") as f:
                game_log = json.load(f)
        except Exception:
            pass
    if os.path.exists(JAX_STATS_PATH):
        try:
            with open(JAX_STATS_PATH, "r") as f:
                jax_stats = json.load(f)
        except Exception:
            pass
    return game_log, jax_stats

game_data, jax_stats_data = load_data()

# Sidebar Layout
st.sidebar.markdown("# 🎩 JAX Monopoly")
menu = st.sidebar.radio("Navigation", ["Replay Latest Game", "Evolution Progress"])

PLAYER_COLORS = {
    "Lord Sterling": "#ff4d4d",      # Red
    "Lady Penelope": "#4da6ff",      # Blue
    "Uncle Cecil": "#4dff4d",        # Green
    "ByteBot": "#e6b800",             # Yellow
    "Industrialist": "#ff4d4d",      # Red
    "Jailbird": "#4da6ff",           # Blue
    "Scrooge": "#f97316",            # Orange
    "Flipper": "#84cc16"             # Lime
}

PLAYER_TOKENS = {
    "Lord Sterling": "🎩",
    "Lady Penelope": "🚗",
    "Uncle Cecil": "🐕",
    "ByteBot": "🤖",
    "Industrialist": "🏭",
    "Jailbird": "🔒",
    "Scrooge": "🪙",
    "Flipper": "🐬"
}

GRID_POSITIONS = {
    0: (11, 11), 1: (11, 10), 2: (11, 9), 3: (11, 8), 4: (11, 7), 5: (11, 6), 6: (11, 5), 7: (11, 4), 8: (11, 3), 9: (11, 2),
    10: (11, 1), 11: (10, 1), 12: (9, 1), 13: (8, 1), 14: (7, 1), 15: (6, 1), 16: (5, 1), 17: (4, 1), 18: (3, 1), 19: (2, 1),
    20: (1, 1), 21: (1, 2), 22: (1, 3), 23: (1, 4), 24: (1, 5), 25: (1, 6), 26: (1, 7), 27: (1, 8), 28: (1, 9), 29: (1, 10),
    30: (1, 11), 31: (2, 11), 32: (3, 11), 33: (4, 11), 34: (5, 11), 35: (6, 11), 36: (7, 11), 37: (8, 11), 38: (9, 11), 39: (10, 11)
}

COLOR_GROUP_HEX = {
    "Brown": "#8B4513",
    "Light Blue": "#87CEEB",
    "Pink": "#FF69B4",
    "Orange": "#FF8C00",
    "Red": "#FF0000",
    "Yellow": "#FFD700",
    "Green": "#008000",
    "Dark Blue": "#000080"
}

# ==============================================================================
# MENU 1: Replay Game
# ==============================================================================
if menu == "Replay Latest Game":
    if not game_data:
        st.warning("No JAX showcase game log found. Please run training script first!")
    else:
        log_entries = game_data["log"]
        total_turns = len(log_entries)
        
        if "step_idx" not in st.session_state:
            st.session_state.step_idx = 0

        def prev_step():
            if st.session_state.step_idx > 0:
                st.session_state.step_idx -= 1

        def next_step():
            if st.session_state.step_idx < total_turns - 1:
                st.session_state.step_idx += 1

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎮 Replay Controls")
        col_prev, col_next = st.sidebar.columns(2)
        with col_prev:
            st.button("⬅️ Previous", on_click=prev_step, width='stretch')
        with col_next:
            st.button("Next ➡️", on_click=next_step, width='stretch')
            
        step_idx = st.sidebar.slider("Select Turn Step", 0, total_turns - 1, key="step_idx")
        frame = log_entries[step_idx]
        
        left_col, right_col = st.columns([3, 2])
        
        player_owned_properties = {p["name"]: [] for p in frame["players_state"]}
        for sid_str, p_state in frame["board_state"].items():
            owner = p_state["owner"]
            if owner in player_owned_properties:
                space_info = BOARD_SPACES[int(sid_str)]
                player_owned_properties[owner].append({
                    "name": space_info["name"],
                    "color_group": space_info["color_group"],
                    "mortgaged": p_state["mortgaged"]
                })
        
        with left_col:
            board_html = """
            <div style="
                display: grid;
                grid-template-columns: repeat(11, 1fr);
                grid-template-rows: repeat(11, 1fr);
                width: 500px;
                height: 500px;
                background-color: #deeef5;
                border: 3px solid #1a202c;
                position: relative;
                font-family: 'Inter', sans-serif;
                margin: auto;
                box-sizing: border-box;
            ">
            """
            
            center_html = f"""
            <div style="
                grid-column: 2 / 11;
                grid-row: 2 / 11;
                background-color: #f7fafc;
                border: 1.5px dashed #cbd5e0;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 5px;
                color: #2d3748;
                font-size: 11px;
                text-align: center;
            ">
                <div style="font-size: 18px; font-weight: bold; color: #1a202c; margin-bottom: 2px;">MONOPOLY</div>
                <div style="background-color: #edf2f7; padding: 4px 8px; border-radius: 6px; width: 90%; border: 1px solid #e2e8f0; font-size: 9px;">
                    <strong>Active:</strong> {frame['player']}<br/>
                    <strong>Roll:</strong> {" + ".join(map(str, frame['dice'])) if frame['dice'] != [0,0] else "N/A"}<br/>
                    <strong>Winner Target:</strong> {game_data['winner']}
                </div>
            </div>
            """
            board_html += center_html
            
            pos_players = {}
            for p in frame["players_state"]:
                if not p["is_bankrupt"]:
                    pos = p["position"]
                    if pos not in pos_players:
                        pos_players[pos] = []
                    pos_players[pos].append(p["name"])
            
            for space in BOARD_SPACES:
                sid = space["id"]
                row, col = GRID_POSITIONS[sid]
                
                b_state = frame["board_state"].get(str(sid), {"owner": None, "houses": 0, "mortgaged": False})
                owner = b_state["owner"]
                houses = b_state["houses"]
                mortgaged = b_state["mortgaged"]
                
                color_banner = ""
                cell_padding = "padding: 1px;"
                if space["color_group"] and space["color_group"] in COLOR_GROUP_HEX:
                    cg_color = COLOR_GROUP_HEX[space["color_group"]]
                    if row == 11:
                        color_banner = f'<div style="position: absolute; top: 0; left: 0; height: 5px; width: 100%; background-color: {cg_color}; border-bottom: 0.5px solid #000;"></div>'
                        cell_padding = "padding: 7px 1px 1px 1px;"
                    elif row == 1:
                        color_banner = f'<div style="position: absolute; bottom: 0; left: 0; height: 5px; width: 100%; background-color: {cg_color}; border-top: 0.5px solid #000;"></div>'
                        cell_padding = "padding: 1px 1px 7px 1px;"
                    elif col == 1:
                        color_banner = f'<div style="position: absolute; right: 0; top: 0; width: 5px; height: 100%; background-color: {cg_color}; border-left: 0.5px solid #000;"></div>'
                        cell_padding = "padding: 1px 7px 1px 1px;"
                    elif col == 11:
                        color_banner = f'<div style="position: absolute; left: 0; top: 0; width: 5px; height: 100%; background-color: {cg_color}; border-right: 0.5px solid #000;"></div>'
                        cell_padding = "padding: 1px 1px 1px 7px;"
                
                players_here = pos_players.get(sid, [])
                tokens_html = "".join([f'<span title="{p}">{PLAYER_TOKENS.get(p, "♟")}</span>' for p in players_here])
                
                house_marker = ""
                if houses > 0:
                    if houses == 5:
                        house_marker = '<span style="color: red; font-size: 6px;">🏨</span>'
                    else:
                        house_marker = f'<span style="color: green; font-size: 6px;">{"🏠" * houses}</span>'
                
                owner_style = ""
                if owner:
                    owner_style = f"background-color: {PLAYER_COLORS.get(owner, '#cccccc')}22; border: 1.2px solid {PLAYER_COLORS.get(owner, '#cccccc')};"
                    if mortgaged:
                        owner_style += "text-decoration: line-through;"
                
                space_name_short = space["name"][:12] + ".." if len(space["name"]) > 14 else space["name"]
                
                space_html = f"""
                <div style="
                    grid-column: {col};
                    grid-row: {row};
                    border: 1px solid #cbd5e0;
                    background-color: white;
                    color: #2d3748;
                    font-size: 6px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: space-between;
                    {cell_padding}
                    box-sizing: border-box;
                    overflow: hidden;
                    position: relative;
                    {owner_style}
                ">
                    {color_banner}
                    <div style="font-weight: bold; text-align: center; line-height: 1.0;">{space_name_short}</div>
                    <div style="display: flex; gap: 1px; justify-content: center; font-size: 8px;">{tokens_html}</div>
                    <div style="display: flex; justify-content: space-between; width: 100%; font-size: 5px;">
                        <span>{house_marker}</span>
                        <span>{f'£{space["cost"]}' if space["cost"] > 0 else ''}</span>
                    </div>
                </div>
                """
                board_html += space_html
                
            board_html += "</div>"
            cleaned_board_html = "".join(line.strip() for line in board_html.split("\n"))
            st.markdown(cleaned_board_html, unsafe_allow_html=True)
            
            if frame.get("dialogue"):
                st.info(f"💬 **{frame['player']}:** \"{frame['dialogue']}\"", icon="💬")
            
        with right_col:
            st.markdown("### Player Assets & Properties")
            for p in frame["players_state"]:
                color = PLAYER_COLORS.get(p['name'], "#cccccc")
                status = "🚨 BANKRUPT" if p['is_bankrupt'] else ("🔒 IN JAIL" if p['in_jail'] else "🟢 Active")
                
                props_owned = player_owned_properties.get(p['name'], [])
                badges_list = []
                for pr in sorted(props_owned, key=lambda x: x["color_group"] or ""):
                    bg_color = COLOR_GROUP_HEX.get(pr["color_group"], "#4a5568")
                    txt_decor = "text-decoration: line-through; opacity: 0.6;" if pr["mortgaged"] else ""
                    short_name = pr["name"][:12]
                    badges_list.append(f'<span style="background-color: {bg_color}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; margin: 2px; display: inline-block; font-weight: bold; {txt_decor}">{short_name}</span>')
                
                badges_html = "".join(badges_list) if badges_list else '<span style="color: #a0aec0; font-size: 11px; font-style: italic;">No properties owned</span>'
                
                st.markdown(f"""
                <div class="card" style="border-left: 5px solid {color}; margin-bottom: 8px; padding: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 15px;">
                        <strong>{PLAYER_TOKENS.get(p['name'], '♟')} {p['name']}</strong>
                        <span style="font-size: 11px; font-weight: bold; color: {color};">{status}</span>
                    </div>
                    <div style="font-size: 13px; margin: 4px 0; color: #cbd5e0;">
                        💰 Cash: <strong style="color: #fff;">£{p['cash']}</strong>
                    </div>
                    <div style="margin-top: 4px; line-height: 1.4;">
                        {badges_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with st.expander("📜 Step Logs Detail"):
                for act in frame["actions"]:
                    st.write(f"- {act}")

# ==============================================================================
# MENU 2: Evolution Progress
# ==============================================================================
elif menu == "Evolution Progress":
    if not jax_stats_data:
        st.warning("No JAX evolutionary statistics found. Run training script to generate progress stats.")
    else:
        st.subheader("JAX Co-Evolution Fitness & Progress")
        
        df_jax = pd.DataFrame(jax_stats_data)
        
        st.markdown("### Population Fitness Across Generations")
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(x=df_jax["generation"], y=df_jax["best_fitness"], name="Best Fitness", mode="lines+markers", line=dict(color="#10b981", width=2)))
        fig_fit.add_trace(go.Scatter(x=df_jax["generation"], y=df_jax["avg_fitness"], name="Average Fitness", mode="lines+markers", line=dict(color="#3b82f6", width=2, dash="dash")))
        fig_fit.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Generation", yaxis_title="Fitness Score")
        st.plotly_chart(fig_fit, width='stretch')


