/* src/dashboard/app.js */

// Helper to strip whitespaces and formatting newlines for clean DOM injection
function clean_html(html_str) {
    return html_str.split('\n').map(line => line.trim()).join('');
}

// Dynamic visual mappings
const PLAYER_COLORS = {
    "Player0": "#FF5E62",       // Coral Red
    "Player1": "#3B82F6",       // Royal Blue
    "Player2": "#10B981",       // Emerald Green
    "Player3": "#FBBF24",       // Amber Gold
    "Lord Sterling": "#FF5E62",
    "Lady Penelope": "#3B82F6",
    "Uncle Cecil": "#10B981",
    "ByteBot": "#FBBF24",
    "Industrialist": "#EC4899", // Soft Rose
    "Jailbird": "#8B5CF6",      // Violet
    "Scrooge": "#F97316",       // Orange
    "Flipper": "#06B6D4"        // Cyan
};

const PLAYER_TOKENS = {
    "Player0": "🎩",
    "Player1": "🚗",
    "Player2": "🐕",
    "Player3": "🪙",
    "Lord Sterling": "🎩",
    "Lady Penelope": "🚗",
    "Uncle Cecil": "🐕",
    "ByteBot": "🤖",
    "Industrialist": "🏭",
    "Jailbird": "🔒",
    "Scrooge": "💰",
    "Flipper": "🐬"
};

const GRID_POSITIONS = {
    0: { r: 11, c: 11 }, 1: { r: 11, c: 10 }, 2: { r: 11, c: 9 }, 3: { r: 11, c: 8 }, 4: { r: 11, c: 7 }, 5: { r: 11, c: 6 }, 6: { r: 11, c: 5 }, 7: { r: 11, c: 4 }, 8: { r: 11, c: 3 }, 9: { r: 11, c: 2 },
    10: { r: 11, c: 1 }, 11: { r: 10, c: 1 }, 12: { r: 9, c: 1 }, 13: { r: 8, c: 1 }, 14: { r: 7, c: 1 }, 15: { r: 6, c: 1 }, 16: { r: 5, c: 1 }, 17: { r: 4, c: 1 }, 18: { r: 3, c: 1 }, 19: { r: 2, c: 1 },
    20: { r: 1, c: 1 }, 21: { r: 1, c: 2 }, 22: { r: 1, c: 3 }, 23: { r: 1, c: 4 }, 24: { r: 1, c: 5 }, 25: { r: 1, c: 6 }, 26: { r: 1, c: 7 }, 27: { r: 1, c: 8 }, 28: { r: 1, c: 9 }, 29: { r: 1, c: 10 },
    30: { r: 1, c: 11 }, 31: { r: 2, c: 11 }, 32: { r: 3, c: 11 }, 33: { r: 4, c: 11 }, 34: { r: 5, c: 11 }, 35: { r: 6, c: 11 }, 36: { r: 7, c: 11 }, 37: { r: 8, c: 11 }, 38: { r: 9, c: 11 }, 39: { r: 10, c: 11 }
};

const COLOR_GROUP_HEX = {
    "Brown": "#8B5A2B",
    "Station": "#94a3b8",
    "Light Blue": "#89D9FF",
    "Pink": "#FF5EA6",
    "Utility": "#9C4DFF",
    "Orange": "#FF9F1C",
    "Red": "#FF3B30",
    "Yellow": "#FFD60A",
    "Green": "#2EC770",
    "Dark Blue": "#0055FF",
    "Other": "#64748B"
};

const BOARD_SPACES = [
    {id: 0, name: "Go", type: "go", color_group: null, cost: 0},
    {id: 1, name: "Old Kent Road", type: "property", color_group: "Brown", cost: 60},
    {id: 2, name: "Community Chest", type: "chest", color_group: null, cost: 0},
    {id: 3, name: "Whitechapel Road", type: "property", color_group: "Brown", cost: 60},
    {id: 4, name: "Income Tax", type: "tax", color_group: null, cost: 200},
    {id: 5, name: "Kings Cross Station", type: "station", color_group: "Station", cost: 200},
    {id: 6, name: "The Angel, Islington", type: "property", color_group: "Light Blue", cost: 100},
    {id: 7, name: "Chance", type: "chance", color_group: null, cost: 0},
    {id: 8, name: "Euston Road", type: "property", color_group: "Light Blue", cost: 100},
    {id: 9, name: "Pentonville Road", type: "property", color_group: "Light Blue", cost: 120},
    {id: 10, name: "Jail / Just Visiting", type: "jail", color_group: null, cost: 0},
    {id: 11, name: "Pall Mall", type: "property", color_group: "Pink", cost: 140},
    {id: 12, name: "Electric Company", type: "utility", color_group: "Utility", cost: 150},
    {id: 13, name: "Whitehall", type: "property", color_group: "Pink", cost: 140},
    {id: 14, name: "Northumberland Avenue", type: "property", color_group: "Pink", cost: 160},
    {id: 15, name: "Marylebone Station", type: "station", color_group: "Station", cost: 200},
    {id: 16, name: "Bow Street", type: "property", color_group: "Orange", cost: 180},
    {id: 17, name: "Community Chest", type: "chest", color_group: null, cost: 0},
    {id: 18, name: "Marlborough Street", type: "property", color_group: "Orange", cost: 180},
    {id: 19, name: "Vine Street", type: "property", color_group: "Orange", cost: 200},
    {id: 20, name: "Free Parking", type: "free_parking", color_group: null, cost: 0},
    {id: 21, name: "Strand", type: "property", color_group: "Red", cost: 220},
    {id: 22, name: "Chance", type: "chance", color_group: null, cost: 0},
    {id: 23, name: "Fleet Street", type: "property", color_group: "Red", cost: 220},
    {id: 24, name: "Trafalgar Square", type: "property", color_group: "Red", cost: 240},
    {id: 25, name: "Fenchurch St. Station", type: "station", color_group: "Station", cost: 200},
    {id: 26, name: "Leicester Square", type: "property", color_group: "Yellow", cost: 260},
    {id: 27, name: "Coventry Street", type: "property", color_group: "Yellow", cost: 260},
    {id: 28, name: "Water Works", type: "utility", color_group: "Utility", cost: 150},
    {id: 29, name: "Piccadilly", type: "property", color_group: "Yellow", cost: 280},
    {id: 30, name: "Go to Jail", type: "go_to_jail", color_group: null, cost: 0},
    {id: 31, name: "Regent Street", type: "property", color_group: "Green", cost: 300},
    {id: 32, name: "Oxford Street", type: "property", color_group: "Green", cost: 300},
    {id: 33, name: "Community Chest", type: "chest", color_group: null, cost: 0},
    {id: 34, name: "Bond Street", type: "property", color_group: "Green", cost: 320},
    {id: 35, name: "Liverpool St. Station", type: "station", color_group: "Station", cost: 200},
    {id: 36, name: "Chance", type: "chance", color_group: null, cost: 0},
    {id: 37, name: "Park Lane", type: "property", color_group: "Dark Blue", cost: 350},
    {id: 38, name: "Super Tax", type: "tax", color_group: null, cost: 100},
    {id: 39, name: "Mayfair", type: "property", color_group: "Dark Blue", cost: 400}
];

// App States
let gameData = null;
let stepIdx = 0;
let totalTurns = 0;

function getDiceUnicode(val) {
    const diceMap = {1: "⚀", 2: "⚁", 3: "⚂", 4: "⚃", 5: "⚄", 6: "⚅"};
    return diceMap[val] || val;
}

// Fetch logs and init controls
async function initApp() {
    try {
        const response = await fetch('/data/latest_game_log.json');
        if (!response.ok) throw new Error('Failed to load JSON data');
        gameData = await response.json();
        
        if (gameData && gameData.log) {
            totalTurns = gameData.log.length;
            stepIdx = 0;
            
            const slider = document.getElementById('timeline-slider');
            slider.max = totalTurns - 1;
            slider.value = 0;
            
            document.getElementById('btn-prev').addEventListener('click', () => step(stepIdx - 1));
            document.getElementById('btn-next').addEventListener('click', () => step(stepIdx + 1));
            
            slider.addEventListener('input', (e) => {
                stepIdx = parseInt(e.target.value);
                renderStep();
            });
            
            renderStep();
        }
    } catch (error) {
        console.error('Initialization error:', error);
        document.getElementById('board-wrapper').innerHTML = `<div style="color:#ef4444; padding: 20px;">Failed to load data. Run training script first!</div>`;
    }
}

function step(targetIdx) {
    if (targetIdx >= 0 && targetIdx < totalTurns) {
        stepIdx = targetIdx;
        document.getElementById('timeline-slider').value = stepIdx;
        renderStep();
    }
}

// Render loop
function renderStep() {
    if (!gameData || !gameData.log) return;
    const frame = gameData.log[stepIdx];
    
    renderBoard(frame);
    renderDialogue(frame);
    renderActionLogs(frame);
    renderPortfolios(frame);
}

// Render Grid Board
function renderBoard(frame) {
    const boardWrapper = document.getElementById('board-wrapper');
    
    const posPlayers = {};
    frame.players_state.forEach(p => {
        if (!p.is_bankrupt) {
            if (!posPlayers[p.position]) posPlayers[p.position] = [];
            posPlayers[p.position].push(p.name);
        }
    });
    
    const activePlayerName = frame.player;
    let activePlayerPos = null;
    frame.players_state.forEach(p => {
        if (p.name === activePlayerName) activePlayerPos = p.position;
    });
    const activeColor = PLAYER_COLORS[activePlayerName] || "#ffffff";
    const activeToken = PLAYER_TOKENS[activePlayerName] || "♟";
    
    // Dice Unicode render
    let diceHTML = "";
    if (frame.dice && frame.dice[0] !== 0) {
        const d1 = getDiceUnicode(frame.dice[0]);
        const d2 = getDiceUnicode(frame.dice[1]);
        diceHTML = `
            <div class="center-dice-container" style="color: ${activeColor}; text-shadow: 0 0 18px ${activeColor}55;">
                <span>${d1}</span>
                <span>${d2}</span>
            </div>`;
    } else {
        diceHTML = `
            <div class="center-dice-container" style="color: rgba(255,255,255,0.08);">
                <span>⚀</span>
                <span>⚀</span>
            </div>`;
    }
    
    // Filter money transactions containing "£", trades and rejected trades
    const moneyTransactions = frame.actions.filter(act => act.includes('£') || act.includes('[TRADE'));
    let transactionHTML = "";
    if (moneyTransactions.length > 0) {
        transactionHTML = `<div class="center-transaction-list">`;
        moneyTransactions.forEach(act => {
            const isRejected = act.includes('[TRADE REJECTED]');
            let formattedAct = act;
            Object.keys(PLAYER_COLORS).forEach(pName => {
                const hex = PLAYER_COLORS[pName];
                formattedAct = formattedAct.replaceAll(pName, `<strong style="color: ${hex};">${pName}</strong>`);
            });
            formattedAct = formattedAct.replaceAll(/£\d+/g, (match) => `<span class="transaction-money-value">${match}</span>`);
            if (isRejected) {
                transactionHTML += `<div class="center-transaction-item center-transaction-rejected">${formattedAct}</div>`;
            } else {
                transactionHTML += `<div class="center-transaction-item">${formattedAct}</div>`;
            }
        });
        transactionHTML += `</div>`;
    } else {
        transactionHTML = `<div class="center-transaction-empty">No transactions this step</div>`;
    }

    let boardContentHTML = `
    <div class="monopoly-board">
        <div class="board-center">
            <div class="board-center-glow" style="background: radial-gradient(circle, ${activeColor} 0%, transparent 70%);"></div>
            <div class="center-title">Current Player</div>
            <div class="center-active-player" style="color: ${activeColor};">
                <span class="center-player-badge" style="background-color: ${activeColor};"></span>
                ${activePlayerName}
            </div>
            ${diceHTML}
            <div class="center-transaction-container">
                ${transactionHTML}
            </div>
            <div class="center-step-info">STEP ${stepIdx + 1} / ${totalTurns}</div>
        </div>
    `;
    
    BOARD_SPACES.forEach(space => {
        const sid = space.id;
        const pos = GRID_POSITIONS[sid];
        
        const bState = frame.board_state[sid] || {owner: null, houses: 0, mortgaged: false};
        const owner = bState.owner;
        const houses = bState.houses;
        const mortgaged = bState.mortgaged;
        
        // Render color band strips
        let colorBandHTML = "";
        let paddingStyle = "padding: 3px;";
        if (space.color_group && COLOR_GROUP_HEX[space.color_group]) {
            const cgHex = COLOR_GROUP_HEX[space.color_group];
            if (pos.r === 11) {
                colorBandHTML = `<div style="position: absolute; top: 0; left: 0; height: 4.5px; width: 100%; background: ${cgHex}; border-radius: 5px 5px 0 0;"></div>`;
                paddingStyle = "padding: 7px 3px 3px 3px;";
            } else if (pos.r === 1) {
                colorBandHTML = `<div style="position: absolute; bottom: 0; left: 0; height: 4.5px; width: 100%; background: ${cgHex}; border-radius: 0 0 5px 5px;"></div>`;
                paddingStyle = "padding: 3px 3px 7px 3px;";
            } else if (pos.c === 1) {
                colorBandHTML = `<div style="position: absolute; right: 0; top: 0; width: 4.5px; height: 100%; background-color: ${cgHex}; border-radius: 0 5px 5px 0;"></div>`;
                paddingStyle = "padding: 3px 7px 3px 3px;";
            } else if (pos.c === 11) {
                colorBandHTML = `<div style="position: absolute; left: 0; top: 0; width: 4.5px; height: 100%; background-color: ${cgHex}; border-radius: 5px 0 0 5px;"></div>`;
                paddingStyle = "padding: 3px 3px 3px 7px;";
            }
        }
        
        // Ribbon pointing flag tags
        let ribbonTagHTML = "";
        if (owner) {
            const oHex = PLAYER_COLORS[owner] || "#cccccc";
            if (pos.r === 1) {
                ribbonTagHTML = `<div class="ribbon-flag ribbon-flag-top" style="background-color: ${oHex};"></div>`;
            } else if (pos.r === 11) {
                ribbonTagHTML = `<div class="ribbon-flag ribbon-flag-bottom" style="background-color: ${oHex};"></div>`;
            } else if (pos.c === 1) {
                ribbonTagHTML = `<div class="ribbon-flag ribbon-flag-left" style="background-color: ${oHex};"></div>`;
            } else if (pos.c === 11) {
                ribbonTagHTML = `<div class="ribbon-flag ribbon-flag-right" style="background-color: ${oHex};"></div>`;
            }
        }
        
        // Active position highlights
        const isActive = (activePlayerPos === sid);
        let activeGlowStyle = "";
        if (isActive) {
            const activeHex = PLAYER_COLORS[activePlayerName] || "#3b82f6";
            activeGlowStyle = `box-shadow: 0 0 15px 4px ${activeHex}, inset 0 0 4px ${activeHex} !important; border: 1.5px solid ${activeHex} !important; z-index: 90;`;
        }
        
        let sNameHTML = "";
        let cellBg = "background-color: rgba(15, 23, 42, 0.4);";
        if (["go", "jail", "free_parking", "go_to_jail"].includes(space.type)) {
            const cornerNames = {
                "Go": "GO",
                "Jail / Just Visiting": "JAIL",
                "Free Parking": "PARK",
                "Go to Jail": "TO JAIL"
            };
            sNameHTML = `<div style="font-weight: 800; font-size: 9.5px; color: #f8fafc;">${cornerNames[space.name] || space.name.toUpperCase()}</div>`;
            cellBg = "background-color: #1e293b !important;";
        } else {
            sNameHTML = "";
        }
        
        if (mortgaged) {
            cellBg = "background-color: rgba(239, 68, 68, 0.15) !important; border: 1.5px dashed rgba(239, 68, 68, 0.5) !important;";
        }
        
        // Render overlapping tokens
        let tokensHTML = "";
        const here = posPlayers[sid] || [];
        if (here.length > 0) {
            tokensHTML = `<div class="cell-tokens">`;
            here.forEach(p => {
                const tokenColor = PLAYER_COLORS[p] || "#cccccc";
                tokensHTML += `<div class="player-token-node" style="background-color: ${tokenColor};" title="${p}"></div>`;
            });
            tokensHTML += `</div>`;
        }
        
        // Render visual building indicators (dots for houses, bar for hotels)
        let houseHTML = "";
        if (houses > 0) {
            if (houses === 5) {
                houseHTML = `<span class="building-dot hotel"></span>`;
            } else {
                for (let i = 0; i < houses; i++) {
                    houseHTML += `<span class="building-dot house"></span>`;
                }
            }
        }
        let mortgageHTML = "";
        
        const costStr = "";
        
        boardContentHTML += `
        <div class="board-cell" style="
            grid-column: ${pos.c};
            grid-row: ${pos.r};
            border-radius: 6px;
            ${cellBg}
            ${paddingStyle}
            ${activeGlowStyle}
        ">
            ${colorBandHTML}
            ${ribbonTagHTML}
            ${sNameHTML}
            ${tokensHTML}
            <div style="display: flex; justify-content: space-between; width: 100%; align-items: center; font-size: 7px; color: #64748b; font-weight: 500;">
                <div class="cell-houses-container">
                    ${houseHTML}
                    ${mortgageHTML}
                </div>
                <div class="cell-cost">${costStr}</div>
            </div>
        </div>
        `;
    });
    
    boardContentHTML += `</div>`;
    boardWrapper.innerHTML = clean_html(boardContentHTML);
}

function renderDialogue(frame) {
    const box = document.getElementById('dialogue-box');
    if (frame.dialogue) {
        box.style.display = "flex";
        document.getElementById('dialogue-header').textContent = `${frame.player.toUpperCase()} SAYING:`;
        document.getElementById('dialogue-text').textContent = `"${frame.dialogue}"`;
    } else {
        box.style.display = "none";
    }
}

// Render action log list inside tooltip popup
function renderActionLogs(frame) {
    const container = document.getElementById('action-log-tooltip');
    
    let logsHTML = `
    <div class="tooltip-header">📜 TURN ACTION LOG</div>
    <div class="tooltip-log-list">
    `;
    
    frame.actions.forEach(act => {
        const isRejected = act.includes('[TRADE REJECTED]');
        const isSuccessTrade = act.includes('[TRADE]') && !isRejected;
        Object.keys(PLAYER_COLORS).forEach(pName => {
            const hex = PLAYER_COLORS[pName];
            act = act.replaceAll(pName, `<strong style="color: ${hex};">${pName}</strong>`);
        });
        if (isRejected) {
            logsHTML += `<div class="log-item log-item-rejected"><span style="color:#ef4444;">✗</span><div>${act}</div></div>`;
        } else if (isSuccessTrade) {
            logsHTML += `<div class="log-item log-item-trade"><span style="color:#10b981;">•</span><div>${act}</div></div>`;
        } else {
            logsHTML += `<div class="log-item"><span style="color:#475569;">•</span><div>${act}</div></div>`;
        }
    });
    
    logsHTML += `</div>`;
    container.innerHTML = clean_html(logsHTML);
}

// Render player portfolios accordion
function renderPortfolios(frame) {
    const accordion = document.getElementById('portfolios-accordion');
    let cardsHTML = "";
    
    const playerOwned = {};
    frame.players_state.forEach(p => {
        playerOwned[p.name] = [];
    });
    
    Object.keys(frame.board_state).forEach(sidStr => {
        const sid = parseInt(sidStr);
        const cellState = frame.board_state[sidStr];
        const owner = cellState.owner;
        if (owner && playerOwned[owner]) {
            const spaceInfo = BOARD_SPACES[sid];
            playerOwned[owner].push({
                name: spaceInfo.name,
                color_group: spaceInfo.color_group,
                mortgaged: cellState.mortgaged,
                houses: cellState.houses,
                id: sid
            });
        }
    });
    
    const activePlayerName = frame.player;
    
    frame.players_state.forEach(p => {
        const color = PLAYER_COLORS[p.name] || "#cccccc";
        const props = playerOwned[p.name] || [];
        
        let cardClasses = "player-card";
        if (p.is_bankrupt) cardClasses += " bankrupt-player";
        if (p.in_jail) cardClasses += " in-jail-player";
        
        let jailBarsHTML = "";
        if (p.in_jail) {
            jailBarsHTML = `
            <div class="jail-bars-overlay">
                <div class="jail-bars-window">
                    <div class="jail-bar"></div>
                    <div class="jail-bar"></div>
                    <div class="jail-bar"></div>
                    <div class="jail-bar"></div>
                </div>
            </div>`;
        }
        
        // Group properties that player owns
        const grouped = {};
        props.forEach(pr => {
            const cg = pr.color_group || "Other";
            if (!grouped[cg]) grouped[cg] = [];
            grouped[cg].push(pr);
        });
        
        // 1. Collapsed property grid: columns per color group, squares stack vertically in board order
        const colorOrder = ["Brown", "Light Blue", "Pink", "Orange", "Red", "Yellow", "Green", "Dark Blue", "Utility", "Station"];
        let collapsedStripsHTML = `<div class="collapsed-inventory">`;
        colorOrder.forEach(cg => {
            const cgHex = COLOR_GROUP_HEX[cg] || "#64748b";
            const groupSpaces = BOARD_SPACES.filter(s => s.color_group === cg).sort((a, b) => a.id - b.id);
            if (groupSpaces.length === 0) return;
            collapsedStripsHTML += `<div class="mini-prop-col">`;
            groupSpaces.forEach(space => {
                const cellState = frame.board_state[space.id] || {owner: null, mortgaged: false};
                const isOwned = cellState.owner === p.name;
                const isMortgaged = isOwned && cellState.mortgaged;
                if (isMortgaged) {
                    collapsedStripsHTML += `<div class="mini-prop-square mini-prop-mortgaged" style="background-color:${cgHex};" title="${space.name} (mortgaged)"></div>`;
                } else if (isOwned) {
                    collapsedStripsHTML += `<div class="mini-prop-square mini-prop-owned" style="background-color:${cgHex};" title="${space.name}"></div>`;
                } else {
                    collapsedStripsHTML += `<div class="mini-prop-square mini-prop-unowned" style="--prop-col:${cgHex}22; --prop-border:${cgHex};" title="${space.name}"></div>`;
                }
            });
            collapsedStripsHTML += `</div>`;
        });
        collapsedStripsHTML += `</div>`;
        
        // 2. Expanded properties list (Visible ONLY on Hover, shows names as tags)
        let expandedHTML = `<div class="expanded-inventory">`;
        
        // Always render all 10 colored bullet points in relative order
        colorOrder.forEach(cg => {
            const cgHex = COLOR_GROUP_HEX[cg] || "#64748b";
            let tagsHTML = "";
            
            // Check if player owns properties in this group
            const items = grouped[cg] || [];
            if (items.length > 0) {
                items.forEach(item => {
                    const isMortgaged = item.mortgaged;
                    const tagClass = isMortgaged ? "portfolio-tag mortgaged" : "portfolio-tag";
                    
                    let badgeStyle = `background-color: ${cgHex}; color: #0f172a;`;
                    if (isMortgaged) {
                        badgeStyle = ""; 
                    }
                    
                    const lockIndicator = isMortgaged ? ' 🔒' : '';
                    tagsHTML += `<span class="${tagClass}" style="${badgeStyle}">${item.name}${lockIndicator}</span>`;
                });
            }
            
            expandedHTML += `
            <div class="portfolio-group-row">
                <div class="group-color-dot" style="background-color: ${cgHex};" title="${cg}"></div>
                <div class="group-tags-list">
                    ${tagsHTML}
                </div>
            </div>
            `;
        });
        
        expandedHTML += `</div>`;
        
        const isCurrentTurn = (p.name === activePlayerName);
        let activeCardStyle = `border-left: 4px solid ${color};`;
        if (isCurrentTurn) {
            const rgb = hex_to_rgb(color);
            activeCardStyle = `border: 1px solid ${color}; border-left: 4px solid ${color}; box-shadow: 0 0 15px rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.08); background: rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.02) !important;`;
        }
        
        cardsHTML += `
        <div class="${cardClasses}" style="${activeCardStyle}">
            ${jailBarsHTML}
            <div class="card-header">
                <span class="card-player-identity">
                    <span class="player-avatar" style="background-color: ${color};"></span>
                    <span class="player-name-text">${p.name}</span>
                </span>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span class="card-cash-value" style="font-size: 15px; font-weight: 800; color: #ffffff;">£${p.cash}</span>
                </div>
            </div>
            <div class="card-cash-progress-bg">
                <div class="card-cash-progress-bar" style="width: ${Math.min(100, Math.max(0, (p.cash / 1500) * 100))}%; background-color: ${color};"></div>
            </div>
            <div class="inventory-wrapper">
                ${collapsedStripsHTML}
                ${expandedHTML}
            </div>
        </div>
        `;
    });
    
    accordion.innerHTML = clean_html(cardsHTML);
}

function hex_to_rgb(hex) {
    hex = hex.replace('#', '');
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);
    return [r, g, b];
}

window.addEventListener('DOMContentLoaded', initApp);
