import streamlit as st
import pandas as pd
import pulp
import altair as alt
from music21 import note

# ==========================================
# 1. DATA & METRIC UTILITIES
# ==========================================
@st.cache_data
def load_trombone_data(file_path, horn_type):
    df = pd.read_csv(file_path)
    df.fillna(0, inplace=True) 
    
    # Logic: Absolute Physical Distance = Position + Stretch + Tuning
    df['Phys_Dist'] = df['Position'] + df['Physical_Offset'] + df['Partial_Offset']
    
    # FIX 1: Explicitly map the UI dropdown to your new CSV columns
    col_map = {
        "Straight Tenor": "Basic_Straight", 
        "Symphony Tenor": "Basic_Symphony"
    }
    basic_col = col_map.get(horn_type, "Basic_Straight")
    
    note_map = {}
    basic_map = {}
    
    for _, row in df.iterrows():
        midi = int(row['MIDI Number'])
        
        # Check if it's a basic position for the selected horn (Default to 0 if column is missing)
        is_b = int(row[basic_col]) if basic_col in df.columns else 0
        
        if midi not in note_map:
            note_map[midi] = []
            
        # We now pack 5 items into the tuple, including the is_b flag for the solver!
        note_map[midi].append((int(row['Valve_ID']), row['Phys_Dist'], int(row['Position']), row['Note (Visual Label)'], is_b))
        
        # Build the chart reference map
        if is_b == 1:
            if midi in basic_map:
                # If there are multiple basic positions, strictly keep the one closest to closed horn (shortest distance)
                if row['Phys_Dist'] < basic_map[midi]:
                    basic_map[midi] = row['Phys_Dist']
            else:
                basic_map[midi] = row['Phys_Dist']
                
    return note_map, basic_map

def calculate_metrics(path):
    """Calculates total energy (distance) and complexity (turns)."""
    total_dist = 0
    direction_changes = 0
    last_dir = 0 # 1=Out, -1=In
    
    for i in range(1, len(path)):
        diff = path[i] - path[i-1]
        total_dist += abs(diff)
        
        if diff > 0: current_dir = 1
        elif diff < 0: current_dir = -1
        else: current_dir = last_dir
            
        if last_dir != 0 and current_dir != last_dir:
            direction_changes += 1
        last_dir = current_dir
            
    return round(total_dist, 2), direction_changes

# ==========================================
# 2. THE OPTIMIZATION ENGINE
# ==========================================
def solve_with_leap_logic(midi_sequence, forced_flags, trombone_notes, basic_notes, valve_pen, leap_pen, trigger_pen, alt_pen):
    valid_sequence = [m for m in midi_sequence if m in trombone_notes]
    n = len(valid_sequence)
    if n == 0: return []

    LEAP_THRESHOLD = 3.5 
    prob = pulp.LpProblem("Trombone_Optimizer", pulp.LpMinimize)

    # Decision Variables
    x = pulp.LpVariable.dicts("x", ((i, t, p_dist) for i in range(n) for t, p_dist, p_int, label, is_b in trombone_notes[valid_sequence[i]]), cat=pulp.LpBinary)
    dist = pulp.LpVariable.dicts("dist", range(n), lowBound=1, upBound=9)
    t_state = pulp.LpVariable.dicts("t_state", range(n), cat=pulp.LpBinary)
    u = pulp.LpVariable.dicts("out", range(1, n), cat=pulp.LpBinary)
    v = pulp.LpVariable.dicts("in", range(1, n), cat=pulp.LpBinary)
    w = pulp.LpVariable.dicts("dir_change", range(1, n-1), cat=pulp.LpBinary)
    z = pulp.LpVariable.dicts("valve", range(1, n), cat=pulp.LpBinary)
    abs_jump = pulp.LpVariable.dicts("abs_jump", range(1, n), lowBound=0)
    is_big_leap = pulp.LpVariable.dicts("is_big_leap", range(1, n), cat=pulp.LpBinary)
    
    # NEW Variable: Flag if the selected position is an alternate
    is_alt = pulp.LpVariable.dicts("is_alt", range(n), cat=pulp.LpBinary)

    # Constraints & Objective
    for i, midi_note in enumerate(valid_sequence):
        options = trombone_notes[midi_note]
        # Notice we are now unpacking 5 items in these loops: t, p_dist, p_int, label, is_b
        prob += pulp.lpSum(x[i, t, p_dist] for t, p_dist, p_int, label, is_b in options) == 1
        prob += dist[i] == pulp.lpSum(p_dist * x[i, t, p_dist] for t, p_dist, p_int, label, is_b in options)
        prob += t_state[i] == pulp.lpSum(t * x[i, t, p_dist] for t, p_dist, p_int, label, is_b in options)
        
        # Soft Constraint: If the chosen position has is_b=0, trigger the is_alt penalty variable
        prob += is_alt[i] == pulp.lpSum((1 - is_b) * x[i, t, p_dist] for t, p_dist, p_int, label, is_b in options)

        # Hard Constraints (The Asterisks)
        flag = forced_flags[i]
        if flag == "Basic" and midi_note in basic_notes:
            # Mode 1: Force Basic Position
            b_dist = basic_notes[midi_note]
            for t, p_dist, p_int, label, is_b in options:
                if p_dist == b_dist:
                    prob += x[i, t, p_dist] == 1 
                    break
                    
        elif isinstance(flag, tuple):
            # Mode 2: Force Specific User Coordinate (Valve, Visual Position)
            req_valve, req_pos = flag
            for t, p_dist, p_int, label, is_b in options:
                if t == req_valve and p_int == req_pos:
                    prob += x[i, t, p_dist] == 1
                    break

    for i in range(1, n):
        prob += dist[i] - dist[i-1] <= 10 * u[i]
        prob += dist[i-1] - dist[i] <= 10 * v[i]
        prob += z[i] >= t_state[i] - t_state[i-1]
        prob += z[i] >= t_state[i-1] - t_state[i]
        prob += abs_jump[i] >= dist[i] - dist[i-1]
        prob += abs_jump[i] >= dist[i-1] - dist[i]
        prob += abs_jump[i] <= LEAP_THRESHOLD + 10 * is_big_leap[i]
    for i in range(1, n - 1):
        prob += w[i] >= u[i] + v[i+1] - 1
        prob += w[i] >= v[i] + u[i+1] - 1

    # Multi-Objective Optimization Math
    prob += (pulp.lpSum(w[i] for i in range(1, n-1)) +
             leap_pen * pulp.lpSum(is_big_leap[i] for i in range(1, n)) +
             valve_pen * pulp.lpSum(z[i] for i in range(1, n)) +
             trigger_pen * pulp.lpSum(t_state[i] for i in range(n)) +
             alt_pen * pulp.lpSum(is_alt[i] for i in range(n)) +     # <--- The New Alternate Position Penalty
             0.05 * pulp.lpSum(abs_jump[i] for i in range(1, n)))

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    optimized_path = []
    for i in range(n):
        for t, p_dist, p_int, label, is_b in trombone_notes[valid_sequence[i]]:
            if pulp.value(x[i, t, p_dist]) == 1:
                optimized_path.append({"Step": i, "Note": label, "Valve": 'Trigger' if t == 1 else 'Open', "Visual Pos": p_int, "Phys Dist": round(p_dist, 2)})
    return optimized_path

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Trombone Slide Path Optimizer", layout="wide")
st.title("Trombone Slide Position Optimizer")

# Sidebar Configuration
st.sidebar.header("Instrument Setup")
horn_choice = st.sidebar.selectbox("Select Instrument", ["Straight Tenor", "Symphony Tenor"])

try:
    # Pass the selected horn to the data loader so it knows which column to check!
    trombone_notes, basic_notes = load_trombone_data('trombone_data.csv', horn_choice)
except FileNotFoundError:
    st.error("⚠️ trombone_data.csv not found!")
    st.stop()

st.sidebar.header("Tuning Parameters")
valve_p = st.sidebar.slider("Valve Change Penalty", 0.0, 5.0, 0.75, 0.25)
trigger_p = st.sidebar.slider("Trigger Usage Penalty", 0.0, 5.0, 1.0, 0.25)
leap_p = st.sidebar.slider("Big Leap Penalty (>3.5)", 0.0, 10.0, 3.0, 0.5)
alt_p = st.sidebar.slider("Alternate Position Penalty", 0.0, 5.0, 1.0, 0.25) 
show_comparison = st.sidebar.toggle("Compare with Basic Positions", value=False)

# Main App
manual_input = st.text_input("Enter Note Sequence (e.g. Bb2 F3 Bb3 D4):")

if st.button("Optimize Route"):
    if manual_input:
        try:
            # --- UPGRADED PARSER: HANDLING DYNAMIC ASTERISKS ---
            raw_notes = manual_input.split()
            midi_sequence = []
            forced_flags = []
            
            for n in raw_notes:
                if '*' in n:
                    parts = n.split('*')
                    prefix = parts[0]
                    clean_note = parts[1]
                    
                    if prefix == "":
                        forced_flags.append("Basic")
                    else:
                        try:
                            v_id = int(prefix[0])
                            p_id = int(prefix[1:])
                            forced_flags.append((v_id, p_id))
                        except ValueError:
                            forced_flags.append("Basic") 
                else:
                    forced_flags.append(False)
                    clean_note = n
                    
                midi_sequence.append(note.Note(clean_note).pitch.midi)
            
            # --- RUN THE SOLVER ---
            results = solve_with_leap_logic(
                midi_sequence, 
                forced_flags, 
                trombone_notes, 
                basic_notes, 
                valve_p, 
                leap_p, 
                trigger_p,
                alt_p  
            )
            
            if results:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader("Slide Routing")
                    st.dataframe(results, hide_index=True)
                
                with col2:
                    st.subheader("Physical Slide Path")
                    opt_dists = [r['Phys Dist'] for r in results]
                    valid_midi = [m for m in midi_sequence if m in trombone_notes]
                    
                    basic_dists = [basic_notes[m] for m in valid_midi if m in basic_notes]
                    
                    chart_list = []
                    for i, d in enumerate(opt_dists): chart_list.append({'Note Index': i, 'Distance': d, 'Path': 'Optimized'})
                    
                    if show_comparison and len(basic_dists) == len(opt_dists):
                        for i, d in enumerate(basic_dists): chart_list.append({'Note Index': i, 'Distance': d, 'Path': 'Basic'})
                    elif show_comparison:
                        # FIX 2: Updated warning message to reflect the dynamic column mapping
                        st.warning(f"Could not plot Basic Path: One or more notes in this sequence doesn't have a basic position assigned for the {horn_choice} in your CSV.")
                    
                    chart_df = pd.DataFrame(chart_list)

                    chart = alt.Chart(chart_df).mark_line(point=True).encode(
                        x='Note Index:O',
                        y=alt.Y('Distance:Q', scale=alt.Scale(domain=[1, 8], reverse=True), title="Slide Position"),
                        color='Path:N'
                    ).properties(height=400)
                    st.altair_chart(chart, use_container_width=True)

                if show_comparison and len(basic_dists) == len(opt_dists):
                    st.divider()
                    st.subheader("Efficiency Metrics")
                    o_dist, o_turn = calculate_metrics(opt_dists)
                    b_dist, b_turn = calculate_metrics(basic_dists)
                    
                    if b_dist > 0:
                        change_pct = ((o_dist - b_dist) / b_dist) * 100
                    else:
                        change_pct = 0.0
                    
                    turn_diff = o_turn - b_turn

                    metric_dist_str = f"{change_pct:.1f}% Distance" 
                    metric_turn_str = f"{turn_diff} Turns"
                    
                    if change_pct <= 0:
                        table_dist_str = f"{abs(change_pct):.1f}% Shorter"
                    else:
                        table_dist_str = f"{abs(change_pct):.1f}% Longer"
                        
                    if turn_diff <= 0:
                        table_turn_str = f"{abs(turn_diff)} Fewer Turns"
                    else:
                        table_turn_str = f"{abs(turn_diff)} More Turns"

                    m1, m2 = st.columns(2)
                    
                    m1.metric("Total Slide Travel", f"{o_dist} units", delta=metric_dist_str, delta_color="inverse")
                    m2.metric("Complexity", f"{o_turn} turns", delta=metric_turn_str, delta_color="inverse")

                    comparison_data = {
                        "Metric": ["Total Distance Moved", "Total Direction Changes"],
                        "Optimized Path": [f"{o_dist} units", o_turn],
                        "Basic Path": [f"{b_dist} units", b_turn],
                        "Change": [table_dist_str, table_turn_str]
                    }
                    st.table(pd.DataFrame(comparison_data))
        except Exception as e:
            st.error(f"Error: {e}")