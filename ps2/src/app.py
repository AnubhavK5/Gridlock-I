import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Intelligence Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSV_PATH = "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e; border-radius: 12px; padding: 1.2rem 1.5rem;
    border-left: 4px solid #7c3aed; margin-bottom: 0.8rem;
}
.metric-label { color: #a1a1aa; font-size: 0.78rem; font-weight: 600;
                letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px; }
.metric-value { color: #f4f4f5; font-size: 2rem; font-weight: 800; line-height: 1; }
.metric-sub   { color: #71717a; font-size: 0.75rem; margin-top: 4px; }
.path-box {
    background: #18181b; border-radius: 10px; padding: 1rem 1.2rem;
    font-family: monospace; font-size: 0.82rem; color: #d4d4d8;
    border: 1px solid #27272a; margin-bottom: 0.6rem; overflow-x: auto;
}
.path-node  { display:inline-block; background:#312e81; color:#c7d2fe;
              border-radius:6px; padding:2px 8px; margin:2px; }
.incident-node { background:#7f1d1d; color:#fecaca; }
.status-ok   { background:#14532d; color:#bbf7d0; border-radius:8px;
               padding:0.6rem 1rem; font-size:0.82rem; margin-top:0.5rem; }
.status-warn { background:#713f12; color:#fef08a; border-radius:8px;
               padding:0.6rem 1rem; font-size:0.82rem; margin-top:0.5rem; }
.header-bar  { background:linear-gradient(135deg,#1e1b4b 0%,#312e81 100%);
               border-radius:14px; padding:1.5rem 2rem; margin-bottom:1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Data loader (cached) ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_dropdown_data():
    try:
        df = pd.read_csv(CSV_PATH, usecols=["police_station", "junction", "event_cause",
                                              "event_type", "priority", "requires_road_closure"])
        stations  = sorted(df["police_station"].dropna().unique().tolist())
        junctions = sorted(df["junction"].dropna().unique().tolist())
        causes    = sorted(df["event_cause"].dropna().unique().tolist())
        types_    = sorted(df["event_type"].dropna().unique().tolist())
        return stations, junctions, causes, types_, True
    except Exception as e:
        st.warning(f"CSV not found — using synthetic dropdown values. ({e})")
        stations  = [f"Station_{i}" for i in range(1, 11)]
        junctions = [f"Junction_{chr(65+i)}" for i in range(15)]
        causes    = ["vehicle_breakdown", "accident", "public_event", "construction",
                     "water_logging", "tree_fall", "procession", "protest",
                     "vip_movement", "congestion", "others"]
        types_    = ["unplanned", "planned"]
        return stations, junctions, causes, types_, False


stations, junctions, causes, event_types, csv_loaded = load_dropdown_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 Incident Parameters")
    st.caption("Configure the incident scenario below.")

    event_type    = st.selectbox("Event Type",    event_types, index=0)
    event_cause   = st.selectbox("Event Cause",   causes,      index=0)
    police_station = st.selectbox("Police Station", stations,  index=0)

    st.divider()
    st.markdown("## 🗺️ Route Configuration")
    origin_junc   = st.selectbox("Origin Junction",      junctions, index=0)
    dest_junc     = st.selectbox("Destination Junction", junctions,
                                  index=min(len(junctions)-1, 10))
    incident_junc = st.selectbox("Incident Junction (to bypass)",
                                  junctions, index=min(len(junctions)-1, 5))

    st.divider()
    simulate_btn = st.button("🔴 Simulate System Impact & Generate Plan",
                              use_container_width=True, type="primary")
    st.caption("ℹ️ First run trains the ML models — may take a few seconds.")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <h1 style="margin:0;color:#e0e7ff;font-size:1.8rem;">
    🚦 Event-Driven Traffic Intelligence System
  </h1>
  <p style="margin:4px 0 0;color:#a5b4fc;font-size:0.9rem;">
    Incident Prediction · Resource Optimization · Real-Time Diversion Routing
  </p>
</div>
""", unsafe_allow_html=True)

if not csv_loaded:
    st.warning("Dataset CSV not found in working directory. Synthetic fallback data is active.")


# ── Utility (must be defined before the if/else block) ───────────────────────
def nx_density(G):
    try:
        import networkx as nx
        return nx.density(G)
    except Exception:
        n = G.number_of_nodes()
        e = G.number_of_edges()
        return e / (n * (n - 1)) if n > 1 else 0.0


# ── Main simulation logic ─────────────────────────────────────────────────────
if simulate_btn:

    # ── Lazy imports (avoids training on cold page load) ─────────────────────
    with st.spinner("🤖 Training ML models & building graph…"):
        from predictor       import predict_event_priority
        from optimizer       import calculate_optimal_manpower_and_hardware, format_resource_summary
        from diversion_engine import generate_detour as _gen_detour, get_engine

    # 1. ML Prediction
    with st.spinner("Running ML inference…"):
        pred = predict_event_priority(event_type, event_cause, police_station)

    priority_label   = pred["priority_label"]
    priority_prob    = pred["priority_prob"]
    requires_closure = pred["requires_closure"]
    closure_prob     = pred["requires_closure_prob"]

    # 2. Resource optimization
    resources = calculate_optimal_manpower_and_hardware(
        priority_label, event_cause, requires_closure
    )

    # 3. Diversion routing
    baseline_path, detour_path, route_status = _gen_detour(
        origin_junc, dest_junc, incident_junc
    )

    # ── Results Layout ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Prediction Results")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        color = "#ef4444" if priority_label == "High" else "#22c55e"
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:{color}">
          <div class="metric-label">Predicted Priority</div>
          <div class="metric-value" style="color:{color}">{priority_label}</div>
          <div class="metric-sub">Confidence: {priority_prob*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        closure_color = "#f97316" if requires_closure else "#3b82f6"
        closure_text  = "YES" if requires_closure else "NO"
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:{closure_color}">
          <div class="metric-label">Road Closure Required</div>
          <div class="metric-value" style="color:{closure_color}">{closure_text}</div>
          <div class="metric-sub">Probability: {closure_prob*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#7c3aed">
          <div class="metric-label">Recommended Manpower</div>
          <div class="metric-value">{resources['manpower']}</div>
          <div class="metric-sub">Personnel to deploy</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#06b6d4">
          <div class="metric-label">Barricades Required</div>
          <div class="metric-value">{resources['barricades']}</div>
          <div class="metric-sub">Units to position</div>
        </div>""", unsafe_allow_html=True)

    # ── Resource breakdown ────────────────────────────────────────────────────
    st.markdown("### 🛠️ Resource Allocation Breakdown")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        rows = []
        for label, vals in resources["breakdown"].items():
            rows.append({
                "Component":  label.replace("_", " ").title(),
                "Manpower +": vals["manpower"],
                "Barricades +": vals["barricades"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with col_b:
        st.info(f"**Event:** `{event_type}` · `{event_cause}`\n\n"
                f"**Station:** `{police_station}`\n\n"
                f"**Priority:** `{priority_label}` ({priority_prob*100:.1f}%)\n\n"
                f"**Closure:** `{requires_closure}` ({closure_prob*100:.1f}%)")

    # ── Routing ───────────────────────────────────────────────────────────────
    st.markdown("### 🗺️ Traffic Diversion Analysis")

    def _render_path(path: list, incident: str, label: str, color: str) -> str:
        if not path:
            return f'<div class="path-box"><i>No path found.</i></div>'
        nodes_html = ""
        for n in path:
            cls = "incident-node" if n == incident else "path-node"
            nodes_html += f'<span class="{cls}">{n}</span> → '
        nodes_html = nodes_html.rstrip(" → ")
        return (f'<div style="color:{color};font-weight:700;margin-bottom:4px;">'
                f'{label} ({len(path)} nodes)</div>'
                f'<div class="path-box">{nodes_html}</div>')

    st.markdown("**Baseline Route** *(no incident)*")
    st.markdown(
        _render_path(baseline_path, incident_junc, "Standard Route", "#22c55e"),
        unsafe_allow_html=True
    )

    st.markdown("**Crisis Detour Route** *(incident junction bypassed)*")
    st.markdown(
        _render_path(detour_path, incident_junc, "Diversion Route", "#f97316"),
        unsafe_allow_html=True
    )

    status_cls = "status-ok" if "active" in route_status.lower() or "identical" in route_status.lower() else "status-warn"
    st.markdown(f'<div class="{status_cls}">📍 {route_status}</div>', unsafe_allow_html=True)

    # ── Path comparison table ─────────────────────────────────────────────────
    if baseline_path or detour_path:
        st.markdown("#### Route Node-by-Node Comparison")
        max_len = max(len(baseline_path), len(detour_path))
        comp_df = pd.DataFrame({
            "Step":          list(range(1, max_len + 1)),
            "Baseline Route": (baseline_path + ["—"] * max_len)[:max_len],
            "Detour Route":   (detour_path   + ["—"] * max_len)[:max_len],
        })
        comp_df["Same?"] = comp_df.apply(
            lambda r: "✅" if r["Baseline Route"] == r["Detour Route"] else "🔀", axis=1
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ── Graph stats ───────────────────────────────────────────────────────────
    engine = get_engine()
    with st.expander("📐 Graph Topology Statistics"):
        G = engine.G
        stat_cols = st.columns(3)
        stat_cols[0].metric("Total Junctions (Nodes)", G.number_of_nodes())
        stat_cols[1].metric("Total Road Links (Edges)", G.number_of_edges())
        stat_cols[2].metric("Graph Density", f"{nx_density(G):.4f}")


else:
    # ── Idle state ────────────────────────────────────────────────────────────
    st.info("👈 Configure the incident parameters in the sidebar and click "
            "**'Simulate System Impact & Generate Plan'** to run the full pipeline.")

    col1, col2, col3 = st.columns(3)
    col1.markdown("""
    <div class="metric-card">
      <div class="metric-label">ML Engine</div>
      <div class="metric-value" style="font-size:1.2rem">RandomForest</div>
      <div class="metric-sub">Priority + Road Closure prediction</div>
    </div>""", unsafe_allow_html=True)
    col2.markdown("""
    <div class="metric-card">
      <div class="metric-label">Optimizer</div>
      <div class="metric-value" style="font-size:1.2rem">Heuristic</div>
      <div class="metric-sub">Manpower & Barricade allocation</div>
    </div>""", unsafe_allow_html=True)
    col3.markdown("""
    <div class="metric-card">
      <div class="metric-label">Router</div>
      <div class="metric-value" style="font-size:1.2rem">NetworkX</div>
      <div class="metric-sub">Dijkstra shortest-path diversion</div>
    </div>""", unsafe_allow_html=True)

    with st.expander("📋 Dataset Schema Preview"):
        try:
            df_prev = pd.read_csv(CSV_PATH, nrows=8,
                                   usecols=["event_type","event_cause","police_station",
                                             "junction","priority","requires_road_closure"])
            st.dataframe(df_prev, use_container_width=True)
        except Exception:
            st.caption("Dataset not loaded yet.")
