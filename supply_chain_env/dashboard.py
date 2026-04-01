import os
import sys
import threading
import time
import pandas as pd
import gradio as gr
from datetime import datetime

# Add parent directory to sys.path so 'supply_chain_env' is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from supply_chain_env.inference import run_task, ENV_URL
from supply_chain_env.client import SupplyChainEnv

# --- UI State ---
class GlobalState:
    def __init__(self):
        self.logs = []
        self.current_obs = None
        self.is_running = False
        self.final_score = 0.0

state = GlobalState()

def format_log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    return f"[{timestamp}] {msg}"

def update_ui_from_step(step, action, obs, done, score):
    state.current_obs = obs
    if action:
        state.logs.append(format_log(f"Step {step}: Agent took action '{action.action_type}' on '{action.target}'"))
    elif step == 0:
        state.logs.append(format_log(f"Episode started. Initial observation received."))
    
    if done and step > 0:
        state.final_score = score
        state.logs.append(format_log(f"Episode finished! Final Score: {score:.4f}"))

def start_simulation(task_id_str):
    task_id = int(task_id_str.split(":")[0])
    state.logs = [format_log(f"Connecting to environment at {ENV_URL}...")]
    state.is_running = True
    state.final_score = 0.0
    
    # Run in a separate thread so Gradio doesn't block
    def worker():
        try:
            run_task(task_id, callback=update_ui_from_step)
        finally:
            state.is_running = False

    thread = threading.Thread(target=worker)
    thread.start()
    
    return gr.update(interactive=False), "Simulation Started..."

def get_live_data():
    if not state.current_obs:
        return (
            "Ready", "0", "$0", "0.0000", 
            pd.DataFrame(columns=["Name", "Status", "Stock", "Price", "Lead Time"]),
            pd.DataFrame(columns=["Order ID", "SKU", "Qty", "Deadline", "Done"]),
            "\n".join(state.logs)
        )
    
    obs = state.current_obs
    
    # Supplier Data
    sup_data = []
    for s in obs.suppliers:
        sup_data.append({
            "Name": s.name,
            "Status": s.status.upper(),
            "Stock": s.stock,
            "Price": f"${s.price_per_unit:.2f}",
            "Lead Time": f"{s.delivery_days} days"
        })
    df_sup = pd.DataFrame(sup_data)
    
    # Order Data
    ord_data = []
    for o in obs.customer_orders:
        ord_data.append({
            "Order ID": o.order_id,
            "SKU": o.sku,
            "Qty": o.quantity,
            "Deadline": f"Day {o.deadline_day}",
            "Done": "✅" if o.fulfilled else "⏳"
        })
    df_ord = pd.DataFrame(ord_data)
    
    status = "🔴 FINISHED" if not state.is_running else "🟢 RUNNING"
    if not state.current_obs and not state.is_running:
        status = "⚪ READY"

    return (
        status,
        str(obs.day),
        f"${obs.budget_remaining:,.0f}",
        f"{state.final_score:.4f}",
        df_sup,
        df_ord,
        "\n".join(state.logs[::-1]) # Show newest logs first
    )

# --- Gradio Layout ---
with gr.Blocks(title="Supply Chain Control Center", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📦 Supply Chain Control Center")
    gr.Markdown("Monitor your AI Agent as it navigates supplier disruptions and customer deadlines in real-time.")
    
    with gr.Row():
        with gr.Column(scale=1):
            task_select = gr.Dropdown(
                choices=["0: Easy (1 Supplier Fail)", "1: Medium (Triage)", "2: Hard (Cascading Crisis)"],
                label="Select Task Difficulty",
                value="0: Easy (1 Supplier Fail)"
            )
            run_btn = gr.Button("🚀 Start Agent Simulation", variant="primary")
            sim_status = gr.Markdown("Status: ⚪ READY")
        
        with gr.Column(scale=3):
            with gr.Row():
                day_stat = gr.Label(label="Current Day", value="0")
                budget_stat = gr.Label(label="Budget Remaining", value="$0")
                score_stat = gr.Label(label="Live/Final Score", value="0.0000")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🏭 Supplier Network Status")
            supplier_table = gr.DataFrame(
                headers=["Name", "Status", "Stock", "Price", "Lead Time"],
                datatype=["str", "str", "number", "str", "str"],
                label="Active/Known Suppliers"
            )
            
        with gr.Column():
            gr.Markdown("### 🛒 Active Customer Orders")
            order_table = gr.DataFrame(
                headers=["Order ID", "SKU", "Qty", "Deadline", "Done"],
                datatype=["str", "str", "number", "str", "str"],
                label="Fulfilment Progress"
            )

    gr.Markdown("### 📜 Real-time Decision Log")
    log_output = gr.Textbox(lines=8, label="Agent Thinking Log", interactive=False)
    
    # Timer for UI updates
    timer = gr.Timer(value=1.5)
    timer.tick(get_live_data, outputs=[sim_status, day_stat, budget_stat, score_stat, supplier_table, order_table, log_output])
    
    # Event Handlers
    run_btn.click(
        start_simulation, 
        inputs=[task_select], 
        outputs=[run_btn, sim_status]
    )
    
    # Reset button state when simulation ends
    def check_btn_state():
        return gr.update(interactive=not state.is_running)
    
    timer.tick(check_btn_state, outputs=[run_btn])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
