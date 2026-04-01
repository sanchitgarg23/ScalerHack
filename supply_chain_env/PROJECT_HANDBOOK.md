# 📦 Supply Chain Disruption AI — Project Handbook

Welcome to the **Supply Chain Disruption Management Environment**. This project combines Reinforcement Learning (RL) with Large Language Models (LLMs) to solve one of the most complex problems in logistics: managing unexpected failures in a global supply chain.

---

## 🎯 The Goal
The objective is to train or test an AI "Procurement Manager" who can:
1. **Discover**: Identify which suppliers are reliable and which are failing.
2. **Negotiate**: Haggle for the best prices to stay within a tight budget.
3. **Fulfill**: Ensure customer orders are delivered on time despite disruptions.
4. **Resilience**: Handle "Black Swan" events like sudden supplier bankruptcies or scams.

---

## 🏗️ Technical Architecture
The project follows a **Client-Server architecture** using the **OpenEnv** framework.

### 1. The Server (`/server/app.py` & `supply_chain_environment.py`)
This is the "World Engine." It runs a discrete-time simulation where each "step" represents a day in the supply chain lifecycle.
*   **Logic**: It calculates stock levels, manages the budget, and randomly triggers "Disruptions" (e.g., a supplier failing on Day 10).
*   **API**: It exposes standard RL endpoints: `/reset` (start new mission), `/step` (take action), and `/state` (get full stats).

### 2. The AI Agent (`inference.py`)
This is the "Brain" of the project.
*   **Engine**: Powered by **Groq AI** (Llama 3.3 70B) for high-speed decision making.
*   **Thinking**: It receives a JSON description of the current situation and must output a valid `SupplyChainAction`.
*   **Actions**: The AI can `query_supplier`, `place_order`, `negotiate_price`, `notify_customer`, `expedite_shipment`, or `declare_done`.

### 3. The Control Center (`dashboard.py`)
This is your **Visual Dashboard**.
*   **Real-time Monitoring**: It connects to the server and the agent simultaneously.
*   **Visuals**: Shows a live map of your Supplier Network, your remaining Budget, and a live "Decision Log" of the AI's thoughts.

---

## 📂 Key Files to Know
| File | Purpose |
| :--- | :--- |
| **`dashboard.py`** | **The Entry Point.** Run this to start the UI and the Server together. |
| **`inference.py`** | Contains the LLM agent's "Thinking" logic and API connection. |
| **`server/supply_chain_environment.py`** | The core simulation code (rewards, disruptions, math). |
| **`models.py`** | The "Language" of the project (defines Suppliers, Orders, and Actions). |
| **`.env`** | Securely stores your Groq API key. |

---

## 🏆 Difficulty Tiers
We have programmed 3 distinct scenarios:
*   **Easy (Task 0)**: One supplier fails. You have a huge budget. Just find the backup.
*   **Medium (Task 1)**: Multi-supplier failure. Tight budget. Requires smart negotiation.
*   **Hard (Task 2)**: **The Crisis.** Cascading failures, a demand spike, and a "Scam Supplier" trying to steal your budget.

---

## 🚀 How to Run
It's designed to be simple:
1. Ensure your Groq Key is in `.env`.
2. Run `python3 dashboard.py`.
3. Open `http://127.0.0.1:7860` and click **Start Simulation**.

---

## 🛠️ Built With
*   **FastAPI**: For the high-performance simulation server.
*   **OpenEnv**: The industry-standard RL environment wrapper.
*   **Gradio**: For the real-time Control Center UI.
*   **Groq / Llama 3.3**: For the cutting-edge reasoning agent.
