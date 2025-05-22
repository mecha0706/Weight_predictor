# import gradio as gr
# from BS_asian_all_combined_MAY_2025 import (
#     run_experiment_no_M,
#     short_strike_range,
#     truncation_range,
#     N,
#     result_data_no_bounds
# )

# # ...existing code...
# def display_results():
#     result_data_no_bounds
#     # Format output for display
#     w_opt = result_data_no_bounds["w_opt"]
#     final_value = result_data_no_bounds["final_value"]
#     target = result_data_no_bounds["target_call_price"]
#     hedge = result_data_no_bounds["hedge_value"]

#     w_str = "\n".join([f"w{i}: {w:.6f}" for i, w in enumerate(w_opt)])

#     return f"""📊 **Optimal Weights:**
# {w_str}

# 🎯 **Final Min-Max Value:** {final_value:.6f}
# 🎯 **Target Call Price:** {target:.6f}
# 💼 **Hedge Value:** {hedge:.6f}
# """
# # ...existing code...
# # Setup the Gradio interface
# iface = gr.Interface(
#     fn=display_results,
#     inputs=[],
#     outputs="text",
#     title="Asian Option Hedging Result Viewer",
#     description="Displays optimal weights and objective value from Gurobi-based min-max optimization."
# )

# # Run the Gradio app
# if __name__ == "__main__":
#     iface.launch()


import streamlit as st
from BS_asian_all_combined_MAY_2025 import (
    run_experiment_no_M,
    short_strike_range,
    truncation_range,
    N
)

st.title("Asian Option Hedging Result Viewer")
st.write("Displays optimal weights and objective value from Gurobi-based min-max optimization.")

# Run the experiment (you can add Streamlit widgets for parameters if needed)
result = run_experiment_no_M(
    M_bound=None,
    solver="gurobi",
    truncation_range=truncation_range,
    N=N,
    no_of_options=4,
    short_strikes=short_strike_range[:4],
    epsilon=0,
    choice="non-uniform",
    x0=[0.0] * 5
)

w_opt = result["w_opt"]
final_value = result["final_value"]
target = result["target_call_price"]
hedge = result["hedge_value"]

st.subheader("📊 Optimal Weights")
for i, w in enumerate(w_opt):
    st.write(f"w{i}: {w:.6f}")

st.subheader("🎯 Final Min-Max Value")
st.write(f"{final_value:.6f}")

st.subheader("🎯 Target Call Price")
st.write(f"{target:.6f}")

st.subheader("💼 Hedge Value")
st.write(f"{hedge:.6f}")