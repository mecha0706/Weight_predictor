# import gradio as gr
# import numpy as np
# from BS_asian_all_combined_MAY_2025 import run_experiment_no_M, short_strike_range, truncation_range, N

# def run_minmax(no_of_options=4):
#     short_strikes = short_strike_range[:no_of_options]

#     result = run_experiment_no_M(
#         solver="gurobi",
#         truncation_range=truncation_range,
#         N=N,
#         no_of_options=no_of_options,
#         short_strikes=short_strikes,
#         epsilon=0,
#         M_bound=0,
#         choice="non-uniform",
#         x0=[0] * (no_of_options + 1)
#     )

#     weights = np.array(result["w_opt"])
#     final_value = result["final_value"]

#     return {
#         "Optimal Weights": weights.tolist(),
#         "Min Weight": float(np.min(weights)),
#         "Max Weight": float(np.max(weights)),
#         "Final Objective Value": float(final_value)
#     }

# iface = gr.Interface(
#     fn=run_minmax,
#     inputs=gr.Slider(1, 7, value=4, step=1, label="Number of Options"),
#     outputs=[
#         gr.JSON(label="Optimal Weights"),
#         gr.Number(label="Min Weight"),
#         gr.Number(label="Max Weight"),
#         gr.Number(label="Final Objective Value")
#     ],
#     title="Asian Option Hedging Optimization",
#     description="Run the min-max hedging model for an Asian option and view the optimal hedge weights."
# )

# if __name__ == "__main__":
#     iface.launch()
from flask import Flask, request, jsonify
from BS_asian_all_combined_MAY_2025 import run_experiment_no_M, short_strike_range, truncation_range, N

app = Flask(__name__)

@app.route('/run', methods=['POST'])
def run_model():
    data = request.get_json()
    no_of_options = int(data.get("no_of_options", 4))
    
    short_strikes = short_strike_range[:no_of_options]

    result = run_experiment_no_M(
        solver="gurobi",
        truncation_range=truncation_range,
        N=N,
        no_of_options=no_of_options,
        short_strikes=short_strikes,
        epsilon=0,
        M_bound=0,
        choice="non-uniform",
        x0=[0] * (no_of_options + 1)
    )

    weights = result["w_opt"]
    return jsonify({
        "weights": [float(w) for w in weights],
        "min_weight": float(min(weights)),
        "max_weight": float(max(weights)),
        "final_objective_value": float(result["final_value"])
    })

if __name__ == '__main__':
    app.run(debug=True)