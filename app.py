import gradio as gr
import numpy as np
from BS_asian_all_combined_MAY_2025 import (
    run_experiment_no_M_without_bounds,
    run_experiment_no_M_with_bounds,
    run_experiment_with_lasso,
    run_experiment_with_ridge
)

def run_all_experiments():
    A = run_experiment_no_M_without_bounds()
    B = run_experiment_no_M_with_bounds()
    C = run_experiment_with_lasso()
    D = run_experiment_with_ridge()

    # Format the output for each experiment
    def format_result(res):
        weights = np.round(res["Optimal_weights"], 6).tolist()
        final_value = res["Final_value"]
        return f"Optimal weights: {weights}\nFinal min-max value: {final_value}"

    return (
        format_result(A),
        format_result(B),
        format_result(C),
        format_result(D)
    )

iface = gr.Interface(
    fn=run_all_experiments,
    inputs=[],
    outputs=[
        gr.outputs.Textbox(label="No Bound Results"),
        gr.outputs.Textbox(label="With Bound Results"),
        gr.outputs.Textbox(label="Lasso Regularization Results"),
        gr.outputs.Textbox(label="Ridge Regularization Results")
    ],
    title="Asian Option Hedging: Optimization Strategies",
    description="Click 'Evaluate' to run: No Bound, With Bound, Lasso, and Ridge experiments."
)

if __name__ == "__main__":
    iface.launch(share=True, server_name="0.0.0.0", server_port=7860)