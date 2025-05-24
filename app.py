# import gradio as gr
# import numpy as np
# from BS_asian_all_combined_MAY_2025 import (
#     run_experiment_no_M_without_bounds,
#     run_experiment_no_M_with_bounds,
#     run_experiment_with_lasso,
#     run_experiment_with_ridge
# )

# def run_all_experiments():
#     A = run_experiment_no_M_without_bounds()
#     B = run_experiment_no_M_with_bounds()
#     C = run_experiment_with_lasso()
#     D = run_experiment_with_ridge()

#     # Format the output for each experiment
#     def format_result(res):
#         weights = np.round(res["Optimal_weights"], 6).tolist()
#         final_value = res["Final_value"]
#         return f"Optimal weights: {weights}\nFinal min-max value: {final_value}"

#     return (
#         format_result(A),
#         format_result(B),
#         format_result(C),
#         format_result(D)
#     )

# iface = gr.Interface(
#     fn=run_all_experiments,
#     inputs=[],
#     outputs=[
#         gr.Textbox(label="No Bound Results"),
#         gr.Textbox(label="With Bound Results"),
#         gr.Textbox(label="Lasso Regularization Results"),
#         gr.Textbox(label="Ridge Regularization Results")
#     ],
#     title="Asian Option Hedging: Optimization Strategies",
#     description="Click 'Evaluate' to run: No Bound, With Bound, Lasso, and Ridge experiments."
# )


# iface.launch(share=True)
import gradio as gr
import numpy as np
from BS_asian_all_combined_MAY_2025 import (
    get_strike_range_for_K,
    run_experiment_no_M_without_bounds,
    run_experiment_no_M_with_bounds,
    run_experiment_with_ridge,
    run_experiment_with_lasso,
    truncation_range,
    N
)

def get_strike_choices(K):
    # Generate the available strikes for this K
    _, _, short_strike_range = get_strike_range_for_K(K, truncation_range, N, choice="non-uniform")
    # Return as a string for display
    return ", ".join([f"{i}: {v}" for i, v in enumerate(short_strike_range)])

def run_all_experiments(K, no_of_options, indices_str):
    # Get available strikes for this K
    _, _, short_strike_range = get_strike_range_for_K(K, truncation_range, N, choice="non-uniform")
    # Parse indices
    try:
        selected_indices = [int(i.strip()) for i in indices_str.split(",") if i.strip()]
    except Exception:
        return "Error: Please enter valid comma-separated indices.", "", "", ""
    # Check length
    if len(selected_indices) != no_of_options:
        return f"Error: You selected {len(selected_indices)} strikes, but no_of_options is {no_of_options}.", "", "", ""
    # Build short_strikes
    try:
        short_strikes = [short_strike_range[i] for i in selected_indices]
    except IndexError:
        return f"Error: One or more indices are out of range. Valid indices: 0 to {len(short_strike_range)-1}", "", "", ""
    # Run experiments
    A = run_experiment_no_M_without_bounds(K,no_of_options, short_strikes)
    B = run_experiment_no_M_with_bounds(K,no_of_options, short_strikes)
    C = run_experiment_with_ridge(K,no_of_options, short_strikes)
    D = run_experiment_with_lasso(K,no_of_options, short_strikes)
    # Format output
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

with gr.Blocks() as demo:
    gr.Markdown("## Asian Option Hedging: Optimization Strategies")
    K = gr.Number(label="Target Strike K", value=1)
    no_of_options = gr.Slider(1, 7, value=3, step=1, label="Number of Options")
    strike_choices = gr.Textbox(label="Available Short Strikes (index: value)", interactive=False)
    indices_str = gr.Textbox(label="Indices of Short Strikes (comma-separated, e.g. 0,2,4)")
    show_btn = gr.Button("Show Available Strikes")
    run_btn = gr.Button("Run Experiments")
    out1 = gr.Textbox(label="No Bound Results")
    out2 = gr.Textbox(label="With Bound Results")
    out3 = gr.Textbox(label="Ridge Results")
    out4 = gr.Textbox(label="Lasso Results")

    show_btn.click(get_strike_choices, inputs=K, outputs=strike_choices)
    run_btn.click(run_all_experiments, inputs=[K, no_of_options, indices_str], outputs=[out1, out2, out3, out4])

demo.launch(share=True)