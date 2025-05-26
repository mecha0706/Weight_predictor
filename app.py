# import gradio as gr
# import numpy as np
# from BS_asian_all_combined_MAY_2025 import (
#     get_strike_range_for_K,
#     run_experiment_no_M_without_bounds,
#     run_experiment_no_M_with_bounds,
#     run_experiment_with_ridge,
#     run_experiment_with_lasso,N,
#     simulate_stock_paths,
# )

# def get_strike_choices(stock,K):
#     # Generate the available strikes for this K
#     _, _, short_strike_range = get_strike_range_for_K(stock,K, N, choice="non-uniform")
#     # Return as a string for display
#     return ", ".join([f"{i}: {v}" for i, v in enumerate(short_strike_range)])

# def run_all_experiments(stock, K, no_of_options, indices_str):
#     time_indices, _= simulate_stock_paths(stock)
#     # Get available strikes for this K
#     _, _, short_strike_range = get_strike_range_for_K(stock,K, N, choice="non-uniform")
#     # Parse indices
#     try:
#         selected_indices = [int(i.strip()) for i in indices_str.split(",") if i.strip()]
#     except Exception:
#         return "Error: Please enter valid comma-separated indices.", "", "", ""
#     # Check length
#     if len(selected_indices) != no_of_options:
#         return f"Error: You selected {len(selected_indices)} strikes, but no_of_options is {no_of_options}.", "", "", ""
#     # Build short_strikes
#     try:
#         short_strikes = [short_strike_range[i] for i in selected_indices]
#     except IndexError:
#         return f"Error: One or more indices are out of range. Valid indices: 0 to {len(short_strike_range)-1}", "", "", ""
#     # Run experiments
#     A = run_experiment_no_M_without_bounds(stock,K,no_of_options, short_strikes)
#     B = run_experiment_no_M_with_bounds(stock,K,no_of_options, short_strikes)
#     C = run_experiment_with_ridge(stock,K,no_of_options, short_strikes)
#     D = run_experiment_with_lasso(stock,K,no_of_options, short_strikes)
#     # Format output
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

# with gr.Blocks() as demo:
#     gr.Markdown("## Asian Option Hedging: Optimization Strategies")
#     stock = gr.Number(label="Stock Price", value=1)
#     K = gr.Number(label="Strike price", value=1)
#     no_of_options = gr.Slider(1, 7, value=3, step=1, label="Number of Options")
#     strike_choices = gr.Textbox(label="Available Short Strikes (index: value)", interactive=False)
#     indices_str = gr.Textbox(label="Indices of Short Strikes (comma-separated, e.g. 0,2,4)")
#     show_btn = gr.Button("Show Available Strikes")
#     run_btn = gr.Button("Run Experiments")
#     out1 = gr.Textbox(label="No Bound Results")
#     out2 = gr.Textbox(label="With Bound Results")
#     out3 = gr.Textbox(label="Ridge Results")
#     out4 = gr.Textbox(label="Lasso Results")

#     show_btn.click(get_strike_choices, inputs=(stock,K), outputs=strike_choices)
#     run_btn.click(run_all_experiments, inputs=[stock,K, no_of_options, indices_str], outputs=[out1, out2, out3, out4])

# demo.launch(share=True)
import gradio as gr
import numpy as np
from BS_asian_all_combined_MAY_2025 import (
    K1_init,
    run_experiment_no_M_without_bounds,
    run_experiment_no_M_with_bounds,
    run_experiment_with_ridge,
    run_experiment_with_lasso,
    N,
    # simulate_stock_paths,
)

def get_strike_choices(stock, K):
    # Use K1_init as the available short strikes
    short_strike_range = K1_init
    return ", ".join([f"{i}: {v}" for i, v in enumerate(short_strike_range)])

def run_all_experiments(stock, K, no_of_options, indices_str):
    # time_indices, _ = simulate_stock_paths(stock)
    # Use K1_init as the available short strikes
    short_strike_range = K1_init
    try:
        selected_indices = [int(i.strip()) for i in indices_str.split(",") if i.strip()]
    except Exception:
        return "Error: Please enter valid comma-separated indices.", "", "", ""
    if len(selected_indices) != no_of_options:
        return f"Error: You selected {len(selected_indices)} strikes, but no_of_options is {no_of_options}.", "", "", ""
    try:
        short_strikes = [short_strike_range[i] for i in selected_indices]
    except IndexError:
        return f"Error: One or more indices are out of range. Valid indices: 0 to {len(short_strike_range)-1}", "", "", ""
    A = run_experiment_no_M_without_bounds(stock, K, no_of_options, short_strikes)
    B = run_experiment_no_M_with_bounds(stock, K, no_of_options, short_strikes)
    C = run_experiment_with_ridge(stock, K, no_of_options, short_strikes)
    D = run_experiment_with_lasso(stock, K, no_of_options, short_strikes)
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
    stock = gr.Number(label="Stock Price", value=1)
    K = gr.Number(label="Strike price", value=1)
    no_of_options = gr.Slider(1, 7, value=3, step=1, label="Number of Options")
    strike_choices = gr.Textbox(label="Available Short Strikes (index: value)", interactive=False)
    indices_str = gr.Textbox(label="Indices of Short Strikes (comma-separated, e.g. 0,2,4)")
    show_btn = gr.Button("Show Available Strikes")
    run_btn = gr.Button("Run Experiments")
    out1 = gr.Textbox(label="No Bound Results")
    out2 = gr.Textbox(label="With Bound Results")
    out3 = gr.Textbox(label="Ridge Results")
    out4 = gr.Textbox(label="Lasso Results")

    show_btn.click(get_strike_choices, inputs=[stock, K], outputs=strike_choices)
    run_btn.click(run_all_experiments, inputs=[stock, K, no_of_options, indices_str], outputs=[out1, out2, out3, out4])

demo.launch(share=True)