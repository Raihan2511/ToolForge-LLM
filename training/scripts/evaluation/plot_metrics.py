import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(run_dir):
    trainer_state_file = os.path.join(run_dir, "trainer_state.json")
    custom_metrics_file = os.path.join(run_dir, "custom_routing_metrics.csv")
    output_image = os.path.join(run_dir, "training_metrics.png")

    if not os.path.exists(trainer_state_file):
        print(f"Error: Could not find {trainer_state_file}")
        return

    # Parse HF trainer state
    with open(trainer_state_file, "r") as f:
        state = json.load(f)

    steps = []
    train_loss = []
    eval_loss = []
    eval_steps = []

    for log in state.get("log_history", []):
        step = log.get("step")
        if "loss" in log:
            steps.append(step)
            train_loss.append(log["loss"])
        if "eval_loss" in log:
            eval_steps.append(step)
            eval_loss.append(log["eval_loss"])

    # Parse custom routing metrics
    custom_steps = []
    json_validity = []
    routing_acc = []

    if os.path.exists(custom_metrics_file):
        df = pd.read_csv(custom_metrics_file)
        custom_steps = df["step"].tolist()
        json_validity = (df["tool_json_validity"] * 100).tolist()
        routing_acc = (df["conv_routing_accuracy"] * 100).tolist()
    else:
        print(f"Warning: {custom_metrics_file} not found. Only plotting loss.")

    # Plotting
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Top plot: Loss
    ax1.plot(steps, train_loss, label="Train Loss", color="tab:blue", alpha=0.7)
    if eval_steps and eval_loss:
        ax1.plot(eval_steps, eval_loss, label="Eval Loss", color="tab:red", marker="o")
    ax1.set_title("Training and Evaluation Loss")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Bottom plot: Custom Routing Metrics
    if custom_steps:
        ax2.plot(custom_steps, json_validity, label="Tool JSON Validity (%)", color="tab:green", marker="s", linestyle="--")
        ax2.plot(custom_steps, routing_acc, label="Conversational Routing Accuracy (%)", color="tab:purple", marker="^", linestyle="-.")
        ax2.set_title("Agentic Behavioral Metrics")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_ylim(-5, 105)
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, "Custom Metrics Not Found", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Successfully generated and saved graphs to: {output_image}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ToolForge-LLM training metrics")
    parser.add_argument("--run_dir", type=str, default="/workspace/training/runs/qwen7b_unsloth", help="Directory containing trainer_state.json")
    args = parser.parse_args()
    
    plot_metrics(args.run_dir)
