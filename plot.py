import json
import matplotlib.pyplot as plt

def plot_loss_curve(json_data):
    log_history = json_data.get("log_history", [])

    epochs = [entry["epoch"] for entry in log_history[:-2]]
    losses = [entry["loss"] for entry in log_history[:-2]]

    plt.plot(epochs, losses, marker='o', linestyle='-', color='b', markersize=0.2, linewidth=0.3)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
    plt.savefig("loss.png")

if __name__ == "__main__":
    # 读取 JSON 文件
    with open("./output/trainer_state.json", "r") as f:
        json_data = json.load(f)

    # 画曲线图
    plot_loss_curve(json_data)
