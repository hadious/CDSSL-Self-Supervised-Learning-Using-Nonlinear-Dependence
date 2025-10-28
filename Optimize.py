import yaml
import optuna
import subprocess
import os
from collections import deque


CONFIG_FILE = "config.yaml"
PYTHON_EXEC = "/home/hadi/anaconda3/envs/DNN/bin/python"
MAIN_SCRIPT = "/home/hadi/Desktop/WorkSpace/SSL/SSL/main.py"

def load_config(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def save_config(config, file_path):
    with open(file_path, "w") as file:
        yaml.dump(config, file)

def objective(trial):
    config = load_config(CONFIG_FILE)

    # Sample weights for optimization
    weights = {
        "cross_correlation_between_samples_weight": trial.suggest_loguniform("cc_samples", 1e-2, 1.5),
        "cross_correlation_between_features_weight": trial.suggest_loguniform("cc_features", 1e-2, 1.5),
        "auto_correlation_between_samples_weight": trial.suggest_loguniform("ac_samples", 1e-7, 1e-2),
        "auto_correlation_between_features_weight": trial.suggest_loguniform("ac_features", 1e-7, 1e-2),
        "cross_dependence_between_samples_weight": trial.suggest_float("cd_samples", 1, 150),
        "cross_dependence_between_features_weight": trial.suggest_float("cd_features", 1, 50),
        "auto_dependence_between_samples_weight": trial.suggest_float("ad_samples", 1e-6, 10),
        "auto_dependence_between_features_weight": trial.suggest_float("ad_features", 1e-6, 10),
    }

    # Update the config file with the sampled weights
    for key, value in weights.items():
        config["loss_weights"][key] = value

    temp_config_path = "Temp_Config.yaml"
    save_config(config, temp_config_path)

    subprocess.run(
        [PYTHON_EXEC, MAIN_SCRIPT],
        env=dict(os.environ, CONFIG_PATH=temp_config_path),
        check=True,
    )


    with open("logs/log_ours_cifar_optimization/classifier_results/classifier.txt", "r") as file:

        last_lines = deque(file, maxlen=10)
        top1_accuracy = None
        for line in last_lines:
            if "Top-1 Accuracy:" in line:
                top1_accuracy = float(line.split("Top-1 Accuracy:")[1].split("%")[0].strip())
                break


    os.remove(temp_config_path)

    return -top1_accuracy

def optimize():
    study = optuna.create_study(direction="minimize")  # Optuna minimizes the objective
    study.optimize(objective, n_trials=100)

    print("Best Weights:", study.best_params)
    print("Best Score:", -study.best_value)  # Convert back to Top-1 Accuracy

    # Save the best configuration
    best_config = load_config(CONFIG_FILE)
    for key, value in study.best_params.items():
        best_config["loss_weights"][key] = value
    save_config(best_config, "Best_Config.yaml")

if __name__ == "__main__":
    optimize()
