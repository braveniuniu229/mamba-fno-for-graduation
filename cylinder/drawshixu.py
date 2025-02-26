import matplotlib.pyplot as plt
import numpy as np
import os

# Function to read a CSV file and return the data as a numpy array
def read_csv(file_path):
    return np.loadtxt(file_path, delimiter=",")

# Function to plot the timeseries of the five points for each model
def plot_predictions_and_true_values(models_files, top_5_coords, true_values_file, output_folder="output_plots"):
    # Read true values
    true_values = read_csv(true_values_file)

    # Ensure that all models have the same number of rows as the true values
    for model in models_files:
        predicted_values = read_csv(model["predicted"])
        assert predicted_values.shape == true_values.shape, f"Shape mismatch between {model['predicted']} and {true_values_file}"

    # Create output directory if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Define a color palette with softer tones
    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#3211bd"]  # Blue, orange, green, red, purple
    markers = ['o', '^', 's', 'D', 'p']  # Different marker shapes for each model

    # Define line styles for different models (all will be dashed)
    line_styles = ['--', '--', '--', '--', '--']

    # Iterate through the five coordinates and plot their timeseries
    for i, coord in enumerate(top_5_coords):
        plt.figure(figsize=(10, 6))

        # Extract the values for the current point (coordinate)
        true_values_for_point = true_values[:, i]

        # Plot the true values with a thicker dashed line to make it more visible
        plt.plot(true_values_for_point, label="Ground Truth", color='black', linestyle='-', linewidth=2)

        # Plot the predictions from each model with a different color and dashed line
        for j, model in enumerate(models_files):
            predicted_values = read_csv(model["predicted"])[:, i]
            plt.plot(predicted_values, label=model["name"], color=color_palette[j], linestyle=line_styles[j], linewidth=2)

            # Add different hollow markers at every multiple of 5 in the x-axis for each model
            for tick in range(0, len(predicted_values), 5):
                plt.plot(tick, predicted_values[tick], marker=markers[j], markerfacecolor='none', markeredgewidth=2, markersize=8, color=color_palette[j])

        # Add hollow circle markers at every multiple of 5 in the x-axis for true values
        for tick in range(0, len(true_values_for_point), 5):
            plt.plot(tick, true_values_for_point[tick], marker='o', markerfacecolor='none', markeredgewidth=2, markersize=8, color='black')

        # Customize x-axis ticks to only display every 5th value
        plt.xticks(ticks=np.arange(0, true_values_for_point.shape[0], 5), labels=np.arange(0, true_values_for_point.shape[0], 5))

        # Set plot labels without title (more suited for publication)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)

        # Move the legend to the left bottom corner with larger font size
        plt.legend(loc="lower left", fontsize=10, frameon=False)

        # Customize grid and background for a cleaner look suitable for publications
        plt.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
        plt.gca().set_facecolor('white')

        # Remove spines for a cleaner look
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # Save the plot
        plt.savefig(os.path.join(output_folder, f"point_{coord[0]}_{coord[1]}.png"), dpi=300, bbox_inches='tight')
        plt.close()

# Define the five points coordinates (these are fixed)
top_5_coords = [(40, 50), (120, 100), (200, 150), (250, 170), (310, 180)]

# List of models and their corresponding CSV files
model_files = [
    {"name": "GappyMLP", "predicted": "gappymlp_predicted_values.csv"},
    {"name": "Ours", "predicted": "TPSSM-FNO_predicted_values.csv"},
    {"name": "Shallow Decoder", "predicted": "SD_predicted_values.csv"},
    {"name": "VoronoiCNN", "predicted": "voronoicnn_predicted_values.csv"},
    {"name": "LSTM", "predicted": "lstm_predicted_values.csv"},
]

# File for true values
true_values_file = "true_values.csv"

# Plot for each model
plot_predictions_and_true_values(model_files, top_5_coords, true_values_file, output_folder="output_plots")
