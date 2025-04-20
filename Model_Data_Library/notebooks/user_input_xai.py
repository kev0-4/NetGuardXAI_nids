import numpy as np
from nids_xai_lib import NIDSXAILib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_user_input():
    """
    Prompt the user to input values for the 20 features, validate against ranges, and return as a numpy array.

    Returns:
        np.ndarray: Array of shape (1, 20) with user-provided feature values.
    """
    # Define feature names and ranges (same as create_dummy_data in nids_xai_lib.py)
    feature_info = [
        ('Avg Packet Size', 50.0, 200.0),
        ('Packet Length Mean', 50.0, 150.0),
        ('Bwd Packet Length Std', 0.0, 100.0),
        ('Packet Length Variance', 0.0, 5000.0),
        ('Bwd Packet Length Max', 0.0, 500.0),
        ('Packet Length Max', 100.0, 1000.0),
        ('Packet Length Std', 0.0, 200.0),
        ('Fwd Packet Length Mean', 50.0, 150.0),
        ('Avg Fwd Segment Size', 50.0, 150.0),
        ('Flow Bytes/s', 1000.0, 20000.0),
        ('Avg Bwd Segment Size', 50.0, 150.0),
        ('Bwd Packet Length Mean', 50.0, 150.0),
        ('Fwd Packets/s', 10.0, 100.0),
        ('Flow Packets/s', 10.0, 200.0),
        ('Init Fwd Win Bytes', 1024.0, 65535.0),
        ('Subflow Fwd Bytes', 100.0, 1000.0),
        ('Fwd Packets Length Total', 100.0, 2000.0),
        ('Fwd Act Data Packets', 1.0, 20.0),
        ('Total Fwd Packets', 1.0, 50.0),
        ('Subflow Fwd Packets', 1.0, 50.0)
    ]

    print("Enter values for the 20 features. Each feature has a specified range.")
    user_data = []

    for feature_name, min_val, max_val in feature_info:
        while True:
            try:
                value = input(
                    f"{feature_name} (range {min_val} to {max_val}): ")
                value = float(value)
                if min_val <= value <= max_val:
                    user_data.append(value)
                    break
                else:
                    print(
                        f"Error: Value must be between {min_val} and {max_val}. Try again.")
            except ValueError:
                print("Error: Please enter a valid number. Try again.")

    # Convert to numpy array with shape (1, 20)
    user_data = np.array([user_data], dtype=np.float32)
    logger.info("User input collected successfully")
    return user_data


def main():
    """
    Main function to collect user input, run XAI analyses, and display results.
    """
    try:
        # Initialize NIDSXAILib with scaler enabled (adjust as needed)
        nids = NIDSXAILib(model_path='nids_model.pth', use_scaler=True)

        # Get user input
        print("\nCollecting feature values for network traffic analysis...")
        user_data = get_user_input()

        # Run Integrated Gradients
        print("\nRunning Integrated Gradients...")
        ig_result = nids.run_integrated_gradients(
            user_data, target_class=None, steps=20)
        print(
            f"Integrated Gradients Prediction: {ig_result['predicted_class']} (Confidence: {ig_result['confidence']:.4f})")
        print(f"IG Heatmap saved to: {ig_result['heatmap_path']}")
        print(f"IG Bar plot saved to: {ig_result['bar_plot_path']}")

        # Run LIME
        print("\nRunning LIME...")
        lime_result = nids.run_lime(
            user_data, training_data=None, sample_idx=0, num_features=10)
        print(
            f"LIME Prediction: {lime_result['predicted_class']} (Confidence: {lime_result['confidence']:.4f})")
        print(f"LIME Plot saved to: {lime_result['plot_path']}")
        print("LIME Explanation:")
        for line in lime_result['explanation_text']:
            print(line)

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
