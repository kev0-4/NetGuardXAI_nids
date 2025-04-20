import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, Tuple, Optional
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set memory allocation config
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class HybridCNNBiLSTM(nn.Module):
    """
    Hybrid CNN-BiLSTM model for intrusion detection.
    """

    def __init__(self, cnn_input_shape, lstm_input_shape, num_classes):
        super(HybridCNNBiLSTM, self).__init__()

        # CNN branch
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(0.25)

        cnn_output_height = cnn_input_shape[1] // 4
        cnn_output_width = cnn_input_shape[2] // 4
        self.cnn_output_size = 64 * cnn_output_height * cnn_output_width

        # BiLSTM branch
        self.lstm1 = nn.LSTM(
            input_size=lstm_input_shape[1],
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )
        self.dropout_lstm = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=32,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layers
        combined_size = self.cnn_output_size + 64
        self.fc1 = nn.Linear(combined_size, 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x_cnn, x_lstm):
        # CNN branch
        x1 = self.conv1(x_cnn)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.pool1(x1)
        x1 = self.dropout1(x1)

        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)
        x1 = self.pool2(x1)
        x1 = self.dropout2(x1)
        x1 = x1.view(x1.size(0), -1)

        # BiLSTM branch
        x2, _ = self.lstm1(x_lstm)
        x2 = self.dropout_lstm(x2)
        x2, _ = self.lstm2(x2)
        x2 = x2[:, -1, :]

        # Combine and classify
        combined = torch.cat((x1, x2), dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x


class NIDSXAILib:
    """
    Library for loading a pre-trained NIDS model, making predictions, and performing XAI analysis.
    """

    def __init__(self, model_path: str = 'nids_model.pth', device: str = None, use_scaler: bool = False):
        """
        Initialize the NIDS XAI library.

        Args:
            model_path: Path to the pre-trained model checkpoint (.pth file)
            device: Device to run on ('cuda' or 'cpu'). If None, auto-detects.
            use_scaler: If True, applies StandardScaler from model_path (default: False)
        """
        self.device = torch.device(
            device if device else "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model = None
        self.scaler = None
        self.use_scaler = use_scaler
        self.feature_names = [
            'Avg Packet Size', 'Packet Length Mean', 'Bwd Packet Length Std', 'Packet Length Variance',
            'Bwd Packet Length Max', 'Packet Length Max', 'Packet Length Std', 'Fwd Packet Length Mean',
            'Avg Fwd Segment Size', 'Flow Bytes/s', 'Avg Bwd Segment Size', 'Bwd Packet Length Mean',
            'Fwd Packets/s', 'Flow Packets/s', 'Init Fwd Win Bytes', 'Subflow Fwd Bytes',
            'Fwd Packets Length Total', 'Fwd Act Data Packets', 'Total Fwd Packets', 'Subflow Fwd Packets'
        ]
        self.class_names = ['Benign', 'DoS', 'DDoS', 'Bruteforce', 'Botnet']

        # Load model
        try:
            logger.info(f"Loading model from {model_path}...")
            self.model = HybridCNNBiLSTM(
                cnn_input_shape=(1, 5, 5),
                lstm_input_shape=(10, 2),
                num_classes=len(self.class_names)
            ).to(self.device)
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if use_scaler and 'scaler' in checkpoint:
                self.scaler = checkpoint['scaler']
                logger.info("Loaded StandardScaler from model checkpoint")
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def create_dummy_data(self, n_samples: int = 1) -> np.ndarray:
        """
        Create dummy input samples with 20 features.

        Args:
            n_samples: Number of samples to generate (default: 1)

        Returns:
            Numpy array of shape (n_samples, 20) with dummy values.
        """
        # Define realistic ranges based on typical network traffic features
        ranges = [
            (50.0, 200.0),   # Avg Packet Size
            (50.0, 150.0),   # Packet Length Mean
            (0.0, 100.0),    # Bwd Packet Length Std
            (0.0, 5000.0),   # Packet Length Variance
            (0.0, 500.0),    # Bwd Packet Length Max
            (100.0, 1000.0),  # Packet Length Max
            (0.0, 200.0),    # Packet Length Std
            (50.0, 150.0),   # Fwd Packet Length Mean
            (50.0, 150.0),   # Avg Fwd Segment Size
            (1000.0, 20000.0),  # Flow Bytes/s
            (50.0, 150.0),   # Avg Bwd Segment Size
            (50.0, 150.0),   # Bwd Packet Length Mean
            (10.0, 100.0),   # Fwd Packets/s
            (10.0, 200.0),   # Flow Packets/s
            (1024.0, 65535.0),  # Init Fwd Win Bytes
            (100.0, 1000.0),  # Subflow Fwd Bytes
            (100.0, 2000.0),  # Fwd Packets Length Total
            (1.0, 20.0),     # Fwd Act Data Packets
            (1.0, 50.0),     # Total Fwd Packets
            (1.0, 50.0)      # Subflow Fwd Packets
        ]

        # Generate samples with uniform distribution within ranges
        dummy_data = np.zeros((n_samples, 20), dtype=np.float32)
        for i, (min_val, max_val) in enumerate(ranges):
            dummy_data[:, i] = np.random.uniform(min_val, max_val, n_samples)

        # Apply scaler if enabled
        if self.use_scaler and self.scaler is not None:
            dummy_data = self.scaler.transform(dummy_data)

        logger.info(f"Created {n_samples} dummy data samples")
        return dummy_data

    def preprocess_data(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess input data by reshaping for CNN and LSTM inputs.

        Args:
            X: Input features, shape (n_samples, 20)

        Returns:
            Tuple of (X_cnn, X_lstm) with shapes (n_samples, 1, 5, 5) and (n_samples, 10, 2)
        """
        try:
            # Validate input
            if X.shape[1] != 20:
                raise ValueError(f"Expected 20 features, got {X.shape[1]}")

            # Apply scaler if enabled
            X_processed = X.copy()
            if self.use_scaler and self.scaler is not None:
                X_processed = self.scaler.transform(X_processed)

            # Reshape for CNN (1, 5, 5)
            n_samples = X_processed.shape[0]
            X_padded = np.pad(X_processed, ((0, 0), (0, 5)),
                              mode='constant')  # Pad to 25 features
            X_cnn = X_padded.reshape(n_samples, 1, 5, 5)

            # Reshape for LSTM (10, 2)
            X_lstm = X_processed.reshape(n_samples, 10, 2)

            logger.info(
                f"Preprocessed {n_samples} samples: CNN shape {X_cnn.shape}, LSTM shape {X_lstm.shape}")
            return X_cnn, X_lstm
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on input data.

        Args:
            X: Input features, shape (n_samples, 20)

        Returns:
            Tuple of (predictions, probabilities) where predictions are class indices
            and probabilities are softmax outputs.
        """
        try:
            X_cnn, X_lstm = self.preprocess_data(X)
            X_cnn = torch.tensor(X_cnn, dtype=torch.float32).to(self.device)
            X_lstm = torch.tensor(X_lstm, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                outputs = self.model(X_cnn, X_lstm)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

            logger.info(f"Generated predictions for {X.shape[0]} samples")
            return preds.cpu().numpy(), probs.cpu().numpy()
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def lime_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediction function for LIME, returning probabilities for all classes.

        Args:
            X: Input features, shape (n_samples, 20)

        Returns:
            Numpy array of shape (n_samples, num_classes) with class probabilities.
        """
        try:
            # Ensure input is 2D
            if X.ndim == 1:
                X = X.reshape(1, -1)

            # Validate input shape
            if X.shape[1] != 20:
                raise ValueError(f"Expected 20 features, got {X.shape[1]}")

            _, probs = self.predict(X)
            logger.info(f"LIME predict probabilities: {probs[0]}")
            return probs
        except Exception as e:
            logger.error(f"Error in LIME prediction: {str(e)}")
            raise

    def run_integrated_gradients(self, X: np.ndarray, target_class: Optional[int] = None,
                                 steps: int = 50) -> Dict[str, Any]:
        """
        Run Integrated Gradients analysis for the input data.

        Args:
            X: Input features, shape (n_samples, 20)
            target_class: Target class index (int) or None for predicted class
            steps: Number of interpolation steps for Integrated Gradients

        Returns:
            Dictionary with predictions, probabilities, and XAI results
        """
        try:
            # Preprocess data
            X_cnn, X_lstm = self.preprocess_data(X)
            X_cnn = torch.tensor(X_cnn, dtype=torch.float32).to(self.device)
            X_lstm = torch.tensor(X_lstm, dtype=torch.float32).to(self.device)

            # Get predictions
            preds, probs = self.predict(X)
            sample_number = 0  # For single-sample analysis
            pred_label = preds[sample_number]
            if target_class is None:
                target_class = pred_label

            # Validate target class
            if target_class >= len(self.class_names):
                raise ValueError(
                    f"Target class {target_class} out of range (max: {len(self.class_names)-1})")

            class_name = self.class_names[target_class]
            logger.info(
                f"Running Integrated Gradients for sample {sample_number}, class {class_name}")

            # Compute Integrated Gradients
            x_cnn = X_cnn[sample_number:sample_number +
                          1].detach().requires_grad_(True)
            x_lstm = X_lstm[sample_number:sample_number +
                            1].detach().requires_grad_(True)

            baseline_cnn = torch.zeros_like(x_cnn, device=self.device)
            baseline_lstm = torch.zeros_like(x_lstm, device=self.device)

            integrated_grad_cnn = torch.zeros_like(x_cnn, device=self.device)
            integrated_grad_lstm = torch.zeros_like(x_lstm, device=self.device)

            for step in range(1, steps + 1):
                alpha = step / steps
                interpolated_cnn = baseline_cnn + \
                    alpha * (x_cnn - baseline_cnn)
                interpolated_lstm = baseline_lstm + \
                    alpha * (x_lstm - baseline_lstm)
                interpolated_cnn = interpolated_cnn.detach().requires_grad_(True)
                interpolated_lstm = interpolated_lstm.detach().requires_grad_(True)

                try:
                    with torch.no_grad():
                        for param in self.model.parameters():
                            param.requires_grad_(False)
                    output = self.model(interpolated_cnn, interpolated_lstm)
                    score = output[0, target_class]
                    score.backward()
                    if interpolated_cnn.grad is not None:
                        integrated_grad_cnn += interpolated_cnn.grad.detach()
                    if interpolated_lstm.grad is not None:
                        integrated_grad_lstm += interpolated_lstm.grad.detach()
                finally:
                    for param in self.model.parameters():
                        param.requires_grad_(True)

                interpolated_cnn.grad = None
                interpolated_lstm.grad = None
                torch.cuda.empty_cache()

            attribution_cnn = (x_cnn - baseline_cnn) * \
                integrated_grad_cnn / steps
            attribution_lstm = (x_lstm - baseline_lstm) * \
                integrated_grad_lstm / steps

            # Visualize attributions
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

            # CNN heatmap
            attr_cnn = attribution_cnn.cpu().detach().numpy()[0, 0]
            feature_names = self.feature_names + \
                ['pad_{}'.format(i) for i in range(5)]
            cnn_labels = np.array(feature_names[:25]).reshape(5, 5)
            sns.heatmap(
                np.abs(attr_cnn),
                annot=cnn_labels,
                fmt='s',
                cmap='YlOrRd',
                ax=ax1,
                cbar_kws={'label': 'Attribution Magnitude'},
                annot_kws={'size': 8, 'rotation': 45},
                square=True
            )
            ax1.set_title(
                f'CNN Attributions (Sample {sample_number}, Class: {class_name})', fontsize=12, pad=15)
            ax1.set_xticks(np.arange(5) + 0.5)
            ax1.set_yticks(np.arange(5) + 0.5)
            ax1.set_xticklabels(range(5), fontsize=10)
            ax1.set_yticklabels(range(5), fontsize=10)

            # LSTM heatmap
            attr_lstm = attribution_lstm.cpu().detach().numpy()[0]
            lstm_labels = np.array(self.feature_names).reshape(10, 2)
            sns.heatmap(
                np.abs(attr_lstm),
                annot=lstm_labels,
                fmt='s',
                cmap='viridis',
                ax=ax2,
                cbar_kws={'label': 'Attribution Magnitude'},
                annot_kws={'size': 8},
                square=True
            )
            ax2.set_title(
                f'LSTM Attributions (Sample {sample_number}, Class: {class_name})', fontsize=12, pad=15)
            ax2.set_xlabel('Feature', fontsize=10)
            ax2.set_ylabel('Timestep', fontsize=10)
            ax2.set_xticks(np.arange(2) + 0.5)
            ax2.set_yticks(np.arange(10) + 0.5)
            ax2.set_xticklabels(['F0', 'F1'], fontsize=10)
            ax2.set_yticklabels(range(10), fontsize=10)

            plt.tight_layout()
            plt.savefig(
                f'ig_heatmap_sample_{sample_number}_class_{class_name}.png', dpi=300, bbox_inches='tight')
            plt.close()

            # LSTM feature importance bar plot
            feature_importance = np.abs(attr_lstm.reshape(-1))
            plt.figure(figsize=(12, 6))
            plt.bar(self.feature_names, feature_importance)
            plt.title(
                f'LSTM Feature Importance (Sample {sample_number}, Class: {class_name})', fontsize=14)
            plt.xlabel('Feature', fontsize=12)
            plt.ylabel('Attribution Magnitude', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.tight_layout()
            plt.savefig(
                f'ig_lstm_bar_sample_{sample_number}_class_{class_name}.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(
                f"Integrated Gradients analysis completed for sample {sample_number}, class {class_name}")
            return {
                'predictions': preds,
                'probabilities': probs,
                'predicted_class': self.class_names[pred_label],
                'confidence': probs[0, pred_label],
                'target_class': target_class,
                'target_class_name': class_name,
                'attributions_cnn': attribution_cnn.detach().cpu().numpy(),
                'attributions_lstm': attribution_lstm.detach().cpu().numpy(),
                'input_cnn': x_cnn.detach().cpu().numpy(),
                'input_lstm': x_lstm.detach().cpu().numpy(),
                'heatmap_path': f'ig_heatmap_sample_{sample_number}_class_{class_name}.png',
                'bar_plot_path': f'ig_lstm_bar_sample_{sample_number}_class_{class_name}.png'
            }
        except Exception as e:
            logger.error(f"Error in Integrated Gradients analysis: {str(e)}")
            raise

    def run_lime(self, X: np.ndarray, training_data: Optional[np.ndarray] = None,
                 sample_idx: int = 0, num_features: int = 10) -> Dict[str, Any]:
        """
        Run LIME analysis for the input data.

        Args:
            X: Input features, shape (n_samples, 20)
            training_data: Training data for LIME explainer, shape (n_train_samples, 20). If None, uses dummy data.
            sample_idx: Index of the sample to explain (default: 0)
            num_features: Number of top features to show in explanation (default: 10)

        Returns:
            Dictionary with predictions, probabilities, and LIME results
        """
        try:
            # Validate input
            if X.shape[1] != 20:
                raise ValueError(f"Expected 20 features, got {X.shape[1]}")
            if sample_idx >= X.shape[0]:
                raise ValueError(
                    f"Sample index {sample_idx} out of range for {X.shape[0]} samples")

            # Use provided training data or generate dummy data
            if training_data is None:
                training_data = self.create_dummy_data(
                    n_samples=1000)  # More samples for better LIME
            if training_data.shape[1] != 20:
                raise ValueError(
                    f"Training data expected 20 features, got {training_data.shape[1]}")

            # Initialize LIME explainer
            explainer = LimeTabularExplainer(
                training_data=training_data,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification',
                discretize_continuous=True
            )

            # Get predictions
            preds, probs = self.predict(X)
            sample_number = sample_idx
            pred_label = preds[sample_number]
            predicted_class = self.class_names[pred_label]
            confidence = probs[sample_number, pred_label]

            logger.info(
                f"Running LIME for sample {sample_number}, predicted class: {predicted_class}")

            # Generate LIME explanation
            exp = explainer.explain_instance(
                X[sample_number],
                self.lime_predict,
                num_features=num_features,
                num_samples=5000  # Increase samples for better stability
            )

            # Extract feature weights
            feature_weights = exp.as_list()
            logger.info(f"LIME feature weights: {feature_weights}")

            # Visualize LIME explanation
            plt.figure(figsize=(10, 6))
            features = [name for name, _ in feature_weights]
            weights = [weight for _, weight in feature_weights]
            if all(w == 0 for w in weights):
                logger.warning("All LIME weights are 0.0, plot may be blank")
                plt.text(0.5, 0.5, "No significant feature contributions",
                         horizontalalignment='center', verticalalignment='center')
            else:
                colors = ['green' if w > 0 else 'red' for w in weights]
                plt.barh(features, weights, color=colors)
            plt.title(
                f'LIME Explanation for Sample {sample_number} (Predicted: {predicted_class})')
            plt.xlabel('Feature Contribution')
            plt.tight_layout()
            plot_path = f'lime_explanation_sample_{sample_number}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Detailed text output
            explanation_text = []
            for feature_name, weight in feature_weights:
                explanation_text.append(f"{feature_name}: {weight:.4f}")

            logger.info(f"LIME analysis completed for sample {sample_number}")
            return {
                'predictions': preds,
                'probabilities': probs,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'lime_feature_weights': feature_weights,
                'explanation_text': explanation_text,
                'plot_path': plot_path
            }
        except Exception as e:
            logger.error(f"Error in LIME analysis: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    nids = NIDSXAILib(use_scaler=True)  # Enable scaler for testing
    dummy_data = nids.create_dummy_data(n_samples=1)
    # Run Integrated Gradients
    ig_result = nids.run_integrated_gradients(dummy_data)
    print(
        f"IG Prediction: {ig_result['predicted_class']} (Confidence: {ig_result['confidence']:.4f})")
    print(f"IG Heatmap saved to: {ig_result['heatmap_path']}")
    print(f"IG Bar plot saved to: {ig_result['bar_plot_path']}")
    # Run LIME
    lime_result = nids.run_lime(dummy_data)
    print(
        f"LIME Prediction: {lime_result['predicted_class']} (Confidence: {lime_result['confidence']:.4f})")
    print(f"LIME Plot saved to: {lime_result['plot_path']}")
    print("LIME Explanation:")
    for line in lime_result['explanation_text']:
        print(line)
