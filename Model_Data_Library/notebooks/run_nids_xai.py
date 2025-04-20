from nids_xai_lib import NIDSXAILib
import numpy as np

# Test with scaler
print("Testing with scaler enabled...")
nids = NIDSXAILib(model_path='nids_model.pth', use_scaler=True)
dummy_data = nids.create_dummy_data(n_samples=1)

# Run LIME
lime_result = nids.run_lime(dummy_data, training_data=None, sample_idx=0, num_features=10)
print(f"LIME Prediction (with scaler): {lime_result['predicted_class']} (Confidence: {lime_result['confidence']:.4f})")
print(f"LIME Plot saved to: {lime_result['plot_path']}")
print("LIME Explanation:")
for line in lime_result['explanation_text']:
    print(line)

# Test without scaler
print("\nTesting without scaler...")
nids_no_scaler = NIDSXAILib(model_path='nids_model.pth', use_scaler=False)
dummy_data_no_scaler = nids_no_scaler.create_dummy_data(n_samples=1)

# Run LIME
lime_result_no_scaler = nids_no_scaler.run_lime(dummy_data_no_scaler, training_data=None, sample_idx=0, num_features=10)
print(f"LIME Prediction (no scaler): {lime_result_no_scaler['predicted_class']} (Confidence: {lime_result_no_scaler['confidence']:.4f})")
print(f"LIME Plot saved to: {lime_result_no_scaler['plot_path']}")
print("LIME Explanation:")
for line in lime_result_no_scaler['explanation_text']:
    print(line)

# Test with custom data
custom_data = np.array([[
    120.0, 100.0, 55.0, 3000.0, 250.0, 350.0, 65.0, 85.0, 85.0, 12000.0,
    95.0, 90.0, 60.0, 110.0, 8192.0, 600.0, 700.0, 6.0, 12.0, 12.0
]], dtype=np.float32)
lime_result_custom = nids.run_lime(custom_data, training_data=None, sample_idx=0)
print(f"\nCustom Data LIME Prediction (with scaler): {lime_result_custom['predicted_class']} (Confidence: {lime_result_custom['confidence']:.4f})")
print(f"Custom Data LIME Plot saved to: {lime_result_custom['plot_path']}")
print("Custom Data LIME Explanation:")
for line in lime_result_custom['explanation_text']:
    print(line)