from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from src.nids_xai_lib import NIDSXAILib
from src.gemini_utils import call_gemini_api
import logging
import os
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verify GOOGLE_API_KEY
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY environment variable not set")
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# Define model and plot paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # C:\Users\lemon\Desktop\nids-xai\Backend
MODEL_PATH = os.path.join(BASE_DIR, 'src', 'nids_model.pth')
PLOT_DIR = os.path.join(BASE_DIR, 'plots')

# Verify model path
if not os.path.exists(MODEL_PATH):
    logger.error(f"Model file not found at: {MODEL_PATH}")
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

logger.info(f"Using model path: {MODEL_PATH}")

# Ensure plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)
logger.info(f"Plot directory: {PLOT_DIR}")

# Initialize NIDSXAILib
try:
    nids = NIDSXAILib(model_path=MODEL_PATH, use_scaler=True)
    logger.info("NIDSXAILib initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize NIDSXAILib: {str(e)}")
    raise

# Define feature names and ranges (same as nids_xai_user_input.py)
FEATURE_INFO = [
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

FEATURE_NAMES = [name for name, _, _ in FEATURE_INFO]

@app.get("/plots/{filename:path}")
async def serve_plot(filename: str):
    """
    Serve plot images from the plots/ directory.

    Args:
        filename: Name of the plot file (e.g., ig_heatmap_sample_0_class_Benign.png).

    Returns:
        The requested image file or a JSON error response.
    """
    try:
        file_path = os.path.join(PLOT_DIR, filename)
        if not os.path.exists(file_path):
            logger.error(f"Plot file not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"Plot not found: {filename}")
        return FileResponse(file_path)
    except Exception as e:
        logger.error(f"Error serving plot {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving plot: {str(e)}")

@app.post("/predict")
async def predict(data: dict = Body(...)):
    """
    FastAPI endpoint to accept 20 feature values, run Integrated Gradients and LIME,
    move plots to plots/ directory, call Gemini API, and return results.

    Request Body:
        JSON object with 20 feature values (keys matching FEATURE_NAMES).

    Returns:
        JSON response with predictions, confidence, Integrated Gradients, LIME results,
        and Gemini API summary.
    """
    try:
        # Validate input features
        if not data:
            logger.error("No input data provided")
            raise HTTPException(status_code=400, detail="No input data provided")

        input_data = []
        for feature_name, min_val, max_val in FEATURE_INFO:
            if feature_name not in data:
                logger.error(f"Missing feature: {feature_name}")
                raise HTTPException(status_code=400, detail=f"Missing feature: {feature_name}")
            try:
                value = float(data[feature_name])
                if not (min_val <= value <= max_val):
                    logger.error(f"Value for {feature_name} out of range")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Value for {feature_name} must be between {min_val} and {max_val}"
                    )
                input_data.append(value)
            except (ValueError, TypeError):
                logger.error(f"Invalid value for {feature_name}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for {feature_name}: must be a number"
                )

        # Convert to numpy array with shape (1, 20)
        input_data = np.array([input_data], dtype=np.float32)
        logger.info("Input data validated successfully")

        # Run Integrated Gradients
        ig_result = nids.run_integrated_gradients(input_data, target_class=None, steps=20)

        # Run LIME
        lime_result = nids.run_lime(input_data, training_data=None, sample_idx=0, num_features=10)

        # Move plots to plots/ directory
        for plot_path in [ig_result['heatmap_path'], ig_result['bar_plot_path'], lime_result['plot_path']]:
            src_path = os.path.join(BASE_DIR, plot_path)
            dest_path = os.path.join(PLOT_DIR, os.path.basename(plot_path))
            if os.path.exists(src_path):
                shutil.move(src_path, dest_path)
                logger.info(f"Moved plot from {src_path} to {dest_path}")
            else:
                logger.warning(f"Plot not found at {src_path}, skipping move")

        # Update plot paths in results
        ig_result['heatmap_path'] = os.path.join('plots', os.path.basename(ig_result['heatmap_path']))
        ig_result['bar_plot_path'] = os.path.join('plots', os.path.basename(ig_result['bar_plot_path']))
        lime_result['plot_path'] = os.path.join('plots', os.path.basename(lime_result['plot_path']))

        # Call Gemini API with plots and text prompt
        image_paths = [
            os.path.join(BASE_DIR, ig_result['heatmap_path']),
            os.path.join(BASE_DIR, ig_result['bar_plot_path']),
            os.path.join(BASE_DIR, lime_result['plot_path'])
        ]
        text_prompt = (
            f"These are explainable AI (XAI) results for a network intrusion detection model.\n\n"
            f"**Prediction Summary:**\n"
            f"- The model predicted **'{lime_result['predicted_class']}'** with **{lime_result['confidence']:.2%}** confidence (LIME).\n"
            f"- Integrated Gradients also predicted **'{ig_result['predicted_class']}'** with **{ig_result['confidence']:.2%}** confidence.\n\n"
            f"**Visual Explanations:**\n"
            f"1. **Integrated Gradients Heatmap:** Shows feature attributions for the CNN branch.\n"
            f"2. **Integrated Gradients Bar Plot:** Displays feature importance for the LSTM branch.\n"
            f"3. **LIME Explanation Plot:** Highlights individual feature contributions:\n"
            f"   - {'; '.join(lime_result['explanation_text'])}\n\n"
            f"Please analyze these visualizations and summarize the most influential features contributing to the model's prediction."
        )

        gemini_response = call_gemini_api(image_paths, text_prompt)
        gemini_summary = gemini_response if gemini_response else "Failed to get Gemini API response"

        # Prepare response
        response = {
            'integrated_gradients': {
                'prediction': ig_result['predicted_class'],
                'confidence': float(ig_result['confidence']),
                'heatmap_path': ig_result['heatmap_path'],
                'bar_plot_path': ig_result['bar_plot_path']
            },
            'lime': {
                'prediction': lime_result['predicted_class'],
                'confidence': float(lime_result['confidence']),
                'explanation': lime_result['explanation_text'],
                'plot_path': lime_result['plot_path']
            },
            'gemini_summary': gemini_summary
        }

        logger.info("Analysis completed successfully")
        return response

    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))