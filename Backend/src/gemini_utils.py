import os
import logging
from typing import List, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY environment variable not set")
    raise ValueError("GOOGLE_API_KEY environment variable not set")

genai.configure(api_key=GOOGLE_API_KEY)

def call_gemini_api(
    image_paths: List[str],
    text_prompt: str,
    model_name: str = 'gemini-2.0-flash',
    max_tokens: int = 1000
) -> Optional[str]:
    """
    Call the Google Gemini API with images and a text prompt, and return the response text.
    
    Args:
        image_paths: List of file paths to three images (e.g., ['plots/ig_heatmap.png', ...]).
        text_prompt: Text prompt to send with the images.
        model_name: Gemini model to use (default: 'gemini-1.5-pro').
        max_tokens: Maximum number of tokens in the response (default: 1000).
    
    Returns:
        Response text from the Gemini API, or None if an error occurs.
    """
    try:
        # Validate image paths
        if len(image_paths) != 3:
            logger.error(f"Expected 3 image paths, got {len(image_paths)}")
            raise ValueError(f"Expected 3 image paths, got {len(image_paths)}")
        
        for image_path in image_paths:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Initialize the Gemini model
        model = genai.GenerativeModel(model_name)
        logger.info(f"Initialized Gemini model: {model_name}")
        
        # Prepare content: text prompt + images
        content = [text_prompt]
        for image_path in image_paths:
            with open(image_path, 'rb') as img_file:
                image_data = img_file.read()
            content.append({
                'mime_type': 'image/png',
                'data': image_data
            })
        
        # Generate response
        response = model.generate_content(
            content,
            generation_config={
                'max_output_tokens': max_tokens,
                'temperature': 0.7  # Balanced creativity
            }
        )
        
        # Extract and return response text
        response_text = response.text
        logger.info("Gemini API call successful")
        return response_text
    
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    image_paths = [
        'plots/ig_heatmap_sample_0_class_Benign.png',
        'plots/ig_lstm_bar_sample_0_class_Benign.png',
        'plots/lime_explanation_sample_0.png'
    ]
    text_prompt = (
        "These are XAI results for a network intrusion detection model. "
        "The first image is an Integrated Gradients heatmap showing feature attributions for the CNN branch. "
        "The second image is an Integrated Gradients bar plot for the LSTM branch. "
        "The third image is a LIME explanation showing feature contributions to the prediction. "
        "The model predicted 'Benign' with 95% confidence. "
        "Please analyze these plots and provide a summary of the key features driving the prediction."
    )
    
    response = call_gemini_api(image_paths, text_prompt)
    if response:
        print("Gemini API Response:")
        print(response)
    else:
        print("Failed to get response from Gemini API")