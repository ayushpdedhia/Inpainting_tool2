# src/interface/app.py
import streamlit as st
from PIL import Image
import os
import numpy as np

from .components.ui_components import UIComponents
from .components.canvas_handler import CanvasHandler
from ..core.model_manager import ModelManager
from ..utils.image_processor import ImageProcessor

class InpaintingApp:
    def __init__(self):
        self.ui = UIComponents()
        self.canvas = CanvasHandler()
        self.image_processor = ImageProcessor()
        self.CANVAS_SIZE = 512

    def initialize_model(self) -> ModelManager:
        """Initialize the model manager with proper error handling"""
        try:
            model_manager = ModelManager()
            
            # Load model weights
            current_dir = os.path.dirname(os.path.abspath(__file__))
            weights_dir = os.path.join(current_dir, '..', '..', 'weights')
            
            # Verify weights directory exists
            if not os.path.exists(weights_dir):
                raise FileNotFoundError(f"Weights directory not found at {weights_dir}")
            
            # Initialize model
            model_manager.initialize()
            return model_manager
            
        except Exception as e:
            self.ui.show_error(e)
            return None

    def process_image(self, image: Image.Image, mask: 'np.ndarray', model_name: str) -> Image.Image:
        """Process the image using the selected model"""
        try:
            # Initialize model
            model_manager = self.initialize_model()
            if model_manager is None:
                return None

            # Preprocess image and mask
            processed_image, processed_mask = self.image_processor.preprocess(
                image, 
                mask, 
                target_size=(self.CANVAS_SIZE, self.CANVAS_SIZE)
            )

            # Run inference
            result = model_manager.inpaint(processed_image, processed_mask, model_name)
            
            # Post-process result
            final_image = self.image_processor.postprocess(result)
            return final_image

        except Exception as e:
            self.ui.show_error(e)
            return None

    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(page_title="Image Inpainting Tool", layout="wide")
        st.title("Image Inpainting Tool")
        
        # Setup UI
        self.ui.setup_styles()
        controls = self.ui.create_sidebar_controls()
        
        if controls['uploaded_file'] is not None:
            try:
                # Load and process image
                image = Image.open(controls['uploaded_file'])
                
                # Display canvas and get mask
                mask, process_clicked = self.canvas.display_canvas_columns(image, controls)
                
                # Process image if requested
                if process_clicked and mask is not None:
                    with st.spinner("Processing..."):
                        result = self.process_image(image, mask, controls['model_name'])
                        if result is not None:
                            st.image(result, width=self.CANVAS_SIZE, caption="Inpainting Result")
                
                # Display instructions
                self.ui.display_instructions()
                
            except Exception as e:
                self.ui.show_error(e)
                st.write("Please try again with a different image or refresh the page")

def main():
    app = InpaintingApp()
    app.run()

if __name__ == "__main__":
    main()