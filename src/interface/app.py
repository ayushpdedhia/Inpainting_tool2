# src/interface/app.py
# cd D:\Inpainting_tool2
# python -m streamlit run src/interface/app.py
import streamlit as st
from PIL import Image
import os
import numpy as np

from src.interface.components.ui_components import UIComponents
from src.interface.components.canvas_handler import CanvasHandler
from src.core.model_manager import ModelManager
from src.utils.image_processor import ImageProcessor

class InpaintingApp:
    def __init__(self):
        self.ui = UIComponents()
        self.canvas = CanvasHandler()
        self.config = self._load_config()
        self.image_processor = ImageProcessor()
        self.CANVAS_SIZE = self.config['interface']['canvas_size']
        self.progress_bar = None

    def _load_config(self):
        """Load configuration file"""
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
        with open(config_path, 'r') as f:
            import yaml
            return yaml.safe_load(f)
    
    @st.cache_resource
    def initialize_model(self) -> ModelManager:
        """Initialize the model manager with proper error handling"""
        try:
            model_manager = ModelManager()

            # Add device selection based on config
            device = self.config['model']['device']
            st.write(f"Using device: {device}")
            
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

    def process_image(self, image: Image.Image, mask: np.ndarray, model_name: str) -> Image.Image:
        """Process the image using the selected model with progress tracking"""
        try:
            # Initialize progress tracking
            self.progress_bar = st.progress(0)
            st.write("Initializing model...")
            
            # Initialize model (20%)
            self.progress_bar.progress(20)
            model_manager = self.initialize_model()
            if model_manager is None:
                st.error("Failed to initialize model")
                return None

            # Preprocess image and mask (40%)
            st.write("Preprocessing image...")
            self.progress_bar.progress(40)
            try:
                processed_image, processed_mask = self.image_processor.preprocess(
                    image, 
                    mask, 
                    target_size=(self.CANVAS_SIZE, self.CANVAS_SIZE)
                )
            except Exception as e:
                st.error("Error during preprocessing")
                raise e

            # Run inference (60%)
            st.write("Running inpainting...")
            self.progress_bar.progress(60)
            result = model_manager.inpaint(processed_image, processed_mask, model_name)
            
            # Post-process result (80%)
            st.write("Postprocessing result...")
            self.progress_bar.progress(80)
            if result is not None:
                final_image = self.image_processor.postprocess(result)
                self.progress_bar.progress(100)
                st.write("Processing complete!")
                return final_image
            else:
                raise ValueError("Model returned None result")

        except Exception as e:
            self.ui.show_error(e)
            return None
        finally:
            # Clean up progress bar
            if hasattr(self, 'progress_bar') and self.progress_bar is not None:
                self.progress_bar.empty()

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