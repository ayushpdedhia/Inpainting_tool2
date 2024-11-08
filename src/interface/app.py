# src/interface/app.py
# cd D:\Inpainting_tool2
# python -m streamlit run src/interface/app.py
import streamlit as st
from PIL import Image
import os
import numpy as np
import logging
from src.interface.components.ui_components import UIComponents
from src.interface.components.canvas_handler import CanvasHandler
from src.core.model_manager import ModelManager
from src.utils.image_processor import ImageProcessor
from src.utils.metrics_evaluator import MetricsEvaluator
import yaml
import plotly.express as px

logger = logging.getLogger(__name__)
class InpaintingApp:

    def __init__(self):
        self.ui = UIComponents()
        self.canvas = CanvasHandler()
        self.config = self._load_config()
        self.image_processor = ImageProcessor()
        self.CANVAS_SIZE = self.config['interface']['canvas_size']
        self.progress_bar = None
        self.metrics_evaluator = MetricsEvaluator()

    def _load_config(self):
        """Load configuration file"""
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @st.cache_resource
    def initialize_model(_self) -> ModelManager:
        """Initialize the model manager with proper error handling"""
        try:
            # Get config path 
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, '..', '..', 'config.yaml')
            
            # Create model manager - this will load models automatically
            model_manager = ModelManager(config_path)

            # Add device selection based on config
            device = _self.config['model']['device']
            st.write(f"Using device: {device}")
            
            # No need to verify weights directory or call initialize()
            # as ModelManager constructor handles all of this
            return model_manager
                
        except Exception as e:
            _self.ui.show_error(e)
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
            
            # First resize the image to match canvas size
            image = image.resize((self.CANVAS_SIZE, self.CANVAS_SIZE), Image.Resampling.LANCZOS)
            
            # Preprocess image and mask (40%)
            print(f"Image shape: {image.size}")
            print(f"Mask shape: {mask.shape}")
            print("=== Image and Mask Dimensions ===")
            print(f"Original Image size (W,H): {image.size}")
            print(f"Original Image mode: {image.mode}")
            print(f"Original Mask shape: {mask.shape}")
            print(f"Original Mask dtype: {mask.dtype}")
            
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
                self.progress_bar.progress(90)

                # Calculate and display metrics (90-100%)
                try:
                    metrics = self.metrics_evaluator.calculate_all_metrics(
                        np.array(image),
                        np.array(final_image),
                        mask
                    )
                    
                    # Display metrics in Streamlit
                    st.subheader("Quality Metrics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("PSNR", f"{metrics['psnr']:.2f} dB")
                        st.metric("SSIM", f"{metrics['ssim']:.3f}")
                        st.metric("Edge Coherence", f"{metrics['edge_coherence']:.3f}")
                        
                    with col2:
                        st.metric("Color Consistency", f"{metrics['color_consistency']:.3f}")
                        st.metric("Texture Similarity", f"{metrics['texture_similarity']:.3f}")
                        
                    # Display metrics history plot
                    st.subheader("Metrics History")
                    plot_data = self.metrics_evaluator.visualize_metrics()
                    if plot_data:
                        
                        
                        for metric in ['psnr', 'ssim', 'edge_coherence']:
                            fig = px.line(
                                x=plot_data['timestamps'],
                                y=plot_data[metric],
                                title=f"{metric.upper()} History"
                            )
                            st.plotly_chart(fig)
                            
                except Exception as e:
                    logger.error(f"Error calculating metrics: {str(e)}")
                    st.warning("Could not calculate image quality metrics")

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