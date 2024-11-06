# src/interface/components/canvas_handler.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from typing import Tuple, Dict, Optional

# Import MaskGenerator and MaskConfig from utils
from ...utils.mask_generator import MaskGenerator, MaskConfig

class CanvasHandler:
    """Handles canvas operations for the inpainting tool"""
    
    def __init__(self, canvas_size: int = 512, config=None):
        self.canvas_size = canvas_size
        self.config = config
        # Add real-time preview
        self.preview_container = None

        # Initialize mask generator with configuration
        try:
            self.mask_generator = MaskGenerator(
                height=canvas_size,
                width=canvas_size,
                channels=1,
                config=MaskConfig(
                    min_num_shapes=1,
                    max_num_shapes=10,
                    min_shape_size=10,
                    max_shape_size=int(canvas_size * 0.5)
                )
            )
            print(f"MaskGenerator initialized with canvas size: {canvas_size}")
        except Exception as e:
            st.error(f"Error initializing MaskGenerator: {str(e)}")
            self.mask_generator = None

    def setup_canvas(self, image: Image.Image, stroke_width: int, drawing_mode: str) -> Dict:
        """Setup the drawing canvas with real-time preview"""
        try:
            # Add real-time preview container
            preview_col = st.empty()

            # Add random mask generation button if mask generator is available
            if self.mask_generator is not None:
                if st.button("Generate Random Mask"):
                    try:
                        random_mask = self.mask_generator.sample()
                        preview_col.image(random_mask, caption="Generated Random Mask")
                        # Store the generated mask for later use
                        self._last_generated_mask = random_mask
                    except Exception as e:
                        st.error(f"Error generating random mask: {str(e)}")
            
            # Resize image with high-quality resampling
            resized_image = image.resize((self.canvas_size, self.canvas_size),
                                    Image.Resampling.LANCZOS)
            
            def on_change(canvas_result):
                """Update mask preview in real-time"""
                if canvas_result.image_data is not None:
                    mask = self.process_canvas_result(canvas_result)
                    if mask is not None:
                        preview_col.image(mask, caption="Real-time Mask Preview")
            
            return st_canvas(
                fill_color="#FFFFFF",
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                background_image=resized_image,
                drawing_mode=drawing_mode,
                height=self.canvas_size,
                width=self.canvas_size,
                key="canvas",
                display_toolbar=True,
                on_change=on_change
            )
        except Exception as e:
            st.error(f"Error setting up canvas: {str(e)}")
            return None

    def process_canvas_result(self, canvas_result: Dict) -> Optional[np.ndarray]:
        """Process canvas result to generate mask"""
        if canvas_result.image_data is None:
            if hasattr(self, '_last_generated_mask'):
                return self._last_generated_mask
            return None
            
        mask_display = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        
        if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
            # Handle rectangle drawing
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "rect":
                    x = int(obj["left"])
                    y = int(obj["top"])
                    w = int(obj["width"])
                    h = int(obj["height"])
                    cv2.rectangle(mask_display, (x, y), (x+w, y+h), 255, -1)
        else:
            # Handle freedraw
            mask_display = canvas_result.image_data[:, :, -1]
        
        return mask_display

    def display_canvas_columns(self, image: Image.Image, controls: Dict) -> Tuple[np.ndarray, bool]:
        """Display canvas columns and handle canvas interaction"""
        # Initialize preview container if needed
        if not hasattr(self, 'preview_container'):
            self.preview_container = st.empty()
        
        def validate_mask(mask):
            """Validate mask properties"""
            if mask is None:
                return False
            if mask.sum() == 0:
                st.warning("Mask is empty. Please draw something.")
                return False
            if mask.sum() == mask.size:
                st.warning("Entire image is masked. Please draw a smaller mask.")
                return False
            return True
        
        # Create layout
        col1, col2 = st.columns([1, 1])
        
        # Original image
        with col1:
            st.subheader("Original Image")
            resized_image = image.resize((self.canvas_size, self.canvas_size))
            st.image(resized_image, width=self.canvas_size)
        
        # Drawing canvas
        with col2:
            st.subheader("Draw Mask Here")
            canvas_result = self.setup_canvas(
                resized_image,
                controls['stroke_width'],
                controls['drawing_mode']
            )
        
        # Process and validate mask
        mask = self.process_canvas_result(canvas_result)
        mask_valid = validate_mask(mask) if mask is not None else False
        
        # Display results
        col3, col4 = st.columns([1, 1])
        
        # Mask preview
        with col3:
            st.subheader("Extracted Mask")
            if mask is not None:
                st.image(mask, width=self.canvas_size)
        
        # Result placeholder and processing button
        with col4:
            st.subheader("Inpainting Result")
            process_clicked = False
            if canvas_result.image_data is not None:
                if st.button("Process Image", key="process_button", disabled=not mask_valid):
                    if not mask_valid:
                        st.error("Please draw a valid mask first")
                    else:
                        process_clicked = True
        
        return mask, process_clicked