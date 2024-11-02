# src/interface/components/canvas_handler.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from typing import Tuple, Dict, Optional

class CanvasHandler:
    """Handles canvas operations for the inpainting tool"""
    
    def __init__(self, canvas_size: int = 512):
        self.canvas_size = canvas_size

    def setup_canvas(self, image: Image.Image, stroke_width: int, drawing_mode: str) -> Dict:
        try:
            resized_image = image.resize((self.canvas_size, self.canvas_size),
                                Image.Resampling.LANCZOS)
            return st_canvas(
                fill_color="#FFFFFF",
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                background_image=resized_image,  # Use resized_image here
                drawing_mode=drawing_mode,
                height=self.canvas_size,
                width=self.canvas_size,
                key="canvas",
                display_toolbar=True
            )
        except Exception as e:
            st.error(f"Error setting up canvas: {str(e)}")
            return None
        

    def process_canvas_result(self, canvas_result: Dict) -> Optional[np.ndarray]:
        """Process canvas result to generate mask"""
        if canvas_result.image_data is None:
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
        
        # Process mask
        mask = self.process_canvas_result(canvas_result)
        
        # Display results
        col3, col4 = st.columns([1, 1])
        
        # Mask preview
        with col3:
            st.subheader("Extracted Mask")
            if mask is not None:
                st.image(mask, width=self.canvas_size)
        
        # Result placeholder
        with col4:
            st.subheader("Inpainting Result")
            process_clicked = False
            if canvas_result.image_data is not None:
                if st.button("Process Image", key="process_button"):
                    if not canvas_result.image_data.any():
                        st.error("Please draw a mask first")
                    else:
                        process_clicked = True
        
        return mask, process_clicked