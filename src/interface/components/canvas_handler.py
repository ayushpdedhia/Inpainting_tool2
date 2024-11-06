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
    '''
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
            
            # Remove the on_change parameter and use the canvas directly
            canvas_result = st_canvas(
                fill_color="#FFFFFF",
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                background_image=resized_image,
                drawing_mode=drawing_mode,
                height=self.canvas_size,
                width=self.canvas_size,
                key="canvas",
                display_toolbar=True
            )

            # Update preview after canvas drawing
            if canvas_result is not None and canvas_result.image_data is not None:
                mask = self.process_canvas_result(canvas_result)
                if mask is not None:
                    preview_col.image(mask, caption="Real-time Mask Preview")

            return canvas_result

        except Exception as e:
            st.error(f"Error setting up canvas: {str(e)}")
            return None
             '''

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
        
        # Create 2x2 grid layout
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        # Original image (top left)
        with col1:
            st.subheader("Original Image")
            resized_image = image.resize((self.canvas_size, self.canvas_size))
            st.image(resized_image, width=self.canvas_size)
        
        # Drawing canvas (top right)
        with col2:
            st.subheader("Draw Mask Here")
            
            # Add random mask generation button
            if self.mask_generator is not None:
                if st.button("Generate Random Mask", key="random_mask_btn"):
                    try:
                        # Generate random mask
                        random_mask = self.mask_generator.sample()
                        print(f"Generated random mask shape: {random_mask.shape}")
                        print(f"Random mask min/max values: {random_mask.min()}, {random_mask.max()}")
                        
                        # Store the original mask for extraction
                        self._last_generated_mask = random_mask.copy()
                        
                        # Convert for display
                        display_mask = (random_mask * 255).astype(np.uint8)
                        if len(display_mask.shape) == 2:
                            display_mask = np.stack([display_mask] * 3 + [display_mask], axis=-1)
                        
                        st.session_state.random_mask = display_mask
                        st.session_state.using_random_mask = True
                        print("Random mask generated and stored successfully")
                        
                    except Exception as e:
                        st.error(f"Error generating random mask: {str(e)}")
                        import traceback
                        print(f"Random mask generation error: {str(e)}")
                        print(f"Traceback: {traceback.format_exc()}")
            
            # Create canvas with random mask overlay if available
            background_image = resized_image
            if 'random_mask' in st.session_state and st.session_state.using_random_mask:
                try:
                    # Create a masked version of the image for display
                    print("Applying random mask overlay")
                    mask_overlay = Image.fromarray(st.session_state.random_mask)
                    background_image = Image.fromarray(np.array(resized_image))
                    background_image.paste(mask_overlay, (0, 0), mask_overlay)
                except Exception as e:
                    print(f"Error applying mask overlay: {str(e)}")
            
            canvas_result = st_canvas(
                fill_color="#FFFFFF",
                stroke_width=controls['stroke_width'],
                stroke_color="#FFFFFF",
                background_color="#000000",
                background_image=background_image,
                drawing_mode=controls['drawing_mode'],
                height=self.canvas_size,
                width=self.canvas_size,
                key="canvas",
                display_toolbar=True
            )
        
        # Process mask
        mask = None
        mask_valid = False
        process_clicked = False

        # Initialize mask storage
        if 'manual_mask' not in st.session_state:
            st.session_state.manual_mask = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)

        # Handle manual mask drawing (rect or freedraw)
        if canvas_result is not None and canvas_result.image_data is not None:
            print("Processing canvas input")
            if controls['drawing_mode'] == "freedraw":
                st.session_state.manual_mask = canvas_result.image_data[:, :, -1]
            elif controls['drawing_mode'] == "rect" and canvas_result.json_data is not None:
                st.session_state.manual_mask = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
                for obj in canvas_result.json_data.get("objects", []):
                    if obj["type"] == "rect":
                        x = int(obj["left"])
                        y = int(obj["top"])
                        w = int(obj["width"])
                        h = int(obj["height"])
                        cv2.rectangle(st.session_state.manual_mask, (x, y), (x+w, y+h), 255, -1)

        # Combine manual and random masks
        mask = st.session_state.manual_mask.copy()

        if ('using_random_mask' in st.session_state and 
            st.session_state.using_random_mask and 
            hasattr(self, '_last_generated_mask')):
            print("Combining with random mask")
            # Convert random mask to proper format
            random_mask = (self._last_generated_mask * 255).astype(np.uint8)
            
            # Print debug information
            print(f"Random mask shape: {random_mask.shape}")
            print(f"Random mask min/max: {random_mask.min()}, {random_mask.max()}")
            print(f"Manual mask shape: {mask.shape}")
            print(f"Manual mask min/max: {mask.min()}, {mask.max()}")
            
            # Combine masks using bitwise OR
            mask = cv2.bitwise_or(mask, random_mask)
            print(f"Combined mask min/max: {mask.min()}, {mask.max()}")

        if mask is not None:
            print(f"Final mask shape: {mask.shape}")
            print(f"Final mask min/max values: {mask.min()}, {mask.max()}")

        mask_valid = validate_mask(mask) if mask is not None else False

        # Extracted mask preview (bottom left)
        with col3:
            st.subheader("Extracted Mask")
            if mask is not None:
                # Ensure mask is properly normalized for display
                display_mask = mask.astype(np.uint8)
                st.image(display_mask, width=self.canvas_size, caption="Combined Mask")
                
                # Debug display of individual masks
                if st.checkbox("Show Individual Masks"):
                    col3a, col3b = st.columns(2)
                    with col3a:
                        st.image(st.session_state.manual_mask, caption="Manual Mask")
                    with col3b:
                        if hasattr(self, '_last_generated_mask'):
                            random_display = (self._last_generated_mask * 255).astype(np.uint8)
                            st.image(random_display, caption="Random Mask")
        
        # Inpainting result placeholder (bottom right)
        with col4:
            st.subheader("Inpainting Result")
            if mask_valid:
                if st.button("Process Image", key="process_button"):
                    process_clicked = True
        
        return mask, process_clicked