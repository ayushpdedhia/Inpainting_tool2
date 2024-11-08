# src/interface/components/canvas_handler.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from typing import Tuple, Dict, Optional
import traceback
# Import MaskGenerator and MaskConfig from utils
from ...utils.mask_generator import MaskGenerator, MaskConfig

class CanvasHandler:
    """Handles canvas operations for the inpainting tool"""
    
    def __init__(self, canvas_size: int = 512, config=None):
        self.canvas_size = canvas_size
        self.config = config
        # Add real-time preview
        self.preview_container = None

        # Initialize all required session state variables
        if 'random_mask_array' not in st.session_state:
            st.session_state.random_mask_array = None
        if 'manual_mask' not in st.session_state:
            # Initialize with all 1s (keep all pixels)
            st.session_state.manual_mask = np.ones((self.canvas_size, self.canvas_size), dtype=np.uint8) * 255
        if 'random_mask' not in st.session_state:
            st.session_state.random_mask = None
        if 'using_random_mask' not in st.session_state:
            st.session_state.using_random_mask = False

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

    def combine_masks(self, manual_mask, random_mask):
        """
        Combine manual and random masks
        
        Mask Convention Throughout Pipeline:
        Input from canvas: white (255) = areas to inpaint, black (0) = keep
        Binary mask: 0 = inpaint, 1 = keep original
        """
        # Debug input masks
        print("\n=== Combining Masks ===")
        print(f"Manual mask range: {np.min(manual_mask) if manual_mask is not None else 'None'} to {np.max(manual_mask) if manual_mask is not None else 'None'}")
        print(f"Random mask range: {np.min(random_mask) if random_mask is not None else 'None'} to {np.max(random_mask) if random_mask is not None else 'None'}")

        # Convert manual mask to proper format
        if manual_mask is not None:
            # Convert to binary where:
            # Canvas input: white (255) → 0 (inpaint)
            #              black (0)   → 1 (keep)
            manual_mask_binary = (manual_mask < 127).astype(np.float32)
        else:
            # Default to all 1s (keep all pixels)
            manual_mask_binary = np.ones((self.canvas_size, self.canvas_size), dtype=np.float32)

        # Convert random mask to proper format
        if random_mask is not None:
            random_mask_binary = random_mask.astype(np.float32)
            if len(random_mask_binary.shape) == 3:
                random_mask_binary = random_mask_binary[:, :, 0]
            # Normalize to [0, 1] if needed
            if random_mask_binary.max() > 1:
                random_mask_binary = (random_mask_binary < 127).astype(np.float32)
        else:
            # Default to all 1s (keep all pixels)
            random_mask_binary = np.ones_like(manual_mask_binary)

        # Debug converted masks
        print("\n=== Converted Masks ===")
        print(f"Manual mask binary unique values: {np.unique(manual_mask_binary)}")
        print(f"Random mask binary unique values: {np.unique(random_mask_binary)}")

        # Combine masks - use minimum because if either mask says to inpaint (0),
        # we want the final result to be 0 (inpaint)
        combined = np.minimum(manual_mask_binary, random_mask_binary)

        # Debug final combined mask
        print("\n=== Combined Mask ===")
        print(f"Shape: {combined.shape}, dtype: {combined.dtype}")
        print(f"Unique values: {np.unique(combined)}")
        print(f"Min: {combined.min()}, Max: {combined.max()}")

        # Validate mask values
        assert combined.min() >= 0 and combined.max() <= 1, "Mask values must be in [0,1] range"
        
        return combined

    def process_canvas_result(self, canvas_result: Dict) -> Optional[np.ndarray]:
        """Process canvas result to generate mask"""
        if canvas_result.image_data is None:
            if hasattr(self, '_last_generated_mask'):
                return self._last_generated_mask
            return None
            
        # Initialize with black (keep all pixels)
        mask_display = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        
        if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
            # Handle rectangle drawing - fill with white (255) for areas to inpaint
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "rect":
                    x = int(obj["left"])
                    y = int(obj["top"])
                    w = int(obj["width"])
                    h = int(obj["height"])
                    cv2.rectangle(mask_display, (x, y), (x+w, y+h), 255, -1)
        else:
            # Handle freedraw - white (255) = inpaint, black (0) = keep
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
            assert resized_image.size == (self.canvas_size, self.canvas_size), f"Image resize failed. Expected {self.canvas_size}x{self.canvas_size}, got {resized_image.size}"
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
                        
                        # Store both the original array and display version
                        st.session_state.random_mask_array = random_mask.copy()
                        display_mask = (random_mask * 255).astype(np.uint8)
                        if len(display_mask.shape) == 2:
                            display_mask = np.stack([display_mask] * 3 + [display_mask], axis=-1)
                        
                        st.session_state.random_mask = display_mask
                        st.session_state.using_random_mask = True
                        
                        print("Random mask stored in session state")
                        
                    except Exception as e:
                        st.error(f"Error generating random mask: {str(e)}")
                        print(f"Random mask generation error: {str(e)}")
                        print(f"Traceback: {traceback.format_exc()}")
            
            # Create canvas with random mask overlay
            background_image = resized_image
            if st.session_state.using_random_mask and st.session_state.random_mask is not None:
                try:
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
        
        # Process manual mask
        if canvas_result is not None and canvas_result.image_data is not None:
            print("\n=== Processing Canvas Input ===")
            if controls['drawing_mode'] == "freedraw":
                st.session_state.manual_mask = canvas_result.image_data[:, :, -1]
                print(f"Freedraw mask unique values: {np.unique(st.session_state.manual_mask)}")
            elif controls['drawing_mode'] == "rect" and canvas_result.json_data is not None:
                st.session_state.manual_mask = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
                for obj in canvas_result.json_data.get("objects", []):
                    if obj["type"] == "rect":
                        x = int(obj["left"])
                        y = int(obj["top"])
                        w = int(obj["width"])
                        h = int(obj["height"])
                        cv2.rectangle(st.session_state.manual_mask, (x, y), (x+w, y+h), 255, -1)
                print(f"Rectangle mask unique values: {np.unique(st.session_state.manual_mask)}")

        # Combine masks using session state
        combined_mask = self.combine_masks(
            st.session_state.manual_mask,
            st.session_state.random_mask_array if st.session_state.using_random_mask else None
        )
        
        with col3:
            st.subheader("Extracted Mask")
            if combined_mask is not None:
                st.image(combined_mask, width=self.canvas_size, caption="Combined Mask")
                
                # Debug display of individual masks
                if st.checkbox("Show Individual Masks"):
                    col3a, col3b = st.columns(2)
                    with col3a:
                        st.image(st.session_state.manual_mask, caption="Manual Mask")
                    with col3b:
                        if st.session_state.using_random_mask and st.session_state.random_mask_array is not None:
                            random_display = (st.session_state.random_mask_array * 255).astype(np.uint8)
                            st.image(random_display, caption="Random Mask")
        
        # Inpainting result placeholder (bottom right)
        with col4:
            st.subheader("Inpainting Result")
            process_clicked = False
            if validate_mask(combined_mask):
                if st.button("Process Image", key="process_button"):
                    process_clicked = True

        print("=== Canvas Handler Output ===")
        print(f"Combined Mask shape: {combined_mask.shape}")
        print(f"Combined Mask dtype: {combined_mask.dtype}")
        print(f"Combined Mask unique values: {np.unique(combined_mask)}")
        
        return combined_mask, process_clicked