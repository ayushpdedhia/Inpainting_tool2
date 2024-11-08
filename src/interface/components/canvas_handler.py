# src/interface/components/canvas_handler.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from typing import Tuple, Dict, Optional
import traceback
from ...utils.mask_generator import MaskGenerator, MaskConfig

class CanvasHandler:
    """
    Handles canvas operations for the inpainting tool
    
    Strict Mask Convention:
    Display Format (Canvas/UI):
        - WHITE (255) = Areas to inpaint
        - BLACK (0) = Areas to keep original
    
    Internal Binary Format (Processing):
        - 0 = Areas to inpaint
        - 1 = Areas to keep original
    """
    
    def __init__(self, canvas_size: int = 512, config=None):
        self.canvas_size = canvas_size
        self.config = config
        self.preview_container = None

        # Initialize session state variables with correct display format (0=keep)
        if 'random_mask_array' not in st.session_state:
            st.session_state.random_mask_array = None
        if 'manual_mask' not in st.session_state:
            # Initialize with black (0 = keep all pixels)
            st.session_state.manual_mask = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        if 'random_mask' not in st.session_state:
            st.session_state.random_mask = None
        if 'using_random_mask' not in st.session_state:
            st.session_state.using_random_mask = False

        # Initialize mask generator
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

    def validate_display_mask(self, mask: np.ndarray, name: str = ""):
        """Validate display format mask values"""
        if mask is None:
            return
        
        unique_values = np.unique(mask)
        print(f"\nValidating display mask {name}:")
        print(f"Shape: {mask.shape}")
        print(f"Unique values: {unique_values}")
        
        # Check for valid display values (should be 0 or 255)
        valid_values = np.isin(unique_values, [0, 255])
        if not np.all(valid_values):
            print(f"WARNING: Invalid display mask values found in {name}: {unique_values[~valid_values]}")
            # Fix invalid values
            mask[mask > 0] = 255
            mask[mask < 255] = 0

    def validate_binary_mask(self, mask: np.ndarray, name: str = ""):
        """Validate binary format mask values"""
        if mask is None:
            return
            
        unique_values = np.unique(mask)
        print(f"\nValidating binary mask {name}:")
        print(f"Shape: {mask.shape}")
        print(f"Unique values: {unique_values}")
        
        # Check for valid binary values (should be 0 or 1)
        valid_values = np.isin(unique_values, [0, 1])
        if not np.all(valid_values):
            print(f"WARNING: Invalid binary mask values found in {name}: {unique_values[~valid_values]}")
            # Fix invalid values
            mask = (mask > 0.5).astype(np.float32)

    def _convert_to_display_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Convert internal binary mask to display format
        Input (binary): 0 = inpaint, 1 = keep
        Output (display): 255 = inpaint, 0 = keep
        """
        if binary_mask is None:
            return None
            
        # Validate input
        self.validate_binary_mask(binary_mask, "pre-conversion binary")
        
        # Convert: Invert and scale (0→255, 1→0)
        display_mask = ((1 - binary_mask) * 255).astype(np.uint8)
        
        # Validate output
        self.validate_display_mask(display_mask, "post-conversion display")
        
        return display_mask

    def _convert_to_binary_mask(self, display_mask: np.ndarray) -> np.ndarray:
        """
        Convert display mask to internal binary format
        Input (display): 255 = inpaint, 0 = keep
        Output (binary): 0 = inpaint, 1 = keep
        """
        if display_mask is None:
            return None
            
        # Validate input
        self.validate_display_mask(display_mask, "pre-conversion display")
        
        # Convert: Threshold and invert (255→0, 0→1)
        binary_mask = (display_mask < 127).astype(np.float32)
        
        # Validate output
        self.validate_binary_mask(binary_mask, "post-conversion binary")
        
        return binary_mask
    
    def combine_masks(self, manual_mask, random_mask):
        """
        Combine manual and random masks
        
        Args:
        - manual_mask: Display format (255 = inpaint, 0 = keep)
        - random_mask: Binary format (0 = inpaint, 1 = keep)
        
        Returns:
        - Binary mask (0 = inpaint, 1 = keep)
        """
        print("\n=== Combining Masks ===")
        print(f"Manual mask (display format) - Shape: {manual_mask.shape if manual_mask is not None else None}")
        print(f"Manual mask range: {np.min(manual_mask) if manual_mask is not None else 'None'} "
              f"to {np.max(manual_mask) if manual_mask is not None else 'None'}")
        print(f"Random mask (binary format) - Shape: {random_mask.shape if random_mask is not None else None}")
        if random_mask is not None:
            print(f"Random mask range: {random_mask.min()} to {random_mask.max()}")

        # Validate input formats
        if manual_mask is not None:
            self.validate_display_mask(manual_mask, "input manual mask")
        if random_mask is not None:
            self.validate_binary_mask(random_mask, "input random mask")

        # Convert manual mask from display to binary
        if manual_mask is not None:
            manual_mask_binary = self._convert_to_binary_mask(manual_mask)
        else:
            manual_mask_binary = np.ones((self.canvas_size, self.canvas_size), dtype=np.float32)
        
        # Process random mask (already in binary)
        if random_mask is not None:
            random_mask_binary = random_mask.astype(np.float32)
            if len(random_mask_binary.shape) == 3:
                random_mask_binary = random_mask_binary[:, :, 0]
        else:
            random_mask_binary = np.ones_like(manual_mask_binary)

        # Combine masks - if either says inpaint (0), result should be inpaint (0)
        combined = np.minimum(manual_mask_binary, random_mask_binary)

        # Validate output
        self.validate_binary_mask(combined, "combined output")

        print(f"\nCombined mask stats:")
        print(f"Shape: {combined.shape}")
        print(f"Unique values: {np.unique(combined)}")
        print(f"Inpaint area: {(combined == 0).sum() / combined.size:.2%}")

        return combined

    def process_canvas_result(self, canvas_result: Dict) -> Optional[np.ndarray]:
        """
        Process canvas result to generate mask
        Returns: Display format mask (255 = inpaint, 0 = keep)
        """
        if canvas_result.image_data is None:
            if hasattr(self, '_last_generated_mask'):
                return self._convert_to_display_mask(self._last_generated_mask)
            return None

        print("\n=== Processing Canvas Result ===")
        print(f"Canvas data shape: {canvas_result.image_data.shape}")
        print(f"Canvas data type: {canvas_result.image_data.dtype}")
        
        # Initialize with black (0 = keep all pixels)
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
                    print(f"Drew rectangle at ({x}, {y}) with size {w}x{h}")
        else:
            # Handle freedraw mode
            print("Processing freedraw")
            # Get the actual image data from the canvas
            canvas_data = canvas_result.image_data
            
            # Print detailed debug info
            print(f"Canvas data min/max: {canvas_data.min()}, {canvas_data.max()}")
            print(f"Canvas channels: {[np.unique(canvas_data[:,:,i]) for i in range(canvas_data.shape[-1])]}")
            
            if canvas_data.shape[-1] == 4:  # RGBA format
                # Look at all channels including RGB and alpha
                any_drawn = np.any(canvas_data > 0, axis=-1)
                mask_display = (any_drawn * 255).astype(np.uint8)
                
                # Debug output for mask
                print(f"Drawn pixels detected: {np.sum(mask_display > 0)}")
                print(f"Mask unique values after conversion: {np.unique(mask_display)}")

        # Validate output format
        self.validate_display_mask(mask_display, "canvas output")
        print(f"Final mask unique values: {np.unique(mask_display)}")
        print(f"Inpaint area: {(mask_display > 0).sum()}/{mask_display.size} pixels")
        
        return mask_display

    def display_canvas_columns(self, image: Image.Image, controls: Dict) -> Tuple[np.ndarray, bool]:
        """Display canvas columns and handle canvas interaction"""
        if not hasattr(self, 'preview_container'):
            self.preview_container = st.empty()
        
        def validate_mask(mask):
            """
            Validate mask for processing
            For binary mask: 0 = inpaint, 1 = keep
            """
            if mask is None:
                return False
            
            inpaint_area = (mask == 0).sum()
            total_area = mask.size
            
            if inpaint_area == 0:
                st.warning("No areas marked for inpainting. Please draw something.")
                return False
            if inpaint_area == total_area:
                st.warning("Entire image marked for inpainting. Please draw a smaller mask.")
                return False
            
            print(f"Mask validation - Inpaint area: {inpaint_area/total_area:.2%}")
            return True

        # Create grid layout
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        # Original image
        with col1:
            st.subheader("Original Image")
            resized_image = image.resize((self.canvas_size, self.canvas_size))
            st.image(resized_image, width=self.canvas_size)

        # Drawing canvas
        with col2:
            st.subheader("Draw Mask Here")
            
            # Random mask generation
            if self.mask_generator is not None:
                if st.button("Generate Random Mask", key="random_mask_btn"):
                    try:
                        # Generate random mask in binary format (0=inpaint, 1=keep)
                        random_mask = self.mask_generator.sample()
                        
                        # Store binary format for combining
                        self._last_generated_mask = random_mask.copy()
                        st.session_state.random_mask_array = random_mask.copy()

                        # Convert to display format for overlay
                        display_mask = self._convert_to_display_mask(random_mask)
                        if len(display_mask.shape) == 2:
                            display_mask = np.stack([display_mask] * 3 + [display_mask], axis=-1)

                        st.session_state.random_mask = display_mask
                        st.session_state.using_random_mask = True
                        
                        print("\n=== Random Mask Generation ===")
                        print(f"Binary format - Shape: {random_mask.shape}")
                        print(f"Binary format - Range: [{random_mask.min()}, {random_mask.max()}]")
                        print(f"Display format - Range: [{display_mask.min()}, {display_mask.max()}]")
                        
                    except Exception as e:
                        st.error(f"Error generating random mask: {str(e)}")
                        print(f"Random mask generation error: {traceback.format_exc()}")

            # Create canvas
            background_image = resized_image
            if st.session_state.using_random_mask and st.session_state.random_mask is not None:
                try:
                    mask_overlay = Image.fromarray(st.session_state.random_mask)
                    background_image = Image.fromarray(np.array(resized_image))
                    background_image.paste(mask_overlay, (0, 0), mask_overlay)
                except Exception as e:
                    print(f"Error applying mask overlay: {str(e)}")

            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1.0)",  # Solid white for inpainting
                stroke_width=controls['stroke_width'],
                stroke_color="rgba(255, 255, 255, 1.0)",  # Solid white for inpainting
                background_color="rgba(0, 0, 0, 0.0)",  # Transparent background
                background_image=background_image,
                drawing_mode=controls['drawing_mode'],
                height=self.canvas_size,
                width=self.canvas_size,
                key="canvas",
                display_toolbar=True
            )

        # Process manual mask (in display format)
        if canvas_result is not None and canvas_result.image_data is not None:
            st.session_state.manual_mask = self.process_canvas_result(canvas_result)

        # Combine masks (converts to binary internally)
        combined_mask = self.combine_masks(
            st.session_state.manual_mask,
            st.session_state.random_mask_array if st.session_state.using_random_mask else None
        )

        # Display masks
        with col3:
            st.subheader("Extracted Mask")
            if combined_mask is not None:
                # Convert binary to display for visualization
                display_mask = self._convert_to_display_mask(combined_mask)
                st.image(display_mask, width=self.canvas_size, caption="Combined Mask")

                if st.checkbox("Show Individual Masks"):
                    col3a, col3b = st.columns(2)
                    with col3a:
                        st.image(st.session_state.manual_mask, caption="Manual Mask")
                    with col3b:
                        if st.session_state.using_random_mask:
                            random_display = self._convert_to_display_mask(
                                st.session_state.random_mask_array)
                            st.image(random_display, caption="Random Mask")

        # Process button
        with col4:
            st.subheader("Inpainting Result")
            process_clicked = False
            if validate_mask(combined_mask):
                if st.button("Process Image", key="process_button"):
                    process_clicked = True

        # Debug info in sidebar
        if combined_mask is not None:
            inpaint_area = (combined_mask == 0).sum()
            total_area = combined_mask.size
            
            st.sidebar.write("Mask Statistics:")
            st.sidebar.write(f"Shape: {combined_mask.shape}")
            st.sidebar.write(f"Unique values: {np.unique(combined_mask)}")
            st.sidebar.write(f"Inpaint area: {inpaint_area/total_area:.2%}")

        return combined_mask, process_clicked