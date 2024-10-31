# src/interface/components/ui_components.py
import streamlit as st
from typing import Tuple, Dict, Any

class UIComponents:
    """UI Components for the Inpainting Tool"""

    @staticmethod
    def setup_styles():
        """Setup CSS styles for the application"""
        st.markdown("""
            <style>
                .element-container { margin: 2rem 0; }
                .row-container { margin: 3rem 0; }
                .stButton>button { margin-top: 1rem; }
                [data-testid="column"] { 
                    padding: 0 3rem !important; 
                }
                [data-testid="stHorizontalBlock"] {
                    margin-left: -3rem;
                    margin-right: -3rem;
                    gap: 3rem !important;
                }
                [data-testid="column"] > div {
                    width: 100%;
                    margin: 0 auto;
                }
            </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def create_sidebar_controls() -> Dict[str, Any]:
        """Create and return sidebar controls"""
        st.sidebar.title("Controls")
        
        controls = {}
        
        # File uploader
        controls['uploaded_file'] = st.sidebar.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            key="file_uploader"
        )
        
        # Model selector
        controls['model_name'] = st.sidebar.selectbox(
            "Select Model",
            ["Partial Convolutions"],
            key="model_selector"
        )
        
        st.sidebar.subheader("Drawing Controls")
        
        # Drawing tools
        controls['drawing_mode'] = st.sidebar.radio(
            "Drawing Tool:",
            ["rect", "freedraw"],
            key="drawing_mode"
        )
        
        # Stroke width
        controls['stroke_width'] = st.sidebar.slider(
            "Stroke width: ",
            1, 100, 30,
            key="stroke_width"
        )
        
        return controls

    @staticmethod
    def display_instructions():
        """Display usage instructions"""
        st.markdown("""
        ### Instructions:
        1. Upload an image using the sidebar
        2. Choose your drawing tool:
           - Rectangle: Click and drag to create a rectangular selection
           - Freehand: Click and drag to draw custom shapes
        3. Draw the area you want to inpaint (will appear in white)
        4. The mask will be displayed in real-time for preview
        5. Click 'Process Image' to perform inpainting
        
        Note: For best results, ensure the masked area is clearly defined.
        """)

    @staticmethod
    def show_error(error: Exception, show_traceback: bool = True):
        """Display error message and optionally show traceback"""
        st.error(f"Error: {str(error)}")
        if show_traceback:
            import traceback
            st.write("Detailed error:")
            st.write(traceback.format_exc())