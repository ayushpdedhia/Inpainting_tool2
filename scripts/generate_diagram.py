from graphviz import Digraph

def create_architecture_diagram():
    # Create a new directed graph
    dot = Digraph(comment='Inpainting Tool Architecture')
    dot.attr(rankdir='TB')
    
    # Define node styles
    dot.attr('node', shape='box', style='filled')
    
    # Color schemes for different component types
    colors = {
        'main': '#ff99ff',      # Pink for main app
        'interface': '#ff9966',  # Orange for interface components
        'utils': '#99ccff',      # Light blue for utilities
        'models': '#99ff99',     # Light green for models
        'layers': '#ffff99'      # Yellow for layers
    }
    
    # Add nodes with appropriate colors
    # Main Application
    dot.node('app', 'app.py', fillcolor=colors['main'])
    
    # Interface Components
    interface_components = {
        'ui': 'ui_components.py',
        'canvas': 'canvas_handler.py'
    }
    for id, label in interface_components.items():
        dot.node(id, label, fillcolor=colors['interface'])
    
    # Utility Components
    utils_components = {
        'imgproc': 'image_processor.py',
        'mask': 'mask_generator.py',
        'weight': 'weight_loader.py',
        'data': 'data_loader.py'
    }
    for id, label in utils_components.items():
        dot.node(id, label, fillcolor=colors['utils'])
    
    # Model Components
    model_components = {
        'modman': 'model_manager.py',
        'pconv': 'pconv_unet.py',
        'loss': 'loss.py',
        'vgg': 'vgg_extractor.py'
    }
    for id, label in model_components.items():
        dot.node(id, label, fillcolor=colors['models'])
    
    # Layer Components
    dot.node('pconv2d', 'partialconv2d.py', fillcolor=colors['layers'])
    
    # Add edges
    # Connections from app.py
    for component in ['ui', 'canvas', 'imgproc', 'modman']:
        dot.edge('app', component)
    
    # Connections from canvas_handler.py
    dot.edge('canvas', 'mask')
    
    # Connections from model_manager.py
    for component in ['weight', 'data', 'pconv', 'loss']:
        dot.edge('modman', component)
    
    # Connections from data_loader.py
    dot.edge('data', 'imgproc')
    dot.edge('data', 'mask')
    
    # Connection from loss.py
    dot.edge('loss', 'vgg')
    
    # Connection from pconv_unet.py
    dot.edge('pconv', 'pconv2d')
    
    # Save the diagram
    dot.render('inpainting_architecture', format='png', cleanup=True)

if __name__ == '__main__':
    create_architecture_diagram()