import numpy as np

def color_palette():
    #Define positive as metal line and negative as background
    return [
        [0, 0, 0], #background, True negative (Black)
        [77, 175, 74], #True positive, metal line predicted as metal line (green)
        [228, 26, 28], #False negative, metal line predicted as background (red)
        [55, 126, 184], #False positve, background predicted as metal line (blue)
    ]
    
def apply_palette(label, palette):
    # Create an empty array to store the RGB values
    colored_label = np.zeros((*label.shape, 3), dtype=np.uint8)
    
    # Map each class to its corresponding color
    for class_index, color in enumerate(palette):
        mask = label == class_index
        colored_label[mask] = color
    
    return colored_label
