import scipy.io
import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
from PIL import Image
from datetime import datetime
from xml.dom import minidom

def convert_mall_gt_to_csv(mat_path, output_csv):
    """
    Final revised version: Handles various possible coordinate array formats.
    
    Function Description:
        This function converts ground truth data from a MAT file (used in the Mall dataset) 
        into a CSV file. It extracts the count of people and their locations for each frame 
        and saves the data in a structured format. The function is designed to handle 
        different coordinate array formats robustly.
        
    Parameters:
        mat_path: str
            Path to the input MAT file containing the ground truth data.
        output_csv: str
            Path where the resulting CSV file will be saved.
    
    Output:
        A CSV file with columns: 'filename', 'count', and 'locations'.
        Each row corresponds to an image frame, with its filename, the number of people, 
        and their locations in the format [(x1,y1), (x2,y2), ...].
    """
    # Load the MAT file
    mat_data = scipy.io.loadmat(mat_path)
    
    # Extract data
    counts = mat_data['count'].flatten()
    frames = mat_data['frame'][0]
    
    data = []
    for i in range(len(frames)):
        img_name = f"seq_{i+1:06d}.jpg"
        frame_struct = frames[i][0]
        
        # More robust coordinate extraction
        locations = []
        if 'loc' in frame_struct.dtype.names:
            loc_data = frame_struct['loc'][0]  # Get the actual coordinate array
            if loc_data.size > 0:
                # Handle coordinate data with different dimensions
                if loc_data.ndim == 2:
                    locations = [tuple(map(float, row)) for row in loc_data]
                elif loc_data.ndim == 1 and len(loc_data) == 2:
                    locations = [tuple(map(float, loc_data))]
        
        # Add to data
        data.append({
            'filename': img_name,
            'count': int(counts[i]),
            'locations': f"[{', '.join(f'({x:.1f},{y:.1f})' for x,y in locations)}]" if locations else "[]"
        })
    
    # Save to CSV
    df = pd.DataFrame(data, columns=['filename', 'count', 'locations'])
    df.to_csv(output_csv, index=False)
    print(f"Conversion complete! Processed {len(data)} images.")

def prettify(elem):
    """XML Beautification Function (Compatible with All Python Versions)"""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")

def generate_xml_annotations(mat_path, jpg_dir, xml_dir, annotator_name="Your Name"):
    """
    Generate XML annotations for images based on ground truth data from a MAT file.

    Parameters:
        mat_path: str
            Path to the input MAT file containing ground truth data.
        jpg_dir: str
            Directory containing the image files.
        xml_dir: str
            Directory where the XML annotation files will be saved.
        annotator_name: str (optional)
            Name of the annotator to include in the metadata.

    Output:
        Generates XML annotation files for each image in the specified directory.
    """
    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_path)
    counts = mat_data['count'].flatten()

    # Create the XML directory if it doesn't exist
    os.makedirs(xml_dir, exist_ok=True)

    # File sorting (Adapted for seq_000001.jpg format)
    jpg_files = sorted(
        [f for f in os.listdir(jpg_dir) if f.endswith('.jpg')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )

    assert len(jpg_files) == len(counts), "Mismatch between images and labels!"

    for idx, jpg_name in enumerate(jpg_files):
        jpg_path = os.path.join(jpg_dir, jpg_name)
        xml_path = os.path.join(xml_dir, os.path.splitext(jpg_name)[0] + '.xml')

        # Open the image to get its properties
        with Image.open(jpg_path) as img:
            width, height = img.size
            mode = img.mode

        # Create the XML root element
        root = ET.Element('annotation')
        # Add all necessary information...
        # 1. Basic information
        ET.SubElement(root, 'folder').text = os.path.basename(jpg_dir)
        ET.SubElement(root, 'filename').text = jpg_name
        ET.SubElement(root, 'path').text = os.path.abspath(jpg_path)

        # 2. Image properties
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text = str(len(mode))  # Number of channels (e.g., RGB=3)
        ET.SubElement(size, 'color_mode').text = mode  # e.g., 'RGB'

        # 3. Label data
        label = ET.SubElement(root, 'label')
        ET.SubElement(label, 'count').text = str(int(counts[idx]))  # Count label

        # 4. Other metadata (optional)
        metadata = ET.SubElement(root, 'metadata')
        ET.SubElement(metadata, 'source').text = 'Mall Dataset'
        ET.SubElement(metadata, 'annotator').text = annotator_name

        # Save the beautified XML
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(prettify(root))
        
        print(f"Generated: {xml_path}")