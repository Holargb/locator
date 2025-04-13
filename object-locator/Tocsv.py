import os
import scipy.io
import csv
import numpy as np

def process_mat_files(input_folder, output_csv):
    """
    Process .mat files in the input folder and save the extracted data into a CSV file.

    Parameters:
        input_folder: str
            Path to the folder containing .mat files.
        output_csv: str
            Path to the output CSV file where the results will be saved.
    """
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['filename', 'count', 'locations'])

        for filename in sorted(os.listdir(input_folder)):  # Sort by filename
            if filename.endswith('.mat'):
                # Generate the first column: remove .mat and GT_ prefix, then add .jpg
                base_name = filename.replace('.mat', '').replace('GT_', '') + '.jpg'
                
                try:
                    mat_data = scipy.io.loadmat(os.path.join(input_folder, filename))
                    image_info = mat_data['image_info'][0, 0]
                    number = int(image_info['number'][0, 0])  # Second column: pure number
                    locations = image_info['location']
                    
                    # Process coordinate data
                    if locations.shape == (1, 1) and isinstance(locations[0, 0], np.ndarray):
                        actual_locations = locations[0, 0]
                        coord_list = [(float(loc[1]), float(loc[0])) for loc in actual_locations]
                    elif locations.ndim == 2 and locations.shape[1] == 2:
                        coord_list = [(float(loc[1]), float(loc[0])) for loc in locations]
                    else:
                        coord_list = []
                    
                    # Format coordinate string
                    coord_str = "[" + ", ".join([f"({x:.2f}, {y:.2f})" for x, y in coord_list]) + "]"
                    
                    # Write the row, ensuring no extra spaces
                    csv_writer.writerow([base_name, str(number), coord_str])
                
                except Exception as e:
                    print(f"Error processing file {filename}: {str(e)}")
                    continue
    
    print(f"Processing complete. Results saved to {output_csv}")

input_folder = 'ground_truth'  # Replace with your folder containing .mat files
output_csv = '1gt.csv'         # Output CSV filename
process_mat_files(input_folder, output_csv)