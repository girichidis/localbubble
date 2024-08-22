import pickle
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='+', help='List of pickle files to process')
args = parser.parse_args()

def convert_to_float32(obj):
    """
    Recursively convert all numpy arrays in a dictionary to float32.
    """
    if isinstance(obj, np.ndarray):
        return obj.astype(np.float32)
    elif isinstance(obj, dict):
        return {k: convert_to_float32(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_float32(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_float32(item) for item in obj)
    else:
        return obj
    
# Process each pickle file
for file in args.files:
    # Load the original pickle file
    print("original file     : ", file)
    with open(file, 'rb') as f:
        data = pickle.load(f)
    # Convert all numpy arrays to float32
    data = convert_to_float32(data)
    # Save the modified data to a new pickle file
    output_file = file.replace('.pkl', '-float32.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("--> wrote new file: ", output_file)

# def process_pickle(input_pickle_path, output_pickle_path):
#     # Load the original pickle file
#     with open(input_pickle_path, 'rb') as f:
#         data = pickle.load(f)

#     # Convert all numpy arrays to float32
#     data = convert_to_float32(data)

#     # Save the modified data to a new pickle file
#     with open(output_pickle_path, 'wb') as f:
#         pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# # Example usage:
# input_pickle = 'input_data.pkl'
# output_pickle = 'output_data.pkl'
# process_pickle(input_pickle, output_pickle)

