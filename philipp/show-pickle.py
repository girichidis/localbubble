import pickle
import numpy as np
import sys

def summarize_array(arr):
    """
    Summarizes a numpy array, printing its shape, data type, min, and max values.
    """
    summary = {
        'shape': arr.shape,
        'dtype': arr.dtype,
        'min': arr.min(),
        'max': arr.max()
    }
    return summary

def print_summary(data, indent=0):
    """
    Recursively prints the content of a nested data structure.
    """
    indent_str = ' ' * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{indent_str}{key}:")
            print_summary(value, indent + 2)
    elif isinstance(data, list):
        print(f"{indent_str}List of length {len(data)}:")
        for index, item in enumerate(data):
            print(f"{indent_str}  [{index}]:")
            print_summary(item, indent + 4)
    elif isinstance(data, tuple):
        print(f"{indent_str}Tuple of length {len(data)}:")
        for index, item in enumerate(data):
            print(f"{indent_str}  ({index}):")
            print_summary(item, indent + 4)
    elif isinstance(data, np.ndarray):
        summary = summarize_array(data)
        print(f"{indent_str}Numpy array: shape={summary['shape']}, dtype={summary['dtype']}, "
              f"min={summary['min']}, max={summary['max']}")
    else:
        print(f"{indent_str}{repr(data)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python summarize_pickle.py <path_to_pickle>")
        sys.exit(1)

    pickle_path = sys.argv[1]

    # Load the pickle file
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    # Print a recursive summary of the pickle contents
    print_summary(data)

if __name__ == "__main__":
    main()

