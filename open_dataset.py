import h5py

# File path
filename = "./Mimic.h5"

# Function to print a snippet of the dataset
def print_dataset_snippet(ppg, labels, subject_idx, num_samples=5, ppg_snippet_length=6):
    """
    Print a snippet of the dataset as it is stored.

    Args:
        ppg: PPG dataset.
        labels: Labels dataset.
        subject_idx: Subject index dataset.
        num_samples: Number of samples to display.
        ppg_snippet_length: Number of PPG data points to display per sample.
    """
    print("\nDataset Snippet:")
    for i in range(num_samples):
        print(f"Sample {i + 1}:")
        print(f"  Subject Index: {subject_idx[i][0]}")
        print(f"  PPG (First {ppg_snippet_length} points of total 875): {ppg[i][:ppg_snippet_length]}")
        print(f"  Label (Systolic, Diastolic): {labels[i]}")
        print()

# Open the HDF5 file
with h5py.File(filename, 'r') as h5:
    labels = h5['label']
    ppg = h5['ppg']
    subject_idx = h5['subject_idx']
    
    print_dataset_snippet(ppg, labels, subject_idx, num_samples=5)
