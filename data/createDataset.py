import os

def create_dataset_structure(base_path):
    # Define the folder structure
    structure = {
        'images': ['train', 'val', 'test'],
        'labels': ['train', 'val', 'test']
    }

    # Loop through the structure dictionary to create directories
    for main_folder, sub_folders in structure.items():
        for sub_folder in sub_folders:
            # Create the directory path
            folder_path = os.path.join(base_path, main_folder, sub_folder)
            # Make the directory if it does not exist
            os.makedirs(folder_path, exist_ok=True)

    print("All specified directories have been created.")

base_path = 'dataset'

# Call the function with your base path
create_dataset_structure(base_path)
