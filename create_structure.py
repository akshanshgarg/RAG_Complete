import os

# Define the folder structure as a nested dictionary
# None indicates a file, dict indicates a directory
structure = {
    'README.md': None,
    'requirements.txt': None,
    'Docs_Assistant': {
        'app.py': None,
        'src': {
            '__init__.py': None,
            'dataloader.py': None,
            'embedding.py': None,
            'search.py': None,
            'vector_store.py': None,
        }
    }
}

def create_structure(base_path, struct):
    """
    Recursively creates the directory structure and files.
    
    :param base_path: The base directory path where to create the structure
    :param struct: The nested dictionary representing the structure
    """
    for name, content in struct.items():
        path = os.path.join(base_path, name)
        if content is None:
            # Create an empty file
            with open(path, 'w') as f:
                pass
        else:
            # Create directory
            os.makedirs(path, exist_ok=True)
            # Recurse into subdirectory
            create_structure(path, content)

if __name__ == '__main__':
    # Create the structure starting from the current directory
    create_structure('.', structure)
    print("Folder structure created successfully!")