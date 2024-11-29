import os
import shutil

from dotenv import load_dotenv

load_dotenv()


def remove_file(rel_path):
    """
    Delete a file at a specified relative path.

    Parameters:
    rel_path (str): The relative path to the file intended for deletion.

    Raises:
    FileNotFoundError: If the file does not exist at the path.
    PermissionError: If the deletion operation lacks necessary permissions.
    Exception: For any other exceptions that occur during file deletion.
    """
    if len(rel_path) < 2:
        print(f"Error with file: {rel_path}")
        return

    base_dir = os.getenv('BASE_DIR')
    file_path = os.path.join(base_dir, rel_path)

    if file_path is None:
        print(f"No file is present: {rel_path}")
        return

    try:
        os.remove(file_path)
        print(f"File {file_path} has been deleted successfully.")
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except PermissionError:
        print(f"Permission denied to delete the file {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")


def remove_folder(rel_path):
    """
    Deletes the directory at the given relative path.

    Parameters:
    rel_path (str): The relative path to the directory to be deleted.

    Raises:
    FileNotFoundError: If the directory does not exist at the path.
    PermissionError: If the deletion operation lacks necessary permissions.
    OSError: For any other OS-related errors that occur during directory deletion.
    """
    if len(rel_path) < 2:
        print(f"Error with folder: {rel_path}")
        return

    base_dir = os.getenv('BASE_DIR')
    folder_path = os.path.join(base_dir, rel_path)

    if folder_path is None:
        print(f"No folder is present: {rel_path}")
        return

    try:
        shutil.rmtree(folder_path)
        print(f"The directory {folder_path} has been deleted successfully.")
    except FileNotFoundError:
        print(f"The directory {folder_path} does not exist.")
    except PermissionError:
        print(f"Permission denied: unable to delete {folder_path}.")
    except OSError as e:
        print(f"Error: {e.strerror} - {e.filename}")
