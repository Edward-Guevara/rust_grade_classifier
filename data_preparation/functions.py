import os


def get_files_with_extension(path, extension):
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(filename)
    return files


def make_folder(folder_name):
    try:
        os.makedirs(folder_name)
    except FileExistsError:
        print(f"The folder {folder_name} already exists.")
