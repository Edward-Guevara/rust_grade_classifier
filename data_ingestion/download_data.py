import os

"""
this python script is to download the hyperespectral images of steel corroded and download masks to prepare
ground truth maps
"""

repo_url = ["https://github.com/Edward-Guevara/samples.git", "https://github.com/Edward-Guevara/masks.git"]

# Clonar el repositorio

for url in repo_url:    
    os.system(f"git clone {url}")