import os
import requests
from zipfile import ZipFile
from io import BytesIO

def download_and_extract_dataset(
    dataset_url='http://cs231n.stanford.edu/tiny-imagenet-200.zip',
    dataset_dir='data'
):
    """
    Scarica ed estrae il dataset Tiny ImageNet nella cartella specificata.
    Se la cartella esiste già, salta il download.
    """
    target_path = os.path.join(dataset_dir, 'tiny-imagenet-200')

    # Se il dataset è già presente, non riscaricare
    if os.path.exists(target_path):
        print(f"✅ Dataset already exists at: {target_path}")
        return target_path

    # Assicurati che la directory esista
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"⬇️  Downloading Tiny ImageNet from {dataset_url} ...")
    response = requests.get(dataset_url)

    if response.status_code == 200:
        with ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall(dataset_dir)
        print(f"✅ Download and extraction complete! Dataset saved to: {target_path}")
    else:
        raise Exception(f"❌ Failed to download dataset (HTTP {response.status_code})")

    return target_path
