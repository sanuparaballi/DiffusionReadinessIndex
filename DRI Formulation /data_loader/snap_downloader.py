#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 20:29:29 2025

@author: sanup
"""


# diffusion_readiness_project/data_loader/snap_downloader.py
# Python 3.9

"""
Utility to download and extract SNAP datasets if they are not found locally.
"""

import os
import requests
import gzip
import shutil
import tarfile

# --- SNAP Dataset URLs ---
# Using the URLs provided by the SNAP repository
SNAP_URLS = {
    "email-Enron": "https://snap.stanford.edu/data/email-Enron.txt.gz",
    "soc-Digg-friends": "https://snap.stanford.edu/data/soc-Digg-friends.txt.gz",
    "com-dblp": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
}


def download_and_extract_snap_dataset(dataset_name, data_dir="."):
    """
    Checks for a dataset file, and if not present, downloads and extracts it.

    Args:
        dataset_name (str): The name of the SNAP dataset. Must be a key in SNAP_URLS.
                            e.g., "email-Enron".
        data_dir (str): The directory where data should be stored. Defaults to current.

    Returns:
        str: The path to the final, extracted text file. Returns None on failure.
    """
    if dataset_name not in SNAP_URLS:
        print(f"Error: Dataset '{dataset_name}' not recognized. Available keys: {list(SNAP_URLS.keys())}")
        return None

    url = SNAP_URLS[dataset_name]

    # Derive filenames from URL
    gz_filename = os.path.basename(url)  # e.g., email-Enron.txt.gz
    txt_filename = gz_filename.replace(".gz", "")  # e.g., email-Enron.txt

    gz_path = os.path.join(data_dir, gz_filename)
    txt_path = os.path.join(data_dir, txt_filename)

    # 1. Check if the final extracted file already exists
    if os.path.exists(txt_path):
        print(f"Dataset '{txt_filename}' already exists at '{txt_path}'. Skipping download.")
        return txt_path

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # 2. Download the .gz file if it doesn't exist
    if not os.path.exists(gz_path):
        print(f"Downloading '{dataset_name}' from '{url}'...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(gz_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Download complete: '{gz_path}'")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading '{url}': {e}")
            if os.path.exists(gz_path):
                os.remove(gz_path)  # Clean up partial download
            return None

    # 3. Extract the .gz file
    print(f"Extracting '{gz_path}'...")
    try:
        with gzip.open(gz_path, "rb") as f_in:
            with open(txt_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extraction successful: '{txt_path}'")

        # 4. Clean up the downloaded .gz file
        os.remove(gz_path)
        print(f"Cleaned up '{gz_path}'.")

    except gzip.BadGzipFile:
        print(f"Error: '{gz_path}' is not a valid gzip file. It might be a tarball.")
        # Handle case for tar.gz if needed (com-dblp might be tarred)
        # For com-dblp, the .gz might actually contain a tar archive.
        try:
            print("Attempting to extract as a tar.gz file...")
            with tarfile.open(gz_path, "r:gz") as tar:
                # Find the actual .txt file within the tar archive
                members = tar.getmembers()
                txt_member = next((m for m in members if m.name.endswith(".txt")), None)
                if txt_member:
                    txt_filename = os.path.basename(txt_member.name)
                    txt_path = os.path.join(data_dir, txt_filename)
                    if not os.path.exists(txt_path):
                        tar.extractall(path=data_dir)
                        # Rename if necessary from subdirectory
                        extracted_path = os.path.join(data_dir, txt_member.name)
                        if extracted_path != txt_path:
                            shutil.move(extracted_path, txt_path)
                            # Clean up empty directory if created by extractall
                            if os.path.isdir(os.path.join(data_dir, os.path.dirname(txt_member.name))):
                                try:
                                    os.rmdir(os.path.join(data_dir, os.path.dirname(txt_member.name)))
                                except OSError:  # Fails if not empty, which is fine
                                    pass

                        print(f"Extraction successful: '{txt_path}'")
                        os.remove(gz_path)
                        print(f"Cleaned up '{gz_path}'.")
                    else:
                        print(f"Target file '{txt_path}' already exists. Skipping extraction.")
                else:
                    print("Error: No .txt file found within the tar archive.")
                    return None
        except tarfile.TarError as e_tar:
            print(f"Error extracting tar file '{gz_path}': {e_tar}")
            return None

    except Exception as e:
        print(f"An unexpected error occurred during extraction: {e}")
        return None

    return txt_path


if __name__ == "__main__":
    print("Testing SNAP downloader utility...")
    # This will create a 'temp_data' directory and download/extract the Enron dataset into it.
    temp_dir = "temp_data_for_download_test"

    print("\n--- Testing 'email-Enron' download ---")
    enron_path = download_and_extract_snap_dataset("email-Enron", data_dir=temp_dir)
    if enron_path and os.path.exists(enron_path):
        print(f"SUCCESS: Enron dataset is available at '{enron_path}'")
        # Check first few lines
        with open(enron_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"  Line {i+1}: {line.strip()}")
    else:
        print("FAILURE: Enron dataset download/extraction failed.")

    print("\n--- Testing 'com-dblp' download (handles potential tar) ---")
    dblp_path = download_and_extract_snap_dataset("com-dblp", data_dir=temp_dir)
    if dblp_path and os.path.exists(dblp_path):
        print(f"SUCCESS: DBLP dataset is available at '{dblp_path}'")
    else:
        print("FAILURE: DBLP dataset download/extraction failed.")

    # Clean up the temporary directory after test
    # import shutil
    # if os.path.exists(temp_dir):
    #     shutil.rmtree(temp_dir)
    #     print(f"\nCleaned up temporary directory: '{temp_dir}'")

    print("\n--- Downloader Test Complete ---")
