import os
import zipfile
import requests
from io import BytesIO

def download_openslide_binaries(dest_dir="openslide_bin"):
    url = "https://github.com/openslide/openslide-winbuild/releases/download/v20171122/openslide-win64-20171122.zip"
    print(f"Downloading OpenSlide binaries from {url}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        print("Download complete. Extracting...")
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall(dest_dir)
            
        print(f"Extracted to {dest_dir}")
        
        # Find the bin directory
        extracted_folder = os.path.join(dest_dir, "openslide-win64-20171122")
        bin_dir = os.path.join(extracted_folder, "bin")
        
        if os.path.exists(bin_dir):
            print(f"Binary directory found at: {bin_dir}")
            return bin_dir
        else:
            print("Error: 'bin' directory not found in extracted files.")
            return None
            
    except Exception as e:
        print(f"Failed to download or extract: {e}")
        return None

if __name__ == "__main__":
    download_openslide_binaries()
