import gdown

FOLDER_URL = "https://drive.google.com/drive/folders/PASTE_FOLDER_ID_HERE?usp=sharing"
gdown.download_folder(url=FOLDER_URL, output="forest_seg", quiet=False)
