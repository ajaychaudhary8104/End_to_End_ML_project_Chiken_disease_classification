import os
import urllib.request as request
import zipfile
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n+{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  


    
    def extract_zip_file(self):
        """
        Extracts the downloaded archive (zip or 7z) into the data directory
        Returns None; raises informative errors if extraction fails or dependencies are missing
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        file_path = self.config.local_data_file
        # Try zip extraction first
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
                logger.info(f"Extracted zip archive to: {unzip_path}")
                return
        except zipfile.BadZipFile:
            logger.info("Not a zip archive, will attempt other extractors (e.g. 7z)")
        except Exception as e:
            logger.exception(f"Failed to extract as zip: {e}")
            raise e

        # Try 7z extraction using py7zr if available
        try:
            import py7zr
        except Exception:
            logger.error("py7zr is not installed; cannot extract .7z archives. Install with 'pip install py7zr' or change the source to a zip archive.")
            raise RuntimeError("py7zr not installed; cannot extract .7z archives.")

        try:
            with py7zr.SevenZipFile(file_path, mode='r') as archive:
                archive.extractall(path=unzip_path)
                logger.info(f"Extracted 7z archive to: {unzip_path}")
        except Exception as e:
            logger.exception(f"Failed to extract 7z archive: {e}")
            raise e