from pathlib import Path
import sys
try:

    
    # Paths
    PROJ_ROOT = Path(__file__).resolve().parents[1]
    # config_file_path = Path(__file__)

    # # Get the parent directory of the current file
    # PROJ_ROOT =  config_file_path.parent
    
    

    DATA_DIR = PROJ_ROOT / "data"
    # RAW_DATA_DIR = DATA_DIR / "raw"
    # INTERIM_DATA_DIR = DATA_DIR / "interim"
    # PROCESSED_DATA_DIR = DATA_DIR / "processed"
    # EXTERNAL_DATA_DIR = DATA_DIR / "external"

    MODELS_DIR = PROJ_ROOT / "models"

    REPORTS_DIR = PROJ_ROOT / "reports"
    FIGURES_DIR = REPORTS_DIR / "figures"

    print("All Path are set")
    
except ModuleNotFoundError:
    pass 