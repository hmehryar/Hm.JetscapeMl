from pathlib import Path
import os
import sys
import tensorflow as tf
import pandas as pd
import sklearn as sk

class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance
    
    def _initialize(self, simulation_name="default_simulation_name"):
        self.PROJ_ROOT = Path(__file__).resolve().parents[1]
        self.DATA_DIR = self.PROJ_ROOT / "data"
        self.MODELS_DIR = self.PROJ_ROOT / "models"
        self.REPORTS_DIR = self.PROJ_ROOT / "reports"
        self.FIGURES_DIR = self.REPORTS_DIR / "figures"
        
        # Set simulation-specific directories
        self.set_config_dirs_by_simulation_name(simulation_name)


    def set_config_dirs_by_simulation_name(self, simulation_name):
        # self.SIMULATION_DATA_DIR = self.DATA_DIR / simulation_name
        self.SIMULATION_MODELS_DIR = self.MODELS_DIR / simulation_name
        self.SIMULATION_REPORTS_DIR = self.REPORTS_DIR / simulation_name
        self.SIMULATION_FIGURES_DIR = self.FIGURES_DIR / simulation_name

        # Check and create directories
        # self.check_make_directory(self.SIMULATION_DATA_DIR)
        self.check_make_directory(self.SIMULATION_MODELS_DIR)
        self.check_make_directory(self.SIMULATION_REPORTS_DIR)
        self.check_make_directory(self.SIMULATION_FIGURES_DIR)

    def check_make_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f'Directory {path} created.')
        else:
            print(f'Directory {path} already exists.')

    def check_environments_versions(self):
        print(f"TensorFlow Version: {tf.__version__}")
        print(f"Keras Version: {tf.keras.__version__}")
        print()
        print(f"Python Version: {sys.version}")
        print(f"Pandas Version: {pd.__version__}")
        print(f"Scikit-Learn Version: {sk.__version__}")
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        print("GPU is", "available" if gpu_available else "NOT AVAILABLE")

    def __str__(self):
        return (f"Project Root: {self.PROJ_ROOT}\n"
                f"Data Directory: {self.DATA_DIR}\n"
                f"Models Directory: {self.MODELS_DIR}\n"
                f"Reports Directory: {self.REPORTS_DIR}\n"
                f"Figures Directory: {self.FIGURES_DIR}\n"
                f"Simulation Data Directory: {self.SIMULATION_DATA_DIR}\n"
                f"Simulation Models Directory: {self.SIMULATION_MODELS_DIR}\n"
                f"Simulation Reports Directory: {self.SIMULATION_REPORTS_DIR}\n"
                f"Simulation Figures Directory: {self.SIMULATION_FIGURES_DIR}\n"
                f"Environment Details:\n"
                f"  TensorFlow Version: {tf.__version__}\n"
                f"  Keras Version: {tf.keras.__version__}\n"
                f"  Python Version: {sys.version}\n"
                f"  Pandas Version: {pd.__version__}\n"
                f"  Scikit-Learn Version: {sk.__version__}\n"
                f"  GPU is {'available' if len(tf.config.list_physical_devices('GPU')) > 0 else 'NOT AVAILABLE'}")
    # # Singleton instance access
    # def get_config_instance(simulation_name="default_simulation_name"):
    #     return Config(simulation_name)

