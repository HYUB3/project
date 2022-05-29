import sys
import os
from src.export import *


path = str(os.getcwd()).replace("src", "")
sys.path.append(path)

export_csv()






