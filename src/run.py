import os
from pathlib import Path
from datareader import *
from loadingxml import *

global name_list
name_list = []

data_reader(name_list)

xml_loader()
