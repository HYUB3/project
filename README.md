
![header](https://capsule-render.vercel.app/api?type=slice&color=002266&height=300&section=header&text=Project%20B3&fontSize=110&fontColor=FFFF&rotate=20&fontAlignY=30&fontAlign=65)
***
## INDEX
***
- [Description](#description)
- [Prerequisite](#prerequisite)
- [Usage](#usage)
- [Environment](#environment)
- [Files](#files)
- [Contributing](#contributing)




## Description
***
We have created a module to extract and analyze customer-supplied data, especially a project to show and save information in graphs and csv files to make modulation performance of the modulator easier to see.


## Prerequisite
***
* Install `pandas` to process data. It is known as an essential library for tasks such as data analysis using Python. <http://pandas.pydata.org/pandas-docs/stable/>
* Install `xml.etree.elementTree`. The module implements a simple and efficient API for parsing and creating XML data.
* Install `numpy`. Numpy is Python package that deals with numerical data. It is mainly used in linear algebra calculations using vectors and matrices via ndarray, a multidimensional matrix data structure called the core of Numpy.
* Install `matplotlib.pyplot`. Used to vissualize data understanding prior to data analysis, or to visualize results after data analysis.
* Install `lmfit`. Lmfit provides a high-level interface to non-linear optimization and curve fitting problems for Python.
  <https://lmfit.github.io/lmfit-py/>
* If you want install all prerequistite, Write `pip install -r requirement.txt`   


## Usage
***
1. We carried out this project using Pycharm. In order to run this program, the user must install the pycharm.
2. Run in Pycharm run.py and open or select the data file you want to analyze.
3. Write wafer or coordinates you want to analyze.


* If you do not run with Pycharm, you may experience an error that fails to stop while the program is in progress.


## Environment
***
* Python 3.9
* Windows 10


## <img src="https://img.shields.io/badge/files-4285F4?style=for-the-badge&logo=Files&logoColor=white">

***
* src (여기 수정해야돼)
  * create.py -
  * data_analysis.py - If there is no directory, it is a code that functions to create a new directory.
  * data_plotting.py - It is a code that extracts information from a given xml data and has the ability to store it by replacing it with a file in csv format.
  * datareader.py - 
  * graph.py - Using the lmfit function, we obtain the polynomial closest to a given data and represent it in graphs.
  * loading.py - 
  * loadingxml.py -
* gitignore   - Files that do not need to be managed in the project were managed using the gitignore file to exclude them from git.
* run.py      - It is the code that executes the project, and it receives several options to execute the Src file.


## Contributing
If you have any errors or questions while using the code, please send an email to the address below.


<img src="https://img.shields.io/badge/Adress-EA4335?style=for-the-badge&logo=Gmail&logoColor=white">

- <chaeyoon20@hanyang.ac.kr>
- <dbgmlcks53@hanyang.ac.kr>
- <fabiankd31@gmail.com>
