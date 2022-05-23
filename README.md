# Project_B3
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
* If you want inastall all prerequistite, Write `pip install -r requirements.txt.`   
<!--
작성한 코드를 실행하기 전에 설치해야할 pakage나 의존성이 걸리는 문제
-->

## Usage
***
1. We carried out this project using Pycharm. In order to run this program, the user must install the pycham.
2. Run in Pycharm Run.py and open or select the data file you want to analyze.
3. Write wafer or coordinates you want to analyze afterwards.
4. Choose whether to show data, save figure, or save csv files.

* If you do not run with Pycharm, you may experience an error that fails to stop while the program is in progress.

<!--
작성한 코드를 어떻게 실행해야 하는지에 대한 가이드라인
Usage Example을 함께 작성
-->

## Environment
***
* Python 3.8 
* Windows 10
<!--
실행환경에 대해 작성하면 된다. OS나 컴파일러 혹은 Hardware와 관련된 환경
-->

## Files
***
* src
  * directory.py - If there is no directory, it is a code that functions to create a new directory.
  * extract.py - It is a code that extracts information from a given xml data and has the ability to store it by replacing it with a file in csv format.
  * graph.py - Using the polyfit function, we obtain the polynomial closest to a given data and represent it in graphs.
  * path.py 
  * process.py - Based on the options received from run.py, the code is executed by selecting specific properties (image, wafer, image) from the data.
* gitignore   - Files that do not need to be managed in the project were managed using the gitignore file to exclude them from git.
* run.py      - It is the code that executes the project, and it receives several options to execute the Src file.
<!--
중요한 코드 파일들 몇 개를 대상으로 해당 파일이 어떠한 역할을 하는 파일인지를 간단히 설명해주면 전반적인 맥락을 파악하기에 좋을 것 같아 추가하였다.
-->

## Contributing
If you have any errors or questions while using the code, please send an email to the address below.
- <dbgmlcks53@hanyang.ac.kr>
- <chaeyoon20@hanyang.ac.kr>
- <fabiankd31@gmail.com>
<!-- 
license 기입하기
-->
