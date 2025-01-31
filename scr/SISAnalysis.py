"""
Lethargus analysis
20240802
Author: Shinichi Miyazaki

This script enable state analysis for SIS data

Work on Windows 10 (64bit)
Python 3.12.3


"""

import os
import tkinter.filedialog
import pandas as pd
from myfunctions import SIS_analyzer

def main():
    # select file
    root = tkinter.Tk()
    root.withdraw()
    filepath = tkinter.filedialog.askopenfilename()
    os.chdir(os.path.dirname(filepath))

    # Load data and make bodysize_list
    data = pd.read_csv(filepath)
    body_size = int(input('bodysize (pixel) : '))
    fig_rnum = int(input('How many rows? : '))
    fig_cnum = int(input('How many columns? : '))

    # SIS analysis

    # analysis type: 1 hour, 3 hour, 6 hour and 12 hour
    SIS_analysis_duration_list = [0.5, 4.5, 5, 12]

    SIS_analyzer(data, body_size, fig_rnum, fig_cnum, SIS_analysis_duration_list)

if __name__ == '__main__':
    main()