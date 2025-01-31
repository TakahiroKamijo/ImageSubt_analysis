"""
Lethargus analysis
20230308
Author: Shinichi Miyazaki

This script enable lethargus analysis for imagesubtraction data

Work on Windows 10 (64bit)
Python 3.12.3
"""

import os
import tkinter.filedialog
import pandas as pd
from myfunctions import lethargus_analyzer

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

    # Lethargus analysis
    lethargus_analyzer(data, body_size, fig_rnum, fig_cnum)

if __name__ == '__main__':
    main()