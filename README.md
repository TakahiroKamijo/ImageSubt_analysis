# ImageSubt_analysis
## Overview
This repository contain following 4 python scripts for analyzing developmentally timed sleep in C. elegans.

- LethargusAnalysis.py  
analyze the DTS data and plot the graph.
- SISAnalysis.py  
analyze the SIS data and plot the graph.
- imagesubtandmeasure_withoutimshow.py  
Subtract the image and measure the activity of worms without showing the subtracted images.
## Working environment
Currently, the scripts work on following environments.
### DTSAnalysis.py
Any python on Windows or Mac OS.
### SISAnalysis.py
Any python on Windows or Mac OS.
### imagesubtandmeasure_withoutimshow.py
Any Python later than 3.9 on Windows or Mac OS.
## Usage
After installing the required libraries, run the scripts imagesubtandmeasure_withoutimshow.py.
The script generate area.csv.
Then, run LethargusAnalysis.py or SISAnalysis and select the area.csv.
## Parameters and default values
### DTSAnalysis and SISAnalysis
- imaging_interval (sec): 2
- imaging_num_per_hour: 1800
- margin_image_num : 60 min
if DTS starts within 60 min, it is not used for analysis
- out_DTS_duration (sec): 600
The script extract above duration after DTS as out of DTS data
- out_DTS_image_num: 300
Time window of rolling average (sec): 600
- rolling_window_image_num: 300
These two parameters are used for calculating the rolling average.
- threshold of FoQ: 0.05
If the FoQ is above this threshold for more than 20 min, the episode is judged as sleep episodes.
## Reference
The definition of DTS followed the paper below. Funato et al., 2016, Nature
## Authors
Taizo Kawano  
Shinichi Miyazaki
