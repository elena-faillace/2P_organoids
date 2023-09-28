# 2P_organoids
Working code for figures in organoids' paper

# Files included
- **env.yml**: used for creating the virtual environment to run the code
- **clean_data**.ipynb: used to load the .csv files with the ROIs information and traces. It saves the traces, the dff, and the spikes trains as .npy
- **generate_figures.ipynb**: used to generate the figures
- **tools.py**, **tools_figures.py**: contain the functions needed for the analysis and produce all the plots

In **tools.py** and **generate_figures.ipynb** you need to change path_to_origin to be where the folder with all the data is (https://imperiallondon-my.sharepoint.com/:f:/g/personal/ef120_ic_ac_uk/Es7cdEjlUF1AiClyo7uNag0B6NMqtGdl7HOHBVxqFW0vAw?e=umgTbM)