# Paper

This repository is forked from [matteozavatteri/certified-nn-ecai24](https://github.com/matteozavatteri/certified-nn-ecai24), that 
contains the supplementary material of the paper:

	Matteo Zavatteri, Davide Bresolin, Nicolò Navarin. 
	Automated Synthesis of Certified Neural Networks. 
	27TH EUROPEAN CONFERENCE ON ARTIFICIAL INTELLIGENCE (ECAI 2024).

We only needed `milp.py`, `conf.py` from there, hence most of the files have been removed. 

# Dataset for original repo

That code worked on the original dataset of the paper:
	
	Jonas Tjomsland, Sinan Kalkan and Hatice Gunes
	Mind Your Manners! A Dataset and A Continual Learning Approach for Assessing Social Appropriateness of Robot Actions
	Frontiers in Robotics and AI, Special Issue on Lifelong Learning and Long-term Human-Robot Interaction, 9:1-18, 2022

[Link to the paper](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.669420/full)

They do not provide the dataset here. However, to get all set up to reproduce the experiments:

- Download [https://github.com/jonastjoms/MANNERS-DB/blob/master/src/data/not_normallized.csv](https://github.com/jonastjoms/MANNERS-DB/blob/master/src/data/not_normallized.csv)
- Run `$ python3 preprocess.py`. This will create the file `dataset.csv`.

This step needs only to be done once.

# Dependencies

- gurobipy
- pytorch (thus numpy too)

To run all the experiments of that paper run:

	$ python3 RunExperiments.py

Recall to download the dataset and preprocess it.

	
	
