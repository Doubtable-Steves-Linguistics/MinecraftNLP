# Minecraft NLP Project
### Presented by:
- Rae Downen, Cristina Lucin, Michael Mesa and John "Chris" Rosenberger

## Deliverables
* A GitHub Repository ("repo") containing project work
* A Readme file with project description and instructions on how to duplicate project

## Project Description

This project focuses on building a prediction model for accurately predicting the coding language of a project using
examination of GitHub repo README files. Our goal is to develop several predictive models utilizing Python and Python libraries,
and select the most effective model for production. Initially, we are utilizing BeautifulSoup to acquire our data, selecting 500
repositories from the 'Minecraft' topic tag from GitHub, taking in all Readme text and repo language information from each repo. 
After gathering the data, we explore the data through questions, visualizations, and statistical tests before developing a model
that can tell us: "What language is this repository most likely to be written in?"

## Initial Thoughts

Keywords, either singularly or through bigrams and trigrams, will help develop features to effectively model prediction
of Repo programming languages.

# Data Dictionary

| Feature | Definition | Manipulations applied|Data Type|
|--------|-----------|-----------|-----------|
|repo| The repository path (URL path) to the directory || string
|langugage| The programming language listed as the top used for the repo || string
|readme_contents| The text contents of the readme file || string
|clean| Cleaned text from the readme_contents | String and REGEX methods | string
|lemmatized| Lemmatized text from readme_contents| Cleaned and Lemmatized | string

## Exploration

* Java was the most frequent language found in the Repos examined
* Javascript Repos had the highest average wordcount, Java Repos had the lowest
* "Install" was the most common word for Python Repos
* "Mod" and "Build" were the most frequently found Java strings
* "Command" was the most frequent word found in JavaScript Repos

## Modeling

* We elected to utilize accuracy as the evaluation metric
* We developed three different models using differnt model types: (Naive Bayes, SKLearn Gradient Booster, XG Boost)
* The model that performs the best was evaluated on test data
* We utilized the mode of 'language' as the baseline (Java, 45.3)
* All models were overfit on the training data
* SKLearn Gradient Boost was chosen for test data
* This model performed with a 76 percent accuracy, a 30 percent improvement from the baseline

## Conclusions

GitHub Repos with different programming languages have significantly different features (Word count and unique words)
Because ReadMe files are written in normal language, the accuracy of any model is limited
Improved cleaning methods may increase model performance
Count Vectorization (CV) in combination with ensemble classification is an effective modeling strategy for NLP/Text Classification problems

## Recommendations 

* Acquire longer Readme text files to feed into algorithm
* Narrow down parameters for classifications (more languages are more difficult to classify)
* Additional hyperparameter tuning may result in better model performance

## Next Steps

* Utilize statistical methods to identify additional stopwords
* Develop and test different model types for performance
* Find alternative methods for pulling repo data from GitHub

## Steps to Reproduce
* Our data comes from the topic category "Minecraft" repositories on GitHub as of January 25, 2023.
1) Clone this repo into your computer.
2) Modules and data files are contained in the project folder.
3) Place env.py file into the cloned repo.
3) Run the ```final_project_notebook.ipynb``` file.