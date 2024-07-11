# SkimLit

Skimlit is a NLP project which helps convert attributes to headings such as Result , Methods , Objectives , Conclusions and Background for easier reading and analysis


#MODEL

1)The architecture used is avaliable in model.png file.

2)The model was trained on pubmed-rct-20k dataset .

3)Model folder contains the trained model can be used using tf.keras.load_model method.

4)It acheived accuracy of 84% on validation split.

Input Format
1)preprocessing fuctions available in utility.py convert sentences to the required input format 

Architecture
![model](https://user-images.githubusercontent.com/97525421/235885818-b2af264d-51fc-4ebb-a45c-347a141ff1b6.png)


Sample
![usecase](https://user-images.githubusercontent.com/97525421/235887381-a21c7bef-2675-44e2-956f-a923fcf6d934.png)
