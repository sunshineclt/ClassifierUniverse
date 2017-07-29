# Classifier Universe
Homework 2 "Option I: Model averaging with online learning algorithms" for 2017 summer course "Advanced Machine Learning: Online learning and Optimization" 

## Includes
A lot of algorithms to solve classifying problem. That's why it is called Classifier Universe:)
And online learning to combine them and compare with the cross-validation strategy. 

## How to run
The paper "**final project.pdf**" explains detailly about what I do. 
To run the code, you need two kinds of MNIST dataset, one is a smaller and the other one is full dataset. The first one will automatically downloaded from Internet and since it is just a small version of MNIST it won't take long. The second one is appended in the project as "mnist.npz". So, as long as all the dependencies (including scikit-learn, numpy, matplotlib) are satisfied, you can run "main.py" to see all the results mensioned in the paper (Though you may need to change some of the parameters). The code is intended for the **section 3.1 and 3.3** in the paper and if you want to run **section 3.2**'s result, just edit line 42 and 43 from ":500*(j+1)" to "5000". 

## Finally
Hope you enjoy the code:)


