# Flight Delay Prediction at Scale

#### Presentation Link: https://docs.google.com/presentation/d/1Qz2XyPpMMYSWpUuMSiwSV4yTi9EIQV-89tcz7LAIA1k/edit?usp=sharing

##### Team:
* Shyamkarthik Rameshbabu
* Sean Campos 
* Sanjay Saravanan
* Devesh Khandelwal


## Introduction

Predicting Flight Delay has been an age old problem troubling travellers, airport administrators, and airline staff alike. After studying various pieces of literature in this space, our team has taken a stab at using **flight**, **weather**, and **airport** data to build machine learning models that will **predict whether a flight will be delayed, or not delayed,** based off a variety of features. We hope this report will shed light on our journey and our discoveries.

## Datasets

* FAA Flight Data - 1.52 GB
* NOAA Weather Data - 23.64 GB 
* Flight Stations Data - 726.46 KB


## Context:
As we are all painfully aware, airline delays are all too common of an experience.  And as airports are one of the most resource intensive shared spaces in modern commerce, delays bring increased costs to crews, airport personnel, limited gate availability, fuel and maintenance cost as well as to the passengers.  The FAA reports that in 2019 flight delays cost the airlines more than $8 billion, passengers more than $18 billion, an additional $6 billion in lost demand and indirect costs.  Every minute of recovered delay time is worth at least $75.  While many of the event sequences that cause these delays are too complicated to fully prevent, predicting them and providing advanced notice to the multitude of interested parties would not only be a convenience, but a significant savings in wasted resources. Passengers would not have to spend as much time occupying expensive shared resources and airport personnel could take the necessary steps to allocate limited resources such as gates and scheduled equipment usage to not only save resources, but also mitigate the propagation of the delay downstream to additional flights, crews and passengers.

Our specific objective is to predict whether or not each flight will depart from the gate more than fifteen minutes after the scheduled departure time based on all of the information available two hours prior to departure.  These predictions are suitable for triggering both a customer centric notification system, such as push notifications through an air carrier’s app, as well as the airport’s internal resource coordination infrastructure. 

While every prediction algorithm strives for accuracy, we think it’s important to note that there may be a high cost to **false positives** associated with this system.  Passengers mistakenly believing that they have more time to arrive at the airport or misallocating airport resources for an incoming flight could be far more costly than the benefits of having a prediction system in the first place.  For this reason, **precision** will be our **most important metric** as it will force us to focus on reducing false positives.

In the previous publications that take a big data approach with similar datasets, we have found that Choi et. al. [1] achieved accuracy=0.80 with precision=0.71 and Patgiri et. al. [2] achieved accuracy=0.82 with precision=0.82.  In both papers, many machine learning algorithms were explored ranging from KNN to logistic regression and various tree based methods, yet the best performance was achieved with random forests.


> 1. S. Choi, Y. Kim, S. Briceno, and D. Mavris. Prediction of weather-induced airline delays based on machine learning algorithms.  In AIAA/IEEE Digital Avionics Systems Confer-ence - Proceedings, volume 2016-December, 2016. [https://ieeexplore.ieee.org/document/7777956]
> 2. R. Patgiri, S. Hussain, and A. Nongmeikapam. Empirical Study on Airline Delay Analysis and Prediction. In EAI Endorsed Transactions 2020. [https://arxiv.org/pdf/2002.10254.pdf]
> 


#### Working Directory 
Many of the working directories are too large to be stored on to GitHub, therefore you will find the following directories shared in compressed format.

    ├── airline-delays-literature
    ├── data_joining
    ├── eda
    │   ├── airlines
    │   ├── stations
    │   └── weather
    ├── feature_engineering
    ├── modelling
    ├── other
    └── slide_visuals
        └── DeveshEDAImages
