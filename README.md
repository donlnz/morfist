# morfist: mixed-output-rf
Multi-target Random Forest implementation that can mix both classification and regression tasks.

Morfist implements the Random Forest algorithm (Breiman, 2001)
with support for mixed-task multi-task learning, i.e., it is possible to train the model on any number
of classification tasks and regression tasks, simultaneously. Morfist's mixed multi-task learning implementation follows that proposed by Linusson (2013). 

* Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
* Linusson, H. (2013). Multi-output random forests.



## TODO:
* Some amount of documentation.
* Speed up the learning algorithm implementation (morfist is currently **much** slower than the Random Forest implementation available in scikit-learn) 
