# Data_Mining
COMP#5440
Name : Varun Kumar Bejjenki
ID : 02070487
Email : varunkumar_bejjenki@student.uml.edu


Project Description:

splitting.py: this file is to split the data set into training, val and testing text files. we will be using pandas.get_dummies() to convert the data set to one hot binary file. Then we will be using numpy.split function to divide the data set into three

train dataset = 70%
Validate Dataset = 15%
Testing Dataset = 15%

this will result in 'training.txt', 'val.txt','testing.txt' fields

formulas.py: In this file the formulas required for model has been implemented using numpy library functions.

models.py: The model functions will be present in this file. The eval() function is we will be using numpy.dot() function and the sigmoid function defined in fromulas.py file. Then the backprop() is calculated using learning rate other formula. 

Proj_test.py: In this file the code required for training and fitting the model will be present. We will be calling eval() function then backprop function then the calculate the error value of the data set.

Outputs.docx : this contains the error values for datasets and their screenshots.

-----------------------------------------------------------------------------------------------

Steps to run the code:

1)command:
python splitting.py

2)python proj_test.py
This will first give you the training error percentage value. 

3)Then press enter to start the validation process. This will result in validation error value

4)Then press enter to start the testing process. the final test error value will be the output.

-------------------------------------------------------------------------------------------------
