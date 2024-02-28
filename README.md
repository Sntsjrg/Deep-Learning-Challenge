# Deep-Learning-Challenge
## Overview
  - The task involves creating a binary classifier for Alphabet Soup, a nonprofit foundation, to predict the success of funding applicants based on provided dataset features. The dataset includes over 34,000 organizations with columns capturing metadata, such as identification, application type, industry affiliation, government classification, funding use case, organization type, active status, income classification, special considerations, funding amount requested (ASK_AMT), and a target variable (IS_SUCCESSFUL) indicating funding effectiveness.
  - 
## Step 1: Preprocess the Data

1. **Upload Starter File**: Upload the starter file to Google Colab.

2. **Read and Preprocess Data**:
   - Read charity_data.csv into a Pandas DataFrame.
   - Identify target and feature variables; drop EIN and NAME columns.

3. **Unique Values and Binning**:
   - Determine unique values for each column.
   - For >10 unique values, bin "rare" categorical variables.
   - Use pd.get_dummies() to encode categorical variables.

4. **Split and Scale Data**:
   - Split data into features array (X) and target array (y).
   - Use train_test_split for training and testing datasets.
   - Scale features using StandardScaler.

# Step 2: Compile, Train, and Evaluate the Model

Using TensorFlow, design a neural network for binary classification predicting Alphabet Soup-funded organization success based on dataset features. Follow these steps:

1. **Load Preprocessed Data**: Continue using the Google Colab file from Step 1.

2. **Create Neural Network Model**:
   - Assign input features and nodes for each layer using TensorFlow and Keras.
   - Design the first hidden layer with an appropriate activation function.
   - Optionally, add a second hidden layer with an appropriate activation function.
   - Create the output layer with an appropriate activation function.

3. **Check Model Structure**: Verify the structure of the designed neural network.

4. **Compile and Train Model**:
   - Compile the model.
   - Train the model using the preprocessed data.
   - Implement a callback to save model weights every five epochs.

5. **Evaluate Model**:
   - Assess model performance using test data.
   - Determine model loss and accuracy.

6. **Save and Export Results**:
   - Save the model results to an HDF5 file.
   - Name the file 'AlphabetSoupCharity.h5'.

# Step 3: Optimize the Model

Optimize the TensorFlow model to achieve a predictive accuracy surpassing 75%. Employ various strategies:

1. **Adjust Input Data**:
   - Experiment with dropping more or fewer columns.
   - Create additional bins for rare occurrences in columns.
   - Adjust the number of values for each bin.

2. **Neural Network Modification**:
   - Add more neurons to existing hidden layers.
   - Introduce more hidden layers.
   - Utilize different activation functions for the hidden layers.

3. **Training Regimen Adjustments**:
   - Experiment with adding or reducing the number of epochs in the training regimen.

Apply these optimizations to enhance model performance and reach the specified target accuracy.

# Step 4: Write a Report on the Neural Network Model

  **For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup**.

The report should contain the following:

 **Overview of the analysis**: Explain the purpose of this analysis.

**Results**: Using bulleted lists and images to support your answers, address the following questions:

**Data Preprocessing**

What variable(s) are the target(s) for your model?
What variable(s) are the features for your model?
What variable(s) should be removed from the input data because they are neither targets nor features?

**Compiling, Training, and Evaluating the Model**

How many neurons, layers, and activation functions did you select for your neural network model, and why?
Were you able to achieve the target model performance?
What steps did you take in your attempts to increase model performance?

**Summary**: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.



