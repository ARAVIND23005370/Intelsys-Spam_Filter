# TITLE:
Spam Detector Using Naive Bayes
# BRIEF EXPLANATION ABOUT THE WORKING MODEL:
This project is a machine learning-based spam filter that predicts whether a given email is spam or not. It processes email content and classifies it using a pre-trained model.



# ML ALGORITHM USED:
1) Algorithm: Naive Bayes (MultinomialNB)
2) How It Works: The Naive Bayes algorithm is a probabilistic classifier based on Bayes' theorem. It calculates the likelihood of an email being spam by analyzing the frequency of words in the email content, using the training data to predict the label (Spam or Not Spam).


# FRONT-END TECHNOLOGY USED:
1) Technology: HTML, CSS, and Flask Framework
2) Description:
   1) Flask: Used to create the backend for handling HTTP requests, routing, and integrating the ML model.
   2) HTML & CSS: Designed a simple, user-friendly interface where users can input email content and receive spam predictions.
   3) jQuery: Added interactivity for dynamic result updates without refreshing the page.

# STEP-BY-STEP EXPLANATION OF THE CODE:
## Model Code Explanation
1) Dataset Loading and Preprocessing:

* Loaded the dataset containing email content and corresponding labels (Spam or Not Spam).
* Renamed or created necessary columns to standardize the dataset (email column renamed to text).
* Converted labels into binary values (1 for Spam, 0 for Not Spam).

2) Training the Model:

* Split the dataset into training and testing sets (75% training, 25% testing).
* Used CountVectorizer to convert email content into a bag-of-words feature matrix.
* Trained a Multinomial Naive Bayes classifier on the training data.
3) Saving and Loading the Model:

* Serialized the trained model and vectorizer using Python’s pickle module for reuse without retraining.
4) Prediction API:

* Set up a Flask route (/predict) to handle POST requests.
* Transformed input email content using the saved vectorizer and made predictions using the model.
## Front-End Code Explanation
1) HTML:

* Created a simple form with a textarea input for users to enter email content.
* Added a button to submit the form data to the backend for prediction.
2) CSS:

* Designed a visually appealing interface with gradient text, glowing buttons, and a responsive layout.
* Used animations (fade-in) and a semi-transparent background for a modern look.
3) JavaScript (jQuery):

* Added functionality to submit the form without reloading the page.
* Displayed the prediction result dynamically based on the backend response.


# OUTPUT:
### INTERFACE
![alt text](interface.png)

### HAM 
![alt text](ham.png)
 ### SPAM 
 ![alt text](spam.png)


# RESULT:

The Spam Detector successfully classifies emails with high accuracy using the Naive Bayes algorithm. Its interactive front-end makes it easy to use and visually appealing. This project demonstrates the effective integration of machine learning with a web-based interface.

