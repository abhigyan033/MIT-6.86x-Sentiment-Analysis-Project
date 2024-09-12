from string import punctuation, digits
import numpy as np
import random



#==============================================================================
#===  PART I  =================================================================
#==============================================================================



def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices



def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        `feature_vector` - numpy array describing the given data point.
        `label` - float, the correct classification of the data
            point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - float representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given data point and
        parameters.
    """
    dot_product = np.dot(theta, feature_vector)
    hinge_loss = max(0, 1 - label * (dot_product + theta_0))
    
    return hinge_loss



def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the hinge loss for given classification parameters averaged over a
    given dataset

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given dataset and
        parameters.  This number should be the average hinge loss across all of
    """

    num_points = feature_matrix.shape[0]
    total_hinge_loss = 0.0
    
    for i in range(num_points):
        feature_vector = feature_matrix[i]
        label = labels[i]
        # Calculate hinge loss for a single data point using the previously defined function
        loss = hinge_loss_single(feature_vector, label, theta, theta_0)
        total_hinge_loss += loss
    
    # Calculate the average hinge loss
    average_hinge_loss = total_hinge_loss / num_points
    
    return average_hinge_loss




def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the perceptron algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.
    Returns a tuple containing two values:
        the updated feature-coefficient parameter `theta` as a numpy array
        the updated offset parameter `theta_0` as a floating point number
    """
    prediction = np.dot(current_theta, feature_vector) + current_theta_0
    
    # Check if the prediction is correct
    if label * prediction <= 0:
        # Update the parameters
        new_theta = current_theta + label * feature_vector
        new_theta_0 = current_theta_0 + label
    else:
        # If correct, return the current parameters
        new_theta = current_theta
        new_theta_0 = current_theta_0
        
    return new_theta, new_theta_0



def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set: we do not stop early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the feature-coefficient parameter `theta` as a numpy array
            (found after T iterations through the feature matrix)
        the offset parameter `theta_0` as a floating point number
            (found also after T iterations through the feature matrix).
    """
    nsamples, nfeatures = feature_matrix.shape
    theta = np.zeros(nfeatures)
    theta_0 = 0.0
    for t in range(T):
        for i in get_order(nsamples):
            feature_vector = feature_matrix[i]
            label = labels[i]
            theta, theta_0 = perceptron_single_step_update(feature_vector, label, theta, theta_0)
    return theta, theta_0



def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given dataset.  Runs `T`
    iterations through the dataset (we do not stop early) and therefore
    averages over `T` many parameter values.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: It is more difficult to keep a running average than to sum and
    divide.

    Args:
        `feature_matrix` -  A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the average feature-coefficient parameter `theta` as a numpy array
            (averaged over T iterations through the feature matrix)
        the average offset parameter `theta_0` as a floating point number
            (averaged also over T iterations through the feature matrix).
    """
    nsamples, nfeatures = feature_matrix.shape
    theta = np.zeros(nfeatures)
    theta_0 = 0.0

    cumulative_theta = np.zeros(nfeatures)
    cumulative_theta_0 = 0.0

    for t in range(T):
        for i in get_order(nsamples):
            feature_vector = feature_matrix[i]
            label = labels[i]
            theta, theta_0 = perceptron_single_step_update(feature_vector, label, theta, theta_0)
            cumulative_theta += theta
            cumulative_theta_0 += theta_0
    
    total_updates = T * nsamples
    average_theta = cumulative_theta / total_updates
    average_theta_0 = cumulative_theta_0 / total_updates
    
    return average_theta, average_theta_0    


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the Pegasos algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        `feature_vector` - A numpy array describing a single data point.
        `label` - The correct classification of the feature vector.
        `L` - The lamba value being used to update the parameters.
        `eta` - Learning rate to update parameters.
        `theta` - The old theta being used by the Pegasos
            algorithm before this update.
        `theta_0` - The old theta_0 being used by the
            Pegasos algorithm before this update.
    Returns:
        a tuple where the first element is a numpy array with the value of
        theta after the old update has completed and the second element is a
        real valued number with the value of theta_0 after the old updated has
        completed.
    """
    decision_value = np.dot(theta, feature_vector) + theta_0
    if label * decision_value >= 1:
        # Update theta without the feature vector
        theta_new = (1 - eta * L) * theta
        theta_0_new = theta_0
    else:
        # Update theta with the feature vector
        theta_new = (1 - eta * L) * theta + eta * label * feature_vector
        theta_0_new = theta_0 + eta * label
    
    return theta_new, theta_0_new



def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T iterations
    through the data set, there is no need to worry about stopping early.  For
    each update, set learning rate = 1/sqrt(t), where t is a counter for the
    number of updates performed so far (between 1 and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.  Do
    not copy paste code from previous parts.

    Args:
        `feature_matrix` - A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        `L` - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns:
        a tuple where the first element is a numpy array with the value of the
        theta, the linear classification parameter, found after T iterations
        through the feature matrix and the second element is a real number with
        the value of the theta_0, the offset classification parameter, found
        after T iterations through the feature matrix.
    """
    m, n = feature_matrix.shape
    theta = np.zeros(n)
    theta_0 = 0.0
    t = 0

    for _ in range(T):
        order = get_order(m)
        for i in order:
            t += 1
            eta = 1 / np.sqrt(t)
            feature_vector = feature_matrix[i]
            label = labels[i]
            theta, theta_0 = pegasos_single_step_update(feature_vector, label, L, eta, theta, theta_0)
    
    return theta, theta_0



#==============================================================================
#===  PART II  ================================================================
#==============================================================================



##  #pragma: coderesponse template
##  def decision_function(feature_vector, theta, theta_0):
##      return np.dot(theta, feature_vector) + theta_0
##  def classify_vector(feature_vector, theta, theta_0):
##      return 2*np.heaviside(decision_function(feature_vector, theta, theta_0), 0)-1
##  #pragma: coderesponse end



def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses given parameters to classify a set of
    data points.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.

    Returns:
        a numpy array of 1s and -1s where the kth element of the array is the
        predicted classification of the kth row of the feature matrix using the
        given theta and theta_0. If a prediction is GREATER THAN zero, it
        should be considered a positive classification.
    """
    # Compute the dot product of each data point with the classifier and add theta_0
    scores = np.dot(feature_matrix, theta) + theta_0
    
    # Classify based on the sign of the score
    classifications = np.where(scores > 0, 1, -1)
    
    return classifications


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.  The classifier is
    trained on the train data.  The classifier's accuracy on the train and
    validation data is then returned.

    Args:
        `classifier` - A learning function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        `train_feature_matrix` - A numpy matrix describing the training
            data. Each row represents a single data point.
        `val_feature_matrix` - A numpy matrix describing the validation
            data. Each row represents a single data point.
        `train_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        `val_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        `kwargs` - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns:
        a tuple in which the first element is the (scalar) accuracy of the
        trained classifier on the training data and the second element is the
        accuracy of the trained classifier on the validation data.
    """
    T_class = classifier(train_feature_matrix, train_labels, **kwargs)
    T_judge = classify(train_feature_matrix, T_class[0], T_class[1])
    
    T_acc = accuracy(T_judge, train_labels)

    # for validation, you should not train again, just apply the classifier from the train set
    V_judge = classify(val_feature_matrix, T_class[0], T_class[1])

    V_acc = accuracy(V_judge, val_labels)

    return T_acc, V_acc



def extract_words(text):
    """
    Helper function for `bag_of_words(...)`.
    Args:
        a string `text`.
    Returns:
        a list of lowercased words in the string, where punctuation and digits
        count as their own words.
    """
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    digits = '0123456789'
    
    # Combine punctuation and digits
    punctuation_digits = punctuation + digits
    
    # Insert spaces around punctuation and digits
    for c in punctuation_digits:
        text = text.replace(c, ' ' + c + ' ')
    
    # Convert to lowercase and split into words
    return text.lower().split()



def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    dictionary = {}  # maps word to unique index

    with open("stopwords.txt", "r") as text_file:
        word_stop = set(text_file.read().split())  # Set of stopwords

    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and word not in word_stop:
                dictionary[word] = len(dictionary)
    return dictionary



def extract_bow_feature_vectors(reviews, indices_by_word, binarize=True):
    """
    Args:
        `reviews` - a list of natural language strings
        `indices_by_word` - a dictionary of uniquely-indexed words.
    Returns:
        a matrix representing each review via bag-of-words features.  This
        matrix thus has shape (n, m), where n counts reviews and m counts words
        in the dictionary.
    """
    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word not in indices_by_word: continue
            feature_matrix[i, indices_by_word[word]] = 1 if binarize else feature_matrix[i, indices_by_word[word]] + 1
    return feature_matrix



def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the fraction of predictions that are correct.
    """
    return (preds == targets).mean()
