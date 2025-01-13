import numpy as np
from typing import Tuple, List
import random
import time
import os

def read_ddata(filename: str, n_rows: int = 28, n_cols: int = 28) -> np.ndarray:
    """
    Read digit image data from file
    Returns numpy array of images, each 28x28 pixels
    """
    try:
        images = []
        with open(os.path.abspath(filename), 'r') as f:
            image = []
            for line in f:
                # Remove newline and convert pixels to binary (0 or 1)
                pixels = [1 if c in '+#' else 0 for c in line.rstrip('\n')]
                image.extend(pixels)
                if len(image) == n_rows * n_cols:
                    images.append(image)
                    image = []
        return np.array(images)
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")
        raise

def read_fdata(filename: str, n_rows: int = 70, n_cols: int = 60) -> np.ndarray:
    """
    Read face image data from file
    Returns numpy array of images, each 70x60 pixels
    """
    try:
        images = []
        with open(os.path.abspath(filename), 'r') as f:
            image = []
            for line in f:
                # Convert pixels to binary (0 or 1)
                pixels = [1 if c == '#' else 0 for c in line.rstrip('\n')]
                image.extend(pixels)
                if len(image) == n_rows * n_cols:
                    images.append(image)
                    image = []
        return np.array(images)
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")
        raise

def read_labels(filename: str) -> np.ndarray:
    """Read labels from file"""
    try:
        with open(os.path.abspath(filename), 'r') as f:
            return np.array([int(line.strip()) for line in f])
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")
        raise

class DataLoader:
    def __init__(self, base_path: str):
        self.base_path = os.path.abspath(base_path)
        
    def load_ddata(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all digit datasets
        Returns (train_images, train_labels, test_images, test_labels, val_images, val_labels)
        """
        digit_path = os.path.join(self.base_path, "digitdata")
        
        print(f"Loading digit data from: {digit_path}")  # Debug print
        
        train_images = read_ddata(os.path.join(digit_path, "trainingimages"))
        train_labels = read_labels(os.path.join(digit_path, "traininglabels"))
        
        test_images = read_ddata(os.path.join(digit_path, "testimages"))
        test_labels = read_labels(os.path.join(digit_path, "testlabels"))
        
        val_images = read_ddata(os.path.join(digit_path, "validationimages"))
        val_labels = read_labels(os.path.join(digit_path, "validationlabels"))
        
        return train_images, train_labels, test_images, test_labels, val_images, val_labels

    def load_fdata(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all face datasets
        Returns (train_images, train_labels, test_images, test_labels, val_images, val_labels)
        """
        face_path = os.path.join(self.base_path, "facedata")
        
        print(f"Loading face data from: {face_path}")  # Debug print
        
        train_images = read_fdata(os.path.join(face_path, "facedatatrain"))
        train_labels = read_labels(os.path.join(face_path, "facedatatrainlabels"))
        
        test_images = read_fdata(os.path.join(face_path, "facedatatest"))
        test_labels = read_labels(os.path.join(face_path, "facedatatestlabels"))
        
        val_images = read_fdata(os.path.join(face_path, "facedatavalidation"))
        val_labels = read_labels(os.path.join(face_path, "facedatavalidationlabels"))
        
        return train_images, train_labels, test_images, test_labels, val_images, val_labels

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}  # P(C)
        self.feature_probs = {}  # P(F|C)
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train Naive Bayes classifier on binary features
        X: array of shape (n_samples, n_features) with binary values
        y: array of shape (n_samples,) with class labels
        """
        n_samples, n_features = X.shape
        classes = np.unique(y)
        
        # Calculate P(C) for each class
        for c in classes:
            self.class_probs[c] = np.sum(y == c) / n_samples
            
        # Calculate P(F|C) for each feature and class
        for c in classes:
            class_samples = X[y == c]
            # Laplace smoothing
            feature_counts = np.sum(class_samples, axis=0) + 1
            total_count = len(class_samples) + 2
            self.feature_probs[c] = feature_counts / total_count
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for new samples
        Returns array of predicted class labels
        """
        predictions = []
        for x in X:
            class_scores = {}
            for c in self.class_probs:
                # Calculate log probability to avoid underflow
                score = np.log(self.class_probs[c])
                score += np.sum(x * np.log(self.feature_probs[c]) + 
                              (1-x) * np.log(1 - self.feature_probs[c]))
                class_scores[c] = score
            predictions.append(max(class_scores.items(), key=lambda x: x[1])[0])
        return np.array(predictions)

class MultiClassPerceptron:
    def __init__(self, n_features: int, n_classes: int, learning_rate: float = 0.1):
        self.n_classes = n_classes
        # One perceptron per class
        self.weights = np.zeros((n_classes, n_features))
        self.biases = np.zeros(n_classes)
        self.lr = learning_rate
        
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50):
        """
        Train multi-class perceptron classifier
        X: array of shape (n_samples, n_features)
        y: array of shape (n_samples,) with class labels
        """
        # Normalize features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        for epoch in range(epochs):
            mistakes = 0
            for xi, yi in zip(X, y):
                # Calculate scores for each class
                scores = np.dot(self.weights, xi) + self.biases
                predicted_class = np.argmax(scores)
                
                # Update weights if prediction is wrong
                if predicted_class != yi:
                    mistakes += 1
                    # Decrease weights for predicted class
                    self.weights[predicted_class] -= self.lr * xi
                    self.biases[predicted_class] -= self.lr
                    # Increase weights for correct class
                    self.weights[yi] += self.lr * xi
                    self.biases[yi] += self.lr
            
            # Early stopping if perfect classification
            if mistakes == 0:
                print(f"Converged after {epoch + 1} epochs")
                break
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X"""
        # Normalize features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        scores = np.dot(X, self.weights.T) + self.biases
        return np.argmax(scores, axis=1)

def train_and_evaluate(classifier_type: str, X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray, 
                      training_sizes: List[float]) -> Tuple[List[float], List[float]]:
    """
    Train and evaluate classifier with different amounts of training data
    Returns accuracies and training times
    """
    accuracies = []
    times = []
    
    for size in training_sizes:
        n_samples = int(len(X_train) * size)
        indices = random.sample(range(len(X_train)), n_samples)
        X_subset = X_train[indices]
        y_subset = y_train[indices]
        
        if classifier_type == 'naive_bayes':
            classifier = NaiveBayesClassifier()
        else:
            # For perceptron, we now use the multi-class version
            n_classes = len(np.unique(y_train))
            classifier = MultiClassPerceptron(X_train.shape[1], n_classes)
        
        start_time = time.time()
        classifier.train(X_subset, y_subset)
        train_time = time.time() - start_time
        
        accuracy = evaluate_classifier(classifier, X_test, y_test)
        accuracies.append(accuracy)
        times.append(train_time)
        
    return accuracies, times

def extract_features(image: np.ndarray, task: str) -> np.ndarray:
    """
    Extract features from image based on task type
    Returns binary feature vector
    """
    if task == 'digit':
        # Use 10x10 grid of binary features
        h, w = image.shape
        grid_h, grid_w = h // 10, w // 10
        features = []
        for i in range(10):
            for j in range(10):
                region = image[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                features.append(1 if np.mean(region) > 0.5 else 0)
        return np.array(features)
    else:  # face detection
        # Use edge density in 8x8 grid cells
        h, w = image.shape
        grid_h, grid_w = h // 8, w // 8
        features = []
        for i in range(8):
            for j in range(8):
                region = image[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                features.append(1 if np.sum(region) > (grid_h * grid_w * 0.1) else 0)
        return np.array(features)

def evaluate_classifier(classifier, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Calculate classification accuracy"""
    predictions = classifier.predict(X_test)
    return np.mean(predictions == y_test)

def train_and_evaluate(classifier_type: str, X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray, 
                      training_sizes: List[float]) -> Tuple[List[float], List[float]]:
    """
    Train and evaluate classifier with different amounts of training data
    Returns accuracies and training times
    """
    accuracies = []
    times = []
    
    for size in training_sizes:
        n_samples = int(len(X_train) * size)
        indices = random.sample(range(len(X_train)), n_samples)
        X_subset = X_train[indices]
        y_subset = y_train[indices]
        
        if classifier_type == 'naive_bayes':
            classifier = NaiveBayesClassifier()
        else:
            # For perceptron, we now use the multi-class version
            n_classes = len(np.unique(y_train))
            classifier = MultiClassPerceptron(X_train.shape[1], n_classes)
        
        start_time = time.time()
        classifier.train(X_subset, y_subset)
        train_time = time.time() - start_time
        
        accuracy = evaluate_classifier(classifier, X_test, y_test)
        accuracies.append(accuracy)
        times.append(train_time)
    
    return accuracies, times

def main():
    import os
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Script directory:", script_dir)
    
    # Initialize data loader with path relative to script location
    loader = DataLoader(os.path.join(script_dir, "data"))
    
    # Load digit data
    digit_train_images, digit_train_labels, digit_test_images, digit_test_labels, \
    digit_val_images, digit_val_labels = loader.load_ddata()
    
    # Load face data
    face_train_images, face_train_labels, face_test_images, face_test_labels, \
    face_val_images, face_val_labels = loader.load_fdata()
    
    # Extract features for digit data
    digit_train_features = np.array([extract_features(img.reshape(28, 28), 'digit') 
                                   for img in digit_train_images])
    digit_test_features = np.array([extract_features(img.reshape(28, 28), 'digit') 
                                  for img in digit_test_images])
    
    # Extract features for face data
    face_train_features = np.array([extract_features(img.reshape(70, 60), 'face') 
                                  for img in face_train_images])
    face_test_features = np.array([extract_features(img.reshape(70, 60), 'face') 
                                 for img in face_test_images])
    
    # Training sizes to test (10% to 100%)
    training_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Evaluate Naive Bayes on digits
    nb_digit_accuracies, nb_digit_times = train_and_evaluate(
        'naive_bayes', digit_train_features, digit_train_labels,
        digit_test_features, digit_test_labels, training_sizes
    )
    
    # Evaluate Perceptron on digits
    perceptron_digit_accuracies, perceptron_digit_times = train_and_evaluate(
        'perceptron', digit_train_features, digit_train_labels,
        digit_test_features, digit_test_labels, training_sizes
    )
    
    # Print results
    print("Digit Classification Results:")
    print("\nNaive Bayes:")
    for size, acc, t in zip(training_sizes, nb_digit_accuracies, nb_digit_times):
        print(f"Training size: {size*100}%, Accuracy: {acc*100:.2f}%, Time: {t:.3f}s")
    
    print("\nPerceptron:")
    for size, acc, t in zip(training_sizes, perceptron_digit_accuracies, perceptron_digit_times):
        print(f"Training size: {size*100}%, Accuracy: {acc*100:.2f}%, Time: {t:.3f}s")
    # ... (keep existing digit classification code) ...
    
    # Evaluate Naive Bayes on faces
    nb_face_accuracies, nb_face_times = train_and_evaluate(
        'naive_bayes', face_train_features, face_train_labels,
        face_test_features, face_test_labels, training_sizes
    )
    
    # Evaluate Perceptron on faces
    perceptron_face_accuracies, perceptron_face_times = train_and_evaluate(
        'perceptron', face_train_features, face_train_labels,
        face_test_features, face_test_labels, training_sizes
    )
    
    # Print face classification results
    print("\nFace Classification Results:")
    print("\nNaive Bayes:")
    for size, acc, t in zip(training_sizes, nb_face_accuracies, nb_face_times):
        print(f"Training size: {size*100}%, Accuracy: {acc*100:.2f}%, Time: {t:.3f}s")
    
    print("\nPerceptron:")
    for size, acc, t in zip(training_sizes, perceptron_face_accuracies, perceptron_face_times):
        print(f"Training size: {size*100}%, Accuracy: {acc*100:.2f}%, Time: {t:.3f}s")

if __name__ == "__main__":
    main()