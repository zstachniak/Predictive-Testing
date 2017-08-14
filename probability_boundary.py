# -*- coding: utf-8 -*-
"""
Probability Boundary Score

@author: zstachniak
"""

# Import Statements
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class probability_boundary:
    '''A class that represents probability scores weighted by true values.'''
    
    def __init__ (self, y_pred_prob, y_true, boundary=0.50):
        '''Function initializes the probability boundary score given 
        a set of prediction probabilities and the true values. Upon 
        initialization, function gathers both an unweighted score and 
        a score that is balanced by class size, in which case the score 
        is calculated separately for each class and then averaged.

        @ Parameters:
        ------------------
            y_pred_prob: array of probability scores for each class
            y_true: array of true class values
            boundary: float value between 0.00 and 1.00
        '''
        # Coerce to ensure values are not imported with unordered indices 
        # (e.g., a shuffled Pandas series)
        self.y_pred_prob = np.asarray(y_pred_prob)
        self.y_true = np.asarray(y_true)
        
        # Preserve boundary value
        self.boundary = boundary
        
        # Unweighted Probability Boundary Score
        # Sample Size
        n = self.y_true.shape[0]
        # Make predictions
        y_pred = np.argmax(self.y_pred_prob, axis=1)
        # Identify correct predictions
        P_correct = self.y_pred_prob[np.argwhere(y_pred == self.y_true)]
        # Identify incorrect predictions
        P_incorrect = self.y_pred_prob[np.argwhere(y_pred != self.y_true)]
        # Caculate Probability Boundary Score
        pb_correct = np.sum(np.absolute(P_correct - boundary))
        pb_incorrect = np.sum(boundary - np.absolute(P_incorrect - boundary))
        self.unweighted = (pb_correct + pb_incorrect) / n
        
        # Weighted Probability Boundary Score
        # Determine class values
        classes = np.unique(self.y_true)
        # Initialize dictionary for storage and code re-use
        pb_dict = {}
        # Iterate through classes
        for c in classes:
            # Separate out classes
            pb_dict[c] = {'y_pred_prob': self.y_pred_prob[np.argwhere(self.y_true == c).flatten()],
                          'y_true': self.y_true[np.where(self.y_true == c)[0]]
                         }
            # Sample size
            pb_dict[c]['n'] = len(pb_dict[c]['y_true'])
            # Make predictions
            pb_dict[c]['y_pred'] = np.argmax(pb_dict[c]['y_pred_prob'], axis=1)
            # Identify correct predictions
            pb_dict[c]['P_correct'] = pb_dict[c]['y_pred_prob'][np.argwhere(pb_dict[c]['y_pred'] == pb_dict[c]['y_true']).flatten()]
            # Identify incorrect predictions
            pb_dict[c]['P_incorrect'] = pb_dict[c]['y_pred_prob'][np.argwhere(pb_dict[c]['y_pred'] != pb_dict[c]['y_true']).flatten()]
            # Calculate Probability Boundary Score
            pb_dict[c]['pb_correct'] = np.sum(np.absolute(pb_dict[c]['P_correct'] - boundary))
            pb_dict[c]['pb_incorrect'] = np.sum(boundary - np.absolute(pb_dict[c]['P_incorrect'] - boundary))
            pb_dict[c]['pb_score'] = (pb_dict[c]['pb_correct'] + pb_dict[c]['pb_incorrect']) / pb_dict[c]['n']
        # Average scores for weighted
        self.weighted = np.mean([pb_dict[c]['pb_score'] for c in classes])

    def __repr__ (self):
        'Canonical representation'
        return 'probability_boundary({0}, {1}, {2})'.format(self.y_pred_prob, self.y_true, self.boundary)
        
    def __str__ (self):
        'String representation'
        return 'Unweighted: {0:.2f} | Weighted {1:.2f}'.format(self.unweighted, self.weighted)
            
    def weighted_score (self):
        'Return the weighted score'
        return self.weighted
    
    def unweighted_score (self):
        'Return the unweighted score'
        return self.unweighted
            
    def plot (self, width=14, height=5, subplot=111):
        '''A function that displays a plot of probability boundaries.
        
        @ Parameters:
        ------------------
            width: width of plot in inches
            height: height of plot in inches
            subplot: three-digit shorthand for subplot
        '''
        
        # Set default plot size
        plt.rcParams['figure.figsize'] = (width, height)

        # Number of samples
        n = self.y_pred_prob.shape[0]
        # x value is the index of array
        x = np.arange(n)
        # y value is the probability of the element being class 1
        y = self.y_pred_prob[:,1]
        # Red indicates 0 (Fail) and blue indicates 1 (Pass)
        colors = ['red', 'blue']

        # Plot probability boundaries
        f = plt.figure()
        sp = f.add_subplot(subplot)
        sp.set_ylim([-0.05, 1.05])
        sp.scatter(x, y, c=self.y_true, cmap=ListedColormap(colors), alpha=0.6)
        sp.axhline(y=self.boundary, color='r', linestyle='--')
        sp.set_title('Probability Boundary (Color = True Class)\nUnweighted: {0:.2f} | Weighted: {1:.2f}'.format(self.unweighted, self.weighted))
        sp.set_xlabel('Observation')
        sp.set_ylabel('Class Probability Assigned by Model')
        return f