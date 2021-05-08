#run command as example
#python classification_metrics.py --n_classes 10 --n_examples 1000 --class_label 0 --seed 42

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Calculating Classification Metrics')
parser.add_argument('--n_classes', type=int, help='Number of classes')
parser.add_argument('--n_examples', type=int, help='Number of examples')
parser.add_argument('--class_label', type=int, help='Class Label')
parser.add_argument('--seed', type=int, help='SEED')

args = parser.parse_args()

class classification_metrics():
    def __init__(self, n_classes, n_examples, class_label, seed):
        self.n_classes = n_classes
        self.n_examples = n_examples
        self.class_label = class_label
        self.seed=seed
        
    def make_data(self):
        
        np.random.seed(self.seed)
        
        classes = np.arange(self.n_classes)
        actual_labels = np.random.choice(classes, self.n_examples)
        predicted_labels = np.random.choice(classes, self.n_examples)
        
        return actual_labels, predicted_labels
    
    def calculate_accuracy(self):
        actual_labels, predicted_labels = self.make_data()
        
        return sum(actual_labels==predicted_labels) / len(actual_labels)
    
    def find_outcome(self,x,y,class_label):
        
        if (x==y):
            if (x==class_label):
                return 'TP'
            else:
                return 'TN'
        else:
            if (x==class_label):
                return 'FN'
            else:
                return 'FP'
        
    def find_confusion_matrix(self,class_label):
        actual_labels, predicted_labels = self.make_data()
        outcomes = np.array(list(map(lambda x,y: self.find_outcome(x,y,class_label), actual_labels, predicted_labels)))
        
        tp = sum(outcomes=='TP')
        tn = sum(outcomes=='TN')
        fp = sum(outcomes=='FP')
        fn = sum(outcomes=='FN')
        
        outcome_dict = {'TP':tp, 'TN':tn, 'FP':fp, 'FN':fn}
        return outcome_dict
    
    def calculate_precision(self, class_label):
        outcome_dict = self.find_confusion_matrix(class_label)
        
        return outcome_dict['TP'] / (outcome_dict['TP'] + outcome_dict['FP'])
    
    def calculate_recall(self, class_label):
        outcome_dict = self.find_confusion_matrix(class_label)
        
        return outcome_dict['TP'] / (outcome_dict['TP'] + outcome_dict['FN'])
    
    def calculate_f1_score(self, class_label):
        prec = self.calculate_precision(class_label)
        rec = self.calculate_recall(class_label)
        
        return 2 * prec * rec / (prec + rec)   
    
    def calculate_balanced_accuracy(self):
        
        classes = np.arange(self.n_classes)
        return np.mean(list(map(lambda x: self.calculate_recall(x), classes)))
    
    
if __name__=='__main__':
    metrics = classification_metrics(args.n_classes, args.n_examples, args.class_label, args.seed)
    
    prec = metrics.calculate_precision(args.class_label).round(4) * 100
    rec = metrics.calculate_recall(args.class_label).round(4) * 100
    f1_score = metrics.calculate_f1_score(args.class_label).round(4) * 100
    acc = metrics.calculate_accuracy().round(4) * 100
    balanced_acc = metrics.calculate_balanced_accuracy().round(4) * 100
    
    print('Precision:', prec, '%')
    print('Recall:', rec, '%')
    print('F1 Score:', f1_score, '%')
    print('Accuracy:', acc, '%')
    print('Balanced Accuracy:', balanced_acc, '%')