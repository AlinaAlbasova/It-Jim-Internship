import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay

classes = os.listdir("dataset/")
class_numbers = range(0,16)


def eval(correct, predicted):
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(correct, predicted)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(correct, predicted, average='weighted')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(correct, predicted, average='weighted')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(correct, predicted, average='weighted')
    print('F1 score: %f' % f1)

    # confusion matrix
    matrix = confusion_matrix(correct, predicted)

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)

    disp.plot(include_values=True, xticks_rotation='vertical')
    plt.show()