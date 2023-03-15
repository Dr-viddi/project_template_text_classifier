"""Functions for model evaluation."""
import keras
import matplotlib.pyplot as plt
import numpy
import seaborn as sb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from typing import Union


class Evaluator:
    """
    The evaluator class used for evaluating the model.

    Args:
        y_pred: Dataframe to be analyzed.
        y_test: Dataframe to be analyzed.
        all_labels: Dataframe to be analyzed.
        model_history: The training history object.
        pipeline_config: The pipeline config containing the storage paths.

    Attributes:
        y_pred: Dataframe to be analyzed.
        y_test: Dataframe to be analyzed.
        all_labels: Dataframe to be analyzed.
        model_history: The training history object.
        pipeline_config: The pipeline config containing the storage paths.
    """

    def __init__(self,
                 y_pred: numpy.ndarray,
                 y_test: numpy.ndarray,
                 all_labels: numpy.ndarray,
                 model_history: Union[keras.callbacks.History, None],
                 pipeline_config: dict) -> None:
        self.y_pred = y_pred
        self.y_test = y_test
        self.all_labels = all_labels
        self.model_history = model_history
        self.pipeline_config = pipeline_config
        self.labels_in_test_set = self.all_labels[numpy.unique(self.y_test)]

    def get_accuracy(self) -> None:
        """Return the accuracy score.

        Args:
            None

        Return:
            None
        """
        print('accuracy %s' % accuracy_score(y_true=self.y_test,
                                             y_pred=self.y_pred)
              )

    def get_classification_report(self) -> None:
        """Return the classification report (precision, recall, f1-score,
        support).

        Args:
            None

        Return:
            None
        """
        class_report = classification_report(y_true=self.y_test,
                                             y_pred=self.y_pred,
                                             target_names=self.labels_in_test_set
                                             )
        print(class_report)
        with open(self.pipeline_config["paths"]["classification_report"], 'w') as f:
            print(class_report, file=f)

    def plot_confusion_matrix(self) -> None:
        """Plot the confusion matrix of a model.

        Args:
            None

        Returns:
            None
        """
        conf_mtx = confusion_matrix(y_true=self.y_test, y_pred=self.y_pred)
        plt.subplots(constrained_layout=True, figsize=(14, 14))
        sb.heatmap(conf_mtx, annot=True, cmap="Blues", fmt='d',
                   xticklabels=self.labels_in_test_set, yticklabels=self.labels_in_test_set)
        plt.ylabel('Actual')
        plt.xlabel('Predicted', fontsize=8)
        plt.title("Confusion matrix", size=16)
        plt.show(block=False)
        plt.savefig(self.pipeline_config["paths"]["confusion_matrix_plot"])

    def plot_accuracy(self) -> None:
        """Plot the train and test accuracy vs epoch number.

        Args:
            None

        Returns:
            None
        """
        if not self.model_history:
            print("No training history provided. Nothing to plot!")
            return
        plt.figure(constrained_layout=True)
        plt.plot(self.model_history.history['categorical_accuracy'])
        plt.plot(self.model_history.history['val_categorical_accuracy'])
        plt.title('accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show(block=False)
        plt.savefig(self.pipeline_config["paths"]["accuracy_plot"])

    def plot_loss(self) -> None:
        """Plot the train and test loss vs epoch number.

        Args:
            None

        Returns:
            None
        """
        if not self.model_history:
            print("No training history provided. Nothing to plot!")
            return
        plt.figure(constrained_layout=True)
        plt.plot(self.model_history.history['loss'])
        plt.plot(self.model_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show(block=False)
        plt.savefig(self.pipeline_config["paths"]["loss_plot"])
