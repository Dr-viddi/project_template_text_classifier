"""Collection of various data analysis utility functions."""
import matplotlib.pyplot as plt
import numpy
import pandas


class Data_analyizer:
    """
    The Data_analyzer object used for exploratory data analysis.

    Args:
        dataframe: Dataframe to be analyzed.

    Attributes:
        training_dataframe: Dataframe to be analyzed.
    """

    def __init__(self, dataframe: pandas.DataFrame) -> None:
        self.training_dataframe = dataframe

    def plot_class_count(self, path: str) -> plt.figure:
        """Plot the Class counts.

        Args:
            path: Storage path to disk.

        Returns:
            Matplotlib figure object.
        """
        fig = plt.figure(constrained_layout=True, figsize=(14, 14))
        self.training_dataframe.groupby('journal').text.count().sort_values().plot.bar(
            ylim=0, title='Number of texts in each journal')
        plt.ylabel('label count', fontsize=10)
        plt.xlabel('label count', fontsize=8)
        plt.show(block=False)
        plt.savefig(path)
        return fig

    def get_labels(self) -> numpy.ndarray:
        """Return all (unique) labels of the dataset.

        Returns:
            Array of unique labels.
        """
        return self.training_dataframe['journal'].unique()

    def get_text(self, id: int) -> str:
        """Return a a text example.

        Args:
            id: Index of the dataframe row.

        Returns:
            Text example.
        """
        return self.training_dataframe["text"].iloc[id]

    def get_label_count(self) -> int:
        """Return the number of labels in the dataset.

        Args:
            None

        Returns:
            The label count.
        """
        return len(self.training_dataframe['journal'].unique())

    def get_text_count(self) -> int:
        """Return the number of text files in the dataset.

        Args:
            None

        Returns:
            The textfile count.
        """
        return len(self.training_dataframe)
