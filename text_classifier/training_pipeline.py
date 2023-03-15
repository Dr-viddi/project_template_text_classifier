import argparse
import keras
import numpy
import pandas as pd
from text_classifier.machine_learning.preprocessing import Preprocessor
from text_classifier.machine_learning.training import (
    Trainer,
    Deep_learning_trainer
)
from text_classifier.machine_learning.evaluation import Evaluator
from text_classifier.machine_learning.data_analysis import Data_analyizer
from text_classifier.machine_learning.utils import (
    save_as_pickle,
    load_config
)
from text_classifier.machine_learning.constants import (
    PIPELINE_CONFIG_PATH_FOR_PREDICTION,
    IMPLEMENTED_DEEP_LEARNING_MODELS,
    IMPLEMENTED_SCIKIT_MODELS
)
from keras.utils import pad_sequences, to_categorical
from numpy import argmax
import shutil
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from typing import Union


def main():
    """Main function for entry point to execute training pipeline.
    """
    parser = argparse.ArgumentParser(
        prog="Text classifier",
        description="Classifies texts with respect to text classes."
        )

    parser.add_argument(
        "--config",
        "-c",
        help="Pipeline config file",
        action="store",
        type=str,
        default="configs/pipeline_config.yml",
    )
    args = parser.parse_args()
    print("\n")
    print(f"Selected pipeline config file: {args.config}")

    start_training_pipeline(pipeline_config_path=args.config)

    # copy pipeline_config to PIPELINE_CONFIG_PATH_FOR_PREDICTION
    # Required for the prediction in the docker container
    if not args.config == PIPELINE_CONFIG_PATH_FOR_PREDICTION:
        print("\n")
        print(f"Copy config to: {PIPELINE_CONFIG_PATH_FOR_PREDICTION}")
        shutil.copy(args.config, PIPELINE_CONFIG_PATH_FOR_PREDICTION)


def start_training_pipeline(pipeline_config_path: str) -> None:
    """Start the training pipeline. This function is called from the main()
    function and executes all parts from the training pipeline, i.e., from
    cleaning data to evaluating the model performance.

    Args:
        pipeline_config_path: Path to pipeline_config file.

    Returns:
        None.
    """
    # Load config
    print("Load training configuration file")
    pipeline_config = load_config(pipeline_config_path)
    print(f"...Model selected: {pipeline_config['model']['name']}")

    # Load data
    print("\n")
    print("Load data")
    trainings_data = pd.read_json(
        path_or_buf=pipeline_config['paths']['data_path'],
        lines=True
        )

    # Clean data / reduce imbalance
    print("\n")
    print("Clean data")
    trainings_data = filter_data(trainings_data, pipeline_config)

    # Data analysis
    print("\n")
    print("Analyize data")
    label_count, text_count = analyize_data(trainings_data, pipeline_config)

    # Preprocessing data
    print("\n")
    print("Preprocessing data")
    features, encoded_labels, vocab_size = preprocess_data(
        trainings_data,
        pipeline_config
        )

    # Create datasets
    print("\n")
    print("Create train and test datasets")
    X_train, X_test, y_train, y_test = create_training_sets(
        features,
        encoded_labels,
        label_count,
        pipeline_config
        )

    # Train model
    print("\n")
    print("Train model")
    model, history = train_model(
        X_train,
        y_train,
        label_count,
        vocab_size,
        pipeline_config
        )

    # Prediction Test set
    print("\n")
    print("Prediction for the test set")
    y_pred = model.predict(X_test)

    # Evaluate model
    print("\n")
    print("Evaluate model")
    all_labels = trainings_data['journal'].unique()
    save_as_pickle(all_labels, pipeline_config['paths']['labels_path'])
    evaluate_model(y_pred, y_test, all_labels, history, pipeline_config)


def filter_data(trainings_data: pd.DataFrame,
                pipeline_config: dict
                ) -> pd.DataFrame:
    """Delete all rows that are not specified in the pipeline_config.

    Args:
        trainings_data: The unprocessed raw data.
        pipeline_config: Pipeline config file.

    Returns:
        trainings_data: The filtered data.
    """
    texts_to_keep = pipeline_config['preprocessings']['text_to_keep']
    if texts_to_keep:
        print(f"...Only the following labels will be considered: {text_to_keep}")
        # build query string
        query_string = ""
        for index, text in enumerate(texts_to_keep):
            if index < len(texts_to_keep)-1:
                str = "text == " + f"'{text}'" + " | "
            else:
                str = "text == " + f"'{text}'"
            query_string = query_string + str
        # filter data
        trainings_data = trainings_data.query(query_string)
    return trainings_data


def analyize_data(trainings_data: pd.DataFrame,
                  pipeline_config: dict
                  ) -> list[int, int]:
    """Apply basic exploratory Data Analysis.

    Args:
        trainings_data: The data to be analyized.
        pipeline_config: Pipeline config file that specifies configuration
            of pipeline.

    Returns:
        A list containing the label count and the text count.
    """
    data_analyizer = Data_analyizer(trainings_data)
    label_count = data_analyizer.get_label_count()
    text_count = data_analyizer.get_text_count()
    data_analyizer.plot_class_count(path=pipeline_config["paths"]["class_count_plot"])
    print(f"...Number of labels in dataset: {label_count}")
    print(f"...Number of textfiles in dataset: {text_count}")
    return label_count, text_count


def preprocess_data(trainings_data: pd.DataFrame,
                    pipeline_config: dict
                    ) -> list[numpy.ndarray, numpy.ndarray, int]:
    """Apply different preprocessing on the data. Possible preprocessings are:
    - remove non_alphabetic_symbols
    - remove single_characters
    - lowercase text
    - remove multiple spaces
    - remove stopwords
    - stem words
    - tokenize
    - padding
    - vectorize
    - encoding (labels)
    Which preprocessings will be applied is specified in the pipeline_config.

    Args:
        trainings_data: Data to be processed.
        pipeline_config: Pipeline configuration file.

    Returns:
        A list containing preprocessed features, encoded labels and the vocab
        size of the dictionary.
    """
    preprocessor = Preprocessor()
    if pipeline_config['preprocessings']['remove_non_alphabetic_symbols']:
        print("...remove non alphabetic symbols")
        trainings_data['text'] = trainings_data['text'].apply(
            preprocessor.remove_non_alphabetic_symbols
            )
    if pipeline_config['preprocessings']['remove_single_characters']:
        print("...remove single characters")
        trainings_data['text'] = trainings_data['text'].apply(
            preprocessor.remove_single_characters
            )
    if pipeline_config['preprocessings']['lowercase_text']:
        print("...lowercase text")
        trainings_data['text'] = trainings_data['text'].apply(
            preprocessor.lowercase_text
            )
    if pipeline_config['preprocessings']['remove_multiple_spaces']:
        print("...remove multiple spaces")
        trainings_data['text'] = trainings_data['text'].apply(
            preprocessor.remove_multiple_spaces
            )
    if pipeline_config['preprocessings']['remove_stopwords']:
        print("...remove stopwords")
        trainings_data['text'] = trainings_data['text'].apply(
            preprocessor.remove_stopwords
            )
    if pipeline_config['preprocessings']['stem_words']:
        print("...stem words")
        trainings_data['text'] = trainings_data['text'].apply(
            preprocessor.stem_words
            )

    print("...tokenize, pad and vectorize")
    # used for scikit models
    if pipeline_config['model']['name'] in IMPLEMENTED_SCIKIT_MODELS:
        features, fitted_vectorizer = preprocessor.Vectorizer(
            trainings_data['text']
            )
        save_as_pickle(fitted_vectorizer,
                       pipeline_config['paths']['tokenizer_path']
                       )
        vocab_size = None
    # used for deep learning keras models
    elif pipeline_config['model']['name'] in IMPLEMENTED_DEEP_LEARNING_MODELS:
        # tokenize
        train_sequences, fitted_tokenizer, vocab_size = preprocessor.tokenize(
            trainings_data['text'],
            pipeline_config['preprocessings']['max_vocab_size']
            )
        save_as_pickle(fitted_tokenizer,
                       pipeline_config['paths']['tokenizer_path']
                       )
        # padding
        features = pad_sequences(
            train_sequences,
            maxlen=pipeline_config['preprocessings']['max_pad_sequ_length'],
            padding=pipeline_config['preprocessings']['pad_type'],
            truncating=pipeline_config['preprocessings']['pad_trunc_type']
            )
        # Note: Embedding happens in the embedding layer in the model
    else:
        print("Unknown model in config")
        print("Stop execution!")
        return
    # encode labels
    encoded_labels = preprocessor.encode_labels(trainings_data['journal'])
    return features, encoded_labels, vocab_size


def create_training_sets(features: numpy.ndarray,
                         encoded_labels: numpy.ndarray,
                         label_count: int,
                         pipeline_config: dict
                         ) -> list[numpy.ndarray,
                                   numpy.ndarray,
                                   numpy.ndarray,
                                   numpy.ndarray
                                   ]:
    """Create the training and testing sets.

    Args:
        features: Preprocessed input features.
        encoded_labels: Encoded labels.
        label_count: The number of available labels.
        pipeline_config: Pipeline config file.

    Returns:
        List contraining the training set (X_train), the testing set(X_test),
        the training labels (y_train), and the testing labels (y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        encoded_labels,
        test_size=pipeline_config['dataset']['test_train_split'],
        random_state=pipeline_config['dataset']['random_state'],
        shuffle=pipeline_config['dataset']['shuffle'],
        stratify=pipeline_config['dataset']['stratify']
        )

    # Deep learning models require transformation to categorical encoded labels
    if pipeline_config['model']['name'] in IMPLEMENTED_DEEP_LEARNING_MODELS:
        y_train = to_categorical(y_train, label_count)
        y_test = to_categorical(y_test, label_count)
    print(f"...X_train size: {X_train.shape}")
    print(f"...y_train size: {y_train.shape}")
    print(f"...X_test size: {X_test.shape}")
    print(f"...y_test size: {y_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train: numpy.ndarray,
                y_train: numpy.ndarray,
                label_count: int,
                vocab_size: int,
                pipeline_config: dict
                ) -> list[Union[keras.Sequential,
                                LinearSVC,
                                LogisticRegression,
                                RandomForestClassifier,
                                MultinomialNB],
                          Union[keras.callbacks.History, None]
                          ]:
    """Create and train the model.

    Args:
        X_train: Training features.
        y_train: Training labels.
        label_count: The number of overall labels.
        vocab_size: The size of the tokenizer vocabulary.
        pipeline_config: Pipeline config file.

    Returns:
        The trained model object and (in case of an deep learning model)
        the training history.
    """
    model_config = load_config(pipeline_config['model']['config'])
    # sci kit models
    if pipeline_config['model']['name'] in IMPLEMENTED_SCIKIT_MODELS:
        trainer = Trainer(pipeline_config['model']['name'],
                          X_train,
                          y_train,
                          model_config
                          )
        model = trainer.fit()
        history = None  # sci kit models do not have training histories
        save_as_pickle(model, pipeline_config['paths']['model_path'])
    # deep learning keras model
    else:
        trainer = Deep_learning_trainer(pipeline_config['model']['name'],
                                        X_train,
                                        y_train,
                                        label_count,
                                        model_config,
                                        vocab_size
                                        )
        model, history = trainer.fit()
        model.save(pipeline_config['paths']['model_path'])
    return model, history


def evaluate_model(y_pred: numpy.ndarray,
                   y_test: numpy.ndarray,
                   all_labels: numpy.ndarray,
                   history: Union[keras.callbacks.History, None],
                   pipeline_config: dict
                   ) -> None:
    """Evaluate the model performance.

    Args:
        y_pred: The predicted labels of the test set.
        y_test: The ground truth labels of the test set.
        all_labels: All available labels.
        history: The training history object. Only the keras deep learning
            models do have a history object.
        pipeline_config: Pipeline config file.

    Returns:
        None
    """
    # Deep learning model output is a vector and has to be transformed back to
    # integer coded labels (inverse to_categorical)
    if pipeline_config['model']['name'] in IMPLEMENTED_DEEP_LEARNING_MODELS:
        y_pred = argmax(y_pred, axis=1)
        y_test = argmax(y_test, axis=1)
    evaluator = Evaluator(y_pred, y_test, all_labels, history, pipeline_config)
    evaluator.get_accuracy()
    evaluator.get_classification_report()
    evaluator.plot_accuracy()
    evaluator.plot_loss()
    evaluator.plot_confusion_matrix()


if __name__ == "__main__":
    main()
