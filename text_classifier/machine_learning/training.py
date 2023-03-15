"""This module contains machine learning models for classifying texts."""
import numpy
from keras.layers import Embedding, LSTM, Conv1D, Flatten
from keras.layers.core import Dropout, Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from tensorflow import keras
from typing import Union


class Trainer:
    """
    The trainer object used for training.

    Args:
        model: The model that is trained.
        X_train: Feature dataset.
        y_train: Label dataset.
        config: Configuration file for training.

    Attributes:
        model: Model object.
        X_train: Feature dataset.
        y_train: Label dataset.
        config: Configuration file for training.
    """
    def __init__(self,
                 model: str,
                 X_train: numpy.ndarray,
                 y_train: numpy.ndarray,
                 config: dict
                 ) -> None:
        self.config = config
        if model == "svm":
            self.model = LinearSVC(
                penalty=self.config["model_params"]["penalty"],
                loss=self.config["model_params"]["loss"],
                dual=self.config["model_params"]["dual"],
                tol=self.config["model_params"]["tol"],
                C=self.config["model_params"]["C"],
                multi_class=self.config["model_params"]["multi_class"],
                fit_intercept=self.config["model_params"]["fit_intercept"],
                intercept_scaling=self.config["model_params"]["intercept_scaling"],
                class_weight=self.config["model_params"]["class_weight"],
                verbose=self.config["model_params"]["verbose"],
                random_state=self.config["model_params"]["random_state"],
                max_iter=self.config["model_params"]["max_iter"]
                )
        elif model == "naive bayes":
            self.model = MultinomialNB(
                alpha=self.config["model_params"]["alpha"],
                fit_prior=self.config["model_params"]["fit_prior"],
                class_prior=self.config["model_params"]["class_prior"]
                )
        elif model == "logistic regression":
            self.model = LogisticRegression(
                penalty=self.config["model_params"]["penalty"],
                dual=self.config["model_params"]["dual"],
                tol=self.config["model_params"]["tol"],
                C=self.config["model_params"]["C"],
                fit_intercept=self.config["model_params"]["fit_intercept"],
                intercept_scaling=self.config["model_params"]["intercept_scaling"],
                class_weight=self.config["model_params"]["class_weight"],
                random_state=self.config["model_params"]["random_state"],
                solver=self.config["model_params"]["solver"],
                max_iter=self.config["model_params"]["max_iter"],
                multi_class=self.config["model_params"]["multi_class"],
                verbose=self.config["model_params"]["verbose"],
                warm_start=self.config["model_params"]["warm_start"],
                n_jobs=self.config["model_params"]["n_jobs"],
                l1_ratio=self.config["model_params"]["l1_ratio"]
                )
        elif model == "random forest":
            self.model = RandomForestClassifier(
                n_estimators=self.config["model_params"]["n_estimators"],
                criterion=self.config["model_params"]["criterion"],
                max_depth=self.config["model_params"]["max_depth"],
                min_samples_split=self.config["model_params"]["min_samples_split"],
                min_samples_leaf=self.config["model_params"]["min_samples_leaf"],
                min_weight_fraction_leaf=self.config["model_params"]["min_weight_fraction_leaf"],
                max_features=self.config["model_params"]["max_features"],
                max_leaf_nodes=self.config["model_params"]["max_leaf_nodes"],
                min_impurity_decrease=self.config["model_params"]["min_impurity_decrease"],
                bootstrap=self.config["model_params"]["bootstrap"],
                oob_score=self.config["model_params"]["oob_score"],
                n_jobs=self.config["model_params"]["n_jobs"],
                random_state=self.config["model_params"]["random_state"],
                verbose=self.config["model_params"]["verbose"],
                warm_start=self.config["model_params"]["warm_start"],
                class_weight=self.config["model_params"]["class_weight"],
                ccp_alpha=self.config["model_params"]["ccp_alpha"],
                max_samples=self.config["model_params"]["max_samples"]
                )
        else:
            print("unknown model!")
        self.X_train = X_train
        self.y_train = y_train

    def fit(self) -> Union[LinearSVC, LogisticRegression,
                           RandomForestClassifier, MultinomialNB]:
        """Fit the model to the data.

        Args:
            None

        Returns:
            Fitted model object.
        """
        self.model.fit(self.X_train, self.y_train)
        return self.model


class Deep_learning_trainer:
    """
    The deep learning trainer object used for training deep learning models.

    Args:
        model: The deep learning model that is trained.
        X_train: Feature dataset.
        y_train: Label dataset.
        config: Configuration file for training.
        number_of_classes: Class count used for training.
        config: Model config file.
        vocab_size: Size of vocab used for embedding

    Attributes:
        model: Deep learning model object.
        X_train: Feature dataset.
        y_train: Label dataset.
        config: Configuration file for training.
        number_of_classes: Class count used for training.
        embedding_dimension: Emebdding dimension.
        epochs: Training epochs count.
        batch_size: Batch size for training.
        validation_split: Validation split for training.
        vocab_size: Vocabulary size after tokenizing.
        max_sequence_length: Max sequence length of input feature
    """
    def __init__(self,
                 model: str,
                 X_train: numpy.ndarray,
                 y_train: numpy.ndarray,
                 number_of_classes: int,
                 config: dict,
                 vocab_size: str
                 ) -> None:
        self.config = config
        self.X_train = X_train
        self.y_train = y_train
        self.number_of_classes = number_of_classes
        self.vocab_size = vocab_size
        if model == "lstm":
            self.model = self.create_lstm_model()
        elif model == "cnn":
            self.model = self.create_cnn_model()
        else:
            print("unknown model!")

    def create_lstm_model(self) -> keras.Sequential:
        """Create a keras lstm model.

        Args:
            None

        Returns:
            Untrained LSTM model.
        """
        model = keras.Sequential(
            [
                Embedding(
                    self.vocab_size,
                    output_dim=self.config["model_params"]["embedding_dim"],
                    input_length=self.config["model_params"]["max_sequ_len"]
                    ),
                LSTM(128, return_sequences=True),
                Dropout(0.5),
                LSTM(64),
                Dropout(0.5),
                Dense(self.number_of_classes, activation='sigmoid')
            ]
        )
        print(model.summary())
        return model

    def create_cnn_model(self) -> keras.Sequential:
        """Create a keras CNN model.

        Args:
            None

        Returns:
            Untrained CNN model.
        """
        model = keras.Sequential(
            [
                Embedding(
                    self.vocab_size,
                    output_dim=self.config["model_params"]["embedding_dim"],
                    input_length=self.config["model_params"]["max_sequ_len"]
                    ),
                Conv1D(64, 3, activation='sigmoid'),
                Conv1D(100, 3, activation='sigmoid'),
                Conv1D(100, 3, activation='sigmoid'),
                Dropout(0.70),
                Conv1D(48, 3, activation='sigmoid'),
                Flatten(),
                Dense(self.number_of_classes, activation='sigmoid')
            ]
        )
        print(model.summary())
        return model

    def fit(self) -> Union[keras.Sequential, str]:
        """Fit the model to the data.

        Args:
            None

        Returns:
            The trained model and its training history.
        """
        self.model.compile(
            optimizer=self.config["compile_params"]["optimizer"],
            loss=self.config["compile_params"]["loss"],
            metrics=[self.config["compile_params"]["metrics"]],
            loss_weights=self.config["compile_params"]["loss_weights"],
            weighted_metrics=self.config["compile_params"]["weighted_metrics"],
            run_eagerly=self.config["compile_params"]["run_eagerly"],
            steps_per_execution=self.config["compile_params"]["steps_per_execution"],
            jit_compile=self.config["compile_params"]["jit_compile"]
            )
        history = self.model.fit(
            x=self.X_train,
            y=self.y_train,
            batch_size=self.config["fit_params"]["batch_size"],
            epochs=self.config["fit_params"]["epochs"],
            verbose=self.config["fit_params"]["verbose"],
            callbacks=self.config["fit_params"]["callbacks"],
            validation_split=self.config["fit_params"]["validation_split"],
            validation_data=self.config["fit_params"]["validation_data"],
            shuffle=self.config["fit_params"]["shuffle"],
            class_weight=self.config["fit_params"]["class_weight"],
            sample_weight=self.config["fit_params"]["sample_weight"],
            initial_epoch=self.config["fit_params"]["initial_epoch"],
            steps_per_epoch=self.config["fit_params"]["steps_per_epoch"],
            validation_steps=self.config["fit_params"]["validation_steps"],
            validation_batch_size=self.config["fit_params"]["validation_batch_size"],
            validation_freq=self.config["fit_params"]["validation_freq"],
            max_queue_size=self.config["fit_params"]["max_queue_size"],
            workers=self.config["fit_params"]["workers"],
            use_multiprocessing=self.config["fit_params"]["use_multiprocessing"]
            )
        return self.model, history
