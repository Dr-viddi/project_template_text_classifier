"""Api file to start prediction server."""
from fastapi import FastAPI, HTTPException
from text_classifier.machine_learning.utils import load_pickle
from text_classifier.machine_learning.constants import (
    PIPELINE_CONFIG_PATH_FOR_PREDICTION,
    IMPLEMENTED_DEEP_LEARNING_MODELS,
    IMPLEMENTED_SCIKIT_MODELS
)
from numpy import argmax
from tensorflow import keras
import uvicorn
import yaml
from keras.utils import pad_sequences


# Load config file
with open(PIPELINE_CONFIG_PATH_FOR_PREDICTION, "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# load trained model
# keras models
if cfg['model']['name'] in IMPLEMENTED_DEEP_LEARNING_MODELS:
    model = keras.models.load_model(cfg['paths']['model_path'])
# sci kit models
else:
    model = load_pickle(cfg['paths']['model_path'])

# load trained tokenizer
tokenizer = load_pickle(cfg['paths']['tokenizer_path'])

# load labels
labels = load_pickle(cfg['paths']['labels_path'])
print(labels[0])

# create server
app = FastAPI()


@app.get("/")
def read_root() -> dict:
    """Display a landing page message.

    Returns:
        The landing page message.
    """
    return {"msg": "Text classifier"}


@app.get("/info")
def get_info() -> dict:
    """Display the model name used for prediction.

    Returns:
        Name of prediction model.
    """
    return {"Model used for prediction": cfg["model"]["name"]}


@app.post("/suggest")
def predict_text(input: str) -> dict:
    """Predict a text for a given input text.

    Args:
        input: Input text string.

    Raises:
        HTTPException: Raises an error 400 if not a string is provided
            as an input.

    Returns:
        The text prediction.
    """
    if not type(input) == str:
        raise HTTPException(status_code=400, detail="String must be provided")
    
    ## add preprocessing here
    
    # stat. models use vectorizer
    if cfg['model']['name'] in IMPLEMENTED_SCIKIT_MODELS:
        test_data = tokenizer.transform([input]).toarray()
    # deep learning models use tokenzizer
    else:
        test_data = tokenizer.texts_to_sequences([input])
        test_data = pad_sequences(
            test_data,
            maxlen=cfg['preprocessings']['max_pad_sequ_length'],
            padding=cfg['preprocessings']['pad_type'],
            truncating=cfg['preprocessings']['pad_trunc_type']
            )    
    prediction = model.predict(test_data)
    if cfg['model']['name'] in IMPLEMENTED_DEEP_LEARNING_MODELS:
        prediction = argmax(prediction, axis=1)
    prediction_str = labels[prediction[0]]
    return {"prediction": prediction_str}


# start server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
