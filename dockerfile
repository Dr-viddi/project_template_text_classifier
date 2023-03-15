FROM python:3.9

WORKDIR /text_classifier

RUN apt-get update

COPY pyproject.toml pyproject.toml
RUN pip3 install .

COPY ./text_classifier ./text_classifier
COPY ./runs ./runs
COPY ./configs ./configs

EXPOSE 8000

CMD ["uvicorn", "text_classifier.api.main:app", "--host", "0.0.0.0", "--port", "8000"]