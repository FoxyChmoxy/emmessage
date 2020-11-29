FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt
RUN export FLASK_APP=app.py
CMD ["flask", "run"]