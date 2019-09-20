# Simple Book Reccomendation Model

This repository is a Proof of Concept (P.O.C) of the simplest form to demonstrate the capabilities of Machine Learning in the public sector. All work here is done on behalf of the Kommune of Aarhus, in Aarhus Denmark.

### Data:
The data used in this proof of concept was found in a public repository, and can be access [here](https://github.com/williamscott701/Information-Retrieval/tree/master/2.%20TF-IDF%20Ranking%20-%20Cosine%20Similarity%2C%20Matching%20Score). 

### Model:
The model is a simple KNN model, converting the texts in this dataset and subsequent queries to vectors using Bag of Words and then applying a TF-IDF transformation on these vectors. Using the TF-IDF vectors, the model uses similarity metric to determine the highest match. 

### APIs: 

#### API access:
Hosted at [sjhan12.pythonanywhere.com/getReccomendation](https://sjhan12.pythonanywhere.com/getReccomendation).

Example POST body for /getReccomendation:
```
{
    "numResponses": 6,
    "query": "love"
}
```

Returns: 
```
`{
    "query": [
        "Poetry: A Song of Love (October 25, 1993)",
        "Under the Weeping Willow Tree, by Patrick Miner",
        "Nightmare in Grey by Fredric Brown",
        "Poem: Through My Eyes",
        "The Chronicles of Astrus II: Father Adler by Ben Blumenberg",
        "Contradiction 1, by Rick Brunet"
    ]
}`
```

#### API setup:

1. Install dependencies:
```pip install -r requirements.txt```

2. Run server on local machine:
```python server.py``` (runs on localhost:3000)


### Python setup:

On a Mac, ensure that xcode is installed.

Install Python 3 from [here](https://www.anaconda.com/distribution/).

All questions can be directed to `sharan@uw.edu`.
