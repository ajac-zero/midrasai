# MidrasAI

MidrasAI provides a simple API for using the Colpali model, which is a multi-modal model for text and image retrieval.
It allows for local access to the model, and integrates a vector database for efficient storage and sematic search.

Setting up the model as a server for remote access is a WIP.

## Getting started

Note: This is an alpha version of MidrasAI. All feedack and suggestions are welcome!

### Local Dependencies

- ColPali access: ColPali is based on PaliGemma, you will need to request access to the model [here](https://huggingface.co/google/paligemma-3b-mix-448). Then you must authenticate through the huggingface-cli to download the model.
- Poppler: Midras uses `pdf2image` to convert pdfs to images. This library requires `poppler` to be installed on your system. Check out the installation instructions [here](https://poppler.freedesktop.org/).
- Hardware: ColPali is a 3B parmeter model, so I recommend using a GPU with at least 8GB of VRAM.

### Installation

If running locally, you can install MidrasAI and its dependencies with pip, poetry, or uv:
```bash
# pip
pip install 'midrasai[local]'

# poetry
poetry install 'midrasai[local]'

# uv
uv install 'midrasai[local]'
```

### Usage

#### Starting the ColPali model

To load the ColPali model locally, you just need to use the `LocalMidras` class:

```python3
from midrasai.local import LocalMidras

midras = LocalMidras() # Make sure you're logged in to HuggingFace so you can download the model
```

#### Creating an index

To create an index, you can use the `create_index` method with the name of the index you want to create:
```python3
midras.create_index("my_index")
```

#### Using the model to embed data

The Midras class provides a couple of convenience methods for embeding data.
You can use the `embed_pdf` method to embed a single pdf, or the `embed_pil_images` method to embed a list of images. Here's how to use them:

```python3
# Embed a single pdf
path_to_pdf = "path/to/pdf.pdf"

pdf_response = midras.embed_pdf(path_to_pdf, include_images=True)
```

```python3
# Embed a list of images
images = [Image.open("path/to/image.png"), Image.open("path/to/another_image.png")]

image_response = midras.embed_pil_images(images)
```

#### Inserting data into an index

Once you have your data embeddings, you can insert a data point into your index with the `add_point` method:

```python3
midras.add_point(
    index="my_index", # name of the index you want to add to
    id=1, # id of this data point, can be any integer or string
    embedding=response.embeddings[0], # the embedding you created in the previous step
    data={ # any additional data you want to store with this point, can be any dictionary
        "something": "hi"
        "something_else": 123
    }
)
```

### Searching an index

After you've added data to your index, you can start searching for relevant data. You can use the `query` method to do this:

```python3
query = "What is the meaing of life?"

results = midras.query(index_name, query=query)

# Top 3 relevant data points
for result in results[:3]:
    # Each result will have a score, which is a measure of how relevant the data is to the query
    print(f"score: {result.score}")
    # Each result will also have any additional data you stored with it
    print(f"data: {result.data}")
```

If you want a more detailed example including RAG, check out the [example vector search notebook](https://github.com/Midras-AI-Systems/midrasai/blob/main/examples/vector_search/vector_search.ipynb).
