# URL-Based Question Answering using Pinecone and OpenAI

This application allows users to input a URL and retrieve relevant textual content from that page using `Selenium` and `BeautifulSoup`. The extracted content is chunked, vectorized, and stored in a Pinecone index. Users can query the content with questions, and relevant context is retrieved from Pinecone and passed to OpenAI's GPT-3.5-turbo to generate answers. The app is built using `Streamlit` for an interactive user interface.

## Features

- **URL Text Extraction**: Extracts the textual content from a webpage using Selenium.
- **Content Chunking**: Splits large blocks of text into overlapping chunks to ensure complete context.
- **Embedding and Vector Storage**: Utilizes OpenAI's `text-embedding-ada-002` model to convert text into vectors, which are stored in Pinecone.
- **Pinecone Integration**: Connects to a Pinecone index for vector-based similarity search.
- **Question Answering**: Retrieves relevant chunks from Pinecone based on user queries and constructs a response using OpenAI's GPT-3.5-turbo.
- **Conversation Memory**: Implements a conversation chain using LangChain’s memory feature for contextual responses.

## Tech Stack

- **Streamlit**: Provides the user interface.
- **Selenium**: Automates the webpage interaction to retrieve dynamic content.
- **BeautifulSoup**: Parses the webpage content for text extraction.
- **OpenAI API**: Provides embedding for text chunks and generates responses using GPT-3.5.
- **Pinecone**: Stores and retrieves vector embeddings for efficient similarity search.
- **LangChain**: Manages conversation and memory for contextual responses.

## Prerequisites

- Python 3.8 or above

## Installation

### Step 1: Clone or download the Repository

```bash
git clone https://github.com/venkatavinayvijjapu/sampl-set-assignment.git
cd project_name
```

Or download it

### Step 2: Set Up a Virtual Environment

You can use `pip` or `conda` to create an isolated environment for this project.

#### Using `pip`

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

#### Using `conda`

```bash
conda create --name project_env python=3.8
conda activate project_env
```

### Step 3: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Frontend

In a new terminal, run the Streamlit app:

```bash
cd part-2
streamlit run app.py
```

## Project Structure

- **Notebooks**: Contains the notebooks used for testing and development in part-1 directory.
- **streamlit_app/**: Contains the Streamlit front-end code in part-2 directory.
- **requirements.txt**: Lists the project dependencies.
