### `main.py`
- **Purpose**: This is the entry point for the FastAPI application. It manages the initialization and lifecycle of the application components.
- **Key Features**:
  - Manages startup and shutdown processes.
  - Provides routes for uploading documents, querying, and session management.
  - Initializes the document reasoning components and handles HTTP requests.

### `db.py`
- **Purpose**: Handles database interactions using MongoDB for logging and retrieving session and query data.
- **Key Features**:
  - Manages connections and operations in MongoDB.
  - Logs document uploads and queries.
  - Provides session management and analytics data retrieval.

### `file_utils.py`
- **Purpose**: Extracts text from various document formats such as PDF, DOCX, and TXT.
- **Key Features**:
  - Uses libraries like PyMuPDF and python-docx for text extraction.
  - Validates document formats and extracts metadata.

### `llm_utils.py`
- **Purpose**: Manages interactions with Language Model (LLM) providers to analyze document queries.
- **Key Features**:
  - Supports multiple LLM providers like Together AI, Groq, and Fireworks.
  - Creates structured prompts and processes LLM responses.

### `vector_retriever.py`
- **Purpose**: Handles document chunking and semantic search using Pinecone for cloud-based vector storage.
- **Key Features**:
  - Uses sentence transformers for embedding document chunks.
  - Supports querying and retrieval of relevant document chunks based on embeddings.

### `retriever.py.backup`
- **Purpose**: A backup file for the retriever that uses FAISS for local vector storage.
- **Key Features**:
  - Similar to `vector_retriever.py` but employs FAISS for indexing and searching document embeddings locally.
