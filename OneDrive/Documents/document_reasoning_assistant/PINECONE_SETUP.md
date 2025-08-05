# Pinecone Integration Setup Guide

## Overview

This guide will help you integrate Pinecone vector database service for handling large documents efficiently. Pinecone provides a managed vector database that scales well and offers better performance for large document collections.

## Step 1: Create a Pinecone Account

1. Go to [Pinecone.io](https://www.pinecone.io/)
2. Sign up for a free account
3. The free tier includes:
   - 1 million 768-dimension vectors
   - 1 index
   - Up to 2 replicas per index

## Step 2: Get Your API Key

1. After signing up, log into the Pinecone console
2. Navigate to "API Keys" in the left sidebar
3. Create a new API key
4. Copy the API key (keep it secure!)

## Step 3: Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit your `.env` file and add your Pinecone credentials:
   ```env
   # Enable cloud retriever
   USE_CLOUD_RETRIEVER=true
   
   # Pinecone Configuration
   PINECONE_API_KEY=your_actual_api_key_here
   PINECONE_INDEX_NAME=document-reasoning
   PINECONE_NAMESPACE=default
   
   # Embedding Model (compatible with 768 dimensions)
   EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
   
   # Chunking Configuration
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   TOP_K_RESULTS=5
   ```

## Step 4: Install Dependencies

The Pinecone client has already been installed, but if you need to reinstall:

```bash
pip install pinecone-client
```

## Step 5: Test the Integration

1. Start your server:
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. Check the logs - you should see:
   ```
   Initializing Cloud Document Retriever (Pinecone)...
   Loading embedding model: BAAI/bge-small-en-v1.5
   Creating Pinecone index: document-reasoning
   Connected to Pinecone index: document-reasoning
   ```

3. Visit the health endpoint: `http://localhost:8000/health`

## Benefits of Using Pinecone

### 1. **Scalability**
- Handle millions of document chunks
- No memory limitations like local FAISS
- Automatic scaling based on usage

### 2. **Performance**
- Optimized for high-speed similarity search
- Distributed architecture
- Low-latency queries

### 3. **Persistence**
- Data persists between application restarts
- No need to re-process documents
- Multiple applications can share the same index

### 4. **Multi-document Support**
- Use namespaces to separate different documents
- Query specific documents or across all documents
- Easy document management (add, update, delete)

## Usage Examples

### Single Document Mode (Default)
```python
# Upload and process document
chunks = retriever.process_document(text, metadata)

# Query the document
results = retriever.query("What are the key terms?")
```

### Multi-document Mode
```python
# Process multiple documents with unique IDs
retriever.process_document(text1, metadata1, document_id="doc1")
retriever.process_document(text2, metadata2, document_id="doc2")

# Query specific document
results = retriever.query("Key terms?", document_id="doc1")

# Query across all documents
results = retriever.query("Key terms?")

# List all documents
documents = retriever.list_documents()

# Delete specific document
success = retriever.delete_document("doc1")
```

## Monitoring and Limits

### Free Tier Limits
- **Storage**: 1 million vectors (768 dimensions)
- **Requests**: Generous free tier allowance
- **Indexes**: 1 index maximum

### Monitoring Usage
1. Check your Pinecone console for usage statistics
2. Monitor vector count in your application logs
3. Use the `/stats` endpoint to see current usage

### Cost Optimization
- Use appropriate chunk sizes (default 1000 characters)
- Clean up old documents when not needed
- Consider upgrading to paid tier for production use

## Troubleshooting

### Common Issues

1. **"Index creation failed"**
   - Check your API key is correct
   - Ensure you haven't exceeded the index limit
   - Verify your account is active

2. **"Dimension mismatch"**
   - Ensure your embedding model matches the index dimension
   - Default BGE model uses 384 dimensions
   - Recreate index if dimension needs to change

3. **"Connection timeout"**
   - Check your internet connection
   - Verify Pinecone service status
   - Try again after a few minutes

### Fallback to Local Mode

If Pinecone is unavailable, the system will automatically fall back to local FAISS:

1. Set `USE_CLOUD_RETRIEVER=false` in `.env`
2. Or remove the `PINECONE_API_KEY` from `.env`
3. Restart the application

## Migration from Local to Cloud

To migrate existing setup to Pinecone:

1. Set up Pinecone as described above
2. Re-upload your documents (they'll be processed into Pinecone)
3. Existing queries will work the same way
4. Optionally, clean up local FAISS files

## Next Steps

1. **Production Setup**: Configure appropriate region and cloud provider
2. **Security**: Use environment-specific API keys
3. **Monitoring**: Set up alerts for usage limits
4. **Scaling**: Consider paid tiers for larger document collections

For more advanced features like filtering and metadata search, see the Pinecone documentation at https://docs.pinecone.io/
