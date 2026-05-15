import ollama

def embed_texts(texts: list[str], model: str = "nomic-embed-text") -> list[list[float]]:
    """
    Given a list of texts, queries the local Ollama instance to generate dense vector embeddings.
    """
    embeddings = []
    for text in texts:
        response = ollama.embeddings(model=model, prompt=text)
        embeddings.append(response["embedding"])
    return embeddings
