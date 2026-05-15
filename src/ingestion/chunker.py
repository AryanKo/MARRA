from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_file(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Loads a text file and splits it into chunks using a recursive character text splitter.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)
