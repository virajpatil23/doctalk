from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

def chunk_fixed(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(docs)

def chunk_semantic(docs, chunk_size=800, chunk_overlap=100):
    """
    Larger chunks, split primarily on paragraph/sentence boundaries.
    Not true embedding-based semantic chunking (that needs extra compute),
    but prioritizes natural breakpoints over rigid character counts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n\n\n", ".\n", ". "]
    )
    return splitter.split_documents(docs)

def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    """Kept for backward compatibility with existing pipeline calls."""
    return chunk_fixed(docs, chunk_size, chunk_overlap)