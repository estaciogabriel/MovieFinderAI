"""
Pipeline to download embeddings from Chroma Cloud and save locally
"""

from pathlib import Path
from src.infrastructure.chroma_repository import ChromaRepository
from src.infrastructure.embedder import DocumentEmbedder
import numpy as np
import pickle
from tqdm import tqdm

def download_from_cloud():
    """Download all embeddings and documents from Chroma Cloud"""
    
    print("Connecting to Chroma Cloud...")
    repo = ChromaRepository()
    collection = repo.get_collection()
    
    total = collection.count()
    print(f"Total documents in cloud: {total:,}")
    
    print("Downloading all data...")
    results = collection.get(
        include=['embeddings', 'documents', 'metadatas']
    )
    
    embeddings = np.array(results['embeddings'])
    documents_text = results['documents']
    metadatas = results['metadatas']
    
    # Convert to format expected by clustering/anomaly detection
    documents = []
    for doc, meta in zip(documents_text, metadatas):
        documents.append({
            'filename': meta.get('filename', 'unknown.txt'),
            'text': doc,
            'filepath': ''
        })
    
    print(f"Downloaded {len(documents)} documents")
    print(f"Embeddings shape: {embeddings.shape}")
    
    return embeddings, documents

def save_locally(embeddings, documents):
    """Save embeddings and documents locally"""
    
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / 'data/processed/embeddings'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_dir}...")
    
    np.save(output_dir / 'embeddings.npy', embeddings)
    print("✓ Saved embeddings.npy")
    
    with open(output_dir / 'documents.pkl', 'wb') as f:
        pickle.dump(documents, f)
    print("✓ Saved documents.pkl")
    
    filenames = [doc['filename'] for doc in documents]
    with open(output_dir / 'filenames.txt', 'w') as f:
        f.write('\n'.join(filenames))
    print("✓ Saved filenames.txt")
    
    print(f"\n✓ Pipeline complete! Saved {len(documents)} documents")

def main():
    print("=" * 60)
    print("PIPELINE: Download from Chroma Cloud → Save Locally")


    embeddings, documents = download_from_cloud()
    save_locally(embeddings, documents)
    print("Now you can use clustering and anomaly detection!")

if __name__ == "__main__":
    main()
