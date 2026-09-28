"""
Avaliação do sistema de busca semântica
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


from src.infrastructure.embedder import DocumentEmbedder
from src.infrastructure.vector_db import VectorDatabase

def evaluate_retrieval(db, embedder, test_docs, test_files, k=5):
    """
    Avalia a qualidade da busca semântica
    
    Args:
        db: VectorDatabase instance
        embedder: DocumentEmbedder instance
        test_docs: Lista de documentos de teste
        test_files: Lista de filenames de teste
        k: Número de resultados a retornar
        
    Returns:
        dict com métricas de avaliação
    """
    results = []
    
    print(f"\nAvaliando busca semântica com {len(test_docs)} documentos de teste...")
    
    for doc, filename in tqdm(zip(test_docs, test_files), total=len(test_docs)):
        search_results = db.search(doc, n_results=k)
        
        if search_results['documents'] and len(search_results['documents']) > 0:
            retrieved_docs = search_results['documents'][0]
            distances = search_results['distances'][0]
            metadatas = search_results['metadatas'][0]
            
            similarities = [1 - d for d in distances]
            
            retrieved_files = [m['filename'] for m in metadatas]
            
            try:
                rank = retrieved_files.index(filename) + 1
                found = True
            except ValueError:
                rank = None
                found = False
            
            results.append({
                'query_file': filename,
                'top1_file': retrieved_files[0] if retrieved_files else None,
                'top1_similarity': similarities[0] if similarities else 0,
                'found_in_top_k': found,
                'rank': rank,
                'avg_similarity': np.mean(similarities) if similarities else 0
            })
        else:
            results.append({
                'query_file': filename,
                'top1_file': None,
                'top1_similarity': 0,
                'found_in_top_k': False,
                'rank': None,
                'avg_similarity': 0
            })
    
    df_results = pd.DataFrame(results)
    
    metrics = {
        'total_queries': len(test_docs),
        'found_in_top_k': df_results['found_in_top_k'].sum(),
        'precision_at_k': df_results['found_in_top_k'].mean(),
        'avg_top1_similarity': df_results['top1_similarity'].mean(),
        'avg_all_similarity': df_results['avg_similarity'].mean(),
        'perfect_matches': (df_results['top1_similarity'] >= 0.99).sum(),
        'high_quality': (df_results['top1_similarity'] >= 0.80).sum(),
    }
    
    return metrics, df_results

def main():
    print("=" * 80)
    print("AVALIAÇÃO DO SISTEMA DE BUSCA SEMÂNTICA - MOVIES")
    print("=" * 80)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    splits_dir = os.path.join(base_dir, 'data/raw/splits')
    output_dir = os.path.join(base_dir, 'data/processed')
    chroma_dir = os.path.join(base_dir, 'chroma_db')
    
    print("\n[1/3] Carregando dados de teste...")
    test_docs = []
    test_files = []
    
    with open(os.path.join(splits_dir, 'test_files.txt'), 'r') as f:
        test_files = [line.strip() for line in f]
    
    for filename in test_files:
        filepath = os.path.join(base_dir, 'data/raw/documents', filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            test_docs.append(f.read())
    
    print(f"✓ Carregados {len(test_docs)} documentos de teste")
    
    print("\n[2/3] Inicializando sistema...")
    embedder = DocumentEmbedder()
    db = VectorDatabase(db_path=chroma_dir, collection_name="movies_docs")
    
    stats = db.get_stats()
    print(f"✓ ChromaDB carregado com {stats['total_documents']} documentos")
    
    print("\n[3/3] Executando avaliação...")
    metrics, df_results = evaluate_retrieval(
        db, embedder, test_docs, test_files, k=5
    )
    
    results_path = os.path.join(output_dir, 'test_results.csv')
    df_results.to_csv(results_path, index=False)
    print(f"\n✓ Resultados salvos em: {results_path}")
    
    print("MÉTRICAS DE AVALIAÇÃO")
    print(f"Total de consultas:              {metrics['total_queries']}")
    print(f"Encontrados no Top-5:            {metrics['found_in_top_k']} ({metrics['precision_at_k']*100:.2f}%)")
    print(f"Similaridade média (Top-1):      {metrics['avg_top1_similarity']*100:.2f}%")
    print(f"Similaridade média (Top-5):      {metrics['avg_all_similarity']*100:.2f}%")
    print(f"Matches perfeitos (≥99%):        {metrics['perfect_matches']} ({metrics['perfect_matches']/metrics['total_queries']*100:.2f}%)")
    print(f"Alta qualidade (≥80%):           {metrics['high_quality']} ({metrics['high_quality']/metrics['total_queries']*100:.2f}%)")
    
    print("\n  Melhores resultados (Top 5):")
    top_results = df_results.nlargest(5, 'top1_similarity')
    for idx, row in top_results.iterrows():
        print(f"  • {row['query_file']}: {row['top1_similarity']*100:.2f}% similaridade")
    
    print("\n Piores resultados (Bottom 5):")
    bottom_results = df_results.nsmallest(5, 'top1_similarity')
    for idx, row in bottom_results.iterrows():
        print(f"  • {row['query_file']}: {row['top1_similarity']*100:.2f}% similaridade")
    
    print("\n✓ Avaliação concluída!")

if __name__ == "__main__":
    main()
