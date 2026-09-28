import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

class EmbeddingVisualizer:
    
    def __init__(self, embeddings_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings = None
        self.documents = None
        self.reduced_embeddings = None
        
        self.load_data()
    
    def load_data(self):
        embeddings_path = self.embeddings_dir / 'embeddings.npy'
        documents_path = self.embeddings_dir / 'documents.pkl'
        
        self.embeddings = np.load(embeddings_path)
        
        with open(documents_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"✓ Loaded {len(self.documents)} documents")
        print(f"  Embedding shape: {self.embeddings.shape}")
    
    def reduce_with_umap(self, n_components: int = 2, n_neighbors: int = 15, 
                        min_dist: float = 0.1, metric: str = 'cosine'):
        print(f"\nReducing dimensions with UMAP to {n_components}D...")
        
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42
        )
        
        self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        
        print(f"✓ UMAP reduction complete: {self.reduced_embeddings.shape}")
        return self.reduced_embeddings
    
    def reduce_with_tsne(self, n_components: int = 2, perplexity: int = 30):
        print(f"\nReducing dimensions with t-SNE to {n_components}D...")
        
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            n_iter=1000
        )
        
        self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        
        print(f"✓ t-SNE reduction complete: {self.reduced_embeddings.shape}")
        return self.reduced_embeddings
    
    def extract_metadata(self) -> pd.DataFrame:
        metadata_list = []
        
        for doc in self.documents:
            filename = doc['filename']
            parts = filename.replace('.txt', '').rsplit('_', 1)
            
            if len(parts) == 2:
                city_state = parts[0]
                doc_id = parts[1]
                
                city_state_parts = city_state.rsplit('_', 1)
                if len(city_state_parts) == 2:
                    city = city_state_parts[0].replace('_', ' ')
                    state = city_state_parts[1]
                else:
                    city = city_state.replace('_', ' ')
                    state = 'Unknown'
            else:
                city = 'Unknown'
                state = 'Unknown'
                doc_id = '0'
            
            metadata_list.append({
                'filename': filename,
                'city': city,
                'state': state,
                'doc_id': doc_id,
                'text_preview': doc['text'][:100]
            })
        
        return pd.DataFrame(metadata_list)
    
    def plot_2d_scatter(self, color_by: str = 'state', figsize: tuple = (14, 10),
                       save_path: Optional[str] = None):
        if self.reduced_embeddings is None or self.reduced_embeddings.shape[1] != 2:
            print("Running UMAP reduction first...")
            self.reduce_with_umap(n_components=2)
        
        metadata = self.extract_metadata()
        
        plt.figure(figsize=figsize)
        
        if color_by == 'state':
            unique_states = metadata['state'].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_states)))
            state_to_color = dict(zip(unique_states, colors))
            
            for state in unique_states[:10]:
                mask = metadata['state'] == state
                plt.scatter(
                    self.reduced_embeddings[mask, 0],
                    self.reduced_embeddings[mask, 1],
                    c=[state_to_color[state]],
                    label=state,
                    alpha=0.6,
                    s=50
                )
            
            mask = ~metadata['state'].isin(unique_states[:10])
            plt.scatter(
                self.reduced_embeddings[mask, 0],
                self.reduced_embeddings[mask, 1],
                c='gray',
                label='Others',
                alpha=0.3,
                s=30
            )
        else:
            plt.scatter(
                self.reduced_embeddings[:, 0],
                self.reduced_embeddings[:, 1],
                alpha=0.6,
                s=50
            )
        
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.title('US Cities Documents - UMAP Visualization', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
        
        return metadata
    
    def plot_interactive_3d(self, save_path: Optional[str] = None):
        if self.reduced_embeddings is None or self.reduced_embeddings.shape[1] != 3:
            print("Running UMAP reduction to 3D first...")
            self.reduce_with_umap(n_components=3)
        
        import plotly.express as px
        
        metadata = self.extract_metadata()
        
        df = pd.DataFrame({
            'x': self.reduced_embeddings[:, 0],
            'y': self.reduced_embeddings[:, 1],
            'z': self.reduced_embeddings[:, 2],
            'city': metadata['city'],
            'state': metadata['state'],
            'text': metadata['text_preview']
        })
        
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='state',
            hover_data=['city', 'text'],
            title='US Cities Documents - 3D Interactive Visualization',
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'}
        )
        
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(height=800)
        
        if save_path:
            fig.write_html(save_path)
            print(f"✓ Interactive plot saved to {save_path}")
        
        fig.show()
        
        return df

def main():
    embeddings_dir = '/home/gabriel/dsml_final_project/data/processed/embeddings'
    
    print("="*70)
    print(" EMBEDDING VISUALIZATION")
    print("="*70)
    
    viz = EmbeddingVisualizer(embeddings_dir)
    
    print("\n[1] Creating 2D UMAP visualization...")
    print("-"*70)
    metadata = viz.plot_2d_scatter(
        color_by='state',
        save_path='/home/gabriel/dsml_final_project/data/processed/umap_2d.png'
    )
    
    print("\n[2] Creating 3D Interactive visualization...")
    print("-"*70)
    df_3d = viz.plot_interactive_3d(
        save_path='/home/gabriel/dsml_final_project/data/processed/umap_3d.html'
    )
    
    print("\n" + "="*70)
    print(" VISUALIZATION COMPLETE!")
    print("="*70)
    print("\nFiles created:")
    print("  • umap_2d.png - Static 2D visualization")
    print("  • umap_3d.html - Interactive 3D visualization")
    print("\nOpen umap_3d.html in your browser for interactive exploration!")

if __name__ == "__main__":
    main()
