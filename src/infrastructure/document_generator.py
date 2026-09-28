import pandas as pd
import json
import ast
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

class MovieDocumentGenerator:
    
    def __init__(self, data_dir: str, include_sections: Optional[List[str]] = None):
        self.data_dir = Path(data_dir)
        self.movies_df = None
        self.credits_df = None
        self.keywords_df = None
        default_sections = ['header', 'genres', 'overview']
        provided_sections = include_sections or default_sections
        self.include_sections = list(dict.fromkeys(provided_sections))
        if 'overview' not in self.include_sections:
            self.include_sections.append('overview')
        self.include_sections = list(dict.fromkeys(self.include_sections))
        self.load_data()
    
    def load_data(self):
        self.movies_df = pd.read_csv(
            self.data_dir / 'movies_metadata.csv',
            low_memory=False
        )
        
        self.credits_df = pd.read_csv(self.data_dir / 'credits.csv')
        self.keywords_df = pd.read_csv(self.data_dir / 'keywords.csv')
        
        self.movies_df['id'] = pd.to_numeric(self.movies_df['id'], errors='coerce')
        self.credits_df['id'] = pd.to_numeric(self.credits_df['id'], errors='coerce')
        self.keywords_df['id'] = pd.to_numeric(self.keywords_df['id'], errors='coerce')
        
        self.movies_df = self.movies_df[self.movies_df['overview'].notna()]
        self.movies_df = self.movies_df[self.movies_df['title'].notna()]
        
        print(f"Carregados {len(self.movies_df)} filmes")
    
    def safe_parse_json(self, json_str):
        if pd.isna(json_str) or json_str == '':
            return []
        try:
            return ast.literal_eval(json_str)
        except:
            return []
    
    def extract_names(self, json_list, key='name', limit=None):
        items = self.safe_parse_json(json_list)
        names = [item.get(key, '') for item in items if isinstance(item, dict)]
        if limit:
            names = names[:limit]
        return names
    
    def get_director(self, crew_json):
        crew = self.safe_parse_json(crew_json)
        for person in crew:
            if isinstance(person, dict) and person.get('job') == 'Director':
                return person.get('name', '')
        return 'Unknown'
    
    def generate_document(self, movie_id):
        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
        title = movie.get('title', 'Unknown')
        year = movie.get('release_date', '')[:4] if pd.notna(movie.get('release_date')) else 'Unknown'
        overview = movie.get('overview', '')
        rating = movie.get('vote_average', 0)
        votes = movie.get('vote_count', 0)
        runtime = movie.get('runtime', 0)

        genres = self.extract_names(movie.get('genres', '[]'))
        genre_str = ', '.join(genres) if genres else 'Unknown'
        credits = self.credits_df[self.credits_df['id'] == movie_id]
        if not credits.empty:
            cast_names = self.extract_names(credits.iloc[0].get('cast', '[]'), limit=5)
            director = self.get_director(credits.iloc[0].get('crew', '[]'))
        else:
            cast_names = []
            director = 'Unknown'

        cast_str = ', '.join(cast_names) if cast_names else 'Unknown'
        parts = []

        if year != 'Unknown':
            parts.append(f"{title} ({year})")
        else:
            parts.append(title)

        if genre_str != 'Unknown':
            parts.append(f"Genres: {genre_str}")

        if director != 'Unknown':
            parts.append(f"Director: {director}")

        if cast_str != 'Unknown':
            parts.append(f"Cast: {cast_str}")

        if pd.notna(rating) and rating > 0:
            if votes >= 1000:
                parts.append(f"Rating: {rating:.1f}/10 ({votes/1000:.0f}K votes)")
            else:
                parts.append(f"Rating: {rating:.1f}/10")

        if runtime > 0:
            parts.append(f"Runtime: {int(runtime)} min")

        if overview:
            parts.append(f"\n{overview}")

        doc_text = "\n".join(parts)

        return {
            'movie_id': int(movie_id),
            'title': title,
            'year': year,
            'genres': genres,
            'rating': float(rating) if pd.notna(rating) else 0.0,
            'text': doc_text
        }
    
    def generate_multiple_documents(self, n_docs: int = 1000) -> List[Dict]:
        valid_ids = self.movies_df['id'].dropna().astype(int).tolist()
        selected_ids = valid_ids[:min(n_docs, len(valid_ids))]
        documents = []
        print(f"Gerando {len(selected_ids)} documentos...")
        
        for movie_id in tqdm(selected_ids):
            try:
                doc = self.generate_document(movie_id)
                documents.append(doc)
            except Exception as e:
                continue
        
        print(f"Gerados {len(documents)} documentos")
        return documents
    
    def save_documents(self, documents: List[Dict], output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for doc in tqdm(documents, desc="Salvando"):
            filename = f"{doc['title'].replace('/', '_').replace(':', '')}_{doc['year']}_{doc['movie_id']}.txt"
            filename = filename.replace(' ', '_')
            filepath = output_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(doc['text'])
        
        metadata_df = pd.DataFrame([{
            'filename': f"{doc['title'].replace('/', '_').replace(':', '')}_{doc['year']}_{doc['movie_id']}.txt".replace(' ', '_'),
            'movie_id': doc['movie_id'],
            'title': doc['title'],
            'year': doc['year'],
            'genres': '|'.join(doc['genres']),
            'rating': doc['rating']
        } for doc in documents])
        
        metadata_df.to_csv(output_path / 'documents_metadata.csv', index=False)
        print(f"Salvos em {output_dir}")

def main(data_dir=None, output_dir=None):
    from pathlib import Path as _Path
    repo_root = _Path(__file__).resolve().parents[2]
    data_dir = data_dir or str(repo_root / 'data/archive')
    output_dir = output_dir or str(repo_root / 'data/raw/documents')
    generator = MovieDocumentGenerator(data_dir)
    doc = generator.generate_document(862)
    print(f"\nExemplo: {doc['title']} ({doc['year']})")
    print(f"Generos: {', '.join(doc['genres'])}")
    print(f"Rating: {doc['rating']}/10\n")
    print(doc['text'])
    documents = generator.generate_multiple_documents(n_docs=44000)
    generator.save_documents(documents, output_dir)
    print(f"\nTotal: {len(documents)} documentos em {output_dir}")

if __name__ == "__main__":
    main()
