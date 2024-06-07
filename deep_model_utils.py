import json
import numpy as np
from sentence_transformers import SentenceTransformer

class DeepModel:
    def __init__(self, model_path, embeddings_path, documents_path) -> None:
        with open(documents_path) as f:
            self.documents = json.loads(f.read())['doc_no']
        self.model = SentenceTransformer.load(model_path)
        self.embeddings = np.load(embeddings_path)
        
    def retrieve_top_documents(self, keyword, result_limit):
        query = self.model.encode([keyword], normalize_embeddings=True)
        sims = self.model.similarity(query, self.embeddings)[0]
        sims = sims.argsort().cpu().detach().numpy()
        topn = sims[::-1][:result_limit]
        documents = []
        for idx in topn:
            documents.append(self.documents[idx])
        
        return documents
