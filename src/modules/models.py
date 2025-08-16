import torch
from torch import Tensor
import torch.nn.functional as F 
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted
from sentence_transformers import SentenceTransformer

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from constants import RANDOM_STATE


@dataclass
class SentenceEmbeddingConfig:
    device: str
    dtype: str
    model_id: str
    truncate_dim: int
    convert_to_numpy: bool
    convert_to_tensor: bool

class SentenceEmbeddingModel:

    def __init__(self, config: SentenceEmbeddingConfig):
        super().__init__()
        self.config = config
        self.model_id = config.model_id
        self.device = config.device
        self.dtype = config.dtype
        self.truncate_dim = config.truncate_dim

        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
            truncate_dim=self.truncate_dim,
            model_kwargs={"torch_dtype": self.dtype}
        )

    def get_embeddings(self, texts: List[str], prompt_name: Optional[str] = None) -> Tensor:
        embeddings = self.model.encode(
            texts, 
            prompt_name=prompt_name, 
            convert_to_numpy=self.config.convert_to_numpy,
            convert_to_tensor=self.config.convert_to_tensor
        )

        return embeddings

    def calculate_scores(self, query_embeddings: Tensor, document_embeddings: Tensor) -> Tensor:
        scores = self.model.similarity(query_embeddings, document_embeddings)

        return scores

    def get_scores(self, queries: List[str], documents: List[str]) -> Tensor:
        query_embeddings = self.get_embeddings(queries, "query")
        document_embeddings = self.get_embeddings(documents)
        scores = self.calculate_scores(query_embeddings, document_embeddings)
        
        return scores


@dataclass
class KMeansModelConfig:
    n_clusters: int
    topk: Optional[int] = None


class KMeansModels:
    
    def __init__(self, config: KMeansModelConfig) -> None:
        self.topk = config.topk
        self.n_clusters = config.n_clusters
        self.kmeans = KMeans(self.n_clusters, random_state=RANDOM_STATE)

    def fit(self, X) -> None:
        self.kmeans.fit(X)

    def cosine_similarity(self, cluster_embeddings: Tensor, classes_embeddings: Tensor) -> Tensor:
        cluster_embeddings = F.normalize(cluster_embeddings, p=2, dim=1)
        classes_embeddings = F.normalize(classes_embeddings, p=2, dim=1)

        scores = (cluster_embeddings @ classes_embeddings.T)

        return scores
    
    def get_topk_classes(self, scores: Tensor) -> Tensor:
        if self.topk is None:
            raise ValueError("You need to assing topk to use this function.")
         
        topk_classes = torch.topk(scores, k=self.topk)

        return topk_classes[1]
    
    def get_classes(self, scores: Tensor) -> Tensor:
        if self.topk is not None:
            classes = self.get_topk_classes(scores)
        else:
            classes = torch.argmax(scores, dim=1)

        return classes
    
    def get_centroid_classes(self, classes_embeddings: Tensor) -> Dict[int, Any]:
        if not check_is_fitted(self.kmeans):
            raise ValueError("You need to fit the model first.")
        
        centroid_classes = {}
        for i, centroid in enumerate(self.kmeans.cluster_centers_):
            centroid = torch.tensor(centroid, dtype=classes_embeddings.dtype, device=classes_embeddings.device).unsqueeze(0)
            scores = self.cosine_similarity(centroid, classes_embeddings)
            classes = self.get_classes(scores)
            centroid_classes[i] = classes
        
        return centroid_classes
    
    def get_cluster_classes(self, product_embeddings: Tensor, class_embeddings: Tensor) -> Dict[Tuple[int, int], Any]:
        if not check_is_fitted(self.kmeans):
            raise ValueError("You need to fit the model first.")
        
        labels = self.kmeans.predict(product_embeddings.tolsit())
        cluster_items = {i: [] for i in range(self.kmeans.n_clusters)}
        for  i, (embedding, label) in enumerate(zip(product_embeddings, labels)):
            cluster_items[label].append((i, embedding))

        cluster_classes = {}
        for cluster, embeddings in cluster_items.items():
            scores = self.cosine_similarity(embeddings, class_embeddings)
            classes = self.get_classes(scores)
            cluster_classes[cluster] = classes
    

@dataclass
class DummyModelConfig:
    strategy: str


class DummyModel:

    def __init__(self, config: DummyModelConfig):
        self.model = DummyClassifier(strategy=config.strategy)

    def fit_model(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
            



