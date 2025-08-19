import torch
import pickle
from torch import Tensor
from tqdm.auto import tqdm
import torch.nn.functional as F 
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.decomposition import PCA
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from constants import RANDOM_STATE, MODEL_PATH


@dataclass
class OpusTranslationModelConfig:
    padding: bool
    model_name: str
    device: str
    dtype: str
    truncation: bool
    skip_special_tokens: bool


class OpusTranslationModel:

    def __init__(self, config: OpusTranslationModelConfig):
        self.config = config
        self.model = MarianMTModel.from_pretrained(
            self.config.model_name, 
            device_map=self.config.device, 
            torch_dtype=self.config.dtype
        )
        self.tokenizer = MarianTokenizer.from_pretrained(self.config.model_name)
        
    def translate(self, text: str) -> str:
        tokens = self.tokenizer(
            text, 
            padding=self.config.padding, 
            truncation=self.config.truncation, 
            return_tensors="pt"
        ).to(self.config.device)
        translated_tokens = self.model.generate(**tokens)
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=self.config.skip_special_tokens)

        return translated_text


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
    model_name: Optional[str] = None


class KMeansModels:
    
    def __init__(self, config: KMeansModelConfig) -> None:
        self.config = config
        self.topk = config.topk
        self.n_clusters = config.n_clusters
        self.model = (
            KMeans(self.n_clusters, random_state=RANDOM_STATE) 
            if config.model_name is None 
            else self.load_model()
        )

    def fit(self, X) -> None:
        self.model = self.model.fit(X)

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
        centroid_classes = {}
        for i, centroid in tqdm(enumerate(self.model.cluster_centers_)):
            centroid = torch.tensor(centroid, dtype=classes_embeddings.dtype, device=classes_embeddings.device).unsqueeze(0)
            scores = self.cosine_similarity(centroid, classes_embeddings)
            classes = self.get_classes(scores)
            centroid_classes[i] = classes
        
        return centroid_classes
    
    def get_cluster_items(self, product_embeddings: Tensor) -> Dict[Tuple[int, int], Any]:
        labels = self.model.predict(product_embeddings.tolist())
        cluster_items = {i: [] for i in range(self.model.n_clusters)}
        for  i, (embedding, label) in enumerate(zip(product_embeddings, labels)):
            cluster_items[label].append((i, embedding))

        return cluster_items

    def get_cluster_classes(self, product_embeddings: Tensor, class_embeddings: Tensor) -> Dict[int, Any]:
        cluster_items = self.get_cluster_items(product_embeddings)

        clusters_class = {}
        for cluster, values in tqdm(cluster_items.items()):
            embeddings = [embedding for _, embedding in values]
            embeddings = torch.stack(embeddings)
            scores = self.cosine_similarity(embeddings, class_embeddings)
            classes = self.get_classes(scores)
            if classes.dim() > 1:
                classes = classes.flatten()
            counter = Counter(classes.tolist())
            mode, count = counter.most_common(1)[0]
            clusters_class[cluster] = mode, count, classes.shape[0], count/classes.shape[0]

        return clusters_class
    
    def save_model(self):
        with open(MODEL_PATH / self.config.model_name, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self) -> None:
        with open(MODEL_PATH / self.config.model_name, "rb") as f:
            model = pickle.load(f)

        return model

@dataclass
class Falcon3EmbeddingConfig:
    model_name: str
    device: str
    dtype: str
    padding: bool
    truncation: bool
    use_4bit_quantization: bool = True
    max_length: int = 512
    convert_to_numpy: bool = False
    convert_to_tensor: bool = True
    pooling_strategy: str = "mean"
    output_dim: int = 1024
    projection_method: str = "linear"


class Falcon3EmbeddingModel:
    def __init__(self, config: Falcon3EmbeddingConfig):
        self.config = config
        self.model_name = config.model_name
        self.device = config.device
        self.pooling_strategy = config.pooling_strategy
        self.output_dim = config.output_dim
        self.projection_method = config.projection_method

        quantization_config = None
        if config.use_4bit_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, config.dtype),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "device_map": self.device if self.device != "cuda" else "auto",
            "quantization_config": quantization_config,
        }

        if not config.use_4bit_quantization:
            model_kwargs["torch_dtype"] = getattr(torch, config.dtype)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        self.model.eval()
        self.projection_layer = None
        self.pca = None
        self._original_dim = None

    def _initialize_projection_layer(self, original_dim: int):
        if self._original_dim is not None:
            return
        self._original_dim = original_dim
        if self.projection_method == "linear" and original_dim != self.output_dim:
            self.projection_layer = torch.nn.Linear(original_dim, self.output_dim, bias=False)
            torch.nn.init.xavier_uniform_(self.projection_layer.weight)
            if self.device == "cuda" and torch.cuda.is_available():
                self.projection_layer = self.projection_layer.cuda()
            self.projection_layer.eval()

    def _project_embeddings(self, embeddings: Tensor) -> Tensor:
        if self.projection_method == "truncate":
            return embeddings[:, :self.output_dim]
        elif self.projection_method == "linear":
            self._initialize_projection_layer(embeddings.shape[1])
            if self.projection_layer is not None:
                with torch.no_grad():
                    return self.projection_layer(embeddings)
            else:
                return embeddings
        elif self.projection_method == "pca":
            was_tensor = torch.is_tensor(embeddings)
            if was_tensor:
                embeddings_np = embeddings.cpu().numpy()
                device = embeddings.device
            else:
                embeddings_np = embeddings
            if self.pca is None:
                self.pca = PCA(n_components=self.output_dim)
                self.pca.fit(embeddings_np)
            projected_np = self.pca.transform(embeddings_np)
            if was_tensor:
                return torch.tensor(projected_np, device=device, dtype=embeddings.dtype)
            else:
                return projected_np
        else:
            raise ValueError(f"Unknown projection method: {self.projection_method}")

    def _pool_embeddings(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        if self.pooling_strategy == "mean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        elif self.pooling_strategy == "cls":
            pooled = hidden_states[:, 0, :]
        elif self.pooling_strategy == "max":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[input_mask_expanded == 0] = -1e9
            pooled = torch.max(hidden_states, 1)[0]
        elif self.pooling_strategy == "last_token":
            batch_size = hidden_states.shape[0]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            pooled = hidden_states[torch.arange(batch_size), sequence_lengths]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        return self._project_embeddings(pooled)

    def get_embeddings(self, texts: List[str], prompt_name: Optional[str] = None) -> Tensor:
        if prompt_name == "query":
            texts = [f"Query: {text}" for text in texts]
        elif prompt_name is not None:
            texts = [f"{prompt_name}: {text}" for text in texts]

        inputs = self.tokenizer(
            texts,
            padding=self.config.padding,
            truncation=self.config.truncation,
            max_length=self.config.max_length,
            return_tensors="pt"
        )

        if self.device == "cuda":
            device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = {k: v.to(device_to_use) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            embeddings = self._pool_embeddings(hidden_states, inputs['attention_mask'])

            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            if embeddings.shape[1] != self.output_dim and self.projection_method != "none":
                embeddings = self._project_embeddings(embeddings)

        if self.config.convert_to_numpy:
            embeddings = embeddings.cpu().numpy()
        elif not self.config.convert_to_tensor:
            embeddings = embeddings.tolist()

        return embeddings

    def calculate_scores(self, query_embeddings: Tensor, document_embeddings: Tensor) -> Tensor:
        if isinstance(query_embeddings, list):
            query_embeddings = torch.tensor(query_embeddings)
        if isinstance(document_embeddings, list):
            document_embeddings = torch.tensor(document_embeddings)

        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        document_embeddings = F.normalize(document_embeddings, p=2, dim=1)
        scores = torch.mm(query_embeddings, document_embeddings.transpose(0, 1))
        return scores

    def get_scores(self, queries: List[str], documents: List[str]) -> Tensor:
        query_embeddings = self.get_embeddings(queries, "query")
        document_embeddings = self.get_embeddings(documents)
        scores = self.calculate_scores(query_embeddings, document_embeddings)
        return scores

    def similarity(self, embeddings1: Tensor, embeddings2: Tensor) -> Tensor:
        return self.calculate_scores(embeddings1, embeddings2)


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
            



