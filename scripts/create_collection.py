from qdrant_client import QdrantClient, models
import os
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from dotenv import load_dotenv
import json

load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), timeout=60)

print("Uploading Models.... ")
dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
late_interaction_model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")

collection_name = "nvidia"


if client.get_collection(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)
    print(f"{collection_name} Deleted!!!")

client.create_collection(
    collection_name=collection_name,
    vectors_config = {
        "dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
        "colbert": models.VectorParams(size=128, distance=models.Distance.COSINE,
         multivector_config=models.MultiVectorConfig(
              comparator=models.MultiVectorComparator.MAX_SIM
              ),
            hnsw_config=models.HnswConfigDiff(m=0) # For Reranking Only
         )
    },
    sparse_vectors_config={"sparse": models.SparseVectorParams()}
)

print(f"{collection_name} Created!!!")

with open('nvidia_structured_docs.json', 'r') as f:
    docs = json.load(f)

texts = [doc["chunk_text"] for doc in docs]

#Vectorice
print("Geneating Vectors....")
dense_embeddings = list(dense_model.embed(texts))
sparse_embeddings = list(sparse_model.embed(texts))
late_interaction_embeddings = list(late_interaction_model.embed(texts))
print("Vectors Generated!!!")


print("Uploading to Qdrant Cloud...")
points = [
    models.PointStruct(
        id=i,
        vector={
            "dense": dense_embeddings[i].tolist(),
            "colbert": late_interaction_embeddings[i].tolist(),
            "sparse": sparse_embeddings[i].as_object()
        },
        payload=docs[i]
     )  for i in range(len(docs))
]

batch_size = 2
for i in range(0, len(points), batch_size):
    batch = points[i:i+batch_size]
    client.upsert(collection_name=collection_name, points=batch, wait=True)
    print(f"✅ Uploaded batch {i//batch_size + 1} (Indices {i} to {min(i + batch_size, len(points))})")
    
print("Points Uploaded!!!")
