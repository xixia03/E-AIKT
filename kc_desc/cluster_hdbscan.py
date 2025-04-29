import json
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
from collections import defaultdict

data_path = ""
write_path = ""
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    return normalized_embeddings

def main():
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    kcs = []
    index_to_key = []
    for key, item in data.items():
        for kc in item["descs"]:
            kcs.append(kc)
            index_to_key.append(key)

    model = SentenceTransformer('', device='cuda')
    embeddings = model.encode(kcs, show_progress_bar=True, convert_to_tensor=True)
    embeddings = embeddings.cpu().numpy()
    embeddings = normalize_embeddings(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric='euclidean')
    clusters = clusterer.fit_predict(embeddings)
    cluster_map = defaultdict(list)
    for idx, cluster_label in enumerate(clusters):
        cluster_map[cluster_label].append(kcs[idx])
    # Sort clusters by size (excluding outliers initially)
    sorted_clusters = {}
    non_outlier_clusters = {clust: members for clust, members in cluster_map.items() if clust != -1}
    sorted_cluster_ids = sorted(non_outlier_clusters, key=lambda x: len(non_outlier_clusters[x]), reverse=True)
    # Assign new cluster IDs to sorted clusters
    for new_id, old_id in enumerate(sorted_cluster_ids):
        sorted_clusters[new_id] = non_outlier_clusters[old_id]
    # Handle outliers by giving each a unique cluster ID
    outlier_count = len(sorted_clusters)
    for idx, kc in enumerate(cluster_map[-1]):
        sorted_clusters[outlier_count + idx] = [kc]
    # Convert and write kc_to_problems object to json
    with open(write_path, "w", encoding='utf-8') as outfile:
        json.dump(sorted_clusters, outfile, ensure_ascii=False,indent=2)


if __name__ == "__main__":
    main()

