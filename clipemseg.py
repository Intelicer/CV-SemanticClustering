from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from transformers import CLIPModel, CLIPProcessor


# Step 1 - Environment Setup
# Install once in your environment:
# pip install torch torchvision transformers scikit-learn matplotlib pillow git+https://github.com/openai/CLIP.git


def load_openai_clip(device: str) -> tuple[Any, Any]:
	"""Load OpenAI CLIP with model name ViT-B/32."""
	try:
		import clip
	except ModuleNotFoundError as exc:
		raise ModuleNotFoundError(
			"OpenAI CLIP is not installed. Run: pip install git+https://github.com/openai/CLIP.git"
		) from exc

	openai_model, openai_preprocess = clip.load("ViT-B/32", device=device)
	openai_model.eval()
	return openai_model, openai_preprocess


def load_clip(device: str) -> tuple[CLIPModel, CLIPProcessor]:
	"""Step 3: Load Hugging Face CLIP model and processor."""
	model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
	model.eval()
	processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
	return model, processor


def load_images_from_folder(root_dir: Path, max_images: int = 500) -> tuple[list[Image.Image], list[str], list[str]]:
	"""Step 4: Load images from class subfolders."""
	valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
	images: list[Image.Image] = []
	image_paths: list[str] = []
	class_names: list[str] = []

	for class_dir in sorted(root_dir.iterdir()):
		if not class_dir.is_dir():
			continue

		for img_path in sorted(class_dir.iterdir()):
			if len(images) >= max_images:
				return images, image_paths, class_names

			if img_path.suffix.lower() not in valid_ext:
				continue

			with Image.open(img_path) as img:
				images.append(img.convert("RGB"))
			image_paths.append(str(img_path))
			class_names.append(class_dir.name)

	return images, image_paths, class_names


def create_embeddings(
	images: list[Image.Image],
	model: CLIPModel,
	processor: CLIPProcessor,
	device: str,
) -> np.ndarray:
	"""Step 5: Convert each image into a semantic embedding vector."""
	embeddings: list[np.ndarray] = []

	with torch.no_grad():
		for img in images:
			inputs = processor(images=img, return_tensors="pt")
			inputs = {k: v.to(device) for k, v in inputs.items()}
			raw_outputs = model.get_image_features(**inputs)

			# Depending on transformers version/model wrapper, this may be a Tensor
			# or a model output object (e.g., BaseModelOutputWithPooling).
			if isinstance(raw_outputs, torch.Tensor):
				features = raw_outputs
			elif hasattr(raw_outputs, "image_embeds"):
				features = raw_outputs.image_embeds
			elif hasattr(raw_outputs, "pooler_output"):
				features = raw_outputs.pooler_output
			elif isinstance(raw_outputs, (tuple, list)) and len(raw_outputs) > 0:
				features = raw_outputs[0]
			else:
				raise TypeError(f"Unsupported CLIP output type: {type(raw_outputs)}")

			embeddings.append(features.detach().cpu().numpy()[0])

	if not embeddings:
		return np.empty((0, model.config.projection_dim), dtype=np.float32)

	return np.array(embeddings, dtype=np.float32)


def cluster_embeddings(x: np.ndarray, n_clusters: int = 5) -> np.ndarray:
	"""Step 6: Cluster embeddings with K-Means."""
	kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
	return kmeans.fit_predict(x)


def reduce_for_plot(x: np.ndarray, method: str = "tsne") -> np.ndarray:
	"""Step 7: Reduce embedding dimensions to 2D for visualization."""
	if method.lower() == "pca":
		reducer = PCA(n_components=2, random_state=42)
		return reducer.fit_transform(x)

	# t-SNE works better with moderate perplexity for small datasets.
	perplexity = min(30, max(5, len(x) // 10))
	reducer = TSNE(n_components=2, random_state=42, init="pca", perplexity=perplexity)
	return reducer.fit_transform(x)


def plot_clusters(x_2d: np.ndarray, labels: np.ndarray, title: str) -> None:
	"""Create a scatter plot for semantic clusters."""
	plt.figure(figsize=(10, 7))
	unique_labels = np.unique(labels)
	colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

	for i, cid in enumerate(unique_labels):
		mask = labels == cid
		plt.scatter(x_2d[mask, 0], x_2d[mask, 1], s=60, color=colors[i], label=f"Cluster {int(cid)}")

	plt.title(title)
	plt.xlabel("Component 1")
	plt.ylabel("Component 2")
	plt.legend(loc="best")
	plt.tight_layout()
	plt.show()


def analyze_clusters(cluster_labels: np.ndarray, true_labels: list[str]) -> dict[int, dict[str, Any]]:
	"""Step 8: Summarize class composition inside each cluster."""
	summary: dict[int, dict[str, Any]] = {}
	grouped: dict[int, list[str]] = defaultdict(list)

	for pred, truth in zip(cluster_labels, true_labels):
		grouped[int(pred)].append(truth)

	for cid, classes in grouped.items():
		counts = Counter(classes)
		top_class, top_count = counts.most_common(1)[0]
		purity = top_count / len(classes)
		summary[cid] = {
			"size": len(classes),
			"top_class": top_class,
			"purity": purity,
			"distribution": dict(counts),
		}

	return summary


def write_report(
	report_path: Path,
	dataset_dir: Path,
	num_images: int,
	embedding_dim: int,
	n_clusters: int,
	silhouette: float,
	cluster_summary: dict[int, dict[str, Any]],
) -> None:
	"""Step 10: Save a concise result summary."""
	lines = [
		"Semantic Image Clustering with CLIP",
		"",
		"1. Dataset description",
		f"- Source folder: {dataset_dir}",
		f"- Number of images: {num_images}",
		"",
		"2. CLIP model explanation",
		"- Model: openai/clip-vit-base-patch32",
		f"- Embedding dimension: {embedding_dim}",
		"",
		"3. Embedding generation",
		"- One embedding vector generated per image using CLIP image encoder.",
		"",
		"4. Clustering method",
		f"- K-Means with n_clusters={n_clusters}",
		"",
		"5. Visualization",
		"- 2D projection using t-SNE/PCA and colored by cluster.",
		"",
		"6. Results and discussion",
		f"- Silhouette Score: {silhouette:.4f}",
	]

	for cid in sorted(cluster_summary):
		item = cluster_summary[cid]
		lines.append(
			f"- Cluster {cid}: size={item['size']}, top_class={item['top_class']}, purity={item['purity']:.2f}"
		)

	report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
	# Step 2 - Choose a small dataset (200-500 images)
	# Uses ./dataset by default, otherwise falls back to ./images.
	dataset_dir = Path("dataset") if Path("dataset").is_dir() else Path("images")
	if not dataset_dir.is_dir():
		raise FileNotFoundError("Expected a 'dataset/' or 'images/' folder in the current directory.")

	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")

	# Explicitly load OpenAI CLIP with ViT-B/32 as requested.
	openai_model, openai_preprocess = load_openai_clip(device)
	print("Loaded OpenAI CLIP model: ViT-B/32")

	model, processor = load_clip(device)
	embedding_dim = int(model.config.projection_dim)
	print(f"CLIP embedding dimension: {embedding_dim}")

	images, image_paths, true_labels = load_images_from_folder(dataset_dir, max_images=500)
	if len(images) < 2:
		raise ValueError("Need at least 2 images to run clustering.")

	print(f"Loaded {len(images)} images from {len(set(true_labels))} classes.")
	if image_paths:
		print(f"Example image: {image_paths[0]}")

	x = create_embeddings(images, model, processor, device)
	print("Embeddings shape:", x.shape)

	n_clusters = min(5, len(images))
	cluster_labels = cluster_embeddings(x, n_clusters=n_clusters)
	print(f"Assigned clusters for {len(cluster_labels)} images.")

	x_2d = reduce_for_plot(x, method="tsne")
	plot_clusters(x_2d, cluster_labels, "Semantic Clustering of Images using CLIP")

	# Step 8 - Analyze clusters
	summary = analyze_clusters(cluster_labels, true_labels)
	print("\nCluster analysis:")
	for cid in sorted(summary):
		item = summary[cid]
		print(
			f"Cluster {cid}: size={item['size']}, top_class={item['top_class']}, purity={item['purity']:.2f}"
		)

	# Step 9 - Evaluate quality
	silhouette = silhouette_score(x, cluster_labels) if len(np.unique(cluster_labels)) > 1 else -1.0
	print(f"\nSilhouette Score: {silhouette:.4f}")

	# Step 10 - Present results
	report_path = Path("clustering_report.txt")
	write_report(
		report_path=report_path,
		dataset_dir=dataset_dir,
		num_images=len(images),
		embedding_dim=embedding_dim,
		n_clusters=n_clusters,
		silhouette=silhouette,
		cluster_summary=summary,
	)
	print(f"Saved report to: {report_path}")


if __name__ == "__main__":
	main()
