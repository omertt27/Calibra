"""
calibra.curation.latent_embed — Latent state embedding extraction.
"""
from __future__ import annotations

import numpy as np
from calibra.schema.episode import EpisodeBatch

def extract_latent_embeddings(batch: EpisodeBatch, model_type: str = "proprio") -> dict[str, np.ndarray]:
    """
    Extract a latent state representation vector per episode.
    
    Supported types:
      - 'proprio': mean and std of proprioceptive observations (default, fast).
      - 'visual': lightweight spatial statistics (pixel mean/std/PCA) of camera observations.
      - 'resnet': extracts ResNet features using PyTorch (if torch/torchvision is installed).
    """
    embeddings = {}
    
    for ep in batch.episodes:
        obs = ep.observations
        
        if model_type == "proprio":
            proprio = obs.get("proprio")
            if proprio is not None:
                mean = np.mean(proprio, axis=0)
                std = np.std(proprio, axis=0)
                emb = np.concatenate([mean, std])
            else:
                emb = np.zeros(10, dtype=np.float32)
                
        elif model_type == "visual":
            cam = obs.get("camera_rgb")
            if cam is not None and cam.size > 0:
                # Shape (T, H, W, C)
                spatial_mean = np.mean(cam, axis=(1, 2))  # (T, C)
                spatial_std = np.std(cam, axis=(1, 2))    # (T, C)
                mean = np.mean(spatial_mean, axis=0)
                std = np.mean(spatial_std, axis=0)
                std_std = np.std(spatial_std, axis=0)
                emb = np.concatenate([mean, std, std_std])
            else:
                emb = np.zeros(12, dtype=np.float32)
                
        elif model_type == "resnet":
            try:
                import torch
                import torchvision.models as models
                import torchvision.transforms as transforms
                
                cam = obs.get("camera_rgb")
                if cam is not None and cam.size > 0:
                    if cam.max() > 1.0:
                        cam = cam.astype(np.float32) / 255.0
                    tensor = torch.tensor(cam.transpose(0, 3, 1, 2), dtype=torch.float32)
                    
                    normalize = transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                    tensor = normalize(tensor)
                    
                    # Disable torchvision download logging
                    import os
                    os.environ["TORCH_HOME"] = "/tmp/torch"
                    resnet = models.resnet18(pretrained=True)
                    resnet.eval()
                    feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
                    
                    with torch.no_grad():
                        feats = feature_extractor(tensor).squeeze()
                        if feats.ndim == 1:
                            feats = feats.unsqueeze(0)
                        mean = torch.mean(feats, dim=0).numpy()
                        std = torch.std(feats, dim=0).numpy()
                        emb = np.concatenate([mean, std])
                else:
                    emb = np.zeros(1024, dtype=np.float32)
            except Exception:
                # Fallback to visual statistics if PyTorch or imports fail
                emb = extract_latent_embeddings(EpisodeBatch(episodes=[ep], dataset_name="fallback"), model_type="visual")[ep.metadata.episode_id]
                
        else:
            raise ValueError(f"Unknown latent space model type: {model_type}")
            
        embeddings[ep.metadata.episode_id] = emb.astype(np.float32)
        
    return embeddings
