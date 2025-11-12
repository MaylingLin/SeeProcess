import argparse
import importlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from seeclick_ext.datasets.trajectory_dataset import TrajectoryDataset
from seeclick_ext.models.traj_encoder import TrajEncoder
from seeclick_ext.models.cross_modal import CrossModalAlign
import os

def dynamic_import(module_colon_class):
    # expects string like "path.to.module:ClassName"
    if ":" in module_colon_class:
        module_path, class_name = module_colon_class.split(":")
    else:
        raise ValueError("model_module must be like 'path.to.module:ClassName'")
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls

class WrappedModel(nn.Module):
    """
    Wrap original model to accept traj_embedding and fuse with text features.
    The wrapper instantiates the original model and expects that original_model.forward(image_feats, text_feats, ...) exists,
    or will call it with only image_feats, text_feats if no traj is given.
    """
    def __init__(self, original_model, text_feat_dim=512, traj_dim=256):
        super().__init__()
        self.original_model = original_model
        self.traj_dim = traj_dim
        self.text_feat_dim = text_feat_dim
        self.traj_fuse = nn.Linear(text_feat_dim + traj_dim, text_feat_dim)

    def forward(self, image_feats, text_feats, traj_embedding=None):
        if traj_embedding is not None:
            # assume text_feats is Tensor [B, text_feat_dim]
            fused_text = self.traj_fuse(torch.cat([text_feats, traj_embedding], dim=-1))
        else:
            fused_text = text_feats
        # Try to call original_model with (image_feats, fused_text, traj_embedding=None)
        try:
            return self.original_model(image_feats, fused_text, traj_embedding=None)
        except TypeError:
            # fallback: original takes only image_feats, text_feats
            return self.original_model(image_feats, fused_text)

def collate_fn(batch):
    # very simple collate; converts lists to tensors
    images = torch.stack([b["image"] for b in batch], dim=0)
    texts = [b["text"] for b in batch]
    traj_feats = torch.stack([b["traj_feats"] for b in batch], dim=0)  # [B,N,T,D]
    traj_mask = torch.stack([b["traj_mask"] for b in batch], dim=0)
    return {"images": images, "texts": texts, "traj_feats": traj_feats, "traj_mask": traj_mask}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_module", type=str, required=True,
                        help="module:path to model class, e.g. seeclick.models.main:SeeClickModel")
    parser.add_argument("--traj_cache_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    ModelClass = dynamic_import(args.model_module)
    # instantiate original model (must match constructor signature; if unusual, user should adapt here)
    try:
        orig_model = ModelClass()
    except Exception as e:
        print("Automatic instantiation failed; please edit run_traj_training.py to pass correct args to the model constructor.")
        raise e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig_model.to(device)
    model = WrappedModel(orig_model, text_feat_dim=512, traj_dim=256).to(device)

    # instantiate TrajEncoder and CrossModal
    traj_encoder = TrajEncoder(in_dim=3, hidden=128, out_dim=256).to(device)
    cross_align = CrossModalAlign(traj_dim=256, text_dim=512, proj_dim=256).to(device)

    dataset = TrajectoryDataset(args.traj_cache_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(traj_encoder.parameters()) + list(cross_align.parameters()), lr=args.lr)
    mse_loss = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train(); traj_encoder.train(); cross_align.train()
        for batch in dataloader:
            images = batch["images"].to(device)  # [B, C, H, W]
            # text embeddings: for prototyping we use random tensor; in practice extract from your model's text encoder
            # Try to get text_feat from orig_model if function exists
            if hasattr(orig_model, "encode_text"):
                text_feats = orig_model.encode_text(batch["texts"])
            else:
                # placeholder random text features
                text_feats = torch.randn(images.size(0), 512, device=device)

            traj_feats = batch["traj_feats"].to(device)  # [B,N,T,D]
            traj_mask = batch["traj_mask"].to(device)

            # compress nodes by mean to [B,T,D] for simple encoding (example)
            traj_feats_nodes_mean = traj_feats.mean(dim=1)  # [B,T,D]
            traj_embedding = traj_encoder(traj_feats_nodes_mean, traj_mask.mean(dim=1))  # [B,256]

            # cross-modal align loss
            align_loss, logits = cross_align(traj_embedding, text_feats)

            # forward through model
            outputs = model(images, text_feats, traj_embedding=traj_embedding)  # depends on orig_model API
            # For prototyping, assume outputs include 'pred_coords' we can supervise; here we create a dummy target
            # Replace with real target extraction
            target = torch.zeros(images.size(0), 2, device=device)
            # suppose outputs is a tensor [B,2]
            if isinstance(outputs, torch.Tensor):
                action_loss = mse_loss(outputs, target)
            else:
                # if outputs is dict...
                if "pred_coords" in outputs:
                    action_loss = mse_loss(outputs["pred_coords"], target)
                else:
                    # fallback dummy 0 loss
                    action_loss = torch.tensor(0.0, device=device)

            loss = action_loss + 0.1 * align_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} done. Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
