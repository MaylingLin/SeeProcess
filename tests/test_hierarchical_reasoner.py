import torch

from synapse.reasoning import HierarchicalReasoningExecutor


def test_hierarchical_reasoner_forward_and_losses():
    batch_size = 3
    reasoner = HierarchicalReasoningExecutor(
        low_dim=16,
        high_dim=16,
        text_dim=8,
        hidden_dim=16,
        action_tokens=["click", "type", "submit"],
        rl_weight=0.5,
        bc_weight=1.0,
        value_weight=0.2,
    )
    reasoner.register_cross_domain_adapter("robotics")
    reasoner.register_cross_domain_adapter("hci")
    low = torch.randn(batch_size, 16)
    high = torch.randn(batch_size, 16)
    text = torch.randn(batch_size, 8)
    ops = [
        [{"type": "click"}],
        [{"type": "search"}],
        [],
    ]
    domains = ["gui", "robotics", "hci"]
    action_labels = reasoner.encode_actions(ops, domains, low.device)
    traj_mask = torch.ones(batch_size, 2, 2)
    advantages, returns = reasoner.estimate_advantages(traj_mask, device=low.device)

    out = reasoner(
        low_level_state=low,
        high_level_context=high,
        text_features=text,
        action_labels=action_labels,
        advantages=advantages,
        returns=returns,
        domains=domains,
    )

    assert out.action_logits.shape == (batch_size, len(reasoner.action_vocab))
    assert out.guided_state.shape == low.shape
    assert out.plan_vector.shape[0] == batch_size
    assert out.total_loss.requires_grad
