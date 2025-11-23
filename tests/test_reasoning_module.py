"""
验证系统性推理框架：层次化规划、RL+BC 联合训练以及跨场景适配。
"""
import torch

from synapse.reasoning import HierarchicalReasoningExecutor


def test_reasoner_joint_losses_and_domain_adapters():
    torch.manual_seed(2)
    batch_size = 3
    reasoner = HierarchicalReasoningExecutor(
        low_dim=32,
        high_dim=32,
        text_dim=16,
        hidden_dim=32,
        action_tokens=["click", "type", "submit"],
        rl_weight=0.5,
        bc_weight=1.0,
        value_weight=0.2,
    )
    reasoner.register_cross_domain_adapter("robotics")
    reasoner.register_cross_domain_adapter("hci")

    low = torch.randn(batch_size, 32)
    high = torch.randn(batch_size, 32)
    text = torch.randn(batch_size, 16)
    ops = [
        [{"type": "click"}],
        [{"type": "type"}],
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
    assert out.plan_vector.shape == (batch_size, reasoner.hidden_dim)
    assert {"bc_loss", "rl_loss", "value_loss"} <= set(out.losses.keys())
    assert out.total_loss.requires_grad

    # 跨场景适配：不同 domain adapter 应产生不同的状态映射
    gui_state = reasoner._apply_domain_adapters(low, ["gui"] * batch_size)
    robotics_state = reasoner._apply_domain_adapters(low, ["robotics"] * batch_size)
    assert not torch.allclose(gui_state, robotics_state)