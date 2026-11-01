import numpy as np
import torch

from solar_k import unet as sku


def _tiny_setup(seed=0):
    torch.manual_seed(seed)
    model = sku.build_model(encoder_name="resnet18", encoder_weights=None)
    optimizer, scheduler = torch.optim.AdamW(model.parameters(), lr=1e-3), None
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 - 0.01 * step)
    return model, optimizer, scheduler


def _train_step(model, optimizer, scheduler, seed_input):
    torch.manual_seed(seed_input)
    x = torch.randn(1, 1, 64, 64)
    y = (torch.rand(1, 1, 64, 64) > 0.5).float()
    optimizer.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()


def test_checkpoint_round_trip_restores_weights_and_iteration(tmp_path):
    model, optimizer, scheduler = _tiny_setup()
    for i in range(3):
        _train_step(model, optimizer, scheduler, seed_input=i)
    ckpt_path = tmp_path / "ckpt.pt"
    sku.save_checkpoint(ckpt_path, model, optimizer, scheduler, iteration=3, best_metric=0.42)

    model2, optimizer2, scheduler2 = _tiny_setup(seed=999)  # different init
    ckpt = sku.load_checkpoint(ckpt_path, model2, optimizer2, scheduler2, restore_rng=False)

    assert ckpt["iteration"] == 3
    assert ckpt["best_metric"] == 0.42
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
    assert optimizer2.state_dict()["param_groups"][0]["lr"] == optimizer.state_dict()["param_groups"][0]["lr"]
    assert scheduler2.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]


def test_checkpoint_resume_continues_training_identically(tmp_path):
    # Train 4 steps straight through.
    model_a, opt_a, sch_a = _tiny_setup()
    for i in range(4):
        _train_step(model_a, opt_a, sch_a, seed_input=i)
    final_params_a = [p.clone() for p in model_a.parameters()]

    # Train 2 steps, checkpoint, resume, train 2 more -- same seeds each step.
    model_b, opt_b, sch_b = _tiny_setup()
    for i in range(2):
        _train_step(model_b, opt_b, sch_b, seed_input=i)
    ckpt_path = tmp_path / "ckpt.pt"
    sku.save_checkpoint(ckpt_path, model_b, opt_b, sch_b, iteration=2, best_metric=0.0)

    model_c, opt_c, sch_c = _tiny_setup(seed=123)  # different init, will be overwritten
    ckpt = sku.load_checkpoint(ckpt_path, model_c, opt_c, sch_c, restore_rng=False)
    start = ckpt["iteration"]
    for i in range(start, 4):
        _train_step(model_c, opt_c, sch_c, seed_input=i)

    for pa, pc in zip(final_params_a, model_c.parameters()):
        assert torch.allclose(pa, pc, atol=1e-6)


def test_rng_state_round_trip():
    import random

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    state = sku.capture_rng_state()
    before = (random.random(), np.random.rand(), torch.rand(1).item())

    # perturb
    random.random()
    np.random.rand()
    torch.rand(1)

    sku.restore_rng_state(state)
    after = (random.random(), np.random.rand(), torch.rand(1).item())
    assert before == after
