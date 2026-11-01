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


def test_ema_update_moves_toward_model_and_respects_decay():
    model, optimizer, scheduler = _tiny_setup()
    ema = sku.EMA(model, decay=0.9)
    shadow_before = {k: v.clone() for k, v in ema.state_dict().items()}

    _train_step(model, optimizer, scheduler, seed_input=0)
    ema.update(model)

    some_key = next(k for k, v in shadow_before.items() if v.dtype.is_floating_point and v.numel() > 0)
    updated = ema.state_dict()[some_key]
    raw = dict(model.state_dict())[some_key]
    before = shadow_before[some_key]
    # decay=0.9 -> shadow should have moved only 10% of the way from its old
    # value toward the raw model's new value, not jumped all the way there.
    assert torch.allclose(updated, 0.9 * before + 0.1 * raw, atol=1e-6)
    assert not torch.allclose(updated, raw)


def test_ema_copy_to_loads_shadow_weights_into_a_model():
    model, optimizer, scheduler = _tiny_setup()
    ema = sku.EMA(model, decay=0.999)
    for i in range(3):
        _train_step(model, optimizer, scheduler, seed_input=i)
        ema.update(model)

    eval_model = sku.build_model(encoder_name="resnet18", encoder_weights=None)
    ema.copy_to(eval_model)
    for k, v in eval_model.state_dict().items():
        assert torch.allclose(v, ema.state_dict()[k])
    # The raw (non-EMA) model should differ from its own shadow by now.
    raw_sd, shadow_sd = model.state_dict(), ema.state_dict()
    some_key = next(k for k, v in shadow_sd.items() if v.dtype.is_floating_point and v.numel() > 0)
    assert not torch.allclose(raw_sd[some_key], shadow_sd[some_key])


def test_checkpoint_round_trip_includes_ema_state(tmp_path):
    model, optimizer, scheduler = _tiny_setup()
    ema = sku.EMA(model, decay=0.999)
    for i in range(2):
        _train_step(model, optimizer, scheduler, seed_input=i)
        ema.update(model)

    ckpt_path = tmp_path / "ckpt_ema.pt"
    sku.save_checkpoint(ckpt_path, model, optimizer, scheduler, iteration=2, best_metric=0.1, ema=ema)

    model2, optimizer2, scheduler2 = _tiny_setup(seed=999)
    ema2 = sku.EMA(model2, decay=0.999)
    sku.load_checkpoint(ckpt_path, model2, optimizer2, scheduler2, restore_rng=False, ema=ema2)

    for k, v in ema.state_dict().items():
        assert torch.allclose(v, ema2.state_dict()[k])


def test_load_checkpoint_without_ema_key_does_not_crash_when_ema_requested(tmp_path):
    # An old (pre-E004) checkpoint has no "ema" key -- loading it with an
    # EMA object passed in should leave that EMA's shadow untouched, not error.
    model, optimizer, scheduler = _tiny_setup()
    ckpt_path = tmp_path / "old_ckpt.pt"
    sku.save_checkpoint(ckpt_path, model, optimizer, scheduler, iteration=1, best_metric=0.0)  # no ema=...

    model2, optimizer2, scheduler2 = _tiny_setup(seed=5)
    ema2 = sku.EMA(model2, decay=0.999)
    shadow_before = {k: v.clone() for k, v in ema2.state_dict().items()}
    sku.load_checkpoint(ckpt_path, model2, optimizer2, scheduler2, restore_rng=False, ema=ema2)
    for k, v in ema2.state_dict().items():
        assert torch.allclose(v, shadow_before[k])


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
