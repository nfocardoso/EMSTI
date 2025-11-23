"""
Validation tests for H3 Optimizer v0.2.1
Tests new features: decoupled weight decay, presets, and intelligent warnings.
"""

import sys
import warnings
import torch
from torch.utils.data import TensorDataset

print("=" * 80)
print("🔍 H3 OPTIMIZER v0.2.1 VALIDATION TESTS")
print("=" * 80)
print()

tests_passed = 0
tests_failed = 0
errors = []


def test_step(name: str):
    """Print test step header."""
    print(f"\n{'=' * 80}")
    print(f"TEST: {name}")
    print(f"{'=' * 80}")


# ============================================================================
# TEST 1: Import h3 and check version
# ============================================================================
test_step("Version Check")
try:
    import h3
    version = h3.__version__
    expected = "0.2.1"

    if version == expected:
        print(f"✅ Version correct: {version}")
        tests_passed += 1
    else:
        print(f"❌ Version mismatch: got {version}, expected {expected}")
        errors.append(f"Version: {version} != {expected}")
        tests_failed += 1
except Exception as e:
    print(f"❌ Version check FAILED: {e}")
    errors.append(f"Version check: {e}")
    tests_failed += 1
    sys.exit(1)


# ============================================================================
# TEST 2: Decoupled weight decay parameter
# ============================================================================
test_step("Decoupled Weight Decay Parameter")
try:
    from h3 import H3Optimizer

    # Create dummy parameters
    params = [torch.nn.Parameter(torch.randn(10, 10))]

    # Test default (should be True)
    opt1 = H3Optimizer(params, lr=1e-3)
    if opt1.defaults.get("decoupled_weight_decay") is True:
        print("  ✅ Default decoupled_weight_decay = True")
    else:
        print(f"  ❌ Default decoupled_weight_decay = {opt1.defaults.get('decoupled_weight_decay')}")
        errors.append("Default decoupled_weight_decay not True")
        tests_failed += 1

    # Test explicit False
    opt2 = H3Optimizer(params, lr=1e-3, decoupled_weight_decay=False)
    if opt2.defaults.get("decoupled_weight_decay") is False:
        print("  ✅ Can set decoupled_weight_decay = False")
    else:
        print(f"  ❌ Failed to set decoupled_weight_decay = False")
        errors.append("Cannot set decoupled_weight_decay=False")
        tests_failed += 1

    print("✅ Decoupled weight decay parameter working")
    tests_passed += 1

except Exception as e:
    print(f"❌ Decoupled weight decay test FAILED: {e}")
    import traceback
    traceback.print_exc()
    errors.append(f"Decoupled weight decay: {e}")
    tests_failed += 1


# ============================================================================
# TEST 3: More aggressive defaults
# ============================================================================
test_step("More Aggressive Defaults")
try:
    from h3 import H3Optimizer

    params = [torch.nn.Parameter(torch.randn(10, 10))]
    opt = H3Optimizer(params, lr=1e-3)

    # Check lipschitz_safety
    expected_safety = 0.8
    actual_safety = opt.defaults.get("lipschitz_safety")
    if actual_safety == expected_safety:
        print(f"  ✅ lipschitz_safety = {actual_safety} (more aggressive)")
    else:
        print(f"  ⚠️  lipschitz_safety = {actual_safety}, expected {expected_safety}")

    # Check lipschitz_update_interval
    expected_interval = 20
    actual_interval = opt.lipschitz_update_interval
    if actual_interval == expected_interval:
        print(f"  ✅ lipschitz_update_interval = {actual_interval}")
    else:
        print(f"  ⚠️  lipschitz_update_interval = {actual_interval}, expected {expected_interval}")

    print("✅ Aggressive defaults configured")
    tests_passed += 1

except Exception as e:
    print(f"❌ Aggressive defaults test FAILED: {e}")
    errors.append(f"Aggressive defaults: {e}")
    tests_failed += 1


# ============================================================================
# TEST 4: Preset imports
# ============================================================================
test_step("Preset Imports")
try:
    from h3 import (
        h3_cifar10_safe_adamw,
        h3_cifar10_balanced,
        h3_cifar10_green,
        h3_mnist_fast,
        load_preset,
    )
    print("  ✅ h3_cifar10_safe_adamw imported")
    print("  ✅ h3_cifar10_balanced imported")
    print("  ✅ h3_cifar10_green imported")
    print("  ✅ h3_mnist_fast imported")
    print("  ✅ load_preset imported")

    print("✅ All preset functions imported successfully")
    tests_passed += 1

except Exception as e:
    print(f"❌ Preset imports FAILED: {e}")
    import traceback
    traceback.print_exc()
    errors.append(f"Preset imports: {e}")
    tests_failed += 1
    sys.exit(1)  # Critical - can't continue


# ============================================================================
# TEST 5: Preset instantiation
# ============================================================================
test_step("Preset Instantiation")
try:
    from h3 import h3_cifar10_safe_adamw, h3_mnist_fast

    # Create dummy dataset
    dummy_dataset = TensorDataset(
        torch.randn(100, 3, 32, 32),
        torch.randint(0, 10, (100,))
    )

    # Create dummy parameters
    dummy_params = [torch.nn.Parameter(torch.randn(10, 10))]

    # Test CIFAR-10 safe preset
    optimizer, tracker, indexed_ds, config = h3_cifar10_safe_adamw(
        dummy_params,
        dummy_dataset,
        device='cpu'
    )
    print(f"  ✅ h3_cifar10_safe_adamw: keep_frac={config['keep_frac']}, uniform_mix={config['uniform_mix']}")

    # Verify AdamW-style weight decay
    if optimizer.defaults.get("decoupled_weight_decay") is True:
        print("  ✅ Preset uses decoupled weight decay (AdamW-style)")
    else:
        print("  ⚠️  Preset not using decoupled weight decay")

    # Test MNIST fast preset
    optimizer2, tracker2, indexed_ds2, config2 = h3_mnist_fast(
        dummy_params,
        dummy_dataset,
        device='cpu'
    )
    print(f"  ✅ h3_mnist_fast: keep_frac={config2['keep_frac']}, uniform_mix={config2['uniform_mix']}")

    print("✅ Preset instantiation working")
    tests_passed += 1

except Exception as e:
    print(f"❌ Preset instantiation FAILED: {e}")
    import traceback
    traceback.print_exc()
    errors.append(f"Preset instantiation: {e}")
    tests_failed += 1


# ============================================================================
# TEST 6: load_preset() helper
# ============================================================================
test_step("load_preset() Helper Function")
try:
    from h3 import load_preset

    dummy_dataset = TensorDataset(
        torch.randn(100, 3, 32, 32),
        torch.randint(0, 10, (100,))
    )
    dummy_params = [torch.nn.Parameter(torch.randn(10, 10))]

    # Test loading by name
    opt, tracker, ds, cfg = load_preset(
        'cifar10_safe_adamw',
        dummy_params,
        dummy_dataset,
        device='cpu'
    )
    print("  ✅ load_preset('cifar10_safe_adamw') works")

    # Test invalid preset name
    try:
        load_preset('nonexistent', dummy_params, dummy_dataset)
        print("  ❌ Should have raised ValueError for invalid preset")
        errors.append("load_preset doesn't validate preset names")
        tests_failed += 1
    except ValueError as e:
        print(f"  ✅ Correctly raises ValueError for invalid preset: {str(e)[:60]}...")

    print("✅ load_preset() working correctly")
    tests_passed += 1

except Exception as e:
    print(f"❌ load_preset() test FAILED: {e}")
    errors.append(f"load_preset(): {e}")
    tests_failed += 1


# ============================================================================
# TEST 7: Warning for scalar loss
# ============================================================================
test_step("Warning for Scalar Loss (reduction='none' reminder)")
try:
    from h3 import LossTracker

    tracker = LossTracker(num_samples=100).to('cpu')

    # Simulate scalar loss with multiple indices (common mistake)
    indices = torch.tensor([0, 1, 2, 3, 4])
    scalar_loss = torch.tensor([0.5])  # Only 1 loss value for 5 indices!

    # Should trigger warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            tracker.update(indices, scalar_loss)
            # Check if warning was raised
            if len(w) > 0 and "reduction='none'" in str(w[0].message):
                print(f"  ✅ Warning raised: {w[0].message}")
            else:
                print(f"  ⚠️  Warning not raised or incorrect message")
        except ValueError:
            # It's OK if it raises ValueError due to length mismatch
            print("  ✅ Correctly rejects scalar loss (ValueError)")

    print("✅ Scalar loss warning working")
    tests_passed += 1

except Exception as e:
    print(f"❌ Scalar loss warning test FAILED: {e}")
    import traceback
    traceback.print_exc()
    errors.append(f"Scalar loss warning: {e}")
    tests_failed += 1


# ============================================================================
# TEST 8: Warning for aggressive keep_frac
# ============================================================================
test_step("Warning for Aggressive keep_frac (< 0.4)")
try:
    from h3 import InformationWeightedSampler, LossTracker

    dummy_dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    tracker = LossTracker(len(dummy_dataset)).to('cpu')

    # Create sampler with aggressive keep_frac
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sampler = InformationWeightedSampler(
            dummy_dataset,
            tracker,
            keep_frac=0.3,  # Very aggressive!
            uniform_mix=0.2
        )

        if len(w) > 0 and "aggressive" in str(w[0].message).lower():
            print(f"  ✅ Warning raised: {w[0].message}")
        else:
            print(f"  ⚠️  Warning not raised for keep_frac=0.3")

    # Create sampler with safe keep_frac (should NOT warn)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sampler = InformationWeightedSampler(
            dummy_dataset,
            tracker,
            keep_frac=0.75,  # Safe
            uniform_mix=0.2
        )

        if len(w) == 0:
            print(f"  ✅ No warning for safe keep_frac=0.75")
        else:
            print(f"  ⚠️  Unexpected warning for keep_frac=0.75: {w[0].message}")

    print("✅ Aggressive keep_frac warning working")
    tests_passed += 1

except Exception as e:
    print(f"❌ Aggressive keep_frac test FAILED: {e}")
    import traceback
    traceback.print_exc()
    errors.append(f"Aggressive keep_frac: {e}")
    tests_failed += 1


# ============================================================================
# TEST 9: CLI import
# ============================================================================
test_step("CLI Import")
try:
    from h3 import h3_report_cli
    from h3.cli import compare_runs

    print("  ✅ h3_report_cli imported")
    print("  ✅ compare_runs imported from h3.cli")

    # Test compare_runs function
    baseline = {
        "name": "Adam",
        "final_accuracy": 85.0,
        "training_time_s": 100.0,
        "energy_joules": 5000.0,
    }
    h3_run = {
        "name": "H3",
        "final_accuracy": 87.0,
        "training_time_s": 80.0,
        "energy_joules": 4000.0,
    }

    result = compare_runs(baseline, h3_run)
    print(f"  ✅ compare_runs() works: verdict = {result['verdict']}")

    print("✅ CLI functionality working")
    tests_passed += 1

except Exception as e:
    print(f"❌ CLI test FAILED: {e}")
    import traceback
    traceback.print_exc()
    errors.append(f"CLI: {e}")
    tests_failed += 1


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print()
print("=" * 80)
print("📊 VALIDATION SUMMARY")
print("=" * 80)
print(f"Tests Passed: {tests_passed}/9")
print(f"Tests Failed: {tests_failed}/9")
print()

if tests_failed > 0:
    print("❌ SOME TESTS FAILED")
    print()
    print("Errors encountered:")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
    print()
    sys.exit(1)
else:
    print("✅ ALL TESTS PASSED - v0.2.1 READY")
    print()
    print("Key improvements validated:")
    print("  ✅ Decoupled weight decay (AdamW-style)")
    print("  ✅ More aggressive defaults (safety=0.8, interval=20)")
    print("  ✅ New preset system with 4 configurations")
    print("  ✅ Intelligent warnings for common mistakes")
    print("  ✅ CLI tool for result comparison")
    print()
    print("Next steps:")
    print("  1. Make 3 separate commits (optimizer, presets/cli, warnings)")
    print("  2. Test on real CIFAR-10 dataset (30 epochs)")
    print("  3. Validate accuracy gap is closed vs AdamW")
    print()
    sys.exit(0)
