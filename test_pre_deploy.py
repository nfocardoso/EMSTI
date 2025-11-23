"""
Pre-deployment validation for H3 Optimizer v0.2.0-alpha1
Tests all imports, version, and basic functionality before PyPI upload.
"""

import sys

print("=" * 80)
print("🔍 PRE-DEPLOY VALIDATION: H3 Optimizer v0.2.0-alpha1")
print("=" * 80)
print()

# Track test results
tests_passed = 0
tests_failed = 0
errors = []

def test_step(name: str):
    """Decorator for test steps."""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")

# ============================================================================
# TEST 1: Core Imports
# ============================================================================
test_step("Core Imports")
try:
    from h3 import (
        H3Optimizer,
        LossTracker,
        InformationWeightedSampler,
        IndexedDataset,
        EnergyTracker
    )
    print("✅ Core imports successful")
    tests_passed += 1
except Exception as e:
    print(f"❌ Core imports FAILED: {e}")
    errors.append(f"Core imports: {e}")
    tests_failed += 1
    sys.exit(1)  # Cannot continue without core

# ============================================================================
# TEST 2: Phase 1 Feature Imports
# ============================================================================
test_step("Phase 1 Feature Imports (Profiler, Controller, AutoH3)")
try:
    from h3 import (
        H3Profiler,
        EtaController,
        CoherenceTarget,
        OperatingMode,
        create_controlled_h3,
        AutoH3,
        auto_h3
    )
    print("✅ Phase 1 feature imports successful")
    tests_passed += 1
except Exception as e:
    print(f"❌ Phase 1 imports FAILED: {e}")
    errors.append(f"Phase 1 imports: {e}")
    tests_failed += 1
    sys.exit(1)  # Critical failure

# ============================================================================
# TEST 3: Presets & Logging Imports
# ============================================================================
test_step("Presets & Logging Imports")
try:
    from h3 import (
        ThermoAuditLogger,
        create_baseline_logger,
        h3_mnist_fast,
        h3_cifar_safe,
        h3_edge,
        explain_thermo_log,
        green_score,
        compare_runs
    )
    print("✅ Presets & logging imports successful")
    tests_passed += 1
except Exception as e:
    print(f"❌ Presets/logging imports FAILED: {e}")
    errors.append(f"Presets/logging: {e}")
    tests_failed += 1
    # Non-critical, continue

# ============================================================================
# TEST 4: Version Check
# ============================================================================
test_step("Version Check")
try:
    import h3
    version = h3.__version__
    expected = "0.2.0a1"

    if version == expected:
        print(f"✅ Version correct: {version}")
        tests_passed += 1
    else:
        print(f"❌ Version mismatch: got {version}, expected {expected}")
        errors.append(f"Version mismatch: {version} != {expected}")
        tests_failed += 1
        sys.exit(1)  # Critical - wrong version
except Exception as e:
    print(f"❌ Version check FAILED: {e}")
    errors.append(f"Version check: {e}")
    tests_failed += 1

# ============================================================================
# TEST 5: Basic Instantiation
# ============================================================================
test_step("Basic Instantiation")
try:
    import torch
    from torch.utils.data import TensorDataset

    # Create dummy data
    dummy_params = [torch.nn.Parameter(torch.randn(10, 10))]
    dummy_dataset = TensorDataset(
        torch.randn(100, 10),
        torch.randint(0, 2, (100,))
    )

    # Test H3Profiler
    profiler = H3Profiler(device='cpu', name='test_profiler')
    print("  ✅ H3Profiler instantiated")

    # Test EtaController
    target = CoherenceTarget(
        mode=OperatingMode.BALANCED,
        max_accuracy_loss_pp=1.0,
        min_efficiency_gain_pct=10.0
    )
    controller = EtaController(target, total_epochs=10)
    print("  ✅ EtaController instantiated")

    # Test AutoH3
    auto = AutoH3(
        dummy_params,
        dummy_dataset,
        mode='balanced',
        name='test_auto',
        device='cpu',
        total_epochs=10
    )
    print("  ✅ AutoH3 instantiated")

    # Test H3Optimizer
    optimizer = H3Optimizer(dummy_params, lr=1e-3)
    print("  ✅ H3Optimizer instantiated")

    print("✅ All instantiations successful")
    tests_passed += 1

except Exception as e:
    print(f"❌ Instantiation FAILED: {e}")
    import traceback
    traceback.print_exc()
    errors.append(f"Instantiation: {e}")
    tests_failed += 1

# ============================================================================
# TEST 6: Operating Modes
# ============================================================================
test_step("Operating Modes Validation")
try:
    modes = [
        OperatingMode.SAFE,
        OperatingMode.BALANCED,
        OperatingMode.GREEN,
        OperatingMode.EXTREME
    ]

    for mode in modes:
        target = CoherenceTarget(
            mode=mode,
            max_accuracy_loss_pp=1.0,
            min_efficiency_gain_pct=10.0
        )
        print(f"  ✅ Mode {mode.value} validated")

    print("✅ All operating modes validated")
    tests_passed += 1

except Exception as e:
    print(f"❌ Operating modes FAILED: {e}")
    errors.append(f"Operating modes: {e}")
    tests_failed += 1

# ============================================================================
# TEST 7: AutoH3 with Convenience Function
# ============================================================================
test_step("AutoH3 Convenience Function")
try:
    auto_instance = auto_h3(
        dummy_params,
        dummy_dataset,
        mode="balanced",
        name="test_convenience",
        device='cpu'
    )
    print("✅ auto_h3() convenience function works")
    tests_passed += 1

except Exception as e:
    print(f"❌ Convenience function FAILED: {e}")
    errors.append(f"auto_h3(): {e}")
    tests_failed += 1

# ============================================================================
# TEST 8: Controlled H3 Factory
# ============================================================================
test_step("Controlled H3 Factory Function")
try:
    opt, ctrl, tracker, dataset = create_controlled_h3(
        dummy_params,
        dummy_dataset,
        mode="balanced",
        lr=1e-3,
        total_epochs=10
    )
    print("  ✅ create_controlled_h3() returns optimizer, controller, tracker, and dataset")
    print(f"  ✅ Optimizer type: {type(opt).__name__}")
    print(f"  ✅ Controller type: {type(ctrl).__name__}")
    print(f"  ✅ Tracker type: {type(tracker).__name__}")
    print(f"  ✅ Dataset type: {type(dataset).__name__}")
    tests_passed += 1

except Exception as e:
    print(f"❌ Factory function FAILED: {e}")
    errors.append(f"create_controlled_h3(): {e}")
    tests_failed += 1

# ============================================================================
# TEST 9: pyproject.toml Metadata Check
# ============================================================================
test_step("pyproject.toml Metadata Validation")
try:
    with open('pyproject.toml', 'r') as f:
        pyproject_content = f.read()

    # Check version
    if 'version = "0.2.0a1"' in pyproject_content:
        print("  ✅ pyproject.toml version is 0.2.0a1")
    else:
        print("  ❌ pyproject.toml version not found or incorrect")
        errors.append("pyproject.toml version incorrect")
        tests_failed += 1

    # Check name
    if 'name = "h3_optimizer"' in pyproject_content:
        print("  ✅ Package name correct")
    else:
        print("  ❌ Package name incorrect")
        errors.append("Package name incorrect")
        tests_failed += 1

    # Check entry points
    if 'h3-report = "h3.cli:main"' in pyproject_content:
        print("  ✅ CLI entry point configured")
    else:
        print("  ⚠️  CLI entry point missing (non-critical)")

    tests_passed += 1

except Exception as e:
    print(f"❌ Metadata check FAILED: {e}")
    errors.append(f"Metadata: {e}")
    tests_failed += 1

# ============================================================================
# TEST 10: CHANGELOG Validation
# ============================================================================
test_step("CHANGELOG Validation")
try:
    with open('CHANGELOG.md', 'r') as f:
        changelog = f.read()

    # Check version entry
    if "[0.2.0-alpha1] - 2025-01-23" in changelog:
        print("  ✅ CHANGELOG has correct version and date")
    else:
        print("  ❌ CHANGELOG missing version entry or wrong date")
        errors.append("CHANGELOG version/date incorrect")
        tests_failed += 1

    # Check key features mentioned
    key_features = ['H3Profiler', 'η-Controller', 'AutoH3']
    for feature in key_features:
        if feature in changelog:
            print(f"  ✅ {feature} documented")
        else:
            print(f"  ⚠️  {feature} not mentioned in CHANGELOG")

    tests_passed += 1

except Exception as e:
    print(f"❌ CHANGELOG check FAILED: {e}")
    errors.append(f"CHANGELOG: {e}")
    tests_failed += 1

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print()
print("=" * 80)
print("📊 VALIDATION SUMMARY")
print("=" * 80)
print(f"Tests Passed: {tests_passed}")
print(f"Tests Failed: {tests_failed}")
print()

if tests_failed > 0:
    print("❌ VALIDATION FAILED - DO NOT DEPLOY")
    print()
    print("Errors encountered:")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
    print()
    print("Fix all errors before proceeding to deployment.")
    sys.exit(1)
else:
    print("✅ ALL VALIDATION TESTS PASSED")
    print()
    print("Package is ready for PyPI deployment.")
    print()
    print("Next steps:")
    print("  1. Review CHANGELOG.md one final time")
    print("  2. Run: python -m build")
    print("  3. Upload to TestPyPI first (recommended)")
    print("  4. Upload to PyPI")
    print()
    sys.exit(0)
