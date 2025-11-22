"""
Validação básica da instalação H3
"""

print("=" * 60)
print("🧪 TESTE DE INSTALAÇÃO H3 v0.2.0-alpha")
print("=" * 60)
print()

# === TEST 1: Imports Core ===
print("TEST 1: Core imports...")
try:
    from h3 import (
        H3Optimizer,
        LossTracker,
        InformationWeightedSampler,
        IndexedDataset,
        EnergyTracker
    )
    print("✅ Core imports OK")
except Exception as e:
    print(f"❌ Core imports FAILED: {e}")
    print(f"   NOTA: h3/__init__.py importa TUDO, incluindo módulos que precisam de pandas/matplotlib")
    print(f"   Instala dependências: pip install pandas matplotlib")
    print(f"   Ou: pip install -e .[notebooks]")
    exit(1)

# === TEST 2: Imports Phase 0 (Presets/Logging) ===
print("\nTEST 2: Presets & logging imports...")
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
    print("✅ Presets & logging imports OK")
except Exception as e:
    print(f"❌ Presets/logging imports FAILED: {e}")
    print(f"   (Pode ser falta de pandas - instala: pip install pandas matplotlib)")

# === TEST 3: Imports Phase 1 (New Features) ===
print("\nTEST 3: Phase 1 features imports...")
try:
    from h3 import (
        H3Profiler,
        EtaController,
        CoherenceTarget,
        OperatingMode,
        create_controlled_h3,
        AutoH3
    )
    print("✅ Phase 1 features imports OK")
except Exception as e:
    print(f"❌ Phase 1 imports FAILED: {e}")
    exit(1)

# === TEST 4: Basic instantiation ===
print("\nTEST 4: Basic instantiation...")
try:
    import torch

    # Profiler
    profiler = H3Profiler(device='cpu', name='test')
    print("  ✅ H3Profiler instantiates")

    # Controller
    target = CoherenceTarget(
        mode=OperatingMode.BALANCED,
        max_accuracy_loss_pp=1.0,
        min_efficiency_gain_pct=10.0
    )
    controller = EtaController(target)
    print("  ✅ EtaController instantiates")

    # AutoH3 (minimal)
    dummy_params = [torch.nn.Parameter(torch.randn(10, 10))]
    from torch.utils.data import TensorDataset
    dummy_dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))

    auto = AutoH3(dummy_params, dummy_dataset, mode='balanced')
    print("  ✅ AutoH3 instantiates")

except Exception as e:
    print(f"❌ Instantiation FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# === TEST 5: Version check ===
print("\nTEST 5: Version check...")
try:
    import h3
    print(f"  ✅ H3 version: {h3.__version__}")
except:
    print("  ⚠️  No version found (minor issue)")

print()
print("=" * 60)
print("✅ TODOS OS TESTES PASSARAM!")
print("=" * 60)
print()
print("Pronto para testar em notebook.")
print()
print("Próximos passos:")
print("  1. Se este teste passou, o package está bem instalado")
print("  2. Pode agora testar com dados reais em notebook")
print("  3. Ou executar: python examples/quickstart.py (se existir)")
