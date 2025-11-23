# H3 Optimizer v0.2.0-alpha1

**⚠️ Alpha Release - Experimental**

This is an alpha release establishing the baseline for H3 development.
Use for research, experimentation, and feedback - not production.

## ✅ Stable Features (Production-Ready)

### H3Profiler - Zero-Risk Profiling
Profile ANY training run (Adam, SGD, etc.) without code changes:
```python
from h3 import H3Profiler
profiler = H3Profiler(device='cuda')
profiler.start()
# ... your normal training ...
print(profiler.get_report())
```

### ThermoAuditLogger - Experiment Tracking
Standardized logging for experiment analysis:
```python
from h3 import ThermoAuditLogger
logger = ThermoAuditLogger("experiment")
logger.log_epoch(epoch, accuracy, loss, energy, eta)
logger.save()
```

### h3-report CLI
```bash
h3-report --h3-log run.csv --baseline-log baseline.csv
```

## ⚠️ Experimental Features (Beta/Alpha)

- **η-Controller** (Beta): Automatic hyperparameter tuning
- **AutoH3** (Beta): Zero-config automation
- **H3Optimizer** (Alpha): Core optimizer under performance tuning

## 📋 Known Limitations

- H3Optimizer performance tuning incomplete for complex datasets
- CIFAR-10 shows higher accuracy trade-offs than target (<1pp)
- Energy measurements on Apple Silicon (MPS) may be inaccurate
- Hyperparameter sensitivity needs documentation

## 🚀 Installation
```bash
pip install h3-optimizer==0.2.0a1
```

## 🎯 What's Next

- Performance tuning for CIFAR-10 and ImageNet
- Hyperparameter sensitivity analysis
- Improved documentation and examples
- User feedback integration

## 📖 Full Changelog

See [CHANGELOG.md](https://github.com/nfocardoso/EMSTI/blob/main/CHANGELOG.md)

## 🙏 Feedback Welcome

This is an alpha release. Please report issues, share results, and contribute:
- Issues: https://github.com/nfocardoso/EMSTI/issues
- Discussions: https://github.com/nfocardoso/EMSTI/discussions
