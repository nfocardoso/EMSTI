"""
Energy measurement and tracking for H3 optimizer.

Implements thermodynamic efficiency tracking from H3 paper Sections 4.2, 4.3, 5.2.
Measures energy consumption E = ∫P(t)dt and information gain ΔI to compute
efficiency η_thermo = ΔI/E (bits per joule).

Reference: H3 paper, Equations 4.3, 4.5, 4.6, 4.7
"""

from typing import Optional, Dict, List, Tuple
import time
import math
import platform
import threading
import subprocess
import re
import torch


class EnergyTracker:
    """
    Tracks energy consumption and thermodynamic efficiency during training.

    Measures instantaneous power P(t) at regular intervals, integrates to compute
    total energy E via trapezoidal rule (Eq. 4.6), and calculates information gain
    ΔI from loss reduction (Eq. 4.3) to determine thermodynamic efficiency
    η_thermo = ΔI/E (Eq. 4.7).

    Supports multiple backends:
    - NVML: NVIDIA GPU power monitoring via pynvml
    - PowerMetrics: Apple Silicon power monitoring via powermetrics
    - Fallback: Constant power estimate based on TDP

    Args:
        device: Torch device string ('cuda', 'cuda:0', 'mps', 'cpu').
        sampling_rate: Power sampling frequency in Hz (default: 1.0).
            Higher rates give more accurate energy estimates but add overhead.
        backend: Force specific backend ('nvml', 'powermetrics', 'fallback')
            or None for auto-detection.
        fallback_power: Constant power estimate in watts if hardware monitoring
            unavailable (default: 250.0W for GPU, 60.0W for CPU/MPS).

    Attributes:
        backend: Active backend name.
        power_samples: List of (timestamp, power_watts) tuples.
        loss_history: List of (timestamp, loss_value) tuples.
        is_measuring: Whether measurement is currently active.

    Example:
        >>> tracker = EnergyTracker(device='cuda', sampling_rate=2.0)
        >>> tracker.start()
        >>> for batch in dataloader:
        ...     loss = train_step(batch)
        ...     tracker.log_loss(loss.item())
        >>> results = tracker.stop()
        >>> print(f"Efficiency: {results['efficiency_bits_per_j']:.2f} bits/J")
    """

    def __init__(
        self,
        device: str = 'cuda',
        sampling_rate: float = 1.0,
        backend: Optional[str] = None,
        fallback_power: Optional[float] = None,
    ) -> None:
        """Initialize energy tracker."""
        # Validate inputs
        if sampling_rate <= 0.0:
            raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
        if backend is not None and backend not in ['nvml', 'powermetrics', 'fallback']:
            raise ValueError(
                f"backend must be 'nvml', 'powermetrics', 'fallback', or None, "
                f"got {backend}"
            )

        self.device = device
        self.sampling_rate = sampling_rate

        # Set default fallback power based on device
        if fallback_power is None:
            if 'cuda' in device.lower():
                self.fallback_power = 250.0  # Typical GPU TDP
            elif 'mps' in device.lower():
                self.fallback_power = 60.0   # Apple Silicon SoC
            else:
                self.fallback_power = 65.0   # CPU TDP
        else:
            if fallback_power <= 0.0:
                raise ValueError(f"fallback_power must be positive, got {fallback_power}")
            self.fallback_power = fallback_power

        # Auto-detect or use specified backend
        self.backend = self._detect_backend() if backend is None else backend

        # Initialize backend-specific resources
        self._init_backend()

        # Measurement state
        self.power_samples: List[Tuple[float, float]] = []
        self.loss_history: List[Tuple[float, float]] = []
        self.start_time: Optional[float] = None
        self.is_measuring: bool = False
        self._sampling_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()  # Thread safety for shared state

    def _detect_backend(self) -> str:
        """
        Auto-detect best available power monitoring backend.

        Returns:
            backend: Name of detected backend ('nvml', 'powermetrics', 'fallback').
        """
        # Try NVML (NVIDIA GPUs)
        if 'cuda' in self.device.lower():
            try:
                import pynvml
                pynvml.nvmlInit()
                pynvml.nvmlShutdown()
                return 'nvml'
            except Exception:
                pass

        # Try PowerMetrics (Apple Silicon)
        if platform.system() == 'Darwin' and 'mps' in self.device.lower():
            try:
                # Test if powermetrics is available
                result = subprocess.run(
                    ['which', 'powermetrics'],
                    capture_output=True,
                    timeout=1.0
                )
                if result.returncode == 0:
                    return 'powermetrics'
            except Exception:
                pass

        # Fallback to constant estimate
        return 'fallback'

    def _init_backend(self) -> None:
        """Initialize backend-specific resources."""
        if self.backend == 'nvml':
            try:
                import pynvml
                pynvml.nvmlInit()

                # Extract GPU index from device string
                gpu_idx = self._parse_gpu_index(self.device)
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)

                # Verify we can read power
                pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)

            except Exception as e:
                # Fall back if NVML initialization fails
                print(f"Warning: NVML backend failed ({e}), using fallback")
                self.backend = 'fallback'

        elif self.backend == 'powermetrics':
            # No initialization needed for powermetrics
            # Note: Requires sudo privileges, will fall back if not available
            pass

        elif self.backend == 'fallback':
            # No initialization needed for fallback
            pass

    def _parse_gpu_index(self, device_str: str) -> int:
        """
        Parse GPU index from device string.

        Args:
            device_str: Device string like 'cuda', 'cuda:0', 'cuda:1'.

        Returns:
            gpu_idx: GPU index (0 if not specified).
        """
        if ':' in device_str:
            try:
                return int(device_str.split(':')[1])
            except (IndexError, ValueError):
                return 0
        return 0

    def start(self) -> None:
        """
        Start energy measurement.

        Initializes measurement state and starts background thread for
        power sampling at specified sampling_rate.
        """
        with self._lock:
            if self.is_measuring:
                raise RuntimeError("Measurement already started")

            self.is_measuring = True
            self.start_time = time.time()
            self.power_samples = []
            self.loss_history = []

        # Start background thread for power sampling
        self._sampling_thread = threading.Thread(
            target=self._sample_power_loop,
            daemon=True
        )
        self._sampling_thread.start()

    def stop(self) -> Dict[str, float]:
        """
        Stop measurement and compute results.

        Computes total energy via trapezoidal integration (Eq. 4.6),
        information gain from loss reduction (Eq. 4.3), and thermodynamic
        efficiency η_thermo = ΔI/E (Eq. 4.7).

        Returns:
            results: Dictionary containing:
                - total_energy_j: Total energy consumed in joules (Eq. 4.6)
                - total_time_s: Total measurement time in seconds
                - average_power_w: Average power in watts
                - info_gain_bits: Information gain in bits (Eq. 4.3)
                - efficiency_bits_per_j: Thermodynamic efficiency η (Eq. 4.7)
                - num_power_samples: Number of power measurements taken
                - num_loss_samples: Number of loss values logged

        Raises:
            RuntimeError: If measurement was not started.
        """
        with self._lock:
            if not self.is_measuring:
                raise RuntimeError("Measurement not started")

            self.is_measuring = False

        # Wait for sampling thread to finish
        if self._sampling_thread is not None and self._sampling_thread.is_alive():
            self._sampling_thread.join(timeout=2.0)

        # Compute total energy via trapezoidal integration (Eq. 4.6)
        total_energy = self._compute_total_energy()

        # Compute information gain in bits (Eq. 4.3)
        info_gain = self._compute_info_gain()

        # Compute thermodynamic efficiency (Eq. 4.7)
        efficiency = info_gain / total_energy if total_energy > 0 else 0.0

        # Compute statistics
        total_time = time.time() - self.start_time if self.start_time else 0.0
        avg_power = total_energy / total_time if total_time > 0 else 0.0

        return {
            'total_energy_j': total_energy,
            'total_time_s': total_time,
            'average_power_w': avg_power,
            'info_gain_bits': info_gain,
            'efficiency_bits_per_j': efficiency,
            'num_power_samples': len(self.power_samples),
            'num_loss_samples': len(self.loss_history),
        }

    def log_loss(self, loss_value: float) -> None:
        """
        Log current loss value for information gain calculation.

        Should be called after each training step with the current loss.
        Information gain is computed as ΔI = [L_initial - L_final] / ln(2).

        Args:
            loss_value: Current loss value (must be non-negative).

        Raises:
            ValueError: If loss_value is negative.
        """
        if loss_value < 0:
            raise ValueError(f"loss_value must be non-negative, got {loss_value}")

        if self.is_measuring:
            with self._lock:
                self.loss_history.append((time.time(), loss_value))

    def get_current_stats(self) -> Dict[str, float]:
        """
        Get current statistics without stopping measurement.

        Useful for monitoring efficiency during training.

        Returns:
            stats: Dictionary with current measurements, or empty dict if not measuring.
        """
        if not self.is_measuring:
            return {}

        with self._lock:
            elapsed = time.time() - self.start_time if self.start_time else 0.0
            current_energy = self._compute_total_energy()
            current_info = self._compute_info_gain()

        current_efficiency = (
            current_info / current_energy if current_energy > 0 else 0.0
        )

        return {
            'elapsed_time_s': elapsed,
            'current_energy_j': current_energy,
            'current_info_bits': current_info,
            'current_efficiency_bits_per_j': current_efficiency,
            'num_power_samples': len(self.power_samples),
            'num_loss_samples': len(self.loss_history),
        }

    def reset(self) -> None:
        """Reset all measurements to initial state."""
        with self._lock:
            if self.is_measuring:
                raise RuntimeError("Cannot reset while measuring")

            self.power_samples = []
            self.loss_history = []
            self.start_time = None

    def _sample_power_loop(self) -> None:
        """
        Background thread that samples power at regular intervals.

        Runs while is_measuring is True, sampling at rate specified by
        sampling_rate (Hz).
        """
        while self.is_measuring:
            timestamp = time.time()
            power = self._measure_power()

            with self._lock:
                self.power_samples.append((timestamp, power))

            # Sleep for sampling interval
            time.sleep(1.0 / self.sampling_rate)

    def _measure_power(self) -> float:
        """
        Measure instantaneous power in watts.

        Uses active backend to query hardware power consumption.

        Returns:
            power: Instantaneous power in watts.
        """
        if self.backend == 'nvml':
            try:
                import pynvml
                # Returns power in milliwatts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)
                return power_mw / 1000.0  # Convert to watts

            except Exception:
                # Fall back to constant estimate
                return self.fallback_power

        elif self.backend == 'powermetrics':
            try:
                # Run powermetrics for single sample
                result = subprocess.run(
                    ['sudo', 'powermetrics', '--samplers', 'cpu_power', '-n', '1'],
                    capture_output=True,
                    text=True,
                    timeout=2.0
                )

                if result.returncode == 0:
                    power = self._parse_powermetrics(result.stdout)
                    if power > 0:
                        return power

                # Fall back if parsing failed
                return self.fallback_power

            except Exception:
                return self.fallback_power

        else:  # fallback backend
            return self.fallback_power

    def _parse_powermetrics(self, output: str) -> float:
        """
        Parse powermetrics output to extract power consumption.

        Args:
            output: Raw stdout from powermetrics command.

        Returns:
            power: Parsed power in watts, or 0.0 if parsing failed.
        """
        try:
            # Look for patterns like "CPU Power: 1234 mW" or "Combined Power: 5678 mW"
            patterns = [
                r'Combined Power.*?(\d+)\s*mW',
                r'CPU Power.*?(\d+)\s*mW',
                r'GPU Power.*?(\d+)\s*mW',
            ]

            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    power_mw = float(match.group(1))
                    return power_mw / 1000.0  # Convert to watts

            return 0.0

        except Exception:
            return 0.0

    def _compute_total_energy(self) -> float:
        """
        Compute total energy using trapezoidal integration.

        Implements Eq. 4.6 from paper:
            E = ∫P(t)dt ≈ Σ[(P_k + P_{k+1})/2] * (t_{k+1} - t_k)

        Returns:
            energy: Total energy in joules.
        """
        if len(self.power_samples) < 2:
            return 0.0

        total_energy = 0.0

        for i in range(len(self.power_samples) - 1):
            t1, p1 = self.power_samples[i]
            t2, p2 = self.power_samples[i + 1]

            # Trapezoidal rule: E += (p1 + p2)/2 * (t2 - t1)
            dt = t2 - t1
            avg_power = (p1 + p2) / 2.0
            total_energy += avg_power * dt

        return total_energy

    def _compute_info_gain(self) -> float:
        """
        Compute information gain in bits.

        Implements Eq. 4.3 from paper:
            ΔI_bits = [L(w_old) - L(w_new)] / ln(2)

        Where L is the cross-entropy loss. Information gain measures
        reduction in average code length (bits needed to encode labels).

        Returns:
            info_gain: Information gain in bits (non-negative).
        """
        if len(self.loss_history) < 2:
            return 0.0

        # ΔI = [L_initial - L_final] / ln(2)
        _, initial_loss = self.loss_history[0]
        _, final_loss = self.loss_history[-1]

        # Convert nats to bits
        info_gain_nats = initial_loss - final_loss
        info_gain_bits = info_gain_nats / math.log(2)

        # Ensure non-negative (loss might increase in some steps)
        return max(0.0, info_gain_bits)

    def __del__(self) -> None:
        """Cleanup: shutdown NVML if initialized."""
        try:
            if self.backend == 'nvml':
                import pynvml
                pynvml.nvmlShutdown()
        except Exception:
            pass

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EnergyTracker(device={self.device}, backend={self.backend}, "
            f"sampling_rate={self.sampling_rate}Hz)"
        )
