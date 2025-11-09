"""
Test file for tools.py functions

This file tests the semi_diurnal_cosine function with known parameters
to verify it produces correct amplitudes and phases for M2 tidal harmonics.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add modules directory to path
sys.path.append(str(Path(__file__).parent.parent / "modules"))

from tools import semi_diurnal_cosine


def test_semi_diurnal_cosine_basic():
    """Test that semi_diurnal_cosine function exists and is callable."""
    assert callable(semi_diurnal_cosine)


def test_semi_diurnal_cosine_period():
    """Test that the function has the correct period (12.42 hours = 0.5175 days)."""
    # Create time array over multiple periods
    period_days = 12.42 / 24  # 0.5175 days
    t = np.linspace(0, 2 * period_days, 1000)
    
    # Test with simple parameters
    amplitude = 1.0
    phase = 0.0
    offset = 0.0
    
    values = semi_diurnal_cosine(t, amplitude, phase, offset)
    
    # Function should return to same value after one period
    t_one_period = np.array([0, period_days])
    values_period = semi_diurnal_cosine(t_one_period, amplitude, phase, offset)
    
    assert np.isclose(values_period[0], values_period[1], rtol=1e-6), \
        "Function should have same value after one period"


def test_semi_diurnal_cosine_amplitude_case1():
    """Test amplitude recovery - Case 1: Simple cosine with amplitude 2.5."""
    # Test parameters
    true_amplitude = 2.5
    true_phase = 0.0
    true_offset = 5.0
    
    # Create time array over one period
    period_days = 12.42 / 24
    t = np.linspace(0, period_days, 100)
    
    # Generate synthetic data
    synthetic_data = semi_diurnal_cosine(t, true_amplitude, true_phase, true_offset)
    
    # Test amplitude: max - min should equal 2 * amplitude
    data_range = np.max(synthetic_data) - np.min(synthetic_data)
    expected_range = 2 * true_amplitude
    
    assert np.isclose(data_range, expected_range, rtol=1e-3), \
        f"Amplitude test failed: expected range {expected_range}, got {data_range}"
    
    # Test offset: mean should equal offset
    data_mean = np.mean(synthetic_data)
    assert np.isclose(data_mean, true_offset, rtol=1e-2), \
        f"Offset test failed: expected {true_offset}, got {data_mean}"


def test_semi_diurnal_cosine_amplitude_case2():
    """Test amplitude recovery - Case 2: Cosine with amplitude 0.8."""
    # Test parameters
    true_amplitude = 0.8
    true_phase = np.pi / 4  # 45 degree phase shift
    true_offset = -1.2
    
    # Create time array
    period_days = 12.42 / 24
    t = np.linspace(0, period_days, 100)
    
    # Generate synthetic data
    synthetic_data = semi_diurnal_cosine(t, true_amplitude, true_phase, true_offset)
    
    # Test amplitude
    data_range = np.max(synthetic_data) - np.min(synthetic_data)
    expected_range = 2 * true_amplitude
    
    assert np.isclose(data_range, expected_range, rtol=1e-3), \
        f"Amplitude test failed: expected range {expected_range}, got {data_range}"
    
    # Test offset
    data_mean = np.mean(synthetic_data)
    assert np.isclose(data_mean, true_offset, rtol=1e-2), \
        f"Offset test failed: expected {true_offset}, got {data_mean}"


def test_semi_diurnal_cosine_phase_case1():
    """Test phase recovery - Case 1: Zero phase."""
    true_amplitude = 1.0
    true_phase = 0.0
    true_offset = 0.0
    
    # At t=0, function should equal amplitude * cos(phase) + offset = amplitude + offset
    t_zero = np.array([0.0])
    value_at_zero = semi_diurnal_cosine(t_zero, true_amplitude, true_phase, true_offset)
    expected_at_zero = true_amplitude * np.cos(true_phase) + true_offset
    
    assert np.isclose(value_at_zero[0], expected_at_zero, rtol=1e-6), \
        f"Phase test failed: at t=0, expected {expected_at_zero}, got {value_at_zero[0]}"


def test_semi_diurnal_cosine_phase_case2():
    """Test phase recovery - Case 2: π/2 phase shift."""
    true_amplitude = 1.5
    true_phase = np.pi / 2  # 90 degree phase shift
    true_offset = 2.0
    
    # At t=0, function should equal amplitude * cos(π/2) + offset = 0 + offset = offset
    t_zero = np.array([0.0])
    value_at_zero = semi_diurnal_cosine(t_zero, true_amplitude, true_phase, true_offset)
    expected_at_zero = true_amplitude * np.cos(true_phase) + true_offset
    
    assert np.isclose(value_at_zero[0], expected_at_zero, atol=1e-6), \
        f"Phase test failed: at t=0 with π/2 phase, expected {expected_at_zero}, got {value_at_zero[0]}"


def test_semi_diurnal_cosine_maximum_minimum():
    """Test that maximum and minimum occur at expected times."""
    true_amplitude = 3.0
    true_phase = 0.0
    true_offset = 1.0
    
    # Create time array over one period
    period_days = 12.42 / 24
    t = np.linspace(0, period_days, 1000)
    
    values = semi_diurnal_cosine(t, true_amplitude, true_phase, true_offset)
    
    # Find maximum and minimum values
    max_value = np.max(values)
    min_value = np.min(values)
    
    # Expected max and min
    expected_max = true_amplitude + true_offset
    expected_min = -true_amplitude + true_offset
    
    assert np.isclose(max_value, expected_max, rtol=1e-6), \
        f"Maximum value test failed: expected {expected_max}, got {max_value}"
    
    assert np.isclose(min_value, expected_min, rtol=1e-5), \
        f"Minimum value test failed: expected {expected_min}, got {min_value}"


def test_semi_diurnal_cosine_with_scipy_fit():
    """Test that scipy.optimize.curve_fit can recover known parameters."""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        pytest.skip("scipy not available for fitting test")
    
    # Known parameters
    true_amplitude = 1.8
    true_phase = np.pi / 3  # 60 degrees
    true_offset = 2.5
    
    # Create synthetic data
    period_days = 12.42 / 24
    t = np.linspace(0, 3 * period_days, 150)  # 3 periods for good fitting
    
    # Add small amount of noise to make it realistic
    np.random.seed(42)  # For reproducible tests
    noise = np.random.normal(0, 0.01, len(t))
    synthetic_data = semi_diurnal_cosine(t, true_amplitude, true_phase, true_offset) + noise
    
    # Fit the function
    initial_guess = [1.0, 0.0, np.mean(synthetic_data)]
    popt, _ = curve_fit(semi_diurnal_cosine, t, synthetic_data, p0=initial_guess)
    
    fitted_amplitude, fitted_phase, fitted_offset = popt
    
    # Check that fitted parameters are close to true parameters
    assert np.isclose(fitted_amplitude, true_amplitude, rtol=0.05), \
        f"Fitted amplitude {fitted_amplitude} not close to true {true_amplitude}"
    
    assert np.isclose(fitted_phase, true_phase, atol=0.1), \
        f"Fitted phase {fitted_phase} not close to true {true_phase}"
    
    assert np.isclose(fitted_offset, true_offset, rtol=0.05), \
        f"Fitted offset {fitted_offset} not close to true {true_offset}"


def test_semi_diurnal_cosine_edge_cases():
    """Test edge cases and boundary conditions."""
    # Test with zero amplitude
    t = np.array([0, 0.25, 0.5])
    values = semi_diurnal_cosine(t, 0.0, 0.0, 5.0)
    expected = np.array([5.0, 5.0, 5.0])  # Should be constant at offset
    
    assert np.allclose(values, expected), \
        "Zero amplitude should give constant offset"
    
    # Test with negative amplitude
    values_neg = semi_diurnal_cosine(t, -2.0, 0.0, 0.0)
    values_pos = semi_diurnal_cosine(t, 2.0, np.pi, 0.0)  # π phase shift
    
    assert np.allclose(values_neg, values_pos), \
        "Negative amplitude should equal positive amplitude with π phase shift"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])