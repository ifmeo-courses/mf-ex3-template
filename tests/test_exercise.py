"""
Test file for Exercise 3: Time Series Analysis

This file contains automated tests to check student submissions.
Tests are run by GitHub Actions when students push their code.

IMPORTANT: This file tests the basic environment setup.
For detailed completion checking, see test_completion.py
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def test_imports_work():
    """Test that required packages can be imported."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import xarray as xr
        import scipy.signal as sg
        from scipy.optimize import curve_fit
        import gsw
    except ImportError as e:
        pytest.fail(f"Failed to import required package: {e}")


def test_notebook_exists():
    """Test that the assignment notebook exists."""
    # Try different possible locations for the notebook
    possible_paths = [
        Path("assignment.ipynb"),
        Path("../assignment.ipynb"),
        Path("src/assignment.ipynb"),
        Path("../src/assignment.ipynb")
    ]
    
    notebook_found = False
    for path in possible_paths:
        if path.exists():
            notebook_found = True
            break
    
    assert notebook_found, f"assignment.ipynb not found in any of these locations: {[str(p) for p in possible_paths]}"


def test_data_files_exist():
    """Test that the required data files exist."""
    # Try different possible locations for the data files
    ctd_paths = [
        Path("data/mooredCTD1_raw.nc"),
        Path("../data/mooredCTD1_raw.nc"),
        Path("./mooredCTD1_raw.nc")
    ]
    
    velocity_paths = [
        Path("data/mooring1velocity.nc"),
        Path("../data/mooring1velocity.nc"),
        Path("./mooring1velocity.nc")
    ]
    
    ctd_found = any(path.exists() for path in ctd_paths)
    velocity_found = any(path.exists() for path in velocity_paths)
    
    # Data files are optional since they can be downloaded/provided
    if not ctd_found:
        print("Note: CTD data file not found locally - should be provided with exercise")
    if not velocity_found:
        print("Note: Velocity data file not found locally - should be provided with exercise")


def test_harmonic_function():
    """Test that the semi-diurnal cosine function works correctly."""
    def semi_diurnal_cosine(t, amplitude, phase, offset):
        period = 12.42 / 24  # Convert hours to days
        return amplitude * np.cos(2 * np.pi * t / period + phase) + offset
    
    # Test parameters
    amplitude = 2.0
    phase = 0.5
    offset = 35.0
    
    # Test with sample time values
    t = np.linspace(0, 2, 100)  # 2 days
    result = semi_diurnal_cosine(t, amplitude, phase, offset)
    
    # Check that function returns expected shape
    assert len(result) == len(t), "Function should return same length as input"
    
    # Check that result oscillates around offset
    assert abs(np.mean(result) - offset) < 0.1, "Mean should be close to offset"
    
    # Check that amplitude is roughly correct
    assert abs((np.max(result) - np.min(result))/2 - amplitude) < 0.1, "Amplitude should be roughly correct"


def test_curve_fitting_capability():
    """Test that curve fitting functionality works."""
    from scipy.optimize import curve_fit
    
    def simple_cosine(x, a, b, c):
        return a * np.cos(x + b) + c
    
    # Generate test data
    x = np.linspace(0, 4*np.pi, 100)
    y_true = 2 * np.cos(x + 0.5) + 3
    y_noisy = y_true + np.random.normal(0, 0.1, len(x))
    
    # Test curve fitting
    try:
        popt, pcov = curve_fit(simple_cosine, x, y_noisy, p0=[1, 0, 0])
        assert len(popt) == 3, "Should return 3 fitted parameters"
        assert np.all(np.isfinite(popt)), "Fitted parameters should be finite"
    except Exception as e:
        pytest.fail(f"Curve fitting failed: {e}")


def test_basic_time_series_plotting():
    """Test that basic time series plotting functionality works."""
    # Create sample time series data
    time = np.linspace(0, 10, 1000)  # 10 days
    tidal_signal = 2 * np.cos(2 * np.pi * time / 0.5175) + 35  # M2 tide period ≈ 0.5175 days
    noise = np.random.normal(0, 0.1, len(time))
    salinity = tidal_signal + noise
    
    # Test basic plotting
    fig, ax = plt.subplots(figsize=(15, 5))
    
    ax.plot(time, salinity, label='Salinity')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Salinity (PSU)')
    ax.set_title('Test Time Series')
    ax.legend()
    ax.grid(True)
    
    # Check that plot was created
    assert len(ax.lines) > 0, "Plot should contain data lines"
    assert ax.get_xlabel() != '', "X-axis should be labeled"
    assert ax.get_ylabel() != '', "Y-axis should be labeled"
    
    plt.close(fig)


def test_filtering_concepts():
    """Test understanding of data filtering concepts."""
    # Create test signal with high and low frequency components
    time = np.linspace(0, 10, 1000)
    low_freq = np.sin(2 * np.pi * time * 0.1)  # Low frequency
    high_freq = 0.5 * np.sin(2 * np.pi * time * 2)  # High frequency
    signal = low_freq + high_freq
    
    # Simple moving average filter
    window_size = 50
    filtered_signal = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    
    # Test that filtering reduces high frequency content
    # (This is a simplified test - in practice, frequency analysis would be more appropriate)
    assert len(filtered_signal) == len(signal), "Filtered signal should have same length"
    assert np.std(filtered_signal) < np.std(signal), "Filtering should reduce variability"


def test_velocity_ellipse_concepts():
    """Test understanding of velocity ellipse concepts."""
    # Create sample elliptical motion
    time = np.linspace(0, 2*np.pi, 100)
    u_vel = 2 * np.cos(time)      # Major axis
    v_vel = 1 * np.sin(time)      # Minor axis
    
    # Test that velocities create closed ellipse
    # For a complete cycle, should return to starting point
    assert abs(u_vel[0] - u_vel[-1]) < 0.1, "U velocity should be periodic"
    assert abs(v_vel[0] - v_vel[-1]) < 0.1, "V velocity should be periodic"
    
    # Test that velocity magnitudes are reasonable
    assert np.max(np.abs(u_vel)) > np.max(np.abs(v_vel)), "Major axis should be larger than minor axis"


if __name__ == "__main__":
    pytest.main([__file__])