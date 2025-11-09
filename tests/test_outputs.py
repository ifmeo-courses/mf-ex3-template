"""
Test actual outputs produced by students for Exercise 3: Time Series Analysis.
This tests what students actually created, not their code.
"""

import pytest
import xarray as xr
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def test_student_information_completed():
    """Test that student filled in their personal information."""
    notebook_path = Path("src/assignment.ipynb")
    if not notebook_path.exists():
        pytest.skip("Assignment notebook not found")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the student information cell 
    info_cell = None
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if 'Your Name:' in source or 'Complete your information' in source:
                info_cell = source
                break
    
    assert info_cell is not None, "Could not find student information cell"
    
    # Check that placeholders have been replaced
    placeholders = [
        '[REPLACE WITH YOUR ACTUAL NAME]',
        '[REPLACE WITH TODAY\'S DATE]',
        '[REPLACE WITH YOUR STUDENT ID]'
    ]
    
    for placeholder in placeholders:
        assert placeholder not in info_cell, f"Student information incomplete: '{placeholder}' still present"


def test_ctd_data_exists():
    """Test that CTD data file exists."""
    data_file = Path("data/mooredCTD1_raw.nc")
    assert data_file.exists(), "CTD data file was not found - required for analysis"


def test_velocity_data_exists():
    """Test that velocity data file exists."""
    data_file = Path("data/mooring1velocity.nc")
    assert data_file.exists(), "Velocity data file was not found - required for analysis"


def test_data_files_valid():
    """Test that the data files contain expected variables."""
    # Test CTD data
    ctd_file = Path("data/mooredCTD1_raw.nc")
    if ctd_file.exists():
        ctd_ds = xr.open_dataset(ctd_file)
        
        # Check required variables exist
        required_ctd_vars = ['PSAL', 'TEMP', 'PRES', 'time']
        for var in required_ctd_vars:
            assert var in ctd_ds, f"Required CTD variable '{var}' missing from dataset"
        
        # Check data is reasonable
        assert len(ctd_ds.time) > 100, "CTD dataset should have substantial time series data"
        assert ctd_ds.PSAL.max() > 15, "Salinity data appears invalid (too low)"
        assert ctd_ds.PSAL.max() < 40, "Salinity data appears invalid (too high)"
        
        ctd_ds.close()
    
    # Test velocity data
    velo_file = Path("data/mooring1velocity.nc")
    if velo_file.exists():
        velo_ds = xr.open_dataset(velo_file)
        
        # Check required variables exist
        required_velo_vars = ['UVEL', 'VVEL', 'time']
        for var in required_velo_vars:
            assert var in velo_ds, f"Required velocity variable '{var}' missing from dataset"
        
        # Check data is reasonable
        assert len(velo_ds.time) > 100, "Velocity dataset should have substantial time series data"
        assert abs(velo_ds.UVEL.max()) < 5, "U velocity data appears unrealistic (too high)"
        assert abs(velo_ds.VVEL.max()) < 5, "V velocity data appears unrealistic (too high)"
        
        velo_ds.close()


def test_all_figures_created():
    """Test that all 4 required figures were created."""
    figures_dir = Path("figures/")
    if not figures_dir.exists():
        figures_dir = Path("../figures/")  # Check parent directory too
    
    if not figures_dir.exists():
        pytest.skip("Figures directory not found")
    
    # Check each required figure exists
    required_figures = [
        'ex3fig1-*-Messfern.png',  # CTD cosine fit and residuals
        'ex3fig2-*-Messfern.png',  # Velocity cosine fit and residuals  
        'ex3fig3-*-Messfern.png',  # Tidal ellipse
        'ex3fig4-*-Messfern.png'   # Filtering effect
    ]
    
    for pattern in required_figures:
        matching_files = list(figures_dir.glob(pattern))
        assert len(matching_files) > 0, f"Required figure not found: {pattern}"
        
        # Check it's not the template name
        for fig_file in matching_files:
            assert 'YourName' not in fig_file.name, f"Figure name not personalized: {fig_file}"


def test_figures_contain_data():
    """Test that figures actually contain plotted data (not just empty plots)."""
    figures_dir = Path("figures/")
    if not figures_dir.exists():
        figures_dir = Path("../figures/")
    
    if not figures_dir.exists():
        pytest.skip("Figures directory not found")
    
    figure_files = list(figures_dir.glob("ex3fig*-*-Messfern.png"))
    if len(figure_files) == 0:
        pytest.skip("No figures found")
    
    # Test each figure we find
    for fig_file in figure_files[:2]:  # Test first 2 to avoid too many tests
        # Load image and check it's not just white/empty
        img = mpimg.imread(fig_file)
        
        # Check image has reasonable dimensions
        assert img.shape[0] > 100, f"Figure height too small: {fig_file}"
        assert img.shape[1] > 100, f"Figure width too small: {fig_file}"
        
        # Check it's not just a white image (mean pixel value should be < 0.96)
        mean_pixel = np.mean(img)
        assert mean_pixel < 0.96, f"Figure appears to be mostly empty/white: {fig_file} (mean pixel: {mean_pixel:.4f})"


def test_figure_file_sizes():
    """Test that figure files have reasonable sizes (not tiny empty files)."""
    figures_dir = Path("figures/")
    if not figures_dir.exists():
        figures_dir = Path("../figures/")
    
    if not figures_dir.exists():
        pytest.skip("Figures directory not found")
    
    figure_files = list(figures_dir.glob("ex3fig*-*-Messfern.png"))
    
    for fig_file in figure_files:
        file_size = fig_file.stat().st_size
        assert file_size > 10000, f"Figure file too small (likely empty): {fig_file} ({file_size} bytes)"
        assert file_size < 5000000, f"Figure file too large: {fig_file} ({file_size} bytes)"


def test_figure_naming_convention():
    """Test that figures follow the correct naming convention."""
    figures_dir = Path("figures/")
    if not figures_dir.exists():
        figures_dir = Path("../figures/")
    
    if not figures_dir.exists():
        pytest.skip("Figures directory not found")
    
    figure_files = list(figures_dir.glob("ex3fig*-*-Messfern.png"))
    
    for fig_file in figure_files:
        # Should match pattern: ex3fig[1-4]-[StudentName]-Messfern.png
        name_parts = fig_file.stem.split('-')
        
        assert len(name_parts) >= 3, f"Figure name format incorrect: {fig_file.name}"
        assert name_parts[0].startswith('ex3fig'), f"Figure should start with 'ex3fig': {fig_file.name}"
        assert name_parts[0][-1] in ['1', '2', '3', '4'], f"Figure number should be 1-4: {fig_file.name}"
        assert name_parts[-1] == 'Messfern', f"Figure should end with 'Messfern': {fig_file.name}"
        assert len(name_parts[1]) > 0, f"Student name appears missing: {fig_file.name}"


def test_harmonic_fitting_results():
    """Test that students completed the harmonic fitting sections."""
    notebook_path = Path("src/assignment.ipynb")
    if not notebook_path.exists():
        pytest.skip("Assignment notebook not found")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Look for evidence of harmonic fitting in code cells
    fitting_indicators = [
        'curve_fit',
        'semi_diurnal_cosine', 
        'amplitude',
        'phase',
        'fitted_'
    ]
    
    code_content = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            code_content.extend(cell['source'])
    
    full_code = '\n'.join(code_content)
    
    for indicator in fitting_indicators:
        assert indicator in full_code, f"Harmonic fitting not completed - '{indicator}' not found in code"


def test_teos10_calculations():
    """Test that TEOS-10 calculations were performed."""
    notebook_path = Path("src/assignment.ipynb")
    if not notebook_path.exists():
        pytest.skip("Assignment notebook not found")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Look for TEOS-10 functions in code
    teos10_indicators = [
        'gsw.SA_from_SP',
        'gsw.CT_from_t',
        "'SA'",
        "'CT'"
    ]
    
    code_content = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            code_content.extend(cell['source'])
    
    full_code = '\n'.join(code_content)
    
    found_indicators = sum(1 for indicator in teos10_indicators if indicator in full_code)
    assert found_indicators >= 2, "TEOS-10 calculations not completed - missing gsw functions"


def test_filtering_analysis():
    """Test that data filtering analysis was performed."""
    notebook_path = Path("src/assignment.ipynb")
    if not notebook_path.exists():
        pytest.skip("Assignment notebook not found")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Look for evidence of filtering
    filtering_indicators = [
        'rolling',
        'filter',
        'SA_filtered',
        'boxcar'
    ]
    
    code_content = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            code_content.extend(cell['source'])
    
    full_code = '\n'.join(code_content)
    
    found_indicators = sum(1 for indicator in filtering_indicators if indicator in full_code)
    assert found_indicators >= 2, "Data filtering analysis not completed"


def test_analysis_questions_answered():
    """Test that analysis questions section exists."""
    notebook_path = Path("src/assignment.ipynb")
    if not notebook_path.exists():
        pytest.skip("Assignment notebook not found")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Look for analysis questions section
    analysis_found = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source']).lower()
            if 'analysis' in source and 'question' in source:
                analysis_found = True
                break
    
    assert analysis_found, "Analysis questions section not found - please complete all questions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])