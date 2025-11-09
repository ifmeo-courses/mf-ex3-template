# Exercise 3: Time Series Analysis

## Overview

This exercise teaches students to analyze oceanographic time series data using harmonic analysis and filtering techniques. Students will work with real mooring CTD and velocity data to understand tidal variability, perform harmonic fitting, create tidal ellipse plots, and apply data filtering methods.

## Learning Objectives

Students will:
- Load and manipulate time series data using xarray and netCDF4
- Understand oceanographic time coordinate systems and data conversion
- Perform harmonic analysis using scipy.optimize.curve_fit
- Fit semi-diurnal tidal harmonics (M2 tide) to CTD and velocity data
- Calculate and interpret tidal ellipses from velocity data
- Apply TEOS-10 oceanographic calculations (absolute salinity, conservative temperature)
- Implement time series filtering using rolling averages (boxcar filters)
- Analyze residuals from harmonic fits
- Create clear time series plots and tidal ellipse visualizations

## Required Packages

- matplotlib
- numpy
- xarray
- netCDF4
- scipy (for signal processing and optimization)
- gsw (for TEOS-10 oceanographic calculations)

## Data Files

- `data/mooredCTD1_raw.nc`: Real CTD time series data from oceanographic mooring
- `data/mooring1velocity.nc`: Velocity time series data (U, V components)
- `modules/tools.py`: Helper functions for harmonic analysis

## Expected Outputs

Students should generate four figures:
1. `ex3fig1-[StudentName]-Messfern.png`: CTD salinity harmonic fit and residuals
2. `ex3fig2-[StudentName]-Messfern.png`: Velocity harmonic fits and residuals (U and V components)
3. `ex3fig3-[StudentName]-Messfern.png`: Tidal ellipse from velocity harmonic analysis
4. `ex3fig4-[StudentName]-Messfern.png`: Effect of time series filtering (original vs filtered data)

## Key Concepts Covered

### Tidal Harmonic Analysis
- M2 lunar tide period (12.42 hours)
- Semi-diurnal cosine function fitting
- Amplitude, phase, and offset parameter estimation
- Residual analysis for model validation

### TEOS-10 Oceanographic Standards
- Conversion from practical salinity to absolute salinity
- Calculation of conservative temperature
- Understanding differences between measurement standards

### Time Series Processing
- Time coordinate conversion (seconds to fractional days)
- Data quality control and outlier removal
- Rolling average (boxcar) filtering
- Understanding sampling rates and filter window sizes

### Data Visualization
- Multi-panel time series plots
- Tidal ellipse plots with velocity vectors
- Overlay of original and filtered data
- Proper axis scaling and labeling

## Assessment

**PASS Requirements:**
- Load time series data and handle time coordinates correctly
- Implement TEOS-10 calculations (absolute salinity, conservative temperature)
- Complete harmonic fitting to CTD salinity data
- Complete harmonic fitting to velocity data (U and V components)
- Generate tidal ellipse from velocity harmonic analysis
- Implement time series filtering with rolling averages
- Create all 4 required figures with proper naming convention
- Answer analysis questions

**PASS+ Requirements:**
- All PASS requirements met
- Demonstrate understanding of tidal harmonic concepts in analysis answers
- Show clear interpretation of residuals and model adequacy
- Provide insightful discussion of filtering effects and time scales
- Explain TEOS-10 vs practical salinity differences appropriately

**FAIL:**
- Missing key figures or data analysis components
- Incorrect implementation of harmonic fitting or filtering
- Incomplete or superficial analysis question responses

## Instructions for Instructors

This exercise uses  mooring data and introduces time series analysis concepts. The exercise is designed to be completed in 3-4 hours and builds important skills for:
- Oceanographic data analysis workflows
- Harmonic analysis and tidal prediction
- Time series processing and filtering
- Scientific data visualization
- Understanding measurement standards in oceanography

The exercise includes a custom `semi_diurnal_cosine` function in the tools module that students must modify for the M2 tide period, reinforcing the importance of accurate tidal constants in oceanographic analysis.