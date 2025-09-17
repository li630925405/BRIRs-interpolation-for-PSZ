# BRIR Interpolation for Personal Sound Zone Control

This repository contains the implementation for BRIR (Binaural Room Impulse Response) interpolation using neural networks and its application to personal sound zone (PSZ) control. The project combines three main components: BRIR dataset generation, neural network-based interpolation, and personal sound zone control evaluation.

## Project Structure

```
├── dataset/           # BRIR dataset generation using RAZR MATLAB toolbox
├── SIREN/            # SIREN neural network for BRIR interpolation
└── PSZ/              # Personal sound zone control implementation
```

## Components

### 1. Dataset Generation (`dataset/`)

Uses the RAZR MATLAB toolbox to generate BRIR datasets for various microphone configurations.

- **`generate_brir.m`**: Main script for generating BRIR data
  - Simulates room acoustics with configurable dimensions (4×5×3 meters)
  - Generates BRIRs for different microphone numbers (21, 41, 61, ..., 221)
  - Uses HRTF spatialization with CIPIC database
  - Outputs `.mat` files containing BRIR data

### 2. SIREN Neural Network (`SIREN/`)

Implements SIREN (Sinusoidal Representation Network) for BRIR interpolation.

**Key Files:**
- **`main.py`**: Main training script
- **`module.py`**: SIREN network architecture
- **`data.py`**: Data loading and preprocessing
- **`setting.py`**: Configuration parameters
- **`utils.py`**: Utility functions
- **`train_all_speakers.py`**: Batch training for multiple speaker positions
- **`evaluate_all_speakers.py`**: Evaluation across all speaker configurations

**Features:**
- Sinusoidal activation functions for implicit neural representations
- Physics-informed neural network (PINN) option with wave equation constraints
- Supports interpolation across spatial dimensions
- GPU acceleration with CUDA support

### 3. Personal Sound Zone Control (`PSZ/`)

Implements personal sound zone control using interpolated BRIRs from three methods (SIREN, linear interpolation, hybrid) with pressure matching algorithm.

**Key Files:**
- **`ctc_frequency_response.py`**: Average acoustic contrast vs frequency for three methods
- **`nmse_frequency_response.py`**: Average NMSE vs frequency for three methods
- **`combined_turning_points.py`**: Analyzes turning frequency vs number of measuremnt microphones
- **`contrast_comparison.py`**: Analyzes acoustic contrast improvement vs number of measuremnt microphones

**Methods Implemented:**
- **Pressure Matching Algorithm**: Primary PSZ control method
- **Linear Interpolation**: Baseline interpolation method
- **SIREN Interpolation**: Neural network-based interpolation
- **Hybrid Method**: Combination of SIREN and linear interpolation 

## Installation

### Prerequisites

**MATLAB Requirements:**
- MATLAB with Audio Toolbox
- RAZR toolbox for room acoustic simulation
- SOFA toolbox for HRTF handling

**Python Requirements:**
```bash
pip install torch torchvision scipy numpy matplotlib
```

## Usage

### 1. Generate BRIR Dataset

```matlab
cd dataset/
generate_brir
```

This will generate BRIR datasets for different microphone configurations and save them as `.mat` files.

### 2. Train SIREN Network

```bash
cd SIREN/
python main.py
```

Configure training parameters in `setting.py`:
- Network architecture (hidden layers, features)
- Training parameters (learning rate, steps)
- PINN weights and constraints

### 3. Evaluate Personal Sound Zone Control

```bash
cd PSZ/
python contrast_comparison.py
python ctc_frequency_response.py
python nmse_frequency_response.py
```

## Performance Metrics

The project evaluates performance using:
- **Acoustic Contrast (AC)**: Sound zone separation quality
- **Normalized Mean Square Error (NMSE)**: Interpolation accuracy
- **Frequency Response Analysis**: Performance across frequency bands
