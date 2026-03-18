# Feature Extraction

## Overview

Extract quantitative features from segmentation masks for tumor aggressiveness classification. Feature selection differs between datasets based on biological structure and computational constraints.

---

## Challenge: 3D Surface Area Calculation

Most features are straightforward (volume = voxel count × voxel size). Surface area computation is non-trivial due to:
1. **Anisotropic resolution:** Z-spacing typically larger than XY-spacing in microscopy
2. **No native Python function** for 3D surface area on voxel grids

---

## Approach 1: Boundary Voxel Counting (Failed)

### Algorithm

1. Isotropic resampling: Interpolate to match Z-resolution to XY-resolution
2. Boundary detection: Count foreground voxels adjacent to background
3. Surface area: Boundary voxel count × voxel face area

### Failure Analysis

Fails validation on synthetic spheres - computed surface area deviates significantly from analytical solution (4πr²).

**Root cause:** Voxel grid discretization creates staircase artifacts at curved boundaries. Boundary voxel count overestimates true surface area.

---

## Approach 2: Marching Cubes (Current)

### Algorithm

1. Isotropic resampling: Interpolate to match Z-resolution to XY-resolution
2. Surface mesh generation: Marching Cubes algorithm extracts triangular mesh approximating isosurface
3. Surface area: Sum of triangle areas in mesh

### Rationale

**Problem:** Segmentation mask is discrete; real tumor boundary is smooth

**Solution:** Marching Cubes reconstructs continuous surface by interpolating within voxels and approximating local surface patches with triangles.

**Advantage:** Triangular mesh better approximates curved surfaces, reduces staircase artifacts.

### Validation

Tested on synthetic spheres: computed surface area matches analytical solution (4πr²) within acceptable error tolerance.

---

## Dataset-Specific Feature Sets

### System A: Animal Model

**Biological structure:**
- Large injection-site tumor + smaller metastatic tumors
- Distinct individual objects with measurable geometry

**Computed features (per sample):**
- Total tumor volume
- Number of tumor objects
- Volume statistics: largest tumor, second-largest tumor, median volume
- Surface area statistics (using marching cubes)
- Sphericity metrics for largest objects

**Rationale:**
- Individual tumor size and shape characteristics meaningful
- Sufficient samples (~dozens) for object-level statistics
- Computational cost acceptable (few large objects per sample)

---

### System B: Organoid Co-culture

**Biological structure:**
- Tens of thousands of small invading cells (individual dots)
- No large metastatic masses - invasion is dispersed

**Challenge:**
Marching cubes on tens of thousands of small objects is computationally prohibitive even after optimization (minutes-hours per sample).

**Computed features (per sample):**
- **Population-level metrics:**
  - Total invading celss volume
  - number of invading celld
  
- **Top-10 largest objects:**
  - Volume statistics for 10 largest invading cell clusters
  - Surface area (marching cubes applied only to these)

**Findings:**
- Top-10 object statistics less informative - invasion is diffuse, not dominated by few large objects
- Simple aggregate metrics outperformed complex geometric analysis

**Rationale for population focus:**
- Invasion pattern is collective behavior of many small cells, not individual cell characteristics
- Computational constraints: cannot apply marching cubes to all objects
- High biological variability: aggregate statistics more robust than individual measurements


## Pipeline
```
Segmentation Masks (3D)
    ↓
┌─────────────────────────────────────────────────┐
│ System A: Object-level analysis                 │
│   - Few large tumors (1-10 per sample)          │
│   - Isotropic resampling                        │
│   - Marching cubes on all objects               │
│   - Volume, surface area, sphericity            │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ System B: Population-level analysis             │
│   - Tens of thousands of small objects          │
│   - Total volumes                               │
│   -Top-10 objects geometric stats               │
│                                                 │
│                                                 │
└─────────────────────────────────────────────────┘
    ↓
Feature Vector → Classification
```

---

## Key Design Principles

**1. Match features to biological structure:**
- Large distinct objects → individual geometric analysis
- Diffuse small objects → population-level statistics

**2. Computational feasibility:**
- Marching cubes viable for ~10 objects/sample
- Prohibitive for 10,000+ objects/sample
- Choose algorithms that scale to your data

**3. Information content over complexity:**
- Complex geometric features (sphericity, surface area) not always more informative
- Simple statistics (mean intensity, total volume) can outperform in high-variability systems

---

## Next Steps

Upon availability of clinical outcome data:
- Statistical analysis to identify features correlated with aggressiveness
- Train feature based classifier (svm, logestic regression, random forest)
