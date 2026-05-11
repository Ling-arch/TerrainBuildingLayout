# TerrainBuildingLayout
Generate building layout automatically matched with terrain

# Terrain-Adaptive Architectural Generation

<p align="center">
  <img src="/images/volume1.gif" width="650"/>
</p>

<p align="center">
  <img src="/images/volume2.gif" width="650"/>
</p>


## Tensorfield
tensorfield with polyline constraint and point attract:
<p align="center">
  <img src="/images/tensorfield_1.png" width="400" />
  <img src="/images/tensorfield_2.png" width="400" />
  <br>*Tensor Field with Polyline Constraint & Point Attraction Effect*
</p>

## Layout with Space Colonization Algorithm

Evaluate high score regions in terrain to generate attractors for a space colonization algorithm used to create road networks.

<p align="center">
  <img src="/images/layoutPlan1.png" width="400"/>
  <img src="/images/layoutBirdView1.png" width="400"/>
</p>

<p align="center">
  <sub><b>Generated Building Layout Result A - Plan View</b></sub>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <sub><b>Generated Building Layout Result A - Axonometric View</b></sub>
</p>

<br>

<p align="center">
  <img src="/images/layoutPlan2.png" width="400"/>
  <img src="/images/layoutBirdView2.png" width="400"/>
</p>

<p align="center">
  <sub><b>Generated Building Layout Result B - Plan View</b></sub>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <sub><b>Generated Building Layout Result B - Axonometric View</b></sub>
</p>

## Terrain-Adaptive Building Mass Generation Based on Automatic Differentiation

Optimization-driven terrain-adaptive building mass generation using differentiable procedural modeling and automatic differentiation.

<!-- ======================= -->
<!-- Input Terrain -->
<!-- ======================= -->

<p align="center">
  <img src="/images/terrainA.png" width="400"/>
  <img src="/images/terrainB.png" width="400"/>
</p>

<p align="center">
  <sub><b>Input Terrain for Experiment A</b></sub>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <sub><b>Input Terrain for Experiment B</b></sub>
</p>

<br>

<!-- ======================= -->
<!-- Experiment A -->
<!-- ======================= -->

### Experiment A Results

<p align="center">
  <img src="/images/terrainA-1.png" width="260"/>
  <img src="/images/terrainA-2.png" width="260"/>
  <img src="/images/terrainA-3.png" width="260"/>
</p>

<p align="center">
  <sub><b>Initial Building Mass Layout</b></sub>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <sub><b>Terrain-Adaptive Optimization Result</b></sub>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <sub><b>Final Architectural Massing Result</b></sub>
</p>

<br>

<p align="center">
  <img src="/images/a_1.png" width="260"/>
  <img src="/images/a_2.png" width="260"/>
  <img src="/images/a_3.png" width="260"/>
</p>

<p align="center">
  <sub><b>Terrain Adaptation Loss</b></sub>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <sub><b>Topology Optimization Loss</b></sub>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <sub><b>Total Optimization Loss</b></sub>
</p>

<br>

<!-- ======================= -->
<!-- Experiment B -->
<!-- ======================= -->

### Experiment B Results

<p align="center">
  <img src="/images/terrainB-1.png" width="260"/>
  <img src="/images/terrainB-2.png" width="260"/>
  <img src="/images/terrainB-3.png" width="260"/>
</p>

<p align="center">
  <sub><b>Initial Building Mass Layout</b></sub>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <sub><b>Terrain-Adaptive Optimization Result</b></sub>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <sub><b>Final Architectural Massing Result</b></sub>
</p>

## Plan
The floor plan generate use differentiable voronoi:
<p align="center">
  <img src="/images/plan_1.png" width="400" />
  <img src="/images/plan_2.png" width="400" />
  <br>*Differentiable Voronoi Generated Floor Plan Layouts*
</p>