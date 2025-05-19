This URDF is taken from dex-urdf (https://github.com/dexsuite/dex-urdf). 

---

### Modifications to URDF

The initial `dae` meshes found in
the [shadow hand repository](https://github.com/shadow-robot/sr_common/tree/noetic-devel/sr_description/meshes/components/f_knuckle)
were identical or very similar for the `f_knuckle`. As a result, we retained only one of these. The same process was
applied to meshes of other parts, with duplicates being removed to enhance the clarity of the file structure.

For the `distal` finger, we have only kept the `pst` version, which does not include the biotac sensor.

In addition, we have increased the inertia for all fingertip links to resolve the position drive problem in SAPIEN and
IsaacGym.

Almost all meshes have been modified to improve visual quality and collision management.