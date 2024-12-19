# CFG Adapter

## Preliminary results

The following shows samples from trained models without CFG, with CFG, and with the adapter. The sampling time of no CFG and with adapter is approximately equivalent. All samples were generated with the same seed for direct comparison.

### DiT-XL/2 (256x256)

<details>
  <summary>Macaw</summary>
  <p align="center">
    <img src="visuals/dit256/no-cfg-88.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/dit256/cfg-88.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/dit256/adapter-88.png" width="320" />
    <p align="center"><b>Fig 1.</b> Samples of DiT-XL/2 (256x256) without CFG (left), with CFG (center), with CFG adapter (right) of <em>Macaw</em> (label 88).</p>
  </p>
</details>

<details>
  <summary>Border Collie</summary>
  <p align="center">
    <img src="visuals/dit256/no-cfg-232.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/dit256/cfg-232.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/dit256/adapter-232.png" width="320" />
    <p align="center"><b>Fig 2.</b> Samples of DiT-XL/2 (256x256) without CFG (left), with CFG (center), with CFG adapter (right) of <em>Border Collie</em> (label 232).</p>
  </p>
</details>

<details>
  <summary>Mushroom</summary>
  <p align="center">
    <img src="visuals/dit256/no-cfg-947.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/dit256/cfg-947.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/dit256/adapter-947.png" width="320" />
    <p align="center"><b>Fig 3.</b> Samples of DiT-XL/2 (256x256) without CFG (left), with CFG (center), with CFG adapter (right) of <em>Mushroom</em> (label 947).</p>
  </p>
</details>

<details>
  <summary>Alp</summary>
  <p align="center">
    <img src="visuals/dit256/no-cfg-970.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/dit256/cfg-970.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/dit256/adapter-970.png" width="320" />
    <p align="center"><b>Fig 4.</b> Samples of DiT-XL/2 (256x256) without CFG (left), with CFG (center), with CFG adapter (right) of <em>Alp</em> (label 970).</p>
  </p>
</details>

### DiT-XL/2 (512x512)

<details>
  <summary>Macaw</summary>
  <p align="center">
    <img src="visuals/dit512/no-cfg-88.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/dit512/cfg-88.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/dit512/adapter-88.png" width="320" />
    <p align="center"><b>Fig 5.</b> Samples of DiT-XL/2 (512x512) without CFG (left), with CFG (center), with CFG adapter (right) of <em>Macaw</em> (label 88).</p>
  </p>
</details>

<details>
  <summary>Border Collie</summary>
  <p align="center">
    <img src="visuals/dit512/no-cfg-232.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/dit512/cfg-232.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/dit512/adapter-232.png" width="320" />
    <p align="center"><b>Fig 6.</b> Samples of DiT-XL/2 (512x512) without CFG (left), with CFG (center), with CFG adapter (right) of <em>Border Collie</em> (label 232).</p>
  </p>
</details>

<details>
  <summary>Mushroom</summary>
  <p align="center">
    <img src="visuals/dit512/no-cfg-947.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/dit512/cfg-947.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/dit512/adapter-947.png" width="320" />
    <p align="center"><b>Fig 7.</b> Samples of DiT-XL/2 (512x512) without CFG (left), with CFG (center), with CFG adapter (right) of <em>Mushroom</em> (label 947).</p>
  </p>
</details>

<details>
  <summary>Alp</summary>
  <p align="center">
    <img src="visuals/dit512/no-cfg-970.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/dit512/cfg-970.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/dit512/adapter-970.png" width="320" />
    <p align="center"><b>Fig 8.</b> Samples of DiT-XL/2 (512x512) without CFG (left), with CFG (center), with CFG adapter (right) of <em>Alp</em> (label 970).</p>
  </p>
</details>

### Stable Diffusion 2.1

<details>
  <summary><em>"A young badger delicately sniffing a yellow rose, richly textured oil painting."</em></summary>
  <p align="center">
    <img src="visuals/sd2.1/no-cfg-badger.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sd2.1/cfg-badger.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sd2.1/adapter-badger.png" width="320" />
    <p align="center"><b>Fig 9.</b> Samples of Stable Diffusion 2.1 without CFG (left), with CFG (center), with CFG adapter (right).</p>
  </p>
</details>

<details>
  <summary><em>"An empty fireplace with a television above it. The TV shows a lion hugging a giraffe."</em></summary>
  <p align="center">
    <img src="visuals/sd2.1/no-cfg-fireplace.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sd2.1/cfg-fireplace.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sd2.1/adapter-fireplace.png" width="320" />
    <p align="center"><b>Fig 10.</b> Samples of Stable Diffusion 2.1 without CFG (left), with CFG (center), with CFG adapter (right).</p>
  </p>
</details>

<details>
  <summary><em>"Monster Baba yaga house with in a forest, dark horror style, black and white."</em></summary>
  <p align="center">
    <img src="visuals/sd2.1/no-cfg-house.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sd2.1/cfg-house.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sd2.1/adapter-house.png" width="320" />
    <p align="center"><b>Fig 11.</b> Samples of Stable Diffusion 2.1 without CFG (left), with CFG (center), with CFG adapter (right).</p>
  </p>
</details>

<details>
  <summary><em>"A grand piano with a white bench."</em></summary>
  <p align="center">
    <img src="visuals/sd2.1/no-cfg-piano.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sd2.1/cfg-piano.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sd2.1/adapter-piano.png" width="320" />
    <p align="center"><b>Fig 12.</b> Samples of Stable Diffusion 2.1 without CFG (left), with CFG (center), with CFG adapter (right).</p>
  </p>
</details>

### Stable Diffusion XL

<details>
  <summary><em>"An astronaut riding a pig, highly realistic dslr photo, cinematic shot."</em></summary>
  <p align="center">
    <img src="visuals/sdxl/no-cfg-astronaut.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sdxl/cfg-astronaut.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sdxl/adapter-astronaut.png" width="320" />
    <p align="center"><b>Fig 13.</b> Samples of Stable Diffusion XL without CFG (left), with CFG (center), with CFG adapter (right).</p>
  </p>
</details>

<details>
  <summary><em>"A young badger delicately sniffing a yellow rose, richly textured oil painting."</em></summary>
  <p align="center">
    <img src="visuals/sdxl/no-cfg-badger.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sdxl/cfg-badger.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sdxl/adapter-badger.png" width="320" />
    <p align="center"><b>Fig 14.</b> Samples of Stable Diffusion XL without CFG (left), with CFG (center), with CFG adapter (right).</p>
  </p>
</details>

<details>
  <summary><em>"A capybara made of voxels sitting in a field."</em></summary>
  <p align="center">
    <img src="visuals/sdxl/no-cfg-capybara.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sdxl/cfg-capybara.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sdxl/adapter-capybara.png" width="320" />
    <p align="center"><b>Fig 15.</b> Samples of Stable Diffusion XL without CFG (left), with CFG (center), with CFG adapter (right).</p>
  </p>
</details>

<details>
  <summary><em>"Cat patting a crystal ball with the number 7 written on it in black marker."</em></summary>
  <p align="center">
    <img src="visuals/sdxl/no-cfg-cat.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sdxl/cfg-cat.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sdxl/adapter-cat.png" width="320" />
    <p align="center"><b>Fig 16.</b> Samples of Stable Diffusion XL without CFG (left), with CFG (center), with CFG adapter (right).</p>
  </p>
</details>

<details>
  <summary><em>"An empty fireplace with a television above it. The TV shows a lion hugging a giraffe."</em></summary>
  <p align="center">
    <img src="visuals/sdxl/no-cfg-fireplace.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sdxl/cfg-fireplace.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sdxl/adapter-fireplace.png" width="320" />
    <p align="center"><b>Fig 17.</b> Samples of Stable Diffusion XL without CFG (left), with CFG (center), with CFG adapter (right).</p>
  </p>
</details>

<details>
  <summary><em>"Monster Baba yaga house with in a forest, dark horror style, black and white."</em></summary>
  <p align="center">
    <img src="visuals/sdxl/no-cfg-house.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sdxl/cfg-house.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sdxl/adapter-house.png" width="320" />
    <p align="center"><b>Fig 18.</b> Samples of Stable Diffusion XL without CFG (left), with CFG (center), with CFG adapter (right).</p>
  </p>
</details>

<details>
  <summary><em>"A close up of a handpalm with leaves growing from it."</em></summary>
  <p align="center">
    <img src="visuals/sdxl/no-cfg-palm.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sdxl/cfg-palm.png" width="320" />
    &nbsp; &nbsp;
    <img src="visuals/sdxl/adapter-palm.png" width="320" />
    <p align="center"><b>Fig 19.</b> Samples of Stable Diffusion XL without CFG (left), with CFG (center), with CFG adapter (right).</p>
  </p>
</details>
