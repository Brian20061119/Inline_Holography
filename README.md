This project identifies individual droplets across frames, tracks their spatial coordinates, and generates comprehensive position maps.<div align="center">
  <img src="https://github.com/user-attachments/assets/bdbf971c-bca0-458f-80aa-ee4e2b72a696" alt="Droplet Tracking Visualization" width="25%">
  <p><i>Visualization of the automated detection and coordinate mapping process.</i></p>
</div>

Key Functions

Pre-processing & Normalization

Normalization(): Custom function to calibrate raw holographic intensity and remove sensor bias.

imgaussfilt(): Applies Gaussian smoothing kernel to suppress high-frequency digital noise.

Two-Stage Approach

Coarse Scan: Reconstructs the complex wave field at discrete, large intervals to identify focal neighborhoods.

fminbnd(): Optimization engine using Golden Section Search and Parabolic Interpolation for precise axial positioning.
