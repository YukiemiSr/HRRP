"""Conditional Diffusion Transformers (DiT) for HRRP view synthesis.

The package implements the DiT architecture of

    W. Peebles and S. Xie, "Scalable Diffusion Models with Transformers",
    ICCV 2023,

adapted to one-dimensional radar range profiles (HRRPs).  The generation task
is

    x(a_t)  ~  p( . | x(a_o), a_o, a_t),

i.e. given a measured HRRP at the original azimuth ``a_o`` and a requested
target azimuth ``a_t``, synthesise the HRRP of the *same target* at ``a_t``.
"""

__all__ = ["data", "model", "diffusion", "classifier", "metrics"]
__version__ = "1.0.0"
