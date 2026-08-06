# Import adapters so they self-register via @register.
# Order matters: more-specific readers must come before generic ones.
# IsaacLabReader probes the HDF5 structure and takes priority over HDF5Reader.
from calibra.ingestion.adapters import grail, isaac_lab, hdf5, lerobot, mcap, rlds  # noqa: F401
