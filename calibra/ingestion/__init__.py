# Import adapters so they self-register via @register.
# Order matters: more-specific readers must come before generic ones.
# IsaacLabReader probes the HDF5 structure and takes priority over HDF5Reader.
from calibra.ingestion.adapters import isaac_lab, hdf5, lerobot, rlds, mcap, grail  # noqa: F401

