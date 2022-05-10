"""General design GNN pipeline using iprotein-ligand interaction graph"""

# Add imports here
from .interaction_fingerprints_based_graph import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
