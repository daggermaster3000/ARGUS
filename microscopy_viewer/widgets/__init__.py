"""Qt dock widgets for the customised napari viewer.

Panels are declared in :mod:`microscopy_viewer.widgets.registry` and built from
there. Heavier panels are imported lazily below so that pulling in this package
does not drag in their dependencies (matplotlib, in the intensity panel's case).
"""

from .measurements_widget import MeasurementsWidget
from .metadata_widget import MetadataWidget, metadata_for_layer
from .registry import PanelSpec, get_panel, iter_panels, register_panel, unregister_panel
from .toolbar import ViewerToolbar

__all__ = [
    "IntensityComparisonWidget",
    "MeasurementsWidget",
    "MetadataWidget",
    "PanelSpec",
    "ViewerToolbar",
    "get_panel",
    "iter_panels",
    "metadata_for_layer",
    "register_panel",
    "unregister_panel",
]


def __getattr__(name: str):
    if name == "IntensityComparisonWidget":
        from .intensity_comparison import IntensityComparisonWidget

        return IntensityComparisonWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
