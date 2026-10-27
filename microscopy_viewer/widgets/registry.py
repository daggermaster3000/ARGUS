"""Panel manifest: the list of dock widgets the viewer builds at startup.

Panels declare themselves here instead of being wired into
:meth:`~microscopy_viewer.app.MicroscopyViewer._build_docks` by hand, so adding
one is a single :func:`register_panel` call and nothing else has to change.

A panel factory receives the :class:`~microscopy_viewer.app.MicroscopyViewer`
instance and returns a ``QWidget``. Construction is lazy — the factory is only
imported and called while the window is being built — which keeps a heavy panel
(segmentation pulls in torch) off the import path of everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class PanelSpec:
    """One dock widget, and where it belongs in the window."""

    #: Stable key used to look the panel up on the app (``app.panels[...]``).
    identifier: str
    #: Dock title shown in the tab and the Panels menu.
    title: str
    #: Builds the widget. Takes the app, returns a QWidget.
    factory: Callable[[Any], Any]
    #: napari dock area: ``left``, ``right``, ``top`` or ``bottom``.
    area: str = "right"
    #: Attribute name to expose the widget under on the app, if any.
    attribute: str | None = None
    #: Identifier of a panel to share a tab stack with.
    tabify_with: str | None = None
    #: Lower numbers are created first.
    order: int = 100
    #: Start closed; it is opened from the Panels menu. Closed is the
    #: default so the window opens on the canvas rather than on a dozen tabs,
    #: and what was open last time comes back with the session.
    start_hidden: bool = True
    #: Extra keyword arguments passed to ``add_dock_widget``.
    dock_kwargs: dict = field(default_factory=dict)


_PANELS: dict[str, PanelSpec] = {}


def register_panel(spec: PanelSpec) -> PanelSpec:
    """Add *spec* to the manifest, replacing any panel with the same identifier."""
    _PANELS[spec.identifier] = spec
    return spec


def unregister_panel(identifier: str) -> None:
    """Drop a panel from the manifest. Mainly useful in tests."""
    _PANELS.pop(identifier, None)


def iter_panels() -> list[PanelSpec]:
    """Registered panels in creation order."""
    return sorted(_PANELS.values(), key=lambda spec: (spec.order, spec.identifier))


def get_panel(identifier: str) -> PanelSpec | None:
    return _PANELS.get(identifier)


def _register_builtin_panels() -> None:
    """Declare the panels that ship with the viewer.

    Imports happen inside the factories so that importing this module stays cheap
    and free of Qt.
    """

    def _metadata(app):
        from .metadata_widget import MetadataWidget

        return MetadataWidget(app.viewer)

    def _cells(app):
        from .cells_widget import CellsWidget

        # Segmentation (one image or the Experiment selection) and the junction
        # count in one panel. It also sets app.segmentation_widget and
        # app.junction_analysis_widget, the parts other panels borrow from.
        return CellsWidget(app)

    def _regions(app):
        from .regions_widget import RegionsWidget

        # Takes the app: saving regions into a sample's .ims has to close and
        # reopen that sample, which goes through ``open_paths``.
        return RegionsWidget(app)

    def _experiment(app):
        from .experiment_widget import ExperimentWidget

        # Takes the app: opening a sample goes through ``open_paths`` and the
        # batch run borrows the segmentation panel's settings.
        return ExperimentWidget(app)

    def _analysis(app):
        from .analysis_widget import AnalysisWidget

        # Takes the app: it analyses whatever the experiment panel has selected.
        return AnalysisWidget(app)

    def _cell_counter(app):
        from .cell_counter_widget import CellCounterWidget

        # Takes the app: saving closes and reopens the sample through open_paths.
        return CellCounterWidget(app)

    def _fusion(app):
        from .fusion_widget import FusionWidget

        # Takes the app: saving into an .ims closes and reopens the sample.
        return FusionWidget(app)

    def _positive_counter(app):
        from .positive_counter_widget import PositiveCounterWidget

        # Takes the app: saving into an .ims closes and reopens the sample.
        return PositiveCounterWidget(app)

    def _junction_review(app):
        from .junction_review_widget import JunctionReviewWidget

        # Takes the app: it opens samples, and drives the Fusion region panel.
        return JunctionReviewWidget(app)

    def _timeseries(app):
        from .timeseries_widget import TimeSeriesWidget

        # Takes the app rather than the viewer: playback reads the local-cache
        # state off ``app.timeline_manager`` and exports remember the directory.
        return TimeSeriesWidget(app)

    register_panel(
        PanelSpec(
            identifier="metadata",
            title="Acquisition metadata",
            factory=_metadata,
            attribute="metadata_widget",
            order=10,
        )
    )
    register_panel(
        PanelSpec(
            identifier="timeseries",
            title="Time series",
            factory=_timeseries,
            attribute="timeseries_widget",
            # Transport controls belong under the canvas, not beside it: the
            # panel is one row of buttons and would waste a whole right-hand dock.
            area="bottom",
            order=35,
        )
    )
    register_panel(
        PanelSpec(
            identifier="segmentation",
            title="Cells",
            factory=_cells,
            attribute="cells_widget",
            # The first of the right-hand tab stack; the others tabify with it.
            order=50,
        )
    )
    register_panel(
        PanelSpec(
            identifier="regions",
            title="Brain regions",
            factory=_regions,
            attribute="regions_widget",
            tabify_with="segmentation",
            order=60,
        )
    )
    register_panel(
        PanelSpec(
            identifier="experiment",
            title="Experiment setup",
            factory=_experiment,
            attribute="experiment_widget",
            # Its own dock rather than another tab on the right: the thumbnail
            # grid is the one panel that wants width, and it is where a session
            # starts rather than something consulted mid-analysis.
            area="left",
            order=5,
            # The one panel open from the start: it is how samples are reached.
            start_hidden=False,
        )
    )
    register_panel(
        PanelSpec(
            identifier="analysis",
            title="Analysis",
            factory=_analysis,
            attribute="analysis_widget",
            # Beside the experiment panel: it works off that panel's selection.
            area="left",
            tabify_with="experiment",
            order=6,
        )
    )

    register_panel(
        PanelSpec(
            identifier="cell_counter",
            title="Cell counter",
            factory=_cell_counter,
            attribute="cell_counter_widget",
            tabify_with="segmentation",
            order=55,
        )
    )
    register_panel(
        PanelSpec(
            identifier="fusion_region",
            title="Fusion region macro",
            factory=_fusion,
            attribute="fusion_widget",
            tabify_with="segmentation",
            order=57,
        )
    )
    register_panel(
        PanelSpec(
            identifier="positive_counter",
            title="Manual positive count",
            factory=_positive_counter,
            attribute="positive_counter_widget",
            tabify_with="segmentation",
            order=56,
        )
    )
    register_panel(
        PanelSpec(
            identifier="junction_review",
            title="Junction review",
            factory=_junction_review,
            attribute="junction_review_widget",
            tabify_with="segmentation",
            order=59,
            # No stretch under the panel: the sample list is what takes the height.
            dock_kwargs={"add_vertical_stretch": False},
        )
    )


_register_builtin_panels()
