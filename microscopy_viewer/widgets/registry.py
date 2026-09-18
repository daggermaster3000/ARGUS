"""Panel manifest: the list of dock widgets the viewer builds at startup.

Panels declare themselves here instead of being wired into
:meth:`~microscopy_viewer.app.MicroscopyViewer._build_docks` by hand, so adding
one is a single :func:`register_panel` call and nothing else has to change.

A panel factory receives the :class:`~microscopy_viewer.app.MicroscopyViewer`
instance and returns a ``QWidget``. Construction is lazy — the factory is only
imported and called while the window is being built — which keeps a heavy panel
(the intensity comparison pulls in matplotlib) off the import path of everything
else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class PanelSpec:
    """One dock widget, and where it belongs in the window."""

    #: Stable key used to look the panel up on the app (``app.panels[...]``).
    identifier: str
    #: Dock title shown in the tab and the Window menu.
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
    #: Start collapsed; the user can reopen it from the Window menu.
    start_hidden: bool = False
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

    def _measurements(app):
        from .measurements_widget import MeasurementsWidget

        return MeasurementsWidget(app.viewer)

    def _intensity(app):
        from .intensity_comparison import IntensityComparisonWidget

        return IntensityComparisonWidget(app.viewer)

    def _registration(app):
        from .registration_widget import RegistrationWidget

        return RegistrationWidget(app.viewer)

    def _segmentation(app):
        from .segmentation_widget import SegmentationWidget

        # The app is passed for the Batch tab, which segments the samples
        # selected in the Experiment setup panel.
        return SegmentationWidget(app.viewer, app=app)

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
            identifier="measurements",
            title="Measurements",
            factory=_measurements,
            attribute="measurements_widget",
            order=20,
        )
    )
    register_panel(
        PanelSpec(
            identifier="intensity_comparison",
            title="ROI intensity comparison",
            factory=_intensity,
            attribute="intensity_widget",
            tabify_with="measurements",
            order=30,
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
            identifier="atlas_registration",
            title="Atlas registration",
            factory=_registration,
            attribute="registration_widget",
            tabify_with="intensity_comparison",
            order=40,
        )
    )
    register_panel(
        PanelSpec(
            identifier="segmentation",
            title="Segmentation",
            factory=_segmentation,
            attribute="segmentation_widget",
            tabify_with="intensity_comparison",
            order=50,
        )
    )
    register_panel(
        PanelSpec(
            identifier="regions",
            title="Brain regions",
            factory=_regions,
            attribute="regions_widget",
            tabify_with="intensity_comparison",
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


_register_builtin_panels()
