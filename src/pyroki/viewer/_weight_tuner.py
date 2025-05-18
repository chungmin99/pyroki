"""Provides a class for interactively tuning named weights using Viser GUIs."""

from __future__ import annotations

from typing import cast

import jax
import viser


class WeightTuner:
    """Creates and manages a set of Viser GUI sliders for tuning named weights.

    This class simplifies the process of adding interactive controls to a Viser
    application, typically used for adjusting weights in optimization problems
    (like inverse kinematics) or other numerical parameters in real-time.
    The sliders are grouped within a Viser GUI folder.
    """

    _server: viser.ViserServer
    _weight_handles: dict[str, viser.GuiSliderHandle]

    _min: dict[str, float]
    _max: dict[str, float]
    _step: dict[str, float]
    _default: dict[str, float]

    def __init__(
        self,
        server: viser.ViserServer,
        default: dict[str, float],
        *,
        folder_name: str = "Costs",
        min: dict[str, float] | None = None,
        max: dict[str, float] | None = None,
        step: dict[str, float] | None = None,
        default_min: float = 0.0,
        default_max: float = 100.0,
        default_step: float = 0.01,
    ):
        """Initializes the tuner and creates the Viser GUI sliders.
        The sliders show up in the order of the keys in the `default` dictionary.
        `default` must be a dictionary.

        Args:
            server: The Viser server instance.
            default: An instance of the dataclass defining the weights.
            folder_name: Name of the Viser GUI folder to contain the sliders.
            min: Minimum value for each slider.
            max: Maximum value for each slider.
            step: Step size for each slider.
            default_min: Minimum value for all sliders. `min` overrides this.
            default_max: Maximum value for all sliders. `max` overrides this.
            default_step: Step size for all sliders. `step` overrides this.
        """
        leaves = jax.tree.leaves(default)
        assert all(isinstance(leaf, (int, float)) for leaf in leaves), (
            "All default parameters must be ints or floats."
        )
        assert isinstance(default, dict)

        self._server = server
        self._weight_handles = {}
        self._max = jax.tree.map(lambda _: default_max, default)
        if max is not None:
            for key, max_val in max.items():
                cast(dict, self._max)[key] = max_val

        self._min = jax.tree.map(lambda _: default_min, default)
        if min is not None:
            for key, min_val in min.items():
                cast(dict, self._min)[key] = min_val

        self._step = jax.tree.map(lambda _: default_step, default)
        if step is not None:
            for key, step_val in step.items():
                cast(dict, self._step)[key] = step_val

        self._default = default

        with server.gui.add_folder(folder_name):
            for field, default_weight in default.items():
                self._weight_handles[field] = server.gui.add_slider(
                    field,
                    min=self._min[field],
                    max=self._max[field],
                    step=self._step[field],
                    initial_value=default_weight,
                )

            reset_button = server.gui.add_button("Reset Weights")
            reset_button.on_click(lambda _: self.reset_weights())

    def get_weights(self) -> dict[str, float]:
        """Retrieves the current values of all tracked weights from the GUI sliders.

        Returns:
            A dictionary mapping weight names to their current float values
            as set by the sliders.
        """
        return {
            field: handle.value
            for field, handle in zip(
                self._weight_handles.keys(),
                self._weight_handles.values(),
            )
        }

    def reset_weights(self):
        """Resets all weights to their initial values."""
        for name, handle in self._weight_handles.items():
            handle.value = self._default[name]
