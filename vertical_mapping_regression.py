#!/usr/bin/env python3
"""
vertical_mapping_regression.py

Utility to fit a regression function to existing vertical mapping points and
optionally normalize inputs. This lets you replace the discrete point lookup
with a smooth polynomial model saved back to JSON.

Example:
    python vertical_mapping_regression.py \
        --input vertical_mapping3.json \
        --output vertical_mapping_regressed.json \
        --degree 3

The output JSON stores the fitted coefficients and normalization metadata, and
includes a few sample predictions for quick inspection.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np

Number = Union[int, float]


@dataclass
class RegressionModel:
    degree: int
    coefficients: List[float]
    x_min: float
    x_max: float
    wheel_min: float
    wheel_max: float
    r2: float

    def predict(self, y_norm: Union[Number, Sequence[Number]]) -> np.ndarray:
        """Predict wheel degrees for one or many normalized Y values.

        Inputs outside the training range are clipped to [x_min, x_max] before
        normalization so the model cannot extrapolate wildly.
        """
        x = np.asarray(y_norm, dtype=float)
        x_clipped = np.clip(x, self.x_min, self.x_max)
        x_norm = (x_clipped - self.x_min) / max(self.x_max - self.x_min, 1e-9)
        poly = np.poly1d(self.coefficients)
        preds = poly(x_norm)
        return np.clip(preds, self.wheel_min, self.wheel_max)


def load_mapping_points(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r") as f:
        data = json.load(f)
    points = data.get("mapping_points", [])
    if not points:
        raise ValueError(f"No mapping_points found in {path}")
    y_norm = np.array([p["arducam_y_normalized"] for p in points], dtype=float)
    wheel = np.array([p["wheel_degrees"] for p in points], dtype=float)
    return y_norm, wheel


def fit_polynomial_mapping(
    y_norm: np.ndarray, wheel: np.ndarray, degree: int = 2
) -> RegressionModel:
    if y_norm.ndim != 1 or wheel.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")
    if y_norm.shape[0] != wheel.shape[0]:
        raise ValueError("y_norm and wheel arrays must be the same length")
    if y_norm.shape[0] < degree + 1:
        raise ValueError("Not enough points to fit the requested degree")

    x_min, x_max = float(np.min(y_norm)), float(np.max(y_norm))
    wheel_min, wheel_max = float(np.min(wheel)), float(np.max(wheel))

    # Normalize X to [0,1] for a well-conditioned fit
    denom = max(x_max - x_min, 1e-9)
    x_norm = (y_norm - x_min) / denom

    coeffs = np.polyfit(x_norm, wheel, deg=degree)
    poly = np.poly1d(coeffs)
    preds = poly(x_norm)
    ss_res = float(np.sum((wheel - preds) ** 2))
    ss_tot = float(np.sum((wheel - wheel.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return RegressionModel(
        degree=degree,
        coefficients=coeffs.tolist(),
        x_min=x_min,
        x_max=x_max,
        wheel_min=wheel_min,
        wheel_max=wheel_max,
        r2=r2,
    )


def save_model(model: RegressionModel, path: Path, source_file: Path):
    payload = {
        "source_mapping": str(source_file),
        "model": asdict(model),
        "note": "Polynomial regression fitted on arducam_y_normalized → wheel_degrees",
        "sample_predictions": sample_predictions(model),
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def sample_predictions(model: RegressionModel):
    xs = np.linspace(model.x_min, model.x_max, num=5)
    preds = model.predict(xs)
    return [
        {
            "arducam_y_normalized": float(x),
            "pred_wheel_degrees": float(p),
        }
        for x, p in zip(xs, preds)
    ]


def cli(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Fit regression to vertical mapping JSON")
    parser.add_argument("--input", required=True, help="Path to existing mapping JSON")
    parser.add_argument("--output", required=True, help="Where to write regression JSON")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree (default: 2)")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    y_norm, wheel = load_mapping_points(input_path)
    model = fit_polynomial_mapping(y_norm, wheel, degree=args.degree)
    save_model(model, output_path, input_path)

    print(
        f"✅ Fitted degree-{args.degree} polynomial on {len(y_norm)} samples | R^2={model.r2:.4f}\n"
        f"   Input range: [{model.x_min:.3f}, {model.x_max:.3f}] → Output range: [{model.wheel_min:.1f}, {model.wheel_max:.1f}]\n"
        f"   Saved to: {output_path}"
    )


if __name__ == "__main__":
    cli()
