from __future__ import annotations
from dataclasses import dataclass
import json
import os
from typing import Optional

@dataclass
class DataPoint:
    """
    This class is a representation of a single data point in the tensorboard json data.
    """
    timestamp: float
    x_value: float
    y_value: float

@dataclass
class LilaqPlot:
    """
    This class is responsible for anything to do with:
    - lq.plot()
    """

    label: str
    points: list[DataPoint]
    every: Optional[int]

    def downsample(
        self, factor: int
    ) -> LilaqPlot:
        # Downsamples each time series by the given factor
        if factor <= 1:
            return self
        
        ds_points: list[DataPoint] = []
        for i in range(0, len(self.points), factor):
            ds_points.append(self.points[i])
        self.points = ds_points
        return self
    
    def smooth(
        self, window: int
    ) -> LilaqPlot:
        if window <= 1:
            return self

        half: int = window // 2

        smoothed_points: list[DataPoint] = []
        for i in range(len(self.points)):
            start: int = max(0, i - half)
            end: int = min(len(self.points), i + half + 1)
            window_points: list[DataPoint] = self.points[start:end]

            avg_x: float = sum(p.x_value for p in window_points) / len(window_points)
            avg_y: float = sum(p.y_value for p in window_points) / len(window_points)

            smoothed_points.append(
                DataPoint(
                    timestamp=self.points[i].timestamp,
                    x_value=avg_x,
                    y_value=avg_y,
                )
            )
        self.points = smoothed_points
        return self  
    
    def _points_to_lilaq_str(self, indent_level: int) -> str:
        lines_x = [
            f'{indent(indent_level)}/* X */', 
            f'{indent(indent_level)}(',            
        ]
        lines_y = [
            f'{indent(indent_level)}/* Y */', 
            f'{indent(indent_level)}(',
        ]

        for point in self.points:
            lines_x.append(f'{indent(indent_level + 1)}{point.x_value},')
            lines_y.append(f'{indent(indent_level + 1)}{point.y_value},')
        
        lines_x.append(f'{indent(indent_level)}),')
        lines_y.append(f'{indent(indent_level)})')

        str_x = "\n".join(lines_x)
        str_y = "\n".join(lines_y)

        lines = [
            str_x,
            '',
            str_y,
        ]

        return "\n".join(lines)

    def to_str(self, indent_level: int) -> str:
        points_str = self._points_to_lilaq_str(indent_level=indent_level + 1)

        lines = [
            f'{indent(indent_level)}lq.plot(',
            f'{indent(indent_level + 1)}label: [{self.label}],',
            f'{indent(indent_level + 1)}every: {self.every if self.every else "none"},',
            f'{points_str},',
            f'{indent(indent_level)})',
        ]

        return "\n".join(lines)

@dataclass
class LilaqLimit:
    """
    This class is responsible for anything to do with:
    - xlim: (min, max)
    - ylim: (min, max)
    """
    min: Optional[float]
    max: Optional[float]

    def to_str(self) -> str:
        return f"({self.min if self.min is not None else 'auto'}, {self.max if self.max is not None else 'auto'})"


@dataclass
class LilaqDiagram:
    """
    This class is responsible for anything to do with:
    - lq.diagram()
    """
    xlim: LilaqLimit
    xlabel: str

    ylim: LilaqLimit
    ylabel: str

    series: list[LilaqPlot]

    def _series_to_lilaq_str(self, indent_level: int) -> str:
        return ",\n\n".join(s.to_str(indent_level=indent_level) for s in self.series)

    def to_str(self, indent_level: int) -> str:
        lines = [            
            f'{indent(indent_level)}#import "@preview/lilaq:0.5.0" as lq',
            f'{indent(indent_level)}#figure(',
            f'{indent(indent_level + 1)}caption: [GIVE ME A CAPTION],',
            f'{indent(indent_level + 1)}lq.diagram(',
            f'{indent(indent_level + 2)}legend: (position: bottom + center),',
            f'{indent(indent_level + 2)}xlim: {self.xlim.to_str()},',
            f'{indent(indent_level + 2)}xlabel: [{self.xlabel}],',
            '',
            f'{indent(indent_level + 2)}ylim: {self.ylim.to_str()},',
            f'{indent(indent_level + 2)}ylabel: [{self.ylabel}],',
            '',
            f'{self._series_to_lilaq_str(indent_level + 2)}',
            f'{indent(indent_level + 1)})',            
            f'{indent(indent_level)})',
        ]

        return "\n".join(lines)
    
    def downsample(
        self, factor: int
    ) -> LilaqDiagram:
        for s in self.series:
            s.downsample(factor)
        return self
    
    def smooth(
        self, window: int
    ) -> LilaqDiagram:
        for s in self.series:
            s.smooth(window)
        return self
    
    def has_multiple(self) -> bool:
        return len(self.series) > 1

def load_json_into_plots(
    directory: str, experiments: list[str], metric: str, every: Optional[int]
) -> list[LilaqPlot]:
    plots: list[LilaqPlot] = []

    for exp in experiments:
        filename: str = f"{exp}.{metric}.json"
        path: str = os.path.join(directory, filename)

        if not os.path.isfile(path):
            continue

        plot = LilaqPlot(label=exp.capitalize(), points=[], every=every)
        with open(path, "r", encoding="utf-8") as f:
            raw: list[list[float]] = json.load(f)
            plot.points.extend(
                DataPoint(entry[0], entry[1], entry[2]) for entry in raw
            )
            plots.append(plot)

    return plots

def generate_lilaq_string(
    directory: str,
    experiments: list[str],
    metric: str,
    resolution: int = 1,
    smooth_window: int = 1,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    every: Optional[int] = None,
) -> str:
    
    diagram = LilaqDiagram(
        xlim=LilaqLimit(min=x_min, max=x_max),
        xlabel="Episodes",
        ylim=LilaqLimit(min=y_min, max=y_max),
        ylabel=metric.capitalize(),
        series=load_json_into_plots(directory, experiments, metric, every),
    ).downsample(resolution).smooth(smooth_window)

    return diagram.to_str(indent_level=0)

def indent(level: int) -> str:
    return "\t" * level
