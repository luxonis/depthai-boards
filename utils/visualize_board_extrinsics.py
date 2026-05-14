#!/usr/bin/env python3
"""Visualize camera and IMU coordinate frames from a depthai_boards JSON file.

Usage:
    python3 utils/visualize_board_extrinsics.py \
        OAK-D-S2-POE.json

    python3 utils/visualize_board_extrinsics.py \
        boards/OAK-D-S2-POE.json

The script:
- Loads camera and IMU extrinsics from a board JSON
- Resolves all frames into one common reference frame
- Prints the resulting poses
- Draws a 3D visualization of frame origins and axis orientation

It accepts both legacy Euler-style `"rotation": {"r","p","y"}` and
matrix-style `"rotationMatrix": [[...], [...], [...]]`.

Printed poses use the raw resolved coordinates directly from JSON.
For plotting, coordinates are shown in a graph-friendly frame:
- graph X = raw X
- graph Y = raw Z
- graph Z = -raw Y

This keeps the graph axes as:
- X: left-right
- Y: forward-back
- Z: up-down
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _rot_x(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _rot_y(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _rot_z(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def parse_rotation(extrinsics: dict) -> np.ndarray:
    if "rotationMatrix" in extrinsics:
        mat = np.array(extrinsics["rotationMatrix"], dtype=float)
        if mat.shape != (3, 3):
            raise ValueError(f"rotationMatrix must be 3x3, got {mat.shape}")
        return mat

    if "rotation" in extrinsics:
        rot = extrinsics["rotation"]
        roll = float(rot.get("r", 0.0))
        pitch = float(rot.get("p", 0.0))
        yaw = float(rot.get("y", 0.0))
        # DepthAI JSON historically stores roll/pitch/yaw in degrees.
        return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)

    return np.eye(3, dtype=float)


def parse_translation(extrinsics: dict) -> np.ndarray:
    t = extrinsics.get("specTranslation", {})
    return np.array(
        [float(t.get("x", 0.0)), float(t.get("y", 0.0)), float(t.get("z", 0.0))],
        dtype=float,
    )


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=float)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    inv = np.eye(4, dtype=float)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ trans
    return inv


def map_for_plot(vec: np.ndarray) -> np.ndarray:
    return np.array([vec[0], vec[2], -vec[1]], dtype=float)


def normalized(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.zeros_like(vec)
    return vec / norm


@dataclass
class FrameSpec:
    key: str
    label: str
    kind: str


@dataclass
class LinkSpec:
    src: str
    dst: str
    transform_src_to_dst: np.ndarray
    label: str


def build_graph(board_config: dict) -> Tuple[Dict[str, FrameSpec], Dict[str, List[Tuple[str, np.ndarray]]], List[LinkSpec]]:
    frames: Dict[str, FrameSpec] = {}
    edges: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    links: List[LinkSpec] = []

    def add_frame(key: str, label: str, kind: str) -> None:
        frames[key] = FrameSpec(key=key, label=label, kind=kind)
        edges.setdefault(key, [])

    def add_edge(src: str, dst: str, transform_src_to_dst: np.ndarray) -> None:
        # Board JSON extrinsics follow the DepthAI convention used by calibration:
        # the stored transform maps points from `src` into the `dst` frame.
        #
        # If the destination pose is known in world coordinates, the source pose is:
        #   world_T_src = world_T_dst @ src_T_dst
        #
        # So traversal from `dst -> src` uses the direct transform, while
        # traversal from `src -> dst` uses the inverse.
        edges.setdefault(src, []).append((dst, invert_transform(transform_src_to_dst)))
        edges.setdefault(dst, []).append((src, transform_src_to_dst))

    cameras = board_config.get("cameras", {})
    for cam_key, cam_data in cameras.items():
        add_frame(cam_key, f"{cam_key} ({cam_data.get('name', 'camera')})", "camera")

    for cam_key, cam_data in cameras.items():
        extr = cam_data.get("extrinsics")
        if not extr:
            continue
        dst = extr.get("to_cam")
        if dst not in cameras:
            continue
        transform = make_transform(parse_rotation(extr), parse_translation(extr))
        add_edge(cam_key, dst, transform)
        links.append(LinkSpec(cam_key, dst, transform, f"{cam_key} -> {dst}"))

    imu_sensors = board_config.get("imuExtrinsics", {}).get("sensors", {})
    for sensor_key, sensor_data in imu_sensors.items():
        frame_key = f"IMU:{sensor_key}"
        add_frame(frame_key, f"{sensor_key} ({sensor_data.get('name', 'imu')})", "imu")

        extr = sensor_data.get("extrinsics", {})
        dst = extr.get("to_cam")
        if dst not in cameras:
            continue
        transform = make_transform(parse_rotation(extr), parse_translation(extr))
        add_edge(frame_key, dst, transform)
        links.append(LinkSpec(frame_key, dst, transform, f"{sensor_key} -> {dst}"))

    return frames, edges, links


def resolve_poses(frames: Dict[str, FrameSpec], edges: Dict[str, List[Tuple[str, np.ndarray]]], reference: str) -> Dict[str, np.ndarray]:
    if reference not in frames:
        raise KeyError(f"Reference frame '{reference}' not found")

    poses: Dict[str, np.ndarray] = {reference: np.eye(4, dtype=float)}
    queue: deque[str] = deque([reference])

    while queue:
        src = queue.popleft()
        world_t_src = poses[src]
        for dst, src_t_dst in edges.get(src, []):
            if dst in poses:
                continue
            poses[dst] = world_t_src @ src_t_dst
            queue.append(dst)

    return poses


def choose_reference(board_config: dict, requested: str | None) -> str:
    if requested:
        return requested

    cameras = board_config.get("cameras", {})
    for preferred in ("CAM_A", "CAM_B", "CAM_C"):
        if preferred in cameras:
            return preferred
    if cameras:
        return next(iter(cameras))

    imu_sensors = board_config.get("imuExtrinsics", {}).get("sensors", {})
    if imu_sensors:
        return f"IMU:{next(iter(imu_sensors))}"

    raise RuntimeError("No camera or IMU frames found in board JSON")


def print_poses(frames: Dict[str, FrameSpec], poses: Dict[str, np.ndarray], reference: str) -> None:
    print(f"Reference frame: {reference}")
    print()
    for key in sorted(poses):
        pose = poses[key]
        pos = pose[:3, 3]
        rot = pose[:3, :3]
        print(f"{frames[key].label}")
        print(f"  position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        print("  rotationMatrix:")
        for row in rot:
            print(f"    [{row[0]: .4f}, {row[1]: .4f}, {row[2]: .4f}]")
        print()


def print_links(frames: Dict[str, FrameSpec], links: List[LinkSpec]) -> None:
    print("Flashed links")
    print()
    for link in links:
        rot = link.transform_src_to_dst[:3, :3]
        trans = link.transform_src_to_dst[:3, 3]
        print(f"{frames[link.src].label} -> {frames[link.dst].label}")
        print("  Applies as: x' = M x")
        print("  Meaning: coordinates expressed in the source frame are transformed into the `to_cam` frame.")
        print(f"  translation: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]")
        print("  rotationMatrix:")
        for row in rot:
            print(f"    [{row[0]: .4f}, {row[1]: .4f}, {row[2]: .4f}]")
        print()


def validate_links(
    frames: Dict[str, FrameSpec],
    poses: Dict[str, np.ndarray],
    links: List[LinkSpec],
    atol: float = 1e-5,
) -> None:
    print("Consistency check")
    print()
    had_issue = False

    for link in links:
        if link.src not in poses or link.dst not in poses:
            continue

        predicted_src = poses[link.dst] @ link.transform_src_to_dst
        actual_src = poses[link.src]

        rot_err = np.max(np.abs(predicted_src[:3, :3] - actual_src[:3, :3]))
        trans_err = np.max(np.abs(predicted_src[:3, 3] - actual_src[:3, 3]))

        status = "OK" if rot_err <= atol and trans_err <= atol else "MISMATCH"
        print(
            f"{status}: {frames[link.src].label} -> {frames[link.dst].label}  "
            f"(rot_err={rot_err:.3e}, trans_err={trans_err:.3e})"
        )
        if status != "OK":
            had_issue = True

    if not links:
        print("No flashed links found.")
    elif not had_issue:
        print("All resolved links are self-consistent with the chosen reference.")
    print()


def set_equal_axes(ax: plt.Axes, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_poses(
    board_name: str,
    frames: Dict[str, FrameSpec],
    poses: Dict[str, np.ndarray],
    links: List[LinkSpec],
    axis_length: float,
    show_labels: bool,
    save_path: Path | None,
    show_plot: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with "
            "`python3 -m pip install matplotlib` or run with `--no-show` and skip `--save` "
            "if you only want the printed poses."
        ) from exc

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    all_points: List[np.ndarray] = []
    kind_colors = {"camera": "#1f77b4", "imu": "#d62728"}

    for key, pose in poses.items():
        origin = pose[:3, 3]
        rot = pose[:3, :3]
        plot_origin = map_for_plot(origin)
        all_points.append(plot_origin)

        marker_color = kind_colors.get(frames[key].kind, "#333333")
        ax.scatter(plot_origin[0], plot_origin[1], plot_origin[2], s=50, color=marker_color)

        x_axis = map_for_plot(rot[:, 0] * axis_length)
        y_axis = map_for_plot(rot[:, 1] * axis_length)
        z_axis = map_for_plot(rot[:, 2] * axis_length)

        ax.quiver(plot_origin[0], plot_origin[1], plot_origin[2], x_axis[0], x_axis[1], x_axis[2], color="r", linewidth=2)
        ax.quiver(plot_origin[0], plot_origin[1], plot_origin[2], y_axis[0], y_axis[1], y_axis[2], color="g", linewidth=2)
        ax.quiver(plot_origin[0], plot_origin[1], plot_origin[2], z_axis[0], z_axis[1], z_axis[2], color="b", linewidth=2)

        if show_labels:
            ax.text(plot_origin[0], plot_origin[1], plot_origin[2], f" {frames[key].label}", fontsize=9)

    # Draw flashed links using resolved frame origins.
    links_by_dst: Dict[str, List[LinkSpec]] = defaultdict(list)
    for link in links:
        links_by_dst[link.dst].append(link)

    for dst_key, dst_links in links_by_dst.items():
        # Stable ordering keeps arches separated consistently.
        dst_links = sorted(dst_links, key=lambda l: l.src)
        for idx, link in enumerate(dst_links):
            if link.src not in poses or link.dst not in poses:
                continue

            src_origin = map_for_plot(poses[link.src][:3, 3])
            dst_origin = map_for_plot(poses[link.dst][:3, 3])
            delta = dst_origin - src_origin
            distance = float(np.linalg.norm(delta))

            if distance < 1e-9:
                continue

            # If multiple links terminate at the same camera, draw them as arches
            # to keep them visually distinct around the destination frame.
            if len(dst_links) > 1:
                direction = normalized(delta)
                up = np.array([0.0, 0.0, 1.0], dtype=float)
                side = np.cross(direction, up)
                if np.linalg.norm(side) < 1e-9:
                    side = np.array([1.0, 0.0, 0.0], dtype=float)
                side = normalized(side)

                offset_index = idx - (len(dst_links) - 1) / 2.0
                arch_height = max(distance * 0.18, axis_length * 0.35)
                side_offset = offset_index * max(distance * 0.06, axis_length * 0.12)

                t_values = np.linspace(0.0, 1.0, 40)
                curve = []
                for t in t_values:
                    base = src_origin * (1.0 - t) + dst_origin * t
                    arch = math.sin(math.pi * t) * arch_height
                    lateral = math.sin(math.pi * t) * side_offset
                    point = base + np.array([0.0, 0.0, arch], dtype=float) + side * lateral
                    curve.append(point)
                curve = np.vstack(curve)

                ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color="#444444", linewidth=1.5, alpha=0.75, linestyle="--")

                arrow_start = curve[-2]
                arrow_delta = curve[-1] - curve[-2]
                ax.quiver(
                    arrow_start[0],
                    arrow_start[1],
                    arrow_start[2],
                    arrow_delta[0],
                    arrow_delta[1],
                    arrow_delta[2],
                    color="#444444",
                    linewidth=1.5,
                    alpha=0.75,
                    arrow_length_ratio=1.0,
                )

                if show_labels:
                    midpoint = curve[len(curve) // 2]
                    ax.text(midpoint[0], midpoint[1], midpoint[2], f" {link.label}", fontsize=8, color="#444444")
                continue

            # Single incoming link: draw as a straight dashed arrow.
            delta = dst_origin - src_origin
            ax.quiver(
                src_origin[0],
                src_origin[1],
                src_origin[2],
                delta[0],
                delta[1],
                delta[2],
                color="#444444",
                linewidth=1.5,
                alpha=0.7,
                arrow_length_ratio=0.08,
                linestyle="--",
            )

            if show_labels:
                midpoint = (src_origin + dst_origin) / 2.0
                ax.text(midpoint[0], midpoint[1], midpoint[2], f" {link.label}", fontsize=8, color="#444444")

    if all_points:
        set_equal_axes(ax, np.vstack(all_points))

    ax.set_title(f"Board Extrinsics Visualization: {board_name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.view_init(elev=18, azim=-62)

    legend_lines = [
        plt.Line2D([0], [0], color="r", lw=2, label="X axis"),
        plt.Line2D([0], [0], color="g", lw=2, label="Y axis"),
        plt.Line2D([0], [0], color="b", lw=2, label="Z axis"),
        plt.Line2D([0], [0], color="#444444", lw=1.5, linestyle="--", label="Flashed link"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=kind_colors["camera"], markersize=8, label="Camera"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=kind_colors["imu"], markersize=8, label="IMU"),
    ]
    ax.legend(handles=legend_lines, loc="upper right")

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
        print(f"Saved plot to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize camera and IMU coordinate systems from a depthai_boards JSON file")
    parser.add_argument("board_json", help="Board JSON filename in depthai_boards/boards or a full path to a board JSON file")
    parser.add_argument("--reference", default=None, help="Reference frame to place at origin, e.g. CAM_A or CAM_B")
    parser.add_argument("--axis-length", type=float, default=2.0, help="Length of drawn coordinate axes")
    parser.add_argument("--save", type=Path, default=None, help="Optional output image path")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
    parser.add_argument("--no-labels", action="store_true", help="Do not render frame labels")
    parser.add_argument("--print-only", action="store_true", help="Resolve and print poses without plotting")
    return parser.parse_args()


def resolve_board_json(board_arg: str) -> Path:
    board_path = Path(board_arg)
    if board_path.exists():
        return board_path

    boards_dir = Path(__file__).resolve().parent.parent / "boards"
    candidate = boards_dir / board_arg
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Board JSON '{board_arg}' not found. Pass a filename from depthai_boards/boards "
        f"or an explicit path."
    )


def main() -> int:
    args = parse_args()
    board_json = resolve_board_json(args.board_json)
    data = json.loads(board_json.read_text())
    board_config = data["board_config"]

    frames, edges, links = build_graph(board_config)
    reference = choose_reference(board_config, args.reference)
    poses = resolve_poses(frames, edges, reference)

    unresolved = sorted(set(frames) - set(poses))
    if unresolved:
        print("Warning: some frames could not be resolved from the chosen reference:")
        for key in unresolved:
            print(f"  - {frames[key].label}")
        print()

    print_poses(frames, poses, reference)
    print_links(frames, links)
    validate_links(frames, poses, links)
    if not args.print_only:
        plot_poses(
            board_name=board_config.get("name", board_json.stem),
            frames=frames,
            poses=poses,
            links=links,
            axis_length=args.axis_length,
            show_labels=not args.no_labels,
            save_path=args.save,
            show_plot=not args.no_show,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
