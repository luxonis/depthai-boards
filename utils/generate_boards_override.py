import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional
from resources.depthai_boards.boards_reader import get_variant_by_id_typed, get_device_by_id_typed


def generate_override_dict(variant_id: str, os: Optional[str] = None) -> Dict[str, Any]:
    override_payload: Dict[str, Any] = {}

    if os is not None:
        override_payload["os"] = os

    return {variant_id: override_payload}


def save_override_file(override_dict: Dict[str, Any], output_path: Optional[Path] = None) -> None:
    if output_path is None:
        print(json.dumps(override_dict, indent=4))
        return

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(override_dict, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Generate depthai_boards override JSON file to override existing configuration."
    )
    parser.add_argument("--device", "-d", required=True, help="Target device ID or title (e.g. 'oak_4_s' or 'OAK4-S')")
    parser.add_argument("--variant", "-v", required=True, help="Target variant ID or title (e.g. 'SL3443_ASM_P10D3_oak4_cs_og05b10')")
    parser.add_argument("--os", required=False, default=None, help="Custom OS to override")
    parser.add_argument("--output", "-o", required=False, default=None, help="Output path for the override JSON file. If not specified, JSON is printed to stdout.")

    args = parser.parse_args()

    device = get_device_by_id_typed(args.device)
    variant = get_variant_by_id_typed(args.variant)
    variant_id = variant.id

    override_dict = generate_override_dict(variant_id=variant_id, os=args.os)

    out_path = Path(args.output) if args.output else None
    save_override_file(override_dict, output_path=out_path)


if __name__ == "__main__":
    main()
