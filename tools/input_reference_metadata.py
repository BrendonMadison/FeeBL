"""Physics meanings, units and conditions for native input keywords."""
import json


def metadata(root):
    """Read maintained metadata; the generator verifies it against the parser."""
    return json.loads((root / "tools/input_reference_metadata.json").read_text(encoding="utf-8"))
