"""New DEV diagnosis of completed coverage endpoints, never old independent layouts."""
from pathlib import Path

from scripts.paper_crax_payload.diagnose_native_terminal_v11 import main


if __name__ == '__main__':
    main(protocol_path=Path(__file__).resolve().parents[2]/
         'configs/paper_crax_payload/simulation_v12_payload_cover_probe.json')
