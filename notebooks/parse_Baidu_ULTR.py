#!/usr/bin/env python
"""Entrypoint script for parsing Baidu ULTR feather files into prebuilt NPZ artifacts.

Configure via environment variables or CLI flags:

  BAIDU_PARTS_DIR   – directory containing part-*.feather and validation.feather
  BAIDU_OUTPUT_DIR  – output directory for train/val/test NPZs and manifest.json
  BAIDU_MAX_LEN     – maximum rank depth per session (default: 20)
  BAIDU_OVERWRITE   – set to "1"/"true" to overwrite existing output (default: 0)

Example:
  BAIDU_PARTS_DIR=/data/baidu/parts \\
  BAIDU_OUTPUT_DIR=/data/baidu/processed \\
  python notebooks/parse_Baidu_ULTR.py --max-len 20
"""

from __future__ import annotations

from feature_based_propensities_for_ULTR.data.parsers.baidu_ultr import (
    build_parse_arg_parser,
    env_default_output_dir,
    env_default_parts_dir,
    parse_and_save_baidu_ultr,
)


def main() -> None:
    parser = build_parse_arg_parser(
        default_parts_dir=env_default_parts_dir(),
        default_output_dir=env_default_output_dir(),
    )
    args = parser.parse_args()

    if args.parts_dir is None:
        parser.error(
            "No parts directory specified. "
            "Pass --parts-dir or set the BAIDU_PARTS_DIR environment variable."
        )
    if args.output_dir is None:
        parser.error(
            "No output directory specified. "
            "Pass --output-dir or set the BAIDU_OUTPUT_DIR environment variable."
        )

    parse_and_save_baidu_ultr(
        parts_dir=args.parts_dir,
        output_dir=args.output_dir,
        max_len=args.max_len,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
