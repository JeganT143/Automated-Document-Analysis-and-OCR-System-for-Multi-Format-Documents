"""
Command-line interface for the Document OCR pipeline.

    python src/main.py path/to/document.png
    python src/main.py invoice.jpg --format txt --output result.txt
    python src/main.py invoice.jpg --no-layout

A thin wrapper around the single pipeline in src/pipeline.py.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import DocumentOCRPipeline


def _build_arg_parser():
    p = argparse.ArgumentParser(description="Automated Document OCR pipeline")
    p.add_argument("image", help="Path to input document image")
    p.add_argument("--format", dest="output_format",
                   choices=["json", "txt", "csv"], default="json",
                   help="Output format (default: json)")
    p.add_argument("--output", default=None,
                   help="Write output to a file (default: stdout)")
    p.add_argument("--lang", default="eng", help="Tesseract language (default: eng)")
    p.add_argument("--no-layout", action="store_true",
                   help="Skip layout/region analysis (faster)")
    return p


def main():
    args = _build_arg_parser().parse_args()
    if not os.path.isfile(args.image):
        print(f"ERROR: image not found: {args.image}")
        sys.exit(1)

    pipeline = DocumentOCRPipeline(lang=args.lang)
    result = pipeline.run(args.image, analyze_layout=not args.no_layout)
    output = pipeline.render(result, fmt=args.output_format)

    print("=== Document OCR Pipeline ===")
    print(f"Words            : {result['word_count']}")
    print(f"Mean confidence  : {result['mean_confidence']}%")
    print(f"PSM used         : {result['psm_used']}")
    print(f"Regions          : {len(result['regions'])}")
    if result["fields"]:
        print(f"Fields           : {result['fields']}")
    print(f"Total time       : {result['metrics']['total_s']} s")
    print()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Output saved to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
