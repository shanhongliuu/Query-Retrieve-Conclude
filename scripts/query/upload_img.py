#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

# dataset
SUPPORTED_DATASETS = (
    "kym",
    "memeintent",
    "memeinterpret",
    "mami",
    "multioff",
    "msd",
    "harmcp",
)

DEFAULT_REQUEST_TIMEOUT = 20
DEFAULT_UPLOAD_SLEEP_SEC = 0.7

IMGBB_UPLOAD_ENDPOINT = "https://api.imgbb.com/1/upload"


def get_project_root() -> Path:

    return Path(__file__).resolve().parents[2]


def resolve_paths(
    dataset: Optional[str],
    meme_id: Optional[str],
    input_json: Optional[Path],
    image_dir: Optional[Path],
    output_json: Optional[Path],
) -> tuple[Path, Path, Path]:

    project_root = get_project_root()

    if dataset:
        if input_json is None:
            input_json = (
                project_root
                / "data"
                / dataset
                / f"{dataset}_qg.json"  # for questions-generator
            )

        if image_dir is None:
            image_dir = (
                project_root
                / "data"
                / dataset
                / "images"
            )

        if output_json is None:

            # one-meme test
            if meme_id is not None:
                output_json = (
                    project_root
                    / "outputs"
                    / "test_runs"
                    / dataset
                    / "query"
                    / "url_map_imgbb.json"
                )

            # full dataset
            else:
                output_json = (
                    project_root
                    / "outputs"
                    / dataset
                    / "question_generation"
                    / "ris"
                    / "url_map_imgbb.json"
                )

    if input_json is None:
        raise ValueError(
            "Missing input JSON. "
            "Use --dataset or --input-json."
        )

    if image_dir is None:
        raise ValueError(
            "Missing image directory. "
            "Use --dataset or --image-dir."
        )

    if output_json is None:
        raise ValueError(
            "Missing output JSON. "
            "Use --dataset or --output-json."
        )

    return (
        input_json.expanduser().resolve(),
        image_dir.expanduser().resolve(),
        output_json.expanduser().resolve(),
    )


def normalize_ws(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def load_json(
    path: Path,
    default: Any = None,
) -> Any:

    if not path.exists():
        if default is not None:
            return default

        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(
    obj: Any,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = Path(
        str(path) + ".tmp"
    )

    with tmp_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
        )

    os.replace(
        tmp_path,
        path,
    )


def normalize_qg_records(
    qg_data: Any,
) -> List[Dict[str, str]]:

    records: List[Dict[str, str]] = []

    if isinstance(qg_data, dict):

        def sort_key(item):
            key = str(item[0])

            if key.isdigit():
                return (0, int(key))

            return (1, key)

        for meme_id, rec in sorted(
            qg_data.items(),
            key=sort_key,
        ):
            if not isinstance(rec, dict):
                continue

            img = normalize_ws(
                rec.get(
                    "img",
                    rec.get("img_path", ""),
                )
            )

            text = normalize_ws(
                rec.get("text", "")
            )

            records.append(
                {
                    "meme_id": str(meme_id),
                    "img": img,
                    "text": text,
                }
            )

        return records

    if isinstance(qg_data, list):

        for index, rec in enumerate(
            qg_data,
            start=1,
        ):
            if not isinstance(rec, dict):
                continue

            meme_id = normalize_ws(
                rec.get(
                    "meme_id",
                    rec.get("id", index),
                )
            )

            img = normalize_ws(
                rec.get(
                    "img",
                    rec.get("img_path", ""),
                )
            )

            text = normalize_ws(
                rec.get("text", "")
            )

            records.append(
                {
                    "meme_id": (
                        meme_id
                        if meme_id
                        else str(index)
                    ),
                    "img": img,
                    "text": text,
                }
            )

        return records

    raise ValueError(
        "QG JSON must contain either a JSON object "
        f"or JSON array. Got: {type(qg_data).__name__}"
    )


def load_qg_records(
    input_json: Path,
) -> List[Dict[str, str]]:

    data = load_json(
        input_json
    )

    records = normalize_qg_records(
        data
    )

    if not records:
        raise ValueError(
            f"No valid records found in: {input_json}"
        )

    return records


# one-meme test
def select_records(
    records: List[Dict[str, str]],
    meme_id: Optional[str],
) -> List[Dict[str, str]]:

    if meme_id is None:
        return records

    meme_id = normalize_ws(meme_id)

    selected = [
        rec
        for rec in records
        if rec["meme_id"] == meme_id
    ]

    if not selected:
        raise ValueError(
            f"Meme ID {meme_id} was not found."
        )

    return selected


def candidate_image_paths(
    image_dir: Path,
    img_name: str,
) -> List[Path]:

    img_name = normalize_ws(
        img_name
    )

    if not img_name:
        return []

    raw_path = Path(
        img_name
    )

    # Absolute path supplied in input.
    if raw_path.is_absolute():
        return [raw_path]

    candidates: List[Path] = []

    candidates.append(
        image_dir / raw_path
    )

    if img_name.startswith("./"):
        candidates.append(
            image_dir
            / img_name[2:]
        )

    if raw_path.parts:
        if raw_path.parts[0] == "images":
            if len(raw_path.parts) > 1:
                stripped_path = Path(
                    *raw_path.parts[1:]
                )

                candidates.append(
                    image_dir
                    / stripped_path
                )

    if raw_path.suffix == "":
        for extension in (
            ".png",
            ".jpg",
            ".jpeg",
        ):
            candidates.append(
                image_dir
                / f"{img_name}{extension}"
            )

    unique_candidates: List[Path] = []
    seen = set()

    for candidate in candidates:
        key = str(candidate)

        if key in seen:
            continue

        seen.add(key)
        unique_candidates.append(
            candidate
        )

    return unique_candidates


def resolve_image_path(
    image_dir: Path,
    img_name: str,
) -> Optional[Path]:

    candidates = candidate_image_paths(
        image_dir=image_dir,
        img_name=img_name,
    )

    for candidate in candidates:
        if (
            candidate.exists()
            and candidate.is_file()
        ):
            return candidate

    return None


def validate_dataset(
    records: List[Dict[str, str]],
    image_dir: Path,
) -> Dict[str, Any]:

    missing_img_field: List[str] = []
    missing_images: List[Dict[str, str]] = []

    for rec in records:

        meme_id = rec["meme_id"]
        img_name = rec["img"]

        if not img_name:
            missing_img_field.append(
                meme_id
            )
            continue

        image_path = resolve_image_path(
            image_dir=image_dir,
            img_name=img_name,
        )

        if image_path is None:
            missing_images.append(
                {
                    "meme_id": meme_id,
                    "img": img_name,
                }
            )

    return {
        "total_records": len(records),
        "missing_img_field": missing_img_field,
        "missing_images": missing_images,
    }


def upload_image_to_imgbb(
    image_path: Path,
    api_key: str,
    timeout: int,
    expiration: Optional[int] = None,
) -> Dict[str, Any]:

    if not image_path.exists():
        return {
            "success": False,
            "error": (
                f"Image file not found: "
                f"{image_path}"
            ),
        }

    try:

        with image_path.open(
            "rb"
        ) as image_file:

            encoded_image = (
                base64.b64encode(
                    image_file.read()
                ).decode("utf-8")
            )

        payload: Dict[str, str] = {
            "key": api_key,
            "image": encoded_image,
        }

        if expiration is not None:
            payload["expiration"] = str(
                expiration
            )

        response = requests.post(
            IMGBB_UPLOAD_ENDPOINT,
            data=payload,
            timeout=timeout,
        )

        response.raise_for_status()

        result = response.json()

        if not result.get(
            "success",
            False,
        ):
            return {
                "success": False,
                "error": (
                    "ImgBB returned success=false."
                ),
            }

        data = result.get(
            "data",
            {},
        )

        if not isinstance(
            data,
            dict,
        ):
            data = {}

        thumb_data = data.get(
            "thumb",
            {},
        )

        if not isinstance(
            thumb_data,
            dict,
        ):
            thumb_data = {}

        medium_data = data.get(
            "medium",
            {},
        )

        if not isinstance(
            medium_data,
            dict,
        ):
            medium_data = {}

        public_url = normalize_ws(
            data.get("url", "")
        )

        if not public_url:
            return {
                "success": False,
                "error": (
                    "ImgBB response did not contain "
                    "a public image URL."
                ),
            }

        return {
            "success": True,
            "url": public_url,
            "display_url": normalize_ws(
                data.get(
                    "display_url",
                    "",
                )
            ),
            "thumb_url": normalize_ws(
                thumb_data.get(
                    "url",
                    "",
                )
            ),
            "medium_url": normalize_ws(
                medium_data.get(
                    "url",
                    "",
                )
            ),
            "imgbb_id": normalize_ws(
                data.get(
                    "id",
                    "",
                )
            ),
        }

    except requests.Timeout:
        return {
            "success": False,
            "error": (
                f"ImgBB request timed out "
                f"after {timeout} seconds."
            ),
        }

    except requests.RequestException as exc:
        return {
            "success": False,
            "error": (
                f"ImgBB request failed: {exc}"
            ),
        }

    except Exception as exc:
        return {
            "success": False,
            "error": (
                f"Unexpected upload error: {exc}"
            ),
        }


def is_completed_record(
    record: Any,
) -> bool:

    if not isinstance(
        record,
        dict,
    ):
        return False

    return bool(
        normalize_ws(
            record.get(
                "url",
                "",
            )
        )
    )


def upload_records(
    records: List[Dict[str, str]],
    image_dir: Path,
    output_json: Path,
    api_key: str,
    timeout: int,
    sleep_sec: float,
    expiration: Optional[int],
    overwrite: bool,
) -> Dict[str, Any]:

    if output_json.exists():
        url_map = load_json(
            output_json,
            default={},
        )
    else:
        url_map = {}

    if not isinstance(
        url_map,
        dict,
    ):
        raise ValueError(
            f"{output_json}"
        )

    total_records = len(
        records
    )

    uploaded_this_run = 0
    skipped_existing = 0
    failed_this_run = 0

    print()
    print(
        f"Existing ImgBB records: "
        f"{len(url_map)}"
    )
    print()

    for rec in tqdm(
        records,
        desc="Uploading to ImgBB",
        unit="image",
    ):

        meme_id = rec["meme_id"]
        img_name = rec["img"]
        meme_text = rec["text"]

        existing_record = url_map.get(
            img_name,
            {},
        )

        if (
            not overwrite
            and is_completed_record(
                existing_record
            )
        ):
            skipped_existing += 1
            continue

        output_record: Dict[str, Any] = {
            "meme_id": meme_id,
            "img": img_name,
            "text": meme_text,
            "url": "",
            "display_url": "",
            "thumb_url": "",
            "medium_url": "",
            "imgbb_id": "",
            "error": "",
        }

        if not img_name:

            output_record["error"] = (
                "Missing img field in QG record."
            )

            record_key = (
                f"__missing_image__{meme_id}"
            )

            url_map[
                record_key
            ] = output_record

            save_json(
                url_map,
                output_json,
            )

            failed_this_run += 1
            continue

        image_path = resolve_image_path(
            image_dir=image_dir,
            img_name=img_name,
        )

        if image_path is None:

            output_record["error"] = (
                "Image file not found under "
                f"image directory: {img_name}"
            )

            url_map[
                img_name
            ] = output_record

            save_json(
                url_map,
                output_json,
            )

            failed_this_run += 1
            continue

        upload_result = upload_image_to_imgbb(
            image_path=image_path,
            api_key=api_key,
            timeout=timeout,
            expiration=expiration,
        )

        if upload_result.get(
            "success",
            False,
        ):

            output_record["url"] = (
                upload_result.get(
                    "url",
                    "",
                )
            )

            output_record[
                "display_url"
            ] = upload_result.get(
                "display_url",
                "",
            )

            output_record[
                "thumb_url"
            ] = upload_result.get(
                "thumb_url",
                "",
            )

            output_record[
                "medium_url"
            ] = upload_result.get(
                "medium_url",
                "",
            )

            output_record[
                "imgbb_id"
            ] = upload_result.get(
                "imgbb_id",
                "",
            )

            uploaded_this_run += 1

        else:

            output_record["error"] = (
                upload_result.get(
                    "error",
                    "Unknown ImgBB upload error.",
                )
            )

            failed_this_run += 1

        url_map[
            img_name
        ] = output_record

        save_json(
            url_map,
            output_json,
        )

        if sleep_sec > 0:
            time.sleep(
                sleep_sec
            )

    completed_total = sum(
        1
        for rec in url_map.values()
        if is_completed_record(rec)
    )

    print()
    print("=" * 80)
    print("ImgBB upload completed")
    print("=" * 80)

    print(
        f"Input records          : "
        f"{total_records}"
    )

    print(
        f"Uploaded this run      : "
        f"{uploaded_this_run}"
    )

    print(
        f"Skipped existing       : "
        f"{skipped_existing}"
    )

    print(
        f"Failed this run        : "
        f"{failed_this_run}"
    )

    print(
        f"Completed URLs in file : "
        f"{completed_total}"
    )

    print(
        f"Output                 : "
        f"{output_json}"
    )

    return url_map


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Upload meme images to ImgBB for the "
            "Query stage of Query-Retrieve-Conclude."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASETS,
        default=None,
        help=(
            "Dataset name. When provided, input/output "
            "paths are derived automatically."
        ),
    )

    # one-meme test
    parser.add_argument(
        "--meme-id",
        type=str,
        default=None,
        help=(
            "Process only one meme ID. "
            "Omit this argument for the full dataset."
        ),
    )

    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help=(
            "Path to *_qg.json. "
            "Overrides the path derived from --dataset."
        ),
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing meme images. "
            "Overrides the path derived from --dataset."
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Path for url_map_imgbb.json. "
            "Overrides the path derived from --dataset."
        ),
    )

    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help=(
            "ImgBB request timeout in seconds. "
            f"Default: {DEFAULT_REQUEST_TIMEOUT}"
        ),
    )

    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=DEFAULT_UPLOAD_SLEEP_SEC,
        help=(
            "Seconds to sleep between uploads. "
            f"Default: {DEFAULT_UPLOAD_SLEEP_SEC}"
        ),
    )

    parser.add_argument(
        "--expiration",
        type=int,
        default=None,
        help=(
            "Optional ImgBB expiration time in seconds. "
            "By default, no expiration value is requested."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-upload images that already have a "
            "successful URL in the output file."
        ),
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help=(
            "Skip the complete local image validation "
            "before uploading."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    api_key = normalize_ws(
        os.environ.get(
            "IMGBB_API_KEY",
            "",
        )
    )

    if not api_key:

        print(
            "ERROR: IMGBB_API_KEY is not set.",
            file=sys.stderr,
        )

        print(
            "",
            file=sys.stderr,
        )

        print(
            "",
            file=sys.stderr,
        )

        print(
            '    export IMGBB_API_KEY="YOUR_IMGBB_API_KEY"',
            file=sys.stderr,
        )

        sys.exit(1)

    try:

        input_json, image_dir, output_json = (
            resolve_paths(
                dataset=args.dataset,
                meme_id=args.meme_id,
                input_json=args.input_json,
                image_dir=args.image_dir,
                output_json=args.output_json,
            )
        )

    except ValueError as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)

    if not input_json.exists():

        print(
            f"ERROR: QG JSON not found: "
            f"{input_json}",
            file=sys.stderr,
        )

        sys.exit(1)

    if not image_dir.exists():

        print(
            f"ERROR: Image directory not found: "
            f"{image_dir}",
            file=sys.stderr,
        )

        sys.exit(1)

    if not image_dir.is_dir():

        print(
            f"ERROR: Image path is not a directory: "
            f"{image_dir}",
            file=sys.stderr,
        )

        sys.exit(1)

    try:

        all_records = load_qg_records(
            input_json
        )

        # one-meme test
        records = select_records(
            records=all_records,
            meme_id=args.meme_id,
        )

    except Exception as exc:

        print(
            f"ERROR: Failed to load QG input: "
            f"{exc}",
            file=sys.stderr,
        )

        sys.exit(1)

    print("=" * 80)
    print("Query Stage - ImgBB Upload")
    print("=" * 80)

    if args.dataset:
        print(
            f"Dataset     : {args.dataset}"
        )

    # one-meme test
    if args.meme_id is not None:
        print(
            f"Meme ID     : {args.meme_id}"
        )

    print(
        f"Input JSON  : {input_json}"
    )

    print(
        f"Image dir   : {image_dir}"
    )

    print(
        f"Output JSON : {output_json}"
    )

    print(
        f"Records     : {len(records)}"
    )

    if not args.skip_validation:

        validation = validate_dataset(
            records=records,
            image_dir=image_dir,
        )

        missing_img_field = validation[
            "missing_img_field"
        ]

        missing_images = validation[
            "missing_images"
        ]

        print()
        print("Local input validation")
        print("-" * 80)

        print(
            f"Total records           : "
            f"{validation['total_records']}"
        )

        print(
            f"Missing img field       : "
            f"{len(missing_img_field)}"
        )

        print(
            f"Missing local image file: "
            f"{len(missing_images)}"
        )

        if missing_img_field:

            print()
            print(
                "Example records with missing img field:"
            )

            for meme_id in (
                missing_img_field[:10]
            ):

                print(
                    f"  meme_id={meme_id}"
                )

        if missing_images:

            print()
            print(
                "Example missing image files:"
            )

            for item in (
                missing_images[:10]
            ):

                print(
                    f"  meme_id={item['meme_id']} "
                    f"img={item['img']}"
                )

            print()
            print(
                "Available images will still be uploaded."
            )

    upload_records(
        records=records,
        image_dir=image_dir,
        output_json=output_json,
        api_key=api_key,
        timeout=args.request_timeout,
        sleep_sec=args.sleep_sec,
        expiration=args.expiration,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()