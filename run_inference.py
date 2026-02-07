#!/usr/bin/env python
"""
Australian Wildlife Conservancy's Detection & Classification Script
Author: Quan Tran
==========================================
A command-line tool for running animal detection and species classification
on camera trap images using AWC Helpers.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

from awc_helpers import DetectAndClassify
from awc_helpers.format_utils import get_all_image_paths


def setup_logging(log_file: str = None) -> logging.Logger:
    """
    Configure logging to write to both console and file.
    
    Args:
        log_file: Path to log file. If None, creates timestamped log file.
    
    Returns:
        Configured logger instance.
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"inference_{timestamp}.log"
    
    logger = logging.getLogger("awc_inference")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging to: {log_file}")
    return logger


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
    
    Returns:
        Dictionary containing configuration settings.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If required config fields are missing.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Check required fields
    required_fields = ["detector_path", "classifier_path"]
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")
    
    return config


def load_labels(label_path: str) -> list:
    """
    Load species label names from a text file.

    Args:
        label_path: Path to the labels text file.
    
    Returns:
        List of label name strings.
    
    Raises:
        FileNotFoundError: If labels file doesn't exist.
        ValueError: If no labels found in file.
    """
    label_file = Path(label_path)
    if not label_file.exists():
        raise FileNotFoundError(f"Labels file not found: {label_path}")
    
    labels = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                labels.append(line)
    
    if not labels:
        raise ValueError(f"No labels found in {label_path}")
    
    return labels


def main():
    """Main entry point for the inference script."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run wildlife detection and classification on camera trap images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "image_folder",
        type=str,
        help="Path to folder containing images to process (searches subfolders too)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output name for results (overrides config file setting)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Load configuration
        logger.info(f"Loading config from: {args.config}")
        config = load_config(args.config)
        
        # Load labels
        logger.info(f"Loading labels from: {config['label_path']}")
        labels = load_labels(config["label_path"])
        logger.info(f"Loaded {len(labels)} species labels")
        
        # Validate image folder
        image_folder = Path(args.image_folder)
        if not image_folder.exists():
            raise FileNotFoundError(f"Image folder not found: {args.image_folder}")
        if not image_folder.is_dir():
            raise ValueError(f"Not a directory: {args.image_folder}")
        
        # Get all images
        logger.info(f"Scanning for images in: {image_folder}")
        image_paths = get_all_image_paths(str(image_folder))
        
        if not image_paths:
            logger.warning("No images found! Supported formats: .jpg, .jpeg, .png")
            sys.exit(0)
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Get config values with defaults
        detector_path = config["detector_path"]
        classifier_path = config["classifier_path"]
        classifier_base = config.get("classifier_base", "tf_efficientnet_b5.ns_jft_in1k")
        detection_threshold = config.get("detection_threshold", 0.1)
        classification_threshold = config.get("classification_threshold", 0.5)
        batch_size = config.get("classification_batch_size", 4)
        topn = config.get("topn", 1)
        
        # Determine output name
        output_name = args.output or config.get("output_name", "results")
        
        logger.info("=" * 50)
        logger.info("Configuration:")
        logger.info(f"  Detector model: {detector_path}")
        logger.info(f"  Classifier model: {classifier_path}")
        logger.info(f"  Detection threshold: {detection_threshold}")
        logger.info(f"  Classification threshold: {classification_threshold}")
        logger.info(f"  Classification batch size: {batch_size}")
        logger.info(f"  Top-N predictions: {topn}")
        logger.info(f"  Output name: {output_name}")
        logger.info("=" * 50)
        
        # Initialize pipeline
        logger.info("Loading models (this may take a moment)...")
        pipeline = DetectAndClassify(
            detector_path=detector_path,
            classifier_path=classifier_path,
            label_names=labels,
            classifier_base=classifier_base,
            detection_threshold=detection_threshold,
            clas_threshold=classification_threshold
        )
        logger.info("Models loaded successfully!")
        
        # Run inference
        logger.info("Starting detection and classification...")
        logger.info("-" * 50)
        

        output_name = Path(output_name).with_suffix('')
        if str(output_name.parent) == '.': # just a filenname
            output_name = image_folder / output_name.name
        else: # path
            output_name.parent.mkdir(parents=True, exist_ok=True)
        output_name = str(output_name)

        pipeline.predict(
            inp=image_paths,
            clas_bs=batch_size,
            topn=topn,
            output_name=output_name,
            show_progress=True,
        )
        
        logger.info("-" * 50)
        logger.info(f"Results saved to: {output_name}.csv and {output_name}.json")
        logger.info("Done!")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
