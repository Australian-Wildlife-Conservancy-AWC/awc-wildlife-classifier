# Wildlife Detection & Classification Tool

A simple command-line tool for detecting and classifying animals in camera trap images.

## Quick Start

### 1. Setup

Make sure you have:
- Python 3.8 or higher installed
- The `awc_helpers` package installed. Follow the instruction [here](https://github.com/Australian-Wildlife-Conservancy-AWC/awc_inference?tab=readme-ov-file#awc-helpers)
- MegaDetector model (`.pt` file). You can view and download them [here](https://github.com/agentmorris/MegaDetector/releases/tag/v1000.0) (v1000-redwood is the recommended version)
- Species classifier model (`.pth` file). You can download it [here](https://github.com/Australian-Wildlife-Conservancy-AWC/awc-wildlife-classifier/releases/download/awc-135/awc-135-v1.pth)

### 2. Configure

Edit `config.yaml` to set your model paths:

```yaml
detector_path: "models/<megadetector_weight>.pt"
classifier_path: "models/awc-135-v1.pth"
```

### 3. Run

Open a terminal/command prompt and run:

```bash
python run_inference.py <path_to_your_images> --config config.yaml
```

**Example:**
```bash
# Windows
python run_inference.py C:\CameraTrap\Photos --config config.yaml

# Mac/Linux
python run_inference.py /home/user/camera_trap_images --config config.yaml
```

### 4. Results

The tool creates two output files:
- `results.csv` - Spreadsheet-friendly format
- `results.json` - Timelapse-compatible JSON format

Plus a log file with timestamps for troubleshooting.

---

## Detailed Instructions

### Command-Line Options

```
python run_inference.py <image_folder> --config <config_file> [--output <output_name>]

Arguments:
  image_folder       Path to folder containing images (searches subfolders too)
  --config, -c       Path to YAML configuration file (required)
  --output, -o       Override output file name from config (optional)
```

### Configuration File

All settings are in `config.yaml`:

| Setting | Description | Default |
|---------|-------------|---------|
| `detector_path` | Path to MegaDetector model | *required* |
| `classifier_path` | Path to species classifier | *required* |
| `label_path` | Path to labels text file | `labels.txt` |
| `output_name` | Name for output files | `results` |
| `detection_threshold` | Min confidence for detection (0-1) | `0.1` |
| `classification_threshold` | Min confidence for classification (0-1) | `0.5` |
| `topn` | Number of top predictions per animal | `1` |
| `classification_batch_size` | Crops processed at once (GPU memory) | `4` |


## Troubleshooting

### "No images found"
- Check that your folder path is correct
- Supported formats: `.jpg`, `.jpeg`, `.png`
- The tool searches subfolders automatically

### "Out of memory" error
- Reduce `classification_batch_size` in config (try 2 or 1)
- Set `force_cpu: true` to use CPU instead of GPU

### "File not found" errors
- Double-check all paths in `config.yaml`
- Use full paths (e.g., `C:\Models\detector.pt`) instead of relative paths


---

## Output Format

### CSV File

| Column | Description |
|--------|-------------|
| Image Path | Full path to the source image |
| Bounding Box Confidence | Detection confidence (0-1) |
| Bounding Box Normalized | Box coordinates (x, y, width, height) |
| Label 1 | Top predicted species |
| Confidence 1 | Classification confidence |
| Label N... | Additional predictions if `topn > 1` |

### JSON File

Compatible with [Timelapse](http://saul.cpsc.ucalgary.ca/timelapse/) image viewer software.

---

## Need Help?

Check the log file (created in the same folder as the script) for detailed error messages. You can also submit an issue in this repository.
