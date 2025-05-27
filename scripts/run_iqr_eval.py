# scripts/run_iqr_eval.py
import argparse
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# --- Add project root to sys.path ---
# This allows a direct run of the script from anywhere and ensures 'tsfm_ad_lib' can be imported.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path modification ---

from tsfm_ad_lib.data_loader import load_raw_lead_data
from tsfm_ad_lib.preprocessing import preprocess_lead_data_by_building
from tsfm_ad_lib.models.statistical import find_best_iqr_k_for_building
from tsfm_ad_lib.utils import setup_logger, set_random_seeds
from tsfm_ad_lib import config as lib_config # Access default K values, etc.

def main(args):
    """
    Main execution function for IQR evaluation script.
    """
    # Setup
    logger = setup_logger("IQREvalScript", log_file=args.log_file, level=args.log_level.upper())
    set_random_seeds(lib_config.RANDOM_SEED) # Though IQR is deterministic, good practice
    logger.info("Starting IQR Evaluation Script")
    logger.info(f"Arguments: {args}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # 1. Load Raw Data
    try:
        raw_df = load_raw_lead_data(csv_file_path=args.data_path)
        logger.info(f"Loaded raw data. Shape: {raw_df.shape}")
    except FileNotFoundError:
        logger.error(f"Data file not found at {args.data_path}. Exiting.")
        return
    except Exception as e:
        logger.error(f"Error loading data: {e}. Exiting.")
        return

    # 2. Preprocess Data (Median imputation per building, NO SCALING at this global stage)
    # The find_best_iqr_k_for_building function handles per-building scaling internally if scale_data=True.
    try:
        imputed_df = preprocess_lead_data_by_building(
            raw_df,
            building_id_col=args.building_id_col,
            meter_reading_col=args.meter_reading_col,
            scale_meter_reading=False # Global preprocessing doesn't scale for IQR script's pattern
        )
        logger.info(f"Preprocessed data (imputation only). Shape: {imputed_df.shape}")
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}. Exiting.")
        return
    
    if imputed_df.empty:
        logger.warning("Preprocessing resulted in an empty DataFrame. No buildings to process. Exiting.")
        return

    # 3. Evaluate IQR per building
    all_building_ids = sorted(imputed_df[args.building_id_col].unique())
    
    overall_f1_scores = []
    overall_precision_scores = []
    overall_recall_scores = []
    detailed_results = [] # To store results per building

    # Parse K search values if provided as string "start,stop,step" or "k1,k2,k3"
    k_search_list = None
    if args.k_search_values:
        try:
            if ',' in args.k_search_values and len(args.k_search_values.split(',')) == 3: # "start,stop,step"
                start, stop, step = map(float, args.k_search_values.split(','))
                k_search_list = list(np.arange(start, stop, step))
            else: # "k1,k2,k3,..."
                k_search_list = [float(k.strip()) for k in args.k_search_values.split(',')]
            logger.info(f"Using custom K search values: {k_search_list}")
        except ValueError:
            logger.warning(f"Could not parse k_search_values '{args.k_search_values}'. Using default. Provide as 'start,stop,step' or 'k1,k2,...'")
            k_search_list = lib_config.DEFAULT_IQR_K_SEARCH_VALUES


    logger.info(f"Processing {len(all_building_ids)} buildings...")
    for b_id in all_building_ids:
        building_data_slice = imputed_df[imputed_df[args.building_id_col] == b_id]
        if building_data_slice.empty or building_data_slice[args.meter_reading_col].isnull().all():
            logger.warning(f"Skipping building ID {b_id} due to empty or all-NaN meter readings.")
            detailed_results.append({
                "building_id": b_id, "f1": 0.0, "precision": 0.0, "recall": 0.0, "best_k": np.nan, "status": "skipped_no_data"
            })
            continue

        logger.debug(f"Processing building ID: {b_id}")
        
        # The find_best_iqr_k_for_building function handles per-building scaling if scale_data=True (default)
        metrics = find_best_iqr_k_for_building(
            building_data=building_data_slice,
            meter_reading_col=args.meter_reading_col,
            anomaly_col=args.anomaly_col,
            k_search_values=k_search_list, # Pass the parsed or default list
            scale_data=args.scale_per_building # Control scaling within the function
        )
        
        overall_f1_scores.append(metrics["f1"])
        overall_precision_scores.append(metrics["precision"])
        overall_recall_scores.append(metrics["recall"])
        detailed_results.append({
            "building_id": b_id, 
            "f1": metrics["f1"], 
            "precision": metrics["precision"], 
            "recall": metrics["recall"], 
            "best_k": metrics["k"],
            "status": "processed"
        })
        logger.debug(f"Building ID {b_id} - Best F1: {metrics['f1']:.4f} with K={metrics['k']:.2f}")

    # 4. Report and Save Results
    if overall_f1_scores: # If any buildings were processed
        mean_f1 = np.mean(overall_f1_scores)
        mean_precision = np.mean(overall_precision_scores)
        mean_recall = np.mean(overall_recall_scores)
        
        logger.info("--- Overall IQR Evaluation Metrics ---")
        logger.info(f"Average F1 Score: {mean_f1:.4f}")
        logger.info(f"Average Precision: {mean_precision:.4f}")
        logger.info(f"Average Recall: {mean_recall:.4f}")
        logger.info(f"Individual F1 scores: {overall_f1_scores}")
        
        # Save detailed results to CSV
        detailed_results_df = pd.DataFrame(detailed_results)
        results_csv_path = output_dir / "iqr_evaluation_detailed_results.csv"
        detailed_results_df.to_csv(results_csv_path, index=False)
        logger.info(f"Detailed results saved to {results_csv_path}")

        # Save summary
        summary_results = {
            "method": "IQR",
            "average_f1": mean_f1,
            "average_precision": mean_precision,
            "average_recall": mean_recall,
            "num_buildings_processed": len(overall_f1_scores),
            "num_buildings_total": len(all_building_ids)
        }
        summary_df = pd.DataFrame([summary_results])
        summary_csv_path = output_dir / "iqr_evaluation_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Summary results saved to {summary_csv_path}")

    else:
        logger.warning("No buildings were processed successfully. No overall metrics to report.")

    logger.info("IQR Evaluation Script Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IQR-based anomaly detection evaluation.")
    
    parser.add_argument('--data_path', type=str, default=lib_config.DEFAULT_DATA_CSV_PATH,
                        help=f"Path to the raw data CSV file. Default: {lib_config.DEFAULT_DATA_CSV_PATH}")
    parser.add_argument('--output_dir', type=str, default=str(Path(lib_config.DEFAULT_RESULTS_DIR) / "iqr_eval"),
                        help="Directory to save evaluation results and logs.")
    parser.add_argument('--building_id_col', type=str, default='building_id',
                        help="Name of the column for building identifiers.")
    parser.add_argument('--meter_reading_col', type=str, default='meter_reading',
                        help="Name of the column for meter readings.")
    parser.add_argument('--anomaly_col', type=str, default='anomaly',
                        help="Name of the column for true anomaly labels.")
    parser.add_argument('--scale_per_building', action='store_true',
                        help="If set, scale meter readings per building before IQR (as in original script).")
    parser.set_defaults(scale_per_building=True) # Default to True to match original script
    
    parser.add_argument('--k_search_values', type=str, default=None,
                        help="Comma-separated list of K values for IQR threshold search (e.g., '0.5,1.0,1.5') "
                             "OR 'start,stop,step' (e.g., '0.5,4.5,0.5'). "
                             f"Default: uses library default (approx {lib_config.DEFAULT_IQR_K_SEARCH_VALUES}).")
    
    parser.add_argument('--log_file', type=str, default=None,
                        help="Path to save log file. If None, logs only to console. "
                             "Default: <output_dir>/run_iqr_eval.log")
    parser.add_argument('--log_level', type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level.")

    args = parser.parse_args()

    # Set default log file path if not provided
    if args.log_file is None:
        args.log_file = str(Path(args.output_dir) / "run_iqr_eval.log")
    
    # Create output directory for log file if it doesn't exist, before logger setup
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    main(args)