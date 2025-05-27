# scripts/run_lof_eval.py
import argparse
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import json # For parsing dict-like arguments

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path modification ---

from tsfm_ad_lib.data_loader import load_raw_lead_data
from tsfm_ad_lib.preprocessing import preprocess_lead_data_by_building
from tsfm_ad_lib.models.neighbors import find_best_lof_params_for_building # Changed import
from tsfm_ad_lib.utils import setup_logger, set_random_seeds # LOF is deterministic, seed not strictly for it
from tsfm_ad_lib import config as lib_config

def main(args):
    """
    Main execution function for Local Outlier Factor (LOF) evaluation script.
    """
    # LOF is generally deterministic once n_neighbors and other params are set.
    # Seed setting is more for general script reproducibility if other random ops were present.
    logger = setup_logger("LOFEvalScript", log_file=args.log_file, level=args.log_level.upper())
    set_random_seeds(lib_config.RANDOM_SEED) 
    logger.info("Starting Local Outlier Factor (LOF) Evaluation Script")
    logger.info(f"Arguments: {args}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # 1. Load Raw Data
    try:
        raw_df = load_raw_lead_data(csv_file_path=args.data_path)
    except Exception as e:
        logger.error(f"Error loading data from {args.data_path}: {e}. Exiting.")
        return

    # 2. Preprocess Data
    try:
        imputed_df = preprocess_lead_data_by_building(
            raw_df,
            building_id_col=args.building_id_col,
            meter_reading_col=args.meter_reading_col,
            scale_meter_reading=False 
        )
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}. Exiting.")
        return
    
    if imputed_df.empty:
        logger.warning("Preprocessing resulted in an empty DataFrame. Exiting.")
        return

    # 3. Evaluate LOF per building
    all_building_ids = sorted(imputed_df[args.building_id_col].unique()) # TEMPORARY: Process only the first 3 buildings
    logger.info(f"PROCESSING ONLY {len(all_building_ids)} buildings for LOF smoke test...") # Add a log message
    
    overall_f1_scores = []
    overall_precision_scores = []
    overall_recall_scores = []
    detailed_results = []

    # Parse contamination search values
    contamination_search_list = None
    if args.contamination_search_values:
        try:
            if ',' in args.contamination_search_values and len(args.contamination_search_values.split(',')) == 3:
                start, stop, step = map(float, args.contamination_search_values.split(','))
                contamination_search_list = list(np.arange(start, stop, step))
            else:
                contamination_search_list = [float(c.strip()) for c in args.contamination_search_values.split(',')]
            logger.info(f"Using custom contamination search values for LOF: {contamination_search_list}")
        except ValueError:
            logger.warning(f"Could not parse contamination_search_values '{args.contamination_search_values}'. Using default.")
            start_c, stop_c, step_c = lib_config.DEFAULT_CONTAMINATION_SEARCH_RANGE
            contamination_search_list = list(np.arange(start_c, stop_c, step_c))
    else: 
        start_c, stop_c, step_c = lib_config.DEFAULT_CONTAMINATION_SEARCH_RANGE
        contamination_search_list = list(np.arange(start_c, stop_c, step_c))

    # Parse base LOF parameters
    base_lof_params_to_use = lib_config.DEFAULT_LOF_BASE_PARAMS.copy()
    if args.lof_base_params:
        try:
            cli_params = json.loads(args.lof_base_params)
            base_lof_params_to_use.update(cli_params)
            logger.info(f"Using custom LOF base parameters: {base_lof_params_to_use}")
        except json.JSONDecodeError:
            logger.error(f"Could not parse lof_base_params JSON: '{args.lof_base_params}'. Using defaults.")
    base_lof_params_to_use['novelty'] = False # Ensure this is set for fit_predict behavior


    logger.info(f"Processing {len(all_building_ids)} buildings for LOF...")
    for b_id in all_building_ids:
        building_data_slice = imputed_df[imputed_df[args.building_id_col] == b_id]
        if building_data_slice.empty or building_data_slice[args.meter_reading_col].isnull().all():
            logger.warning(f"Skipping building ID {b_id} (LOF) due to empty or all-NaN meter readings.")
            detailed_results.append({
                "building_id": b_id, "f1": 0.0, "precision": 0.0, "recall": 0.0, "best_contamination": np.nan, "status": "skipped_no_data"
            })
            continue
            
        logger.debug(f"Processing building ID {b_id} (LOF)")
        
        metrics = find_best_lof_params_for_building( # Changed function call
            building_data=building_data_slice,
            meter_reading_col=args.meter_reading_col,
            anomaly_col=args.anomaly_col,
            base_lof_params=base_lof_params_to_use,         # Changed param name
            contamination_search_values=contamination_search_list,
            scale_data=args.scale_per_building
        )
        
        overall_f1_scores.append(metrics["f1"])
        overall_precision_scores.append(metrics["precision"])
        overall_recall_scores.append(metrics["recall"])
        detailed_results.append({
            "building_id": b_id, 
            "f1": metrics["f1"], 
            "precision": metrics["precision"], 
            "recall": metrics["recall"], 
            "best_contamination": metrics["contamination"],
            "status": "processed"
        })
        logger.debug(f"Building ID {b_id} (LOF) - Best F1: {metrics['f1']:.4f} with Contamination={metrics['contamination']:.4f}")

    # 4. Report and Save Results
    if overall_f1_scores:
        mean_f1 = np.mean(overall_f1_scores)
        mean_precision = np.mean(overall_precision_scores)
        mean_recall = np.mean(overall_recall_scores)
        
        logger.info("--- Overall Local Outlier Factor (LOF) Evaluation Metrics ---") # Changed
        logger.info(f"Average F1 Score: {mean_f1:.4f}")
        logger.info(f"Average Precision: {mean_precision:.4f}")
        logger.info(f"Average Recall: {mean_recall:.4f}")
        
        detailed_results_df = pd.DataFrame(detailed_results)
        results_csv_path = output_dir / "lof_evaluation_detailed_results.csv" # Changed
        detailed_results_df.to_csv(results_csv_path, index=False)
        logger.info(f"Detailed LOF results saved to {results_csv_path}")

        summary_results = {
            "method": "LocalOutlierFactor", # Changed
            "average_f1": mean_f1,
            "average_precision": mean_precision,
            "average_recall": mean_recall,
            "num_buildings_processed": len(overall_f1_scores),
            "num_buildings_total": len(all_building_ids),
            "base_params_used": base_lof_params_to_use # Log the base params used
        }
        summary_df = pd.DataFrame([summary_results])
        summary_csv_path = output_dir / "lof_evaluation_summary.csv" # Changed
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Summary LOF results saved to {summary_csv_path}")
    else:
        logger.warning("No buildings were processed successfully for LOF. No overall metrics to report.")

    logger.info("Local Outlier Factor (LOF) Evaluation Script Finished.") # Changed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Local Outlier Factor (LOF) based anomaly detection evaluation.") # Changed
    
    parser.add_argument('--data_path', type=str, default=lib_config.DEFAULT_DATA_CSV_PATH)
    parser.add_argument('--output_dir', type=str, default=str(Path(lib_config.DEFAULT_RESULTS_DIR) / "lof_eval")) # Changed
    parser.add_argument('--building_id_col', type=str, default='building_id')
    parser.add_argument('--meter_reading_col', type=str, default='meter_reading')
    parser.add_argument('--anomaly_col', type=str, default='anomaly')
    parser.add_argument('--scale_per_building', action='store_true', help="Scale data per building before LOF.")
    parser.set_defaults(scale_per_building=True)

    parser.add_argument('--contamination_search_values', type=str, default=None,
                        help="Contamination values: 'start,stop,step' or 'c1,c2,...'. Default uses library config.")
    parser.add_argument('--lof_base_params', type=str, default=None, # Changed
                        help="JSON string for base LOF parameters (e.g., "
                             "'{\"n_neighbors\": 20, \"metric\": \"euclidean\"}'). "
                             "Overrides library defaults. 'novelty' will be set to False.")
    # No specific --seed argument for LOF itself as it's deterministic given params.
    # Global seed from set_random_seeds() handles other potential randomness if any.

    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--log_level', type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    if args.log_file is None:
        args.log_file = str(Path(args.output_dir) / "run_lof_eval.log") # Changed
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    
    main(args)