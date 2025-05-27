# scripts/run_isolation_forest_eval.py
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
from tsfm_ad_lib.models.tree_based import find_best_iforest_params_for_building
from tsfm_ad_lib.utils import setup_logger, set_random_seeds
from tsfm_ad_lib import config as lib_config

def main(args):
    """
    Main execution function for Isolation Forest evaluation script.
    """
    logger = setup_logger("IForestEvalScript", log_file=args.log_file, level=args.log_level.upper())
    set_random_seeds(args.seed if args.seed is not None else lib_config.RANDOM_SEED)
    logger.info("Starting Isolation Forest Evaluation Script")
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

    # 2. Preprocess Data (Median imputation per building, NO SCALING globally)
    # Scaling is handled per-building within the IForest function if scale_data=True
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

    # 3. Evaluate Isolation Forest per building
    all_building_ids = sorted(imputed_df[args.building_id_col].unique())
    
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
            logger.info(f"Using custom contamination search values for IForest: {contamination_search_list}")
        except ValueError:
            logger.warning(f"Could not parse contamination_search_values '{args.contamination_search_values}'. Using default.")
            start_c, stop_c, step_c = lib_config.DEFAULT_CONTAMINATION_SEARCH_RANGE
            contamination_search_list = list(np.arange(start_c, stop_c, step_c))
    else: # Use default if not provided
        start_c, stop_c, step_c = lib_config.DEFAULT_CONTAMINATION_SEARCH_RANGE
        contamination_search_list = list(np.arange(start_c, stop_c, step_c))


    # Parse base IForest parameters
    base_if_params = lib_config.DEFAULT_IFOREST_BASE_PARAMS.copy()
    if args.iforest_base_params:
        try:
            cli_params = json.loads(args.iforest_base_params)
            base_if_params.update(cli_params) # Override defaults with CLI params
            logger.info(f"Using custom IForest base parameters: {base_if_params}")
        except json.JSONDecodeError:
            logger.error(f"Could not parse iforest_base_params JSON: '{args.iforest_base_params}'. Using defaults.")
    # Ensure seed is set in params if provided via CLI, overriding potential default
    if args.seed is not None:
         base_if_params['random_state'] = args.seed

    logger.info(f"PROCESSING ONLY {len(all_building_ids)} buildings for smoke test...") # Add a log
    for b_id in all_building_ids:
        building_data_slice = imputed_df[imputed_df[args.building_id_col] == b_id]
        if building_data_slice.empty or building_data_slice[args.meter_reading_col].isnull().all():
            logger.warning(f"Skipping building ID {b_id} (IForest) due to empty or all-NaN meter readings.")
            detailed_results.append({
                "building_id": b_id, "f1": 0.0, "precision": 0.0, "recall": 0.0, "best_contamination": np.nan, "status": "skipped_no_data"
            })
            continue
            
        logger.debug(f"Processing building ID {b_id} (IForest)")
        
        metrics = find_best_iforest_params_for_building(
            building_data=building_data_slice,
            meter_reading_col=args.meter_reading_col,
            anomaly_col=args.anomaly_col,
            base_iforest_params=base_if_params,
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
        logger.debug(f"Building ID {b_id} (IForest) - Best F1: {metrics['f1']:.4f} with Contamination={metrics['contamination']:.4f}")

    # 4. Report and Save Results
    if overall_f1_scores:
        mean_f1 = np.mean(overall_f1_scores)
        mean_precision = np.mean(overall_precision_scores)
        mean_recall = np.mean(overall_recall_scores)
        
        logger.info("--- Overall Isolation Forest Evaluation Metrics ---")
        logger.info(f"Average F1 Score: {mean_f1:.4f}")
        logger.info(f"Average Precision: {mean_precision:.4f}")
        logger.info(f"Average Recall: {mean_recall:.4f}")
        
        detailed_results_df = pd.DataFrame(detailed_results)
        results_csv_path = output_dir / "iforest_evaluation_detailed_results.csv"
        detailed_results_df.to_csv(results_csv_path, index=False)
        logger.info(f"Detailed IForest results saved to {results_csv_path}")

        summary_results = {
            "method": "IsolationForest",
            "average_f1": mean_f1,
            "average_precision": mean_precision,
            "average_recall": mean_recall,
            "num_buildings_processed": len(overall_f1_scores),
            "num_buildings_total": len(all_building_ids),
            "base_params_used": base_if_params # Log the base params used
        }
        summary_df = pd.DataFrame([summary_results])
        summary_csv_path = output_dir / "iforest_evaluation_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Summary IForest results saved to {summary_csv_path}")
    else:
        logger.warning("No buildings were processed successfully for IForest. No overall metrics to report.")

    logger.info("Isolation Forest Evaluation Script Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Isolation Forest based anomaly detection evaluation.")
    
    parser.add_argument('--data_path', type=str, default=lib_config.DEFAULT_DATA_CSV_PATH)
    parser.add_argument('--output_dir', type=str, default=str(Path(lib_config.DEFAULT_RESULTS_DIR) / "iforest_eval"))
    parser.add_argument('--building_id_col', type=str, default='building_id')
    parser.add_argument('--meter_reading_col', type=str, default='meter_reading')
    parser.add_argument('--anomaly_col', type=str, default='anomaly')
    parser.add_argument('--scale_per_building', action='store_true', help="Scale data per building before IForest.")
    parser.set_defaults(scale_per_building=True)

    parser.add_argument('--contamination_search_values', type=str, default=None,
                        help="Contamination values: 'start,stop,step' or 'c1,c2,...'. Default uses library config.")
    parser.add_argument('--iforest_base_params', type=str, default=None,
                        help="JSON string for base Isolation Forest parameters (e.g., "
                             "'{\"n_estimators\": 100, \"max_samples\": 0.1}'). "
                             "Overrides library defaults.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for reproducibility (sets IForest random_state). Overrides config/default.")

    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--log_level', type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    if args.log_file is None:
        args.log_file = str(Path(args.output_dir) / "run_iforest_eval.log")
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    
    main(args)