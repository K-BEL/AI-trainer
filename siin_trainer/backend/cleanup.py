import shutil
from pathlib import Path
import logging

logger = logging.getLogger("trainer")

def cleanup_old_runs(base_dir="runs/ultralytics", keep_n=3):
    """
    Deletes old training run directories, keeping only the 'keep_n' most recent ones.
    
    Args:
        base_dir (str): Base directory where runs are stored.
        keep_n (int): Number of most recent runs to preserve.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"Cleanup directory {base_dir} does not exist. Skipping.")
        return

    # Find all run-* directories
    run_dirs = sorted(list(base_path.glob("run-*")))
    
    if len(run_dirs) <= keep_n:
        # Nothing to clean up
        return

    # Directories to delete (all but the last keep_n)
    to_delete = run_dirs[:-keep_n]
    
    deleted_count = 0
    for run_dir in to_delete:
        try:
            if run_dir.is_dir():
                shutil.rmtree(run_dir)
                deleted_count += 1
                logger.debug(f"Deleted old run directory: {run_dir}")
        except Exception as e:
            logger.error(f"Failed to delete {run_dir}: {e}")

    if deleted_count > 0:
        logger.info(f"Cleanup complete: Removed {deleted_count} old run directories. Kept the latest {keep_n}.")
