use crate::calibrate::{CalibrationMode, Calibrator};
use crate::format::TernaryCheckpoint;
use crate::merkle::MerkleTree;
use crate::pack;
use crate::unpack;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors for checkpoint operations.
#[derive(Error, Debug)]
pub enum CheckpointError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Integrity check failed: merkle root mismatch")]
    IntegrityFailed,
    #[error("No checkpoints found")]
    NoCheckpoints,
    #[error("Invalid checkpoint: {0}")]
    Invalid(String),
}

/// Metadata for a saved checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CheckpointMeta {
    /// Unique identifier (timestamp-based).
    pub id: String,
    /// Step number.
    pub step: u64,
    /// Validation loss.
    pub validation_loss: f32,
    /// File path.
    pub path: PathBuf,
    /// Timestamp (seconds since epoch).
    pub timestamp: u64,
}

/// Manages saving, loading, and pruning ternary model checkpoints.
pub struct CheckpointManager {
    /// Directory for storing checkpoints.
    directory: PathBuf,
    /// Maximum number of checkpoints to keep.
    max_checkpoints: usize,
    /// Calibration mode.
    calibration_mode: CalibrationMode,
}

impl CheckpointManager {
    /// Create a new checkpoint manager.
    pub fn new(directory: impl Into<PathBuf>, max_checkpoints: usize) -> Self {
        Self {
            directory: directory.into(),
            max_checkpoints,
            calibration_mode: CalibrationMode::PerTensor,
        }
    }

    /// Set the calibration mode.
    pub fn with_calibration_mode(mut self, mode: CalibrationMode) -> Self {
        self.calibration_mode = mode;
        self
    }

    /// Save a checkpoint with the given weights and metadata.
    pub fn save(
        &self,
        weights: &[f32],
        shape: &[usize],
        step: u64,
        validation_loss: f32,
    ) -> Result<CheckpointMeta, CheckpointError> {
        fs::create_dir_all(&self.directory)?;

        let calibrator = Calibrator::new(self.calibration_mode.clone());
        let threshold = calibrator.find_optimal_threshold(weights);
        let calibrator = calibrator.with_threshold(threshold);

        let scales = calibrator.per_channel_scales(weights);
        let packed = pack::pack_with_threshold(weights, threshold);
        let num_trits = weights.len();

        let merkle = MerkleTree::build(&packed);

        let checkpoint = TernaryCheckpoint::new(
            shape.to_vec(),
            num_trits,
            threshold,
            scales,
            packed,
            merkle.root(),
        );

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let id = format!("ckpt_step{}_{}", step, timestamp);
        let filename = format!("{}.bin", id);
        let path = self.directory.join(&filename);

        let data = checkpoint
            .serialize()
            .map_err(|e| CheckpointError::Serialization(e.to_string()))?;
        fs::write(&path, data)?;

        let meta = CheckpointMeta {
            id,
            step,
            validation_loss,
            path,
            timestamp,
        };

        // Save metadata alongside
        let meta_path = self.directory.join(format!("{}.meta", meta.id));
        let meta_data = serde_json::to_vec_pretty(&meta)
            .map_err(|e| CheckpointError::Serialization(e.to_string()))?;
        fs::write(meta_path, meta_data)?;

        // Prune old checkpoints
        self.prune()?;

        Ok(meta)
    }

    /// Load a checkpoint from the given path.
    pub fn load(&self, path: &Path) -> Result<Vec<f32>, CheckpointError> {
        let data = fs::read(path)?;
        let checkpoint = TernaryCheckpoint::deserialize(&data)
            .map_err(|e| CheckpointError::Serialization(e.to_string()))?;

        // Verify integrity
        if !MerkleTree::verify(&checkpoint.packed_data, &checkpoint.merkle_root) {
            return Err(CheckpointError::IntegrityFailed);
        }

        // Unpack with scales
        let scales = if checkpoint.scales.len() == 1 {
            // Per-tensor scale
            vec![checkpoint.scales[0]; checkpoint.header.num_trits]
        } else {
            checkpoint.scales.clone()
        };

        let unpacked = if scales.len() == checkpoint.header.num_trits {
            unpack::unpack_with_scales(&checkpoint.packed_data, checkpoint.header.num_trits, Some(&scales))
        } else {
            unpack::unpack_with_scale(
                &checkpoint.packed_data,
                checkpoint.header.num_trits,
                scales.first().copied().unwrap_or(1.0),
            )
        };

        Ok(unpacked)
    }

    /// List all saved checkpoints sorted by validation loss (best first).
    pub fn list(&self) -> Result<Vec<CheckpointMeta>, CheckpointError> {
        let mut checkpoints = Vec::new();

        let entries = fs::read_dir(&self.directory)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "meta") {
                let data = fs::read(&path)?;
                let meta: CheckpointMeta = serde_json::from_slice(&data)
                    .map_err(|e| CheckpointError::Serialization(e.to_string()))?;
                checkpoints.push(meta);
            }
        }

        // Sort by validation loss (ascending = best first)
        checkpoints.sort_by(|a, b| {
            a.validation_loss
                .partial_cmp(&b.validation_loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(checkpoints)
    }

    /// Get the best checkpoint (lowest validation loss).
    pub fn best(&self) -> Result<Option<CheckpointMeta>, CheckpointError> {
        let list = self.list()?;
        Ok(list.into_iter().next())
    }

    /// Prune checkpoints, keeping only the N best by validation loss.
    fn prune(&self) -> Result<(), CheckpointError> {
        let mut checkpoints = self.list()?;

        if checkpoints.len() <= self.max_checkpoints {
            return Ok(());
        }

        // Keep the best N, remove the rest
        let to_remove = checkpoints.split_off(self.max_checkpoints);

        for meta in to_remove {
            let _ = fs::remove_file(&meta.path);
            let meta_path = self.directory.join(format!("{}.meta", meta.id));
            let _ = fs::remove_file(meta_path);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_manager(max: usize) -> (TempDir, CheckpointManager) {
        let dir = TempDir::new().unwrap();
        let mgr = CheckpointManager::new(dir.path().to_path_buf(), max);
        (dir, mgr)
    }

    #[test]
    fn test_save_and_load() {
        let (_dir, mgr) = make_manager(5);
        let weights: Vec<f32> = vec![1.0, -1.0, 0.05, 2.0, -0.5, 0.8, -1.2, 3.0];
        let meta = mgr.save(&weights, &[2, 4], 100, 0.5).unwrap();
        let loaded = mgr.load(&meta.path).unwrap();
        // Should have same length
        assert_eq!(loaded.len(), weights.len());
    }

    #[test]
    fn test_list_checkpoints() {
        let (_dir, mgr) = make_manager(10);
        mgr.save(&[1.0f32, -1.0, 0.5], &[3], 1, 0.8).unwrap();
        mgr.save(&[1.0f32, -1.0, 0.5], &[3], 2, 0.5).unwrap();
        mgr.save(&[1.0f32, -1.0, 0.5], &[3], 3, 0.3).unwrap();

        let list = mgr.list().unwrap();
        assert_eq!(list.len(), 3);
        // Should be sorted by loss ascending
        assert!(list[0].validation_loss <= list[1].validation_loss);
        assert!(list[1].validation_loss <= list[2].validation_loss);
    }

    #[test]
    fn test_best_checkpoint() {
        let (_dir, mgr) = make_manager(10);
        mgr.save(&[1.0f32, -1.0], &[2], 1, 0.9).unwrap();
        mgr.save(&[1.0f32, -1.0], &[2], 2, 0.3).unwrap();
        mgr.save(&[1.0f32, -1.0], &[2], 3, 0.6).unwrap();

        let best = mgr.best().unwrap().unwrap();
        assert!((best.validation_loss - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_prune() {
        let (_dir, mgr) = make_manager(2);
        mgr.save(&[1.0f32], &[1], 1, 0.9).unwrap();
        mgr.save(&[1.0f32], &[1], 2, 0.3).unwrap();
        mgr.save(&[1.0f32], &[1], 3, 0.6).unwrap();

        let list = mgr.list().unwrap();
        assert_eq!(list.len(), 2);
        // Should keep the 2 best (0.3 and 0.6)
        assert!((list[0].validation_loss - 0.3).abs() < 1e-5);
        assert!((list[1].validation_loss - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_empty_weights() {
        let (_dir, mgr) = make_manager(5);
        let result = mgr.save(&[], &[], 1, 1.0);
        // Should succeed (empty weights are valid)
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_nonexistent() {
        let (_dir, mgr) = make_manager(5);
        let result = mgr.load(Path::new("/nonexistent/file.bin"));
        assert!(result.is_err());
    }

    #[test]
    fn test_no_checkpoints() {
        let (_dir, mgr) = make_manager(5);
        let best = mgr.best().unwrap();
        assert!(best.is_none());
    }
}
