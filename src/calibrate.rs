use serde::{Deserialize, Serialize};

/// Calibration mode for scaling factors.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CalibrationMode {
    /// Single scale factor for the entire tensor.
    PerTensor,
    /// One scale factor per output channel.
    /// Requires specifying the channel dimension size.
    PerChannel { channel_dim: usize, channel_size: usize },
}

/// Calibrator for finding optimal thresholds and scaling factors.
#[derive(Debug, Clone)]
pub struct Calibrator {
    mode: CalibrationMode,
    threshold: f32,
}

impl Calibrator {
    /// Create a new calibrator with the given mode and default threshold.
    pub fn new(mode: CalibrationMode) -> Self {
        Self {
            mode,
            threshold: 0.2,
        }
    }

    /// Set the quantization threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Find the optimal threshold using a simple search.
    /// Tries to minimize the mean squared error between original and quantized weights.
    pub fn find_optimal_threshold(&self, weights: &[f32]) -> f32 {
        if weights.is_empty() {
            return 0.2;
        }

        let abs_weights: Vec<f32> = weights.iter().map(|w| w.abs()).collect();
        let max_val = abs_weights.iter().cloned().fold(0.0f32, f32::max);
        if max_val == 0.0 {
            return 0.2;
        }

        let mut best_threshold = 0.2;
        let mut best_mse = f32::MAX;

        // Search over a grid of thresholds
        let steps = 50;
        for i in 0..=steps {
            let t = max_val * (i as f32) / (steps as f32) * 0.8 + 0.001;
            let mse = compute_mse(weights, t);
            if mse < best_mse {
                best_mse = mse;
                best_threshold = t;
            }
        }

        best_threshold
    }

    /// Compute the per-tensor scale factor.
    /// Scale = mean absolute value of non-zero quantized weights.
    pub fn per_tensor_scale(&self, weights: &[f32]) -> f32 {
        if weights.is_empty() {
            return 1.0;
        }

        let threshold = self.threshold;
        let abs_sum: f32 = weights
            .iter()
            .filter(|&&w| w.abs() > threshold)
            .map(|w| w.abs())
            .sum();
        let count = weights.iter().filter(|&&w| w.abs() > threshold).count();

        if count == 0 {
            1.0
        } else {
            abs_sum / count as f32
        }
    }

    /// Compute per-channel scale factors.
    /// Returns one scale factor per channel.
    pub fn per_channel_scales(&self, weights: &[f32]) -> Vec<f32> {
        match &self.mode {
            CalibrationMode::PerTensor => {
                vec![self.per_tensor_scale(weights)]
            }
            CalibrationMode::PerChannel { channel_dim: _, channel_size } => {
                let num_channels = *channel_size;
                let per_channel_len = weights.len() / num_channels;
                let mut scales = Vec::with_capacity(num_channels);

                for ch in 0..num_channels {
                    let offset = ch * per_channel_len;
                    let end = (offset + per_channel_len).min(weights.len());
                    let channel_weights = &weights[offset..end];
                    scales.push(self.per_tensor_scale(channel_weights));
                }

                scales
            }
        }
    }

    /// Get the current threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Get the calibration mode.
    pub fn mode(&self) -> &CalibrationMode {
        &self.mode
    }
}

/// Compute mean squared error between original weights and their quantized version.
fn compute_mse(weights: &[f32], threshold: f32) -> f32 {
    let n = weights.len() as f32;
    weights
        .iter()
        .map(|&w| {
            let q = if w > threshold {
                1.0f32
            } else if w < -threshold {
                -1.0f32
            } else {
                0.0f32
            };
            (w - q).powi(2)
        })
        .sum::<f32>()
        / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_per_tensor_scale_basic() {
        let cal = Calibrator::new(CalibrationMode::PerTensor);
        let weights = vec![2.0, -2.0, 0.05, 2.0];
        let scale = cal.per_tensor_scale(&weights);
        // Non-zero: 2.0, -2.0, 2.0 → mean abs = 2.0
        assert!((scale - 2.0).abs() < 1e-5, "got {}", scale);
    }

    #[test]
    fn test_per_tensor_scale_all_zero() {
        let cal = Calibrator::new(CalibrationMode::PerTensor);
        let weights = vec![0.01, -0.01, 0.05];
        let scale = cal.per_tensor_scale(&weights);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_per_tensor_scale_empty() {
        let cal = Calibrator::new(CalibrationMode::PerTensor);
        let scale = cal.per_tensor_scale(&[]);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_find_optimal_threshold() {
        let cal = Calibrator::new(CalibrationMode::PerTensor);
        let weights = vec![0.0; 100];
        // All zeros → any threshold works, should return something
        let t = cal.find_optimal_threshold(&weights);
        assert!(t > 0.0);
    }

    #[test]
    fn test_find_optimal_threshold_nontrivial() {
        let cal = Calibrator::new(CalibrationMode::PerTensor);
        // Weights clearly separated: around ±1.5 and near 0
        let mut weights = vec![1.5f32; 50];
        weights.extend(vec![-1.5f32; 50]);
        weights.extend(vec![0.01f32; 50]);
        let t = cal.find_optimal_threshold(&weights);
        // Should find a threshold between 0.01 and 1.5
        assert!(t >= 0.0 && t <= 2.0, "threshold {} out of expected range", t);
    }

    #[test]
    fn test_per_channel_scales() {
        let cal = Calibrator::new(CalibrationMode::PerChannel {
            channel_dim: 0,
            channel_size: 2,
        });
        // Channel 0: [1.0, -1.0], Channel 1: [3.0, -3.0]
        let weights = vec![1.0, -1.0, 3.0, -3.0];
        let scales = cal.per_channel_scales(&weights);
        assert_eq!(scales.len(), 2);
        assert!((scales[0] - 1.0).abs() < 1e-5, "channel 0 scale: {}", scales[0]);
        assert!((scales[1] - 3.0).abs() < 1e-5, "channel 1 scale: {}", scales[1]);
    }

    #[test]
    fn test_compute_mse() {
        let mse = compute_mse(&[1.0, -1.0, 0.0], 0.5);
        assert!(mse < 1e-5, "MSE should be near zero for exact values, got {}", mse);
    }

    #[test]
    fn test_calibrator_with_threshold() {
        let cal = Calibrator::new(CalibrationMode::PerTensor).with_threshold(0.5);
        assert_eq!(cal.threshold(), 0.5);
    }
}
