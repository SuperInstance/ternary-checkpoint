use crate::merkle::Hash;
use serde::{Deserialize, Serialize};

/// Magic bytes for the checkpoint format: "TERN" in ASCII.
const MAGIC: [u8; 4] = [b'T', b'E', b'R', b'N'];

/// Current format version.
const VERSION: u32 = 1;

/// Checkpoint header containing metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CheckpointHeader {
    /// Magic bytes for format identification.
    pub magic: [u8; 4],
    /// Format version.
    pub version: u32,
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Number of ternary values (trits).
    pub num_trits: usize,
    /// Quantization threshold used.
    pub threshold: f32,
    /// Number of scaling factors.
    pub num_scales: usize,
    /// Number of packed u32 words.
    pub num_packed: usize,
}

impl CheckpointHeader {
    /// Create a new header.
    pub fn new(
        shape: Vec<usize>,
        num_trits: usize,
        threshold: f32,
        num_scales: usize,
        num_packed: usize,
    ) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            shape,
            num_trits,
            threshold,
            num_scales,
            num_packed,
        }
    }

    /// Validate the header.
    pub fn validate(&self) -> Result<(), FormatError> {
        if self.magic != MAGIC {
            return Err(FormatError::InvalidMagic(self.magic));
        }
        if self.version != VERSION {
            return Err(FormatError::UnsupportedVersion(self.version));
        }
        Ok(())
    }
}

/// Errors for format operations.
#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("invalid magic bytes: {0:?}")]
    InvalidMagic([u8; 4]),
    #[error("unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("data truncated or corrupt")]
    TruncatedData,
}

/// A complete ternary checkpoint: header + data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryCheckpoint {
    /// Header metadata.
    pub header: CheckpointHeader,
    /// Scaling factors.
    pub scales: Vec<f32>,
    /// Packed ternary data.
    pub packed_data: Vec<u32>,
    /// Merkle root hash for integrity.
    pub merkle_root: Hash,
}

impl TernaryCheckpoint {
    /// Create a new checkpoint.
    pub fn new(
        shape: Vec<usize>,
        num_trits: usize,
        threshold: f32,
        scales: Vec<f32>,
        packed_data: Vec<u32>,
        merkle_root: Hash,
    ) -> Self {
        let header = CheckpointHeader::new(
            shape,
            num_trits,
            threshold,
            scales.len(),
            packed_data.len(),
        );
        Self {
            header,
            scales,
            packed_data,
            merkle_root,
        }
    }

    /// Serialize the checkpoint to bytes.
    pub fn serialize(&self) -> Result<Vec<u8>, FormatError> {
        self.header.validate()?;

        bincode::serialize(self).map_err(|e| FormatError::Serialization(e.to_string()))
    }

    /// Deserialize a checkpoint from bytes.
    pub fn deserialize(data: &[u8]) -> Result<Self, FormatError> {
        let checkpoint: Self =
            bincode::deserialize(data).map_err(|e| FormatError::Serialization(e.to_string()))?;

        checkpoint.header.validate()?;

        Ok(checkpoint)
    }

    /// Get the compression ratio compared to f32 storage.
    pub fn compression_ratio(&self) -> f64 {
        let original_bytes = self.header.num_trits as f64 * 4.0; // f32 = 4 bytes each
        let packed_bytes = self.packed_data.len() as f64 * 4.0;
        if packed_bytes == 0.0 {
            return 0.0;
        }
        original_bytes / packed_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_creation() {
        let header = CheckpointHeader::new(vec![2, 4], 8, 0.2, 1, 1);
        assert_eq!(header.magic, MAGIC);
        assert_eq!(header.version, VERSION);
        assert_eq!(header.shape, vec![2, 4]);
    }

    #[test]
    fn test_header_validate_valid() {
        let header = CheckpointHeader::new(vec![2, 4], 8, 0.2, 1, 1);
        assert!(header.validate().is_ok());
    }

    #[test]
    fn test_header_validate_bad_magic() {
        let mut header = CheckpointHeader::new(vec![2, 4], 8, 0.2, 1, 1);
        header.magic = [0xFF; 4];
        assert!(header.validate().is_err());
    }

    #[test]
    fn test_checkpoint_serialize_deserialize() {
        let merkle_root = [0xAB_u8; 32];
        let checkpoint = TernaryCheckpoint::new(
            vec![2, 4],
            8,
            0.2,
            vec![1.5],
            vec![0xDEADBEEF],
            merkle_root,
        );

        let bytes = checkpoint.serialize().unwrap();
        let restored = TernaryCheckpoint::deserialize(&bytes).unwrap();

        assert_eq!(restored.header.shape, vec![2, 4]);
        assert_eq!(restored.header.num_trits, 8);
        assert_eq!(restored.scales, vec![1.5]);
        assert_eq!(restored.packed_data, vec![0xDEADBEEF]);
        assert_eq!(restored.merkle_root, merkle_root);
    }

    #[test]
    fn test_compression_ratio() {
        let checkpoint = TernaryCheckpoint::new(
            vec![16],
            16,
            0.2,
            vec![1.0],
            vec![0u32], // 16 trits in 1 u32
            [0u8; 32],
        );
        // 16 f32s = 64 bytes, 1 u32 = 4 bytes → ratio = 16.0
        let ratio = checkpoint.compression_ratio();
        assert!((ratio - 16.0).abs() < 0.01, "got {}", ratio);
    }

    #[test]
    fn test_deserialize_garbage() {
        let result = TernaryCheckpoint::deserialize(&[0xFF; 100]);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_checkpoint() {
        let checkpoint = TernaryCheckpoint::new(
            vec![],
            0,
            0.2,
            vec![1.0],
            vec![],
            [0u8; 32],
        );
        let bytes = checkpoint.serialize().unwrap();
        let restored = TernaryCheckpoint::deserialize(&bytes).unwrap();
        assert_eq!(restored.header.num_trits, 0);
        assert!(restored.packed_data.is_empty());
    }
}
