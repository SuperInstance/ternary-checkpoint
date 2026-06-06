//! # ternary-checkpoint
//!
//! Model checkpointing optimized for ternary networks.
//! Trits pack 16 to a u32 (2 bits each), giving 16× compression over float32.

/// A single ternary value.
pub type Trit = i8;

/// Pack 16 trits into a u32 (2 bits per trit: -1→00, 0→01, +1→10, 11=invalid).
pub fn pack_trits(trits: &[Trit]) -> Vec<u32> {
    assert!(trits.len() <= 16, "pack_trits takes at most 16 trits");
    let mut packed: u32 = 0;
    for (i, &t) in trits.iter().enumerate() {
        let bits = match t {
            -1 => 0b00u32,
            0 => 0b01u32,
            1 => 0b10u32,
            _ => panic!("Invalid trit"),
        };
        packed |= bits << (i * 2);
    }
    vec![packed]
}

/// Pack a slice of trits into packed u32s (16 trits per u32).
pub fn pack_trits_batch(trits: &[Trit]) -> Vec<u32> {
    let full_chunks = trits.len() / 16;
    let remainder = trits.len() % 16;
    let mut result = Vec::with_capacity(full_chunks + if remainder > 0 { 1 } else { 0 });

    for chunk in trits.chunks(16) {
        let mut packed: u32 = 0;
        for (i, &t) in chunk.iter().enumerate() {
            let bits = match t {
                -1 => 0b00u32,
                0 => 0b01u32,
                1 => 0b10u32,
                _ => panic!("Invalid trit"),
            };
            packed |= bits << (i * 2);
        }
        result.push(packed);
    }
    result
}

/// Unpack u32s back to trits.
pub fn unpack_trits_batch(packed: &[u32], total_trits: usize) -> Vec<Trit> {
    let mut result = Vec::with_capacity(total_trits);
    for &p in packed {
        for i in 0..16 {
            if result.len() >= total_trits {
                break;
            }
            let bits = (p >> (i * 2)) & 0b11;
            let trit = match bits {
                0b00 => -1,
                0b01 => 0,
                0b10 => 1,
                _ => 0, // treat invalid as 0
            };
            result.push(trit);
        }
    }
    result
}

/// Simple rolling checksum for data integrity.
pub fn trit_checksum(trits: &[Trit]) -> u32 {
    let mut hash: u32 = 0x811c9dc5; // FNV offset basis
    for &t in trits {
        let byte = match t {
            -1 => 0u8,
            0 => 1u8,
            1 => 2u8,
            _ => 3u8,
        };
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x01000193); // FNV prime
    }
    hash
}

/// A model checkpoint with metadata.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Model name / identifier
    pub name: String,
    /// Training step this checkpoint was taken at
    pub step: usize,
    /// Packed ternary weights
    pub packed_weights: Vec<u32>,
    /// Total number of trits (may not be multiple of 16)
    pub total_trits: usize,
    /// Checksum of original trits
    pub checksum: u32,
    /// Learning rate at checkpoint time
    pub learning_rate: f64,
    /// Training loss at checkpoint time
    pub loss: f64,
}

impl Checkpoint {
    /// Create a new checkpoint from ternary weights.
    pub fn new(name: &str, step: usize, weights: &[Trit], lr: f64, loss: f64) -> Self {
        let checksum = trit_checksum(weights);
        let packed_weights = pack_trits_batch(weights);
        Self {
            name: name.to_string(),
            step,
            packed_weights,
            total_trits: weights.len(),
            checksum,
            learning_rate: lr,
            loss,
        }
    }

    /// Verify checkpoint integrity.
    pub fn verify(&self) -> bool {
        let unpacked = self.unpack();
        trit_checksum(&unpacked) == self.checksum
    }

    /// Unpack weights back to trits.
    pub fn unpack(&self) -> Vec<Trit> {
        unpack_trits_batch(&self.packed_weights, self.total_trits)
    }

    /// Compression ratio vs float32.
    pub fn compression_ratio(&self) -> f64 {
        let float_bytes = self.total_trits as f64 * 4.0;
        let packed_bytes = self.packed_weights.len() as f64 * 4.0;
        float_bytes / packed_bytes
    }

    /// Size in bytes of packed weights.
    pub fn packed_size_bytes(&self) -> usize {
        self.packed_weights.len() * 4
    }

    /// Serialize to a simple binary format.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Header: name_len (u16), step (u64), total_trits (u64), checksum (u32), lr (f64), loss (f64)
        let name_bytes = self.name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&(self.step as u64).to_le_bytes());
        buf.extend_from_slice(&(self.total_trits as u64).to_le_bytes());
        buf.extend_from_slice(&self.checksum.to_le_bytes());
        buf.extend_from_slice(&self.learning_rate.to_le_bytes());
        buf.extend_from_slice(&self.loss.to_le_bytes());
        // Packed weights count
        buf.extend_from_slice(&(self.packed_weights.len() as u64).to_le_bytes());
        // Packed weights
        for &w in &self.packed_weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf
    }

    /// Deserialize from binary format.
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < 46 { return None; }
        let mut pos = 0;
        let name_len = u16::from_le_bytes(data[pos..pos+2].try_into().ok()?) as usize;
        pos += 2;
        let name = String::from_utf8(data[pos..pos+name_len].to_vec()).ok()?;
        pos += name_len;
        let step = u64::from_le_bytes(data[pos..pos+8].try_into().ok()?) as usize;
        pos += 8;
        let total_trits = u64::from_le_bytes(data[pos..pos+8].try_into().ok()?) as usize;
        pos += 8;
        let checksum = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?);
        pos += 4;
        let learning_rate = f64::from_le_bytes(data[pos..pos+8].try_into().ok()?);
        pos += 8;
        let loss = f64::from_le_bytes(data[pos..pos+8].try_into().ok()?);
        pos += 8;
        let pw_count = u64::from_le_bytes(data[pos..pos+8].try_into().ok()?) as usize;
        pos += 8;
        let mut packed_weights = Vec::with_capacity(pw_count);
        for _ in 0..pw_count {
            packed_weights.push(u32::from_le_bytes(data[pos..pos+4].try_into().ok()?));
            pos += 4;
        }
        Some(Self { name, step, packed_weights, total_trits, checksum, learning_rate, loss })
    }
}

/// Checkpoint manager with keep-best semantics.
#[derive(Debug)]
pub struct CheckpointManager {
    checkpoints: Vec<Checkpoint>,
    max_keep: usize,
    best_loss: f64,
    best_index: usize,
}

impl CheckpointManager {
    pub fn new(max_keep: usize) -> Self {
        Self {
            checkpoints: Vec::new(),
            max_keep,
            best_loss: f64::INFINITY,
            best_index: 0,
        }
    }

    /// Save a checkpoint. Returns true if this is the best so far.
    pub fn save(&mut self, cp: Checkpoint) -> bool {
        let is_best = cp.loss < self.best_loss;
        if is_best {
            self.best_loss = cp.loss;
            self.best_index = self.checkpoints.len();
        }
        self.checkpoints.push(cp);
        // Evict worst if over limit (but never evict the best)
        while self.checkpoints.len() > self.max_keep {
            let worst_idx = self.find_worst();
            self.checkpoints.remove(worst_idx);
            if worst_idx < self.best_index {
                self.best_index -= 1;
            } else if worst_idx == self.best_index {
                // Shouldn't happen — we never evict the best
                self.best_index = self.checkpoints.len().saturating_sub(1);
            }
        }
        is_best
    }

    fn find_worst(&self) -> usize {
        self.checkpoints.iter().enumerate()
            .filter(|(i, _)| *i != self.best_index)
            .max_by(|(_, a), (_, b)| a.loss.partial_cmp(&b.loss).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get the best checkpoint (lowest loss).
    pub fn best(&self) -> Option<&Checkpoint> {
        self.checkpoints.get(self.best_index)
    }

    /// Get the latest checkpoint.
    pub fn latest(&self) -> Option<&Checkpoint> {
        self.checkpoints.last()
    }

    /// Number of stored checkpoints.
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let original: Vec<Trit> = vec![-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1];
        let packed = pack_trits_batch(&original);
        let unpacked = unpack_trits_batch(&packed, original.len());
        assert_eq!(original, unpacked);
    }

    #[test]
    fn test_pack_unpack_partial() {
        let original: Vec<Trit> = vec![-1, 0, 1, -1, 0];
        let packed = pack_trits_batch(&original);
        let unpacked = unpack_trits_batch(&packed, original.len());
        assert_eq!(original, unpacked);
    }

    #[test]
    fn test_compression_ratio() {
        // 16 trits → 1 u32 = 16× compression vs f32
        let weights: Vec<Trit> = vec![1; 16];
        let cp = Checkpoint::new("test", 0, &weights, 0.01, 1.0);
        assert!((cp.compression_ratio() - 16.0).abs() < 0.01);
    }

    #[test]
    fn test_large_weights_compression() {
        let weights: Vec<Trit> = vec![1; 1024];
        let cp = Checkpoint::new("test", 0, &weights, 0.01, 1.0);
        assert!((cp.compression_ratio() - 16.0).abs() < 0.01);
        assert_eq!(cp.packed_size_bytes(), 256); // 1024/16 * 4
    }

    #[test]
    fn test_checksum_deterministic() {
        let w1: Vec<Trit> = vec![-1, 0, 1, 0, -1, 1];
        let w2: Vec<Trit> = vec![-1, 0, 1, 0, -1, 1];
        assert_eq!(trit_checksum(&w1), trit_checksum(&w2));
    }

    #[test]
    fn test_checksum_different() {
        let w1: Vec<Trit> = vec![-1, 0, 1];
        let w2: Vec<Trit> = vec![1, 0, -1];
        assert_ne!(trit_checksum(&w1), trit_checksum(&w2));
    }

    #[test]
    fn test_checkpoint_verify() {
        let weights: Vec<Trit> = vec![-1, 0, 1, -1, 0, 1, -1, 0, 1];
        let cp = Checkpoint::new("test", 100, &weights, 0.001, 0.5);
        assert!(cp.verify());
    }

    #[test]
    fn test_checkpoint_unpack_matches() {
        let weights: Vec<Trit> = vec![-1, 0, 1, -1, 0, 1, 1, 1, -1, 0];
        let cp = Checkpoint::new("model", 42, &weights, 0.01, 0.25);
        assert_eq!(cp.unpack(), weights);
    }

    #[test]
    fn test_serialize_deserialize() {
        let weights: Vec<Trit> = vec![-1, 0, 1, -1, 0, 1, 1, 1, -1, 0, -1, -1, 0, 0, 1, 1];
        let cp = Checkpoint::new("test-model", 500, &weights, 0.001, 0.42);
        let serialized = cp.serialize();
        let restored = Checkpoint::deserialize(&serialized).unwrap();
        assert_eq!(restored.name, "test-model");
        assert_eq!(restored.step, 500);
        assert_eq!(restored.total_trits, 16);
        assert_eq!(restored.checksum, cp.checksum);
        assert!((restored.learning_rate - 0.001).abs() < 1e-10);
        assert!((restored.loss - 0.42).abs() < 1e-10);
        assert_eq!(restored.unpack(), weights);
    }

    #[test]
    fn test_manager_keeps_best() {
        let mut mgr = CheckpointManager::new(3);
        mgr.save(Checkpoint::new("m", 1, &vec![1; 16], 0.01, 1.0));
        mgr.save(Checkpoint::new("m", 2, &vec![1; 16], 0.01, 0.5));
        mgr.save(Checkpoint::new("m", 3, &vec![1; 16], 0.01, 0.8));
        mgr.save(Checkpoint::new("m", 4, &vec![1; 16], 0.01, 0.3)); // best
        assert_eq!(mgr.len(), 3);
        assert!((mgr.best().unwrap().loss - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_manager_latest() {
        let mut mgr = CheckpointManager::new(5);
        mgr.save(Checkpoint::new("m", 1, &vec![1; 16], 0.01, 1.0));
        mgr.save(Checkpoint::new("m", 2, &vec![1; 16], 0.01, 0.5));
        assert_eq!(mgr.latest().unwrap().step, 2);
    }
}
