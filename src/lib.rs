//! # ternary-checkpoint
//!
//! **Ternary model checkpointing with 16× compression and integrity verification.**
//!
//! Neural networks with ternary weights (−1, 0, +1) need only 2 bits per weight to store
//! losslessly. This crate packs 16 trits into a single `u32`, achieving a 16× compression
//! ratio over float32 — with perfect round-trip fidelity verified by
//! [`verify_integrity`].
//!
//! # Packing Scheme
//!
//! | Trit | Binary | Meaning            |
//! |------|--------|--------------------|
//! | −1   | `00`   | Negative connection|
//! |  0   | `01`   | Pruned / inactive  |
//! | +1   | `10`   | Positive connection|
//! |  —   | `11`   | Invalid (unused)   |
//!
//! # Quick Example
//!
//! ```
//! use ternary_checkpoint::*;
//!
//! // Pack a weight matrix
//! let weights = vec![
//!     vec![-1_i8, 0, 1],
//!     vec![1_i8, -1, 0],
//! ];
//! let compressed = pack_matrix(&weights);
//!
//! // Verify lossless round-trip
//! let restored = unpack_matrix(&compressed, 2, 3);
//! assert_eq!(weights, restored);
//!
//! // Check compression ratio — 6 trits pack into 1 u32 (24 bytes → 4 bytes)
//! let ratio = compression_ratio(6 * 4, compressed.len() * 4); // float32 vs packed
//! assert!(ratio > 5.0);
//! ```
//!
//! # Key Functions
//!
//! - [`pack_trits`] / [`unpack_trits`] — Pack/unpack ≤16 trits per `u32`
//! - [`pack_matrix`] / [`unpack_matrix`] — Full weight matrix compression
//! - [`verify_integrity`] — Lossless round-trip verification
//! - [`keep_top_k`] — Sparsify by keeping only top-k magnitude weights
//! - [`compression_ratio`] — Compute compression ratio
//!
//! # Ecosystem
//!
//! Part of the [SuperInstance](https://github.com/SuperInstance) ecosystem.
//! Depends on [`ternary-types`](https://github.com/SuperInstance/ternary-types) for shared type definitions.

/// A single ternary value: -1, 0, or +1.
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

/// Unpack a u32 into 16 trits.
pub fn unpack_trits(packed: &[u32], count: usize) -> Vec<Trit> {
    let mut trits = Vec::with_capacity(count);
    for &p in packed {
        for i in 0..16.min(count - trits.len()) {
            let bits = (p >> (i * 2)) & 0b11;
            let t = match bits {
                0b00 => -1,
                0b01 => 0,
                0b10 => 1,
                _ => 0, // invalid → treat as 0
            };
            trits.push(t);
        }
    }
    trits.truncate(count);
    trits
}

/// Verify integrity of a checkpoint by re-packing and comparing.
pub fn verify_integrity(weights: &[Trit], compressed: &[u32]) -> bool {
    let unpacked = unpack_trits(compressed, weights.len());
    weights == unpacked
}

/// Pack an entire weight matrix.
pub fn pack_matrix(matrix: &[Vec<Trit>]) -> Vec<u32> {
    let flat: Vec<Trit> = matrix.iter().flat_map(|r| r.iter().copied()).collect();
    let mut result = Vec::new();
    for chunk in flat.chunks(16) {
        result.extend(pack_trits(chunk));
    }
    result
}

/// Unpack a compressed weight matrix.
pub fn unpack_matrix(packed: &[u32], rows: usize, cols: usize) -> Vec<Vec<Trit>> {
    let total = rows * cols;
    let flat = unpack_trits(packed, total);
    let mut matrix = Vec::with_capacity(rows);
    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        matrix.push(flat[start..end].to_vec());
    }
    matrix
}

/// Compression ratio achieved.
pub fn compression_ratio(original_bytes: usize, compressed_bytes: usize) -> f64 {
    if compressed_bytes == 0 {
        return 0.0;
    }
    original_bytes as f64 / compressed_bytes as f64
}

/// Keep only the top-k highest magnitude weights.
pub fn keep_top_k(weights: &mut [Trit], k: usize) {
    if k >= weights.len() {
        return;
    }
    let mut indices: Vec<usize> = (0..weights.len()).collect();
    indices.sort_by(|&a, &b| {
        let wa = weights[a].abs();
        let wb = weights[b].abs();
        wb.cmp(&wa)
    });
    for &i in &indices[k..] {
        weights[i] = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let trits = vec![-1, 0, 1, -1, 0, 1, -1, 1, 0, 0, 1, -1, 1, -1, 0, 1];
        let packed = pack_trits(&trits);
        let unpacked = unpack_trits(&packed, trits.len());
        assert_eq!(trits, unpacked);
    }

    #[test]
    fn test_verify_integrity() {
        let weights = vec![-1, 0, 1, 1, 0, -1];
        let compressed = pack_trits(&weights);
        assert!(verify_integrity(&weights, &compressed));
    }

    #[test]
    fn test_pack_matrix() {
        let matrix = vec![
            vec![-1, 0, 1],
            vec![1, -1, 0],
        ];
        let packed = pack_matrix(&matrix);
        let unpacked = unpack_matrix(&packed, 2, 3);
        assert_eq!(matrix, unpacked);
    }

    #[test]
    fn test_keep_top_k() {
        let mut weights = vec![1, 0, -1, 0, 1, -1];
        keep_top_k(&mut weights, 2);
        let nonzero: Vec<i8> = weights.into_iter().filter(|&t| t != 0).collect();
        assert_eq!(nonzero.len(), 2);
    }

    #[test]
    fn test_compression_ratio() {
        let ratio = compression_ratio(64, 8);
        assert!((ratio - 8.0).abs() < 1e-6);
    }
}
