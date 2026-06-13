use sha2::{Digest, Sha256};
use std::fmt;

/// Hash output for Merkle tree nodes.
pub type Hash = [u8; 32];

/// Chunk size for Merkle tree leaves (in u32 words).
const CHUNK_SIZE: usize = 256;

/// Merkle tree for integrity verification of packed ternary data.
///
/// Each leaf hashes a chunk of packed u32 data. Internal nodes hash
/// the concatenation of their children's hashes. The root hash can
/// be stored alongside checkpoints to detect bit rot or corruption.
#[derive(Clone)]
pub struct MerkleTree {
    /// Leaf hashes (one per chunk).
    leaves: Vec<Hash>,
    /// All levels of the tree. levels[0] = leaves, levels[last] = root.
    levels: Vec<Vec<Hash>>,
}

impl MerkleTree {
    /// Build a Merkle tree from packed u32 data.
    pub fn build(packed: &[u32]) -> Self {
        if packed.is_empty() {
            // Empty tree has a zero hash root
            let empty_hash = Self::hash_bytes(&[]);
            return Self {
                leaves: vec![empty_hash],
                levels: vec![vec![empty_hash]],
            };
        }

        let mut leaves = Vec::new();
        for chunk in packed.chunks(CHUNK_SIZE) {
            leaves.push(Self::hash_chunk(chunk));
        }

        let mut levels = vec![leaves.clone()];
        let mut current = leaves.clone();

        while current.len() > 1 {
            let mut next = Vec::new();
            for pair in current.chunks(2) {
                let hash = if pair.len() == 2 {
                    Self::hash_pair(&pair[0], &pair[1])
                } else {
                    pair[0]
                };
                next.push(hash);
            }
            levels.push(next.clone());
            current = next;
        }

        Self { leaves, levels }
    }

    /// Get the root hash.
    pub fn root(&self) -> Hash {
        self.levels
            .last()
            .and_then(|l| l.first())
            .copied()
            .unwrap_or([0u8; 32])
    }

    /// Verify that the given packed data produces the expected root hash.
    pub fn verify(packed: &[u32], expected_root: &Hash) -> bool {
        let tree = Self::build(packed);
        &tree.root() == expected_root
    }

    /// Verify a specific chunk at the given index hasn't been corrupted.
    /// Returns true if the chunk's hash matches the one in the tree.
    pub fn verify_chunk(&self, chunk_index: usize, chunk_data: &[u32]) -> bool {
        if chunk_index >= self.leaves.len() {
            return false;
        }
        Self::hash_chunk(chunk_data) == self.leaves[chunk_index]
    }

    /// Get the number of leaves (chunks).
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    /// Get the chunk size used for the tree.
    pub fn chunk_size() -> usize {
        CHUNK_SIZE
    }

    /// Hash a chunk of u32 data.
    fn hash_chunk(chunk: &[u32]) -> Hash {
        let bytes: Vec<u8> = chunk.iter().flat_map(|&w| w.to_le_bytes()).collect();
        Self::hash_bytes(&bytes)
    }

    /// Hash a pair of child hashes to produce a parent hash.
    fn hash_pair(a: &Hash, b: &Hash) -> Hash {
        let mut combined = [0u8; 64];
        combined[..32].copy_from_slice(a);
        combined[32..].copy_from_slice(b);
        Self::hash_bytes(&combined)
    }

    /// Hash arbitrary bytes with SHA-256.
    fn hash_bytes(data: &[u8]) -> Hash {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().into()
    }
}

impl fmt::Debug for MerkleTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MerkleTree")
            .field("leaf_count", &self.leaves.len())
            .field("levels", &self.levels.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::pack;

    #[test]
    fn test_merkle_empty() {
        let tree = MerkleTree::build(&[]);
        assert_eq!(tree.leaf_count(), 1); // empty has a sentinel leaf
        let root = tree.root();
        assert_ne!(root, [0u8; 32]); // should be SHA256 of empty
    }

    #[test]
    fn test_merkle_single_chunk() {
        let packed = pack(&[1.0, -1.0, 0.05, 2.0, -3.0]);
        let tree = MerkleTree::build(&packed);
        assert_eq!(tree.leaf_count(), 1);
    }

    #[test]
    fn test_merkle_multiple_chunks() {
        // Need > 256 u32s to get multiple chunks
        let weights: Vec<f32> = (0..5000).map(|i| {
            if i % 2 == 0 { 1.0f32 } else { -1.0f32 }
        }).collect();
        let packed = pack(&weights);
        let tree = MerkleTree::build(&packed);
        assert!(tree.leaf_count() > 1);
    }

    #[test]
    fn test_merkle_verify_valid() {
        let packed = pack(&[1.0, -1.0, 0.5, -0.5, 2.0]);
        let tree = MerkleTree::build(&packed);
        let root = tree.root();
        assert!(MerkleTree::verify(&packed, &root));
    }

    #[test]
    fn test_merkle_detect_corruption() {
        let packed = pack(&[1.0, -1.0, 0.5, -0.5, 2.0]);
        let tree = MerkleTree::build(&packed);
        let root = tree.root();

        // Corrupt a value
        let mut corrupted = packed.clone();
        if !corrupted.is_empty() {
            corrupted[0] ^= 0xFF;
        }
        assert!(!MerkleTree::verify(&corrupted, &root));
    }

    #[test]
    fn test_merkle_verify_chunk() {
        // Create data large enough for multiple chunks
        let weights: Vec<f32> = (0..5000).map(|i| {
            if i % 3 == 0 { 1.0f32 } else if i % 3 == 1 { -1.0f32 } else { 0.05f32 }
        }).collect();
        let packed = pack(&weights);
        let tree = MerkleTree::build(&packed);

        // Verify the first chunk
        let chunk: Vec<u32> = packed.iter().take(CHUNK_SIZE).copied().collect();
        assert!(tree.verify_chunk(0, &chunk));
    }

    #[test]
    fn test_merkle_detect_chunk_corruption() {
        let weights: Vec<f32> = (0..5000).map(|i| {
            if i % 2 == 0 { 1.0f32 } else { -1.0f32 }
        }).collect();
        let packed = pack(&weights);
        let tree = MerkleTree::build(&packed);

        // Corrupt the first chunk
        let mut chunk: Vec<u32> = packed.iter().take(CHUNK_SIZE).copied().collect();
        chunk[0] ^= 0xFFFF;
        assert!(!tree.verify_chunk(0, &chunk));
    }

    #[test]
    fn test_merkle_deterministic() {
        let packed = pack(&[1.0, -2.0, 0.5, 3.0, -1.0, 0.1]);
        let tree1 = MerkleTree::build(&packed);
        let tree2 = MerkleTree::build(&packed);
        assert_eq!(tree1.root(), tree2.root());
    }
}
