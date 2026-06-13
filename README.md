# ternary-checkpoint

Ternary model checkpointing with 16× compression and Merkle integrity.

## The Problem

Neural network checkpoints are big. A model with 100M parameters produces a 400 MB checkpoint file at f32 precision. For agent fleets running on edge hardware, this is untenable — storage is limited, network transfer is expensive, and most of that precision is wasted.

Ternary quantization (weights constrained to {-1, 0, +1}) has emerged as an effective extreme quantization scheme for inference. But the checkpoint format for ternary models is usually an afterthought — a directory of numpy arrays or a raw binary blob with no integrity checking.

## The Insight

A ternary weight needs exactly 2 bits to store (3 states: 00=-1, 01=0, 10=+1, 11=unused). A u32 holds 16 such values. That's 16× compression from f32 (32 bits per weight → 2 bits per weight). This isn't lossless — you're discarding magnitude information — but that's the whole point of ternary quantization. You already decided to throw away precision. The checkpoint format should reflect that decision end-to-end.

The missing piece is integrity. If a single bit flips in a 400 MB checkpoint, you get silent model corruption. A SHA-256 Merkle tree over the packed data gives you O(log n) corruption detection — check the root hash once at load time, and if it fails, drill down to the specific 256-word chunk that's corrupted.

## How It Works

### Pack (f32 → 2-bit trits)

Each f32 weight passes through a threshold comparison:

```
value > threshold  → 10 (+1)
value < -threshold → 00 (-1)
otherwise          → 01 (0)
```

Trits are packed into u32 words, 16 per word:

```
word = trit[0] | (trit[1] << 2) | (trit[2] << 4) | ... | (trit[15] << 30)
```

A 100M-parameter model: 100M × 2 bits = 200 Mb = 25 MB. From 400 MB.

### Calibrate (optimal threshold)

The default threshold of 0.2 works for weights initialized with standard methods, but the optimal threshold minimizes MSE between original and quantized weights. The `Calibrator` does a grid search over 50 threshold values between 0 and 80% of the maximum absolute weight, picking the one with lowest MSE.

Scaling factors preserve magnitude information:
- **Per-tensor**: mean |w| over all non-zero-quantized weights
- **Per-channel**: mean |w| per output channel (for Conv2D layers where channel magnitudes vary)

### Merkle Tree

Packed data is split into 256-word chunks (1024 bytes each). Each chunk is SHA-256 hashed to form a leaf. Internal nodes hash the concatenation of their children. The root hash is stored in the checkpoint header.

On load, the tree is rebuilt and the root is compared. A single bit flip in any chunk changes that leaf's hash, which propagates up to the root. Corrupted chunks can be identified individually via `verify_chunk()`.

### Binary Format

```
┌─────────────────────────────┐
│ TERN magic (4 bytes)        │
│ Version: u32                │
│ Shape: Vec<usize>           │
│ Num trits: usize            │
│ Threshold: f32              │
│ Num scales: usize           │
│ Num packed words: usize     │
│ Scales: Vec<f32>            │
│ Packed data: Vec<u32>       │
│ Merkle root: [u8; 32]      │
└─────────────────────────────┘
```

Serialized with bincode (compact, no schema overhead). The `TERN` magic header prevents loading garbage files — deserialization fails immediately if the first 4 bytes don't match.

### Checkpoint Manager

`CheckpointManager` handles the full lifecycle:
- **Save**: calibrate threshold → pack → build Merkle tree → write binary + JSON metadata
- **Load**: read binary → verify Merkle root → unpack with scales → return f32 weights
- **Prune**: keep only the N checkpoints with lowest validation loss
- **Best**: return the checkpoint with lowest validation loss

Metadata (step, validation loss, timestamp) is stored alongside the binary as a `.meta` JSON file. Listing and sorting by loss enables the keep-N-best pruning strategy.

## Code

```rust
use ternary_checkpoint::{CheckpointManager, CalibrationMode};

let manager = CheckpointManager::new("./checkpoints", 3)
    .with_calibration_mode(CalibrationMode::PerChannel {
        channel_dim: 0,
        channel_size: 64,
    });

// Save
let weights: Vec<f32> = model.get_weights(); // your model's f32 weights
let meta = manager.save(&weights, &[64, 512], step, val_loss)?;

// Load best
if let Some(best) = manager.best()? {
    let restored: Vec<f32> = manager.load(&best.path)?;
    model.set_weights(&restored);
}
```

```rust
use ternary_checkpoint::{pack, unpack, MerkleTree};

// Pack 16 weights into 1 u32
let weights: Vec<f32> = vec![1.5, -0.8, 0.01, 2.3, -1.1, 0.4, -0.02, 0.9,
                              1.2, -0.5, 0.08, 1.7, -2.1, 0.3, -0.6, 1.0];
let packed = pack(&weights);  // → Vec<u32> of length 1

let tree = MerkleTree::build(&packed);
let root = tree.root();

// Later: verify integrity
assert!(MerkleTree::verify(&packed, &root));

// Unpack back to f32 (with scaling)
let unpacked = unpack(&packed, 16);
// → [1.0, -1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0, ...]
```

## Module Map

| Module | Responsibility | Key Types |
|---|---|---|
| `pack` | f32 → {-1,0,+1} → 2-bit packed u32 | `pack()`, `pack_with_threshold()`, `extract_trit()` |
| `unpack` | u32 → trits → f32 with optional scaling | `unpack()`, `unpack_with_scale()`, `unpack_with_scales()` |
| `calibrate` | Optimal threshold search (MSE minimization), scaling factors | `Calibrator`, `CalibrationMode` |
| `merkle` | SHA-256 Merkle tree over 256-word chunks | `MerkleTree` |
| `checkpoint` | Save/load/prune lifecycle, keep-N-best | `CheckpointManager`, `CheckpointMeta` |
| `format` | Binary format with TERN magic header, version check | `TernaryCheckpoint`, `CheckpointHeader` |

## Design Decisions

**Why 2 bits and not 1.58?** Some ternary schemes use log₂(3) ≈ 1.58 bits via arithmetic coding. The problem: you lose random access. With 2-bit packing, extracting trit #5 from a packed word is `(word >> 10) & 0b11` — O(1), branchless, no bitstream state. Arithmetic coding requires sequential decoding. For checkpoint loading (where you need random access to specific layers), 2-bit trit extraction is strictly better despite the 27% overhead.

**Why 256-word chunks for the Merkle tree?** 256 words = 1024 bytes = a single page on most systems. This means leaf hashing never crosses a page boundary, and corrupted chunks align with filesystem blocks. Smaller chunks (e.g., 16 words) would increase the tree depth without adding detection granularity — you still need to re-transmit or re-derive the entire chunk.

**Why bincode and not a custom binary format?** Bincode gives us compact serialization with no framing overhead, and it's deterministic — the same struct always produces the same bytes, which is essential for reproducible Merkle roots. A custom format would need to handle endianness, alignment, and padding manually. Bincode handles all of that. The `TERN` magic header provides format identification at zero cost.

**Why JSON metadata and not embedded metadata?** The `.meta` JSON file is human-readable and independently parseable. You can `cat` it, `jq` it, or list checkpoints without deserializing the binary. Embedding metadata in the binary would require partial deserialization to answer simple questions like "what's the validation loss?" The trade-off is two files per checkpoint — acceptable given that metadata files are ~200 bytes.

**Why per-channel scaling?** In Conv2D layers, different output channels can have wildly different magnitude distributions. Channel 0 might have weights around ±0.5 while channel 31 has weights around ±3.0. A single scale factor would under-represent one and over-represent the other. Per-channel scaling (one scale per output channel) captures this variance with negligible overhead (one f32 per channel vs. one per weight).

## Stats

- 46 tests, all passing
- Pure safe Rust, zero unsafe blocks
- Dependencies: `serde`, `serde_json`, `thiserror`, `sha2`, `bincode`
- Dev dependencies: `tempfile`

## License

MIT
