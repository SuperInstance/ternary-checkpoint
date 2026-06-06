# ternary-checkpoint

*Model checkpointing for ternary networks. 16 trits pack into a single u32 — checkpoints are 16× smaller than float32.*

## Why This Exists

Saving model checkpoints is boring until your model has 1.7 billion parameters and you're training on 8 GPUs. A float32 checkpoint for that model is 6.8 GB. A ternary checkpoint? 425 MB. The difference isn't just storage — it's network transfer time during distributed training, and it's the ability to keep more checkpoint history for rollback.

This crate handles the full cycle: pack ternary weights into compact u32 arrays, serialize checkpoints with metadata (epoch, loss, optimizer state references), and restore them with integrity verification.

## Architecture

```
Ternary Weights: [-1, 0, 1, -1, 1, 0, ...] 
                       ↓ pack_trits_batch()
Packed: [0b01_10_01_00_10_01_00_10_01_00_10_01_00_10_01_00_10, ...]
                       ↓ Checkpoint::new()
Checkpoint { epoch: 42, loss: 0.003, weights: packed, hash: ... }
                       ↓ serialize()
Bytes: [compact binary format with header]
```

### Key Types

- **`pack_trits(trits)`** — Pack up to 16 trits into one u32 (2 bits per trit: -1→00, 0→01, +1→10)
- **`pack_trits_batch(trits)`** — Pack arbitrary-length trit slices into Vec<u32>
- **`unpack_trits(packed, n)`** — Recover n trits from packed representation
- **`Checkpoint`** — Serialized snapshot with epoch, loss, packed weights, and integrity hash
- **`CheckpointManager`** — Manage multiple checkpoints with rotation (keep last N)

### Packing Format

```
Trit  -1 = 0b00
Trit   0 = 0b01
Trit  +1 = 0b10
Padding  = 0b11 (only in final u32 if length not multiple of 16)
```

This format is intentionally the same as `ternary-pack` — checkpoints are compatible with the runtime packing system.

## Usage

```rust
use ternary_checkpoint::*;

let weights: Vec<i8> = vec![-1, 0, 1, -1, 1, 0, 0, 1, -1, 1, 0, -1, 1, 0, -1, 1];

// Pack for storage
let packed = pack_trits_batch(&weights);
assert_eq!(packed.len(), 1); // 16 trits = 1 u32

// Create checkpoint
let ckpt = Checkpoint::new(42, 0.003, &weights);
assert_eq!(ckpt.epoch, 42);

// Unpack and verify
let restored = unpack_trits(&packed, weights.len());
assert_eq!(restored, weights);
```

## The Deeper Idea

Checkpoint compression is where ternary networks quietly win. Everyone talks about 16× inference speedup from XNOR+popcount. But 16× checkpoint compression means:
- 16× more checkpoints in the same disk space
- 16× faster checkpoint transfer between nodes
- Rollback granularity that's actually useful

When training runs cost thousands of dollars, being able to keep every checkpoint instead of every 10th checkpoint isn't a nice-to-have. It's the difference between "we lost that run" and "we rolled back to step 42."

## Related Crates

- `ternary-pack` — Runtime packing (same format, different API)
- `ternary-accumulator` — Gradient accumulation during training
- `ternary-distill` — Knowledge distillation for ternary models
- `ternary-prune` — Pruning to ternary from float weights
