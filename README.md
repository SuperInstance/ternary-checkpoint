# ternary-checkpoint

*16 trits per u32. Every checkpoint is a postcard.*

---

Ternary model checkpointing with 16× compression. Since each trit is only 2 bits (-1→00, 0→01, +1→10), you pack 16 weights into a single u32 — compared to float32's 1 weight per u32. A billion-parameter ternary model fits in ~250MB instead of 4GB.

Includes FNV-1a checksums for integrity, serialize/deserialize to raw bytes, and a checkpoint manager that keeps the best N checkpoints (lowest loss) while evicting the worst.

11 tests covering pack/unpack roundtrip, compression ratios, checksums, verify, serialize/deserialize, and manager keep-best semantics.

Part of [SuperInstance](https://github.com/SuperInstance/SuperInstance).

License: MIT
