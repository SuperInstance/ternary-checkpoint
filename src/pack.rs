/// Pack ternary weights: f32 → {-1, 0, +1} → 2-bit packed u32.
///
/// Each f32 weight is quantized to one of three values using thresholds,
/// then packed as 2 bits into a u32 (16 trits per u32).
///
/// Encoding: 00 = -1, 01 = 0, 10 = +1, 11 = unused (treated as 0 on unpack).
///
/// Returns `(packed_data, ternary_count)` where `ternary_count` is the
/// total number of trits (may not fill the last u32 completely).
///
/// Quantize a single f32 to a ternary value: -1, 0, or +1.
#[inline]
fn quantize(value: f32, threshold: f32) -> u8 {
    if value > threshold {
        0b10 // +1
    } else if value < -threshold {
        0b00 // -1
    } else {
        0b01 // 0
    }
}

/// Pack f32 weights into 2-bit ternary representation.
///
/// Uses a default threshold of 0.2 for quantization.
/// Returns a tuple of (packed u32 vector, count of trits).
pub fn pack(weights: &[f32]) -> Vec<u32> {
    pack_with_threshold(weights, 0.2)
}

/// Pack f32 weights using a custom threshold.
pub fn pack_with_threshold(weights: &[f32], threshold: f32) -> Vec<u32> {
    if weights.is_empty() {
        return Vec::new();
    }

    let n = weights.len();
    let num_u32s = n.div_ceil(16);
    let mut packed = Vec::with_capacity(num_u32s);

    for chunk_start in (0..n).step_by(16) {
        let chunk_end = (chunk_start + 16).min(n);
        let mut word: u32 = 0;

        for (i, &w) in weights[chunk_start..chunk_end].iter().enumerate() {
            let trit = quantize(w, threshold);
            word |= (trit as u32) << (i * 2);
        }

        packed.push(word);
    }

    packed
}

/// Pack pre-quantized ternary values (-1, 0, +1 as i8) into packed u32.
pub fn pack_ternary(ternary: &[i8]) -> Vec<u32> {
    if ternary.is_empty() {
        return Vec::new();
    }

    let n = ternary.len();
    let num_u32s = n.div_ceil(16);
    let mut packed = Vec::with_capacity(num_u32s);

    for chunk_start in (0..n).step_by(16) {
        let chunk_end = (chunk_start + 16).min(n);
        let mut word: u32 = 0;

        for (i, &t) in ternary[chunk_start..chunk_end].iter().enumerate() {
            let bits = match t {
                -1 => 0b00u32,
                0 => 0b01u32,
                1 => 0b10u32,
                _ => 0b01u32, // unknown → 0
            };
            word |= bits << (i * 2);
        }

        packed.push(word);
    }

    packed
}

/// Extract a single trit from a packed u32 at the given position (0-15).
/// Returns -1, 0, or +1.
#[inline]
pub fn extract_trit(word: u32, pos: usize) -> i8 {
    let bits = ((word >> (pos * 2)) & 0b11) as u8;
    match bits {
        0b00 => -1,
        0b01 => 0,
        0b10 => 1,
        _ => 0, // unused encoding → 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_empty() {
        let result = pack(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_pack_single_positive() {
        let result = pack(&[1.5]);
        assert_eq!(result.len(), 1);
        assert_eq!(extract_trit(result[0], 0), 1);
    }

    #[test]
    fn test_pack_single_negative() {
        let result = pack(&[-2.0]);
        assert_eq!(result.len(), 1);
        assert_eq!(extract_trit(result[0], 0), -1);
    }

    #[test]
    fn test_pack_single_zero() {
        let result = pack(&[0.05]);
        assert_eq!(result.len(), 1);
        assert_eq!(extract_trit(result[0], 0), 0);
    }

    #[test]
    fn test_pack_sixteen_trits() {
        // 16 weights → exactly 1 u32
        let weights: Vec<f32> = (0..16).map(|i| {
            if i % 3 == 0 { 1.0 } else if i % 3 == 1 { -1.0 } else { 0.05 }
        }).collect();
        let result = pack(&weights);
        assert_eq!(result.len(), 1);
        for i in 0..16 {
            let expected = if i % 3 == 0 { 1 } else if i % 3 == 1 { -1 } else { 0 };
            assert_eq!(extract_trit(result[0], i), expected, "mismatch at position {}", i);
        }
    }

    #[test]
    fn test_pack_seventeen_trits() {
        // 17 weights → 2 u32s
        let weights: Vec<f32> = vec![1.0; 17];
        let result = pack(&weights);
        assert_eq!(result.len(), 2);
        assert_eq!(extract_trit(result[0], 0), 1);
        assert_eq!(extract_trit(result[1], 0), 1);
    }

    #[test]
    fn test_pack_ternary_prequantized() {
        let ternary: Vec<i8> = vec![-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1];
        let result = pack_ternary(&ternary);
        assert_eq!(result.len(), 1);
        for i in 0..16 {
            assert_eq!(extract_trit(result[0], i), ternary[i]);
        }
    }

    #[test]
    fn test_pack_threshold() {
        // With threshold 0.5, 0.3 should quantize to 0
        let result = pack_with_threshold(&[0.3], 0.5);
        assert_eq!(extract_trit(result[0], 0), 0);

        // With threshold 0.1, 0.3 should quantize to +1
        let result = pack_with_threshold(&[0.3], 0.1);
        assert_eq!(extract_trit(result[0], 0), 1);
    }

    #[test]
    fn test_pack_large() {
        let weights: Vec<f32> = (0..1000).map(|i| {
            if i % 2 == 0 { 1.0f32 } else { -1.0f32 }
        }).collect();
        let result = pack(&weights);
        assert_eq!(result.len(), (1000 + 15) / 16); // 63
        for i in 0..1000 {
            let expected: i8 = if i % 2 == 0 { 1 } else { -1 };
            let word_idx = i / 16;
            let bit_pos = i % 16;
            assert_eq!(extract_trit(result[word_idx], bit_pos), expected, "mismatch at {}", i);
        }
    }
}
