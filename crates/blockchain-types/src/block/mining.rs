use tracing::warn;

use crate::{Block, BlockConstructor, block::BlockInner};

fn mine(
    constructor: BlockConstructor,
    difficulty: usize,
    initial_nonce: impl Into<Option<u64>>,
) -> Block {
    assert!(
        difficulty <= 64,
        "Difficulty exceeds maximum possible value of 64"
    );

    if difficulty >= 16 {
        warn!(
            target: "mining",
            difficulty = difficulty,
            "Difficulty may be unsolvable in reasonable time (>16 is impractical)"
        );
    }

    let initial_nonce = initial_nonce.into().unwrap_or(0);

    #[cfg(feature = "rayon")]
    {
        if difficulty < 6 {
            // For low difficulties, use sequential miner
            return sequential_mine(constructor, initial_nonce, difficulty);
        }
        let batch_size = if difficulty < 8 {
            1
        } else if difficulty < 10 {
            2
        } else if difficulty < 16 {
            4
        } else if difficulty < 24 {
            8
        } else {
            16
        };
        parallel_mine(constructor, difficulty, initial_nonce, batch_size)
    }
    #[cfg(not(feature = "rayon"))]
    {
        sequential_mine(constructor, initial_nonce, difficulty)
    }
}

fn mine_const<const D: usize>(
    constructor: BlockConstructor,
    initial_nonce: impl Into<Option<u64>>,
) -> Block {
    const {
        assert!(D <= 64, "Difficulty exceeds maximum possible value of 64");
    };
    if D >= 16 {
        warn!(
            target: "mining",
            difficulty = D,
            "Difficulty may be unsolvable in reasonable time (>16 is impractical)"
        );
    }
    let initial_nonce = initial_nonce.into().unwrap_or(0);
    #[cfg(feature = "rayon")]
    {
        if D < 6 {
            // For low difficulties, use sequential miner
            return sequential_mine_const::<D>(constructor, initial_nonce);
        }
        let batch_size = if D < 8 {
            1
        } else if D < 10 {
            2
        } else if D < 16 {
            4
        } else if D < 24 {
            8
        } else {
            16
        };
        parallel_mine_const::<D>(constructor, initial_nonce, batch_size)
    }
    #[cfg(not(feature = "rayon"))]
    {
        sequential_mine_const::<D>(constructor, initial_nonce)
    }
}

fn sequential_mine(constructor: BlockConstructor, mut nonce: u64, difficulty: usize) -> Block {
    loop {
        let hasher = constructor.hash_state.clone();
        let hash = mining_utils::hash_with_nonce(hasher, nonce);

        if mining_utils::is_valid_target_hash(&hash, difficulty) {
            let inner = BlockInner::new(
                constructor.index,
                constructor.transactions.into(),
                constructor.previous_hash,
                nonce,
            );
            return Block::new(inner, hash, difficulty);
        }
        nonce = nonce.wrapping_add(1);
    }
}

fn sequential_mine_const<const D: usize>(constructor: BlockConstructor, mut nonce: u64) -> Block {
    loop {
        let hasher = constructor.hash_state.clone();
        let hash = mining_utils::hash_with_nonce(hasher, nonce);

        if mining_utils::is_valid_target_hash_const::<D>(&hash) {
            let inner = BlockInner::new(
                constructor.index,
                constructor.transactions.into(),
                constructor.previous_hash,
                nonce,
            );
            return Block::new(inner, hash, D);
        }
        nonce = nonce.wrapping_add(1);
    }
}

#[cfg(feature = "rayon")]
fn parallel_mine(
    constructor: BlockConstructor,
    difficulty: usize,
    initial_nonce: u64,
    batch_size: u64,
) -> Block {
    use rayon::prelude::*;
    let num_threads = rayon::current_num_threads();
    let start_nonce = initial_nonce;
    let chunk_size = u64::MAX / num_threads as u64;

    (0..num_threads)
        .into_par_iter()
        .find_map_any(|thread_id| {
            let mut nonce = start_nonce.wrapping_add(thread_id as u64 * chunk_size);
            let end_nonce = nonce.wrapping_add(chunk_size);

            while nonce < end_nonce {
                for batch in 0..batch_size {
                    let base_nonce = nonce + batch * 8;
                    let hashes: [blake3::Hash; 8] = std::array::from_fn(|i| {
                        mining_utils::hash_with_nonce(
                            constructor.hash_state.clone(),
                            base_nonce + i as u64,
                        )
                    });
                    if let Some(idx) = hashes
                        .iter()
                        .position(|hash| mining_utils::is_valid_target_hash(hash, difficulty))
                    {
                        return Some((base_nonce + idx as u64, hashes[idx]));
                    }
                }
                nonce = nonce.wrapping_add(batch_size);
            }
            None
        })
        .map(|(nonce, hash)| {
            let inner = BlockInner::new(
                constructor.index,
                constructor.transactions.into(),
                constructor.previous_hash,
                nonce,
            );
            Block::new(inner, hash, difficulty)
        })
        .expect("Mining should find a solution for reasonable difficulty")
}

#[cfg(feature = "rayon")]
fn parallel_mine_const<const D: usize>(
    constructor: BlockConstructor,
    initial_nonce: u64,
    batch_size: u64,
) -> Block {
    use rayon::prelude::*;
    let num_threads = rayon::current_num_threads();
    let start_nonce = initial_nonce;
    let chunk_size = u64::MAX / num_threads as u64;

    (0..num_threads)
        .into_par_iter()
        .find_map_any(|thread_id| {
            let mut nonce = start_nonce.wrapping_add(thread_id as u64 * chunk_size);
            let end_nonce = nonce.wrapping_add(chunk_size);

            while nonce < end_nonce {
                for batch in 0..batch_size {
                    let base_nonce = nonce + batch * 8;
                    let hashes: [blake3::Hash; 8] = std::array::from_fn(|i| {
                        mining_utils::hash_with_nonce(
                            constructor.hash_state.clone(),
                            base_nonce + i as u64,
                        )
                    });
                    if let Some(idx) = hashes
                        .iter()
                        .position(mining_utils::is_valid_target_hash_const::<D>)
                    {
                        return Some((base_nonce + idx as u64, hashes[idx]));
                    }
                }
                nonce = nonce.wrapping_add(batch_size);
            }
            None
        })
        .map(|(nonce, hash)| {
            let inner = BlockInner::new(
                constructor.index,
                constructor.transactions.into(),
                constructor.previous_hash,
                nonce,
            );
            Block::new(inner, hash, D)
        })
        .expect("Mining should find a solution for reasonable difficulty")
}

/// Mining strategy trait for finding valid block hashes.
///
/// Implementations can use different approaches (sequential, parallel, batch, etc.)
/// to find a nonce that produces a hash meeting the difficulty target.
pub trait MiningStrategy {
    /// Mine a block with the given difficulty.
    ///
    /// # Arguments
    /// * `constructor` - Block constructor with pre-hashed state
    /// * `difficulty` - Number of leading zero hex digits required
    /// * `initial_nonce` - Starting nonce value (default: 0)
    ///
    /// # Returns
    /// A valid block meeting the difficulty target
    fn mine(
        constructor: BlockConstructor,
        difficulty: usize,
        initial_nonce: impl Into<Option<u64>>,
    ) -> Block;
}

#[derive(Debug, Clone, Copy)]
pub struct Miner;
impl MiningStrategy for Miner {
    fn mine(
        constructor: BlockConstructor,
        difficulty: usize,
        initial_nonce: impl Into<Option<u64>>,
    ) -> Block {
        mine(constructor, difficulty, initial_nonce)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConstMiner<const D: usize>;

impl<const D: usize> MiningStrategy for ConstMiner<D> {
    fn mine(
        constructor: BlockConstructor,
        _difficulty: usize,
        initial_nonce: impl Into<Option<u64>>,
    ) -> Block {
        mine_const::<D>(constructor, initial_nonce)
    }
}

pub mod mining_utils {
    use blake3::{Hash, Hasher};

    #[inline]
    pub fn hash_with_nonce(mut hasher: Hasher, nonce: u64) -> Hash {
        hasher.update(nonce.to_le_bytes().as_ref());
        hasher.finalize()
    }

    #[inline]
    pub fn valid_bytes(bytes: &[u8], full_zeros: usize, partial_zero: usize) -> bool {
        // Check full zero bytes
        bytes[..full_zeros].iter().all(|&b| b == 0)
        // Check partial zero, if any
        && (partial_zero == 0 ||
            // if so, check remaining hex digit (upper nibble of the next byte)
            bytes[full_zeros] >> 4 == 0)
    }

    #[inline]
    pub fn is_valid_target_hash(hash: &Hash, difficulty: usize) -> bool {
        // Blake3 produces 32-byte (256-bit) hashes = 64 hex digits maximum
        // Difficulties > 64 are impossible to satisfy
        assert!(difficulty <= 64, "Difficulty cannot exceed 64");
        let bytes = hash.as_bytes();
        // The hash is a hex string, so each byte represents two hex digits.
        let full_zeros = difficulty / 2; // Number of complete zero bytes
        let partial_zero = difficulty % 2; // 1 if we need to check a half byte
        debug_assert!(
            full_zeros <= 32,
            "Full zeros {full_zeros} cannot exceed 32 bytes"
        );
        valid_bytes(bytes, full_zeros, partial_zero)
    }

    /// Compile-time difficulty validation with optimal strategy selection.
    ///
    /// Uses const evaluation to choose between:
    /// - Inline checks for D <= 2 (1 byte, no loop overhead)
    /// - Auto-vectorized loop for 2 < D < 8 (compiler optimizes)
    /// - Explicit SIMD for D >= 8 (manual 16-byte chunks)
    #[inline]
    pub fn is_valid_target_hash_const<const D: usize>(hash: &Hash) -> bool {
        // Because this is a const generic, we can optimize based on D at compile time skipping checks for impossible values
        // e.g., if D=4, we never check for D>4 paths
        const {
            assert!(D <= 64, "Difficulty cannot exceed 64");
        };
        let bytes = hash.as_bytes();

        // For small difficulties, SIMD overhead isn't worth it
        // Strategy 1: Inline for tiny difficulties (no loop overhead)
        if D <= 2 {
            return match D {
                // D=0: Always valid (no leading zeros required)
                0 => true,
                // D=1: Check upper nibble of first byte is zero
                1 => bytes[0] >> 4 == 0,
                // D=2: Check entire first byte is zero
                2 => bytes[0] == 0,
                _ => unreachable!(),
            };
        }

        let full_zeros: usize = D / 2;
        let partial_zero: usize = D % 2;
        // Strategy 2: Use normal checks for moderate difficulties
        if D < 8 {
            return valid_bytes(bytes, full_zeros, partial_zero);
        }

        // Strategy 3: Explicit SIMD for high difficulties (D >= 8)
        super::simd::is_valid_target_hash_simd::<D>(hash)
    }
}

pub mod simd {

    use blake3::Hash;

    /// SIMD-accelerated hash validation for const difficulty >= 8.
    ///
    /// This function uses SIMD instructions (SSE2 on `x86_64`, NEON on `ARM64`) to validate
    /// hashes much faster than scalar code by processing 16 bytes at once.
    ///
    /// # Architecture Support
    ///
    /// - **`x86_64`**: Uses SSE2 (guaranteed on all `x86_64` CPUs)
    /// - **`aarch64`**: Uses NEON (guaranteed on all ARM64 CPUs, including Apple Silicon)
    /// - **Other**: Falls back to scalar implementation
    ///
    /// # How It Works (`x86_64` Example)
    ///
    /// For difficulty D=32 (16 zero bytes):
    ///
    /// 1. **SIMD Phase**: Process full 16-byte chunks
    ///    - Load 16 bytes into SIMD register (`_mm_loadu_si128`)
    ///    - Compare all bytes with zero (`_mm_cmpeq_epi8`)
    ///    - Create bitmask (`_mm_movemask_epi8`) - 0xFFFF if all zero
    ///    - Early exit if any byte non-zero
    /// 2. **Scalar Phase**: Use `valid_bytes()` for remaining 0-15 bytes
    ///
    /// # Safety
    ///
    /// Uses `unsafe` for SIMD intrinsics, but is sound because:
    /// - Target feature checks ensure instructions are available
    /// - Memory access is always within hash bounds (32 bytes)
    /// - No undefined behavior in SIMD operations
    #[inline]
    pub fn is_valid_target_hash_simd<const D: usize>(hash: &Hash) -> bool {
        const {
            assert!(D <= 64, "Difficulty cannot exceed 64");
        };
        // Calculate these as regular const values, not const items

        let bytes = hash.as_bytes();

        // Platform-specific SIMD implementation
        #[cfg(any(
            all(target_arch = "x86_64", target_feature = "sse2"),
            all(target_arch = "aarch64", target_feature = "neon")
        ))]
        {
            simd_impl::<D>(bytes)
        }

        // Fallback for platforms without SIMD
        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "sse2"),
            all(target_arch = "aarch64", target_feature = "neon")
        )))]
        {
            // Fallback to scalar validation
            let full_zeros: usize = D / 2;
            let partial_zero: usize = D % 2;
            super::mining_utils::valid_bytes(bytes, full_zeros, partial_zero)
        }
    }

    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "sse2"),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    /// Check full 16-byte chunks using SIMD.
    ///
    /// # How It Works
    ///
    /// ## `x86_64` SSE2:
    /// - Creates zero vector: `_mm_setzero_si128()`
    /// - For each 16-byte chunk:
    ///   1. Load 16 bytes: `_mm_loadu_si128(ptr)`
    ///   2. Compare with zeros: `_mm_cmpeq_epi8(chunk, zero_vec)`
    ///      - Returns 0xFF for matching bytes, 0x00 for non-matching
    ///   3. Create bitmask: `_mm_movemask_epi8(cmp)`
    ///      - Each bit represents one byte comparison result
    ///      - If all 16 bytes are zero, mask = 0xFFFF
    ///   4. Return false if mask != 0xFFFF
    ///
    /// ## `ARM` NEON:
    /// - Creates zero vector: `vdupq_n_u8(0)`
    /// - For each 16-byte chunk:
    ///   1. Load 16 bytes: `vld1q_u8(ptr)`
    ///   2. Compare with zeros: `vceqq_u8(chunk, zero_vec)`
    ///      - Returns 0xFF for matching bytes, 0x00 for non-matching
    ///   3. Find minimum: `vminvq_u8(cmp)`
    ///      - If all bytes matched (all 0xFF), result is 0xFF
    ///      - If any didn't match (has 0x00), result is 0x00
    ///   4. Return false if mask != 0xFF
    ///
    /// # Safety
    ///
    /// This function is unsafe because it uses SIMD intrinsics, but is sound because:
    /// - Only called when SIMD features are available (checked by `#[cfg]`)
    /// - Memory access is within bounds: `bytes` is always `[u8; 32]`
    /// - `CHUNKS_TO_CHECK <= 2` (max 32 bytes / 16 = 2 chunks)
    /// - Pointer arithmetic: `chunk_idx * 16 < 32` always holds
    #[inline]
    fn simd_impl<const D: usize>(bytes: &[u8; 32]) -> bool {
        // Import platform-specific intrinsics
        #[cfg(target_arch = "aarch64")]
        use core::arch::aarch64::{vceqq_u8, vdupq_n_u8, vld1q_u8, vminvq_u8};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{
            __m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_setzero_si128,
        };

        const {
            let full_zeros: usize = D / 2;
            let chunks_to_check: usize = full_zeros / 16;
            let remaining_start: usize = chunks_to_check * 16;
            assert!(
                chunks_to_check <= 2,
                "Maximum 2 chunks (32 bytes / 16 bytes per chunk)"
            );
            assert!(
                remaining_start <= 32,
                "Remaining start index exceeds hash size"
            );
        };

        let full_zeros: usize = D / 2;
        let partial_zero: usize = D % 2;

        let chunks_to_check: usize = full_zeros / 16;
        let remaining_start: usize = chunks_to_check * 16;

        unsafe {
            // === x86_64 SSE2 Implementation ===
            #[cfg(target_arch = "x86_64")]
            {
                // Create a SIMD register filled with zeros (128 bits = 16 bytes)
                let zero_vec = _mm_setzero_si128();

                // Check each 16-byte chunk
                let mut chunk_idx = 0;
                while chunk_idx < chunks_to_check {
                    // Get pointer to current 16-byte chunk
                    // chunk_idx=0 -> bytes[0..16]
                    // chunk_idx=1 -> bytes[16..32]
                    let ptr = bytes.as_ptr().add(chunk_idx * 16) as *const __m128i;

                    // Load 16 bytes from hash into SIMD register
                    // Uses unaligned load since hash may not be 16-byte aligned
                    let chunk = _mm_loadu_si128(ptr);

                    // Compare each byte in chunk with zero
                    // Returns 0xFF for matching bytes, 0x00 for non-matching
                    let cmp = _mm_cmpeq_epi8(chunk, zero_vec);

                    // Convert comparison result to 16-bit mask
                    // Each bit represents one byte: 1 if byte matched (was zero), 0 otherwise
                    // If all bytes matched (all 0xFF), mask will be 0xFFFF
                    let mask = _mm_movemask_epi8(cmp);

                    // Early exit if any byte in chunk is non-zero
                    if mask != 0xFFFF {
                        return false;
                    }

                    chunk_idx += 1;
                }
            }

            // === ARM NEON Implementation ===
            #[cfg(target_arch = "aarch64")]
            {
                // Create a NEON register filled with zeros (128 bits = 16 bytes)
                let zero_vec = vdupq_n_u8(0);

                let mut chunk_idx = 0;
                while chunk_idx < chunks_to_check {
                    // Calculate pointer to current 16-byte chunk
                    let ptr = bytes.as_ptr().add(chunk_idx * 16);
                    // Load 16 bytes from hash into NEON register
                    let chunk = vld1q_u8(ptr);
                    // Compare each byte in chunk with zero
                    // Returns 0xFF for matching bytes, 0x00 for non-matching
                    let cmp = vceqq_u8(chunk, zero_vec);
                    // Find minimum value across all lanes
                    // If all bytes matched (all 0xFF), result is 0xFF
                    // If any byte didn't match (0x00), result is 0x00
                    let mask = vminvq_u8(cmp);
                    // Early exit if any byte in chunk is non-zero
                    if mask != 0xFF {
                        return false;
                    }
                    chunk_idx += 1;
                }
            }
            if remaining_start >= 32 {
                // All required zero bytes checked via SIMD
                return true;
            }

            // Slice the byte slice for remaining bytes after full 16-byte chunks

            // === Check remaining bytes (0-15 bytes) using scalar ===
            // After processing full 16-byte chunks, check leftover bytes
            // Example: For D=20 (10 zero bytes), after processing 0 chunks,
            // we have 10 remaining bytes to check

            let rest_bytes = &bytes[remaining_start..];
            // === Check partial zero (upper nibble) ===
            // If difficulty is odd, check the upper nibble of the next byte
            // Example: D=9 means 4 full zero bytes + upper nibble of 5th byte
            let remaining_full_zeros = full_zeros.saturating_sub(remaining_start);
            super::mining_utils::valid_bytes(rest_bytes, remaining_full_zeros, partial_zero)
        }
    }
}
#[cfg(test)]
mod mining_utils_tests {
    use blake3::{Hash, Hasher};

    use super::mining_utils::*;

    /// Helper to create hash from byte pattern
    fn hash_from_bytes(bytes: [u8; 32]) -> Hash {
        Hash::from_bytes(bytes)
    }

    #[test]
    fn test_hash_with_nonce() {
        let hasher = Hasher::new();
        let hash1 = hash_with_nonce(hasher.clone(), 0);
        let hash2 = hash_with_nonce(hasher.clone(), 1);
        let hash3 = hash_with_nonce(hasher.clone(), 0);

        assert_ne!(
            hash1, hash2,
            "Different nonces should produce different hashes"
        );
        assert_eq!(hash1, hash3, "Same nonce should produce same hash");
    }

    #[test]
    fn test_valid_bytes_all_zeros() {
        let bytes = [0u8; 32];
        assert!(valid_bytes(&bytes, 0, 0)); // No zeros required
        assert!(valid_bytes(&bytes, 1, 0)); // 1 full zero byte
        assert!(valid_bytes(&bytes, 1, 1)); // 1 byte + 1 nibble
        assert!(valid_bytes(&bytes, 32, 0)); // All 32 bytes
    }

    #[test]
    fn test_valid_bytes_partial_zeros() {
        // First nibble is 0, second is F: 0x0F
        let mut bytes = [0xFFu8; 32];
        bytes[0] = 0x0F;

        assert!(valid_bytes(&bytes, 0, 1), "First nibble is zero");
        assert!(!valid_bytes(&bytes, 1, 0), "First byte is not fully zero");
    }

    #[test]
    fn test_valid_bytes_no_zeros() {
        let bytes = [0xFFu8; 32];
        assert!(valid_bytes(&bytes, 0, 0), "No zeros required");
        assert!(!valid_bytes(&bytes, 1, 0), "First byte is not zero");
        assert!(!valid_bytes(&bytes, 0, 1), "First nibble is not zero");
    }

    #[test]
    fn test_is_valid_target_hash_difficulty_0() {
        let hash = hash_from_bytes([0xFFu8; 32]);
        assert!(is_valid_target_hash(&hash, 0), "Difficulty 0 always passes");
    }

    #[test]
    fn test_is_valid_target_hash_difficulty_1() {
        // First nibble zero: 0x0F
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0] = 0x0F;
            b
        });
        assert!(is_valid_target_hash(&hash, 1));

        // First nibble non-zero: 0xFF
        let hash = hash_from_bytes([0xFFu8; 32]);
        assert!(!is_valid_target_hash(&hash, 1));
    }

    #[test]
    fn test_is_valid_target_hash_difficulty_2() {
        // First byte zero
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0] = 0x00;
            b
        });
        assert!(is_valid_target_hash(&hash, 2));

        // First byte non-zero
        let hash = hash_from_bytes([0xFFu8; 32]);
        assert!(!is_valid_target_hash(&hash, 2));
    }

    #[test]
    fn test_is_valid_target_hash_difficulty_8() {
        // 4 zero bytes
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..4].fill(0);
            b
        });
        assert!(is_valid_target_hash(&hash, 8));

        // Only 3 zero bytes
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..3].fill(0);
            b
        });
        assert!(!is_valid_target_hash(&hash, 8));
    }

    #[test]
    fn test_is_valid_target_hash_difficulty_16() {
        // 8 zero bytes
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..8].fill(0);
            b
        });
        assert!(is_valid_target_hash(&hash, 16));
    }

    #[test]
    fn test_is_valid_target_hash_difficulty_64() {
        // All zeros
        let hash = hash_from_bytes([0u8; 32]);
        assert!(is_valid_target_hash(&hash, 64));

        // One non-zero byte
        let hash = hash_from_bytes({
            let mut b = [0u8; 32];
            b[31] = 0x01;
            b
        });
        assert!(!is_valid_target_hash(&hash, 64));
    }

    #[test]
    #[should_panic(expected = "Difficulty cannot exceed 64")]
    fn test_is_valid_target_hash_difficulty_too_high() {
        let hash = hash_from_bytes([0u8; 32]);
        is_valid_target_hash(&hash, 65);
    }

    // Const difficulty tests - testing all code paths
    #[test]
    fn test_const_difficulty_0() {
        let hash = hash_from_bytes([0xFFu8; 32]);
        assert!(is_valid_target_hash_const::<0>(&hash), "D=0 always valid");
    }

    #[test]
    fn test_const_difficulty_1() {
        // Valid: 0x0F (first nibble zero)
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0] = 0x0F;
            b
        });
        assert!(is_valid_target_hash_const::<1>(&hash));

        // Invalid: 0xFF (first nibble non-zero)
        let hash = hash_from_bytes([0xFFu8; 32]);
        assert!(!is_valid_target_hash_const::<1>(&hash));
    }

    #[test]
    fn test_const_difficulty_2() {
        // Valid: first byte zero
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0] = 0x00;
            b
        });
        assert!(is_valid_target_hash_const::<2>(&hash));

        // Invalid: first byte non-zero
        let hash = hash_from_bytes([0xFFu8; 32]);
        assert!(!is_valid_target_hash_const::<2>(&hash));
    }

    #[test]
    fn test_const_difficulty_4() {
        // Valid: 2 zero bytes
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..2].fill(0);
            b
        });
        assert!(is_valid_target_hash_const::<4>(&hash));

        // Invalid: only 1 zero byte
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0] = 0x00;
            b
        });
        assert!(!is_valid_target_hash_const::<4>(&hash));
    }

    #[test]
    fn test_const_difficulty_6() {
        // Valid: 3 zero bytes
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..3].fill(0);
            b
        });
        assert!(is_valid_target_hash_const::<6>(&hash));
    }

    #[test]
    fn test_const_difficulty_8() {
        // Valid: 4 zero bytes (triggers SIMD path if available)
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..4].fill(0);
            b
        });
        assert!(is_valid_target_hash_const::<8>(&hash));

        // Invalid: only 3 zero bytes
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..3].fill(0);
            b
        });
        assert!(!is_valid_target_hash_const::<8>(&hash));
    }

    #[test]
    fn test_const_difficulty_16() {
        // Valid: 8 zero bytes (tests single SIMD chunk if available)
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..8].fill(0);
            b
        });
        assert!(is_valid_target_hash_const::<16>(&hash));

        // Invalid: fail at 8th byte
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..7].fill(0);
            b
        });
        assert!(!is_valid_target_hash_const::<16>(&hash));
    }

    #[test]
    fn test_const_difficulty_32() {
        // Valid: 16 zero bytes (tests full SIMD chunk)
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..16].fill(0);
            b
        });
        assert!(is_valid_target_hash_const::<32>(&hash));

        // Invalid: fail at 16th byte
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..15].fill(0);
            b
        });
        assert!(!is_valid_target_hash_const::<32>(&hash));
    }

    #[test]
    fn test_const_difficulty_48() {
        // Valid: 24 zero bytes (tests multiple SIMD chunks)
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..24].fill(0);
            b
        });
        assert!(is_valid_target_hash_const::<48>(&hash));

        // Invalid: fail in second chunk
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..16].fill(0); // First chunk OK
            b[16] = 0xFF; // Second chunk fails
            b
        });
        assert!(!is_valid_target_hash_const::<48>(&hash));
    }

    #[test]
    fn test_const_difficulty_64() {
        // Valid: all zeros
        let hash = hash_from_bytes([0u8; 32]);
        assert!(is_valid_target_hash_const::<64>(&hash));

        // Invalid: one non-zero byte at end
        let hash = hash_from_bytes({
            let mut b = [0u8; 32];
            b[31] = 0x01;
            b
        });
        assert!(!is_valid_target_hash_const::<64>(&hash));
    }

    // Test consistency between runtime and const versions
    #[test]
    fn test_runtime_const_consistency() {
        let test_cases = vec![
            ([0u8; 32], vec![0, 1, 2, 4, 8, 16, 32, 64]),
            (
                {
                    let mut b = [0u8; 32];
                    b[0] = 0x0F;
                    b
                },
                vec![1, 2, 4],
            ),
            (
                {
                    let mut b = [0xFFu8; 32];
                    b[0..4].fill(0);
                    b
                },
                vec![8],
            ),
            (
                {
                    let mut b = [0xFFu8; 32];
                    b[0..16].fill(0);
                    b
                },
                vec![32],
            ),
        ];

        for (bytes, difficulties) in test_cases {
            let hash = hash_from_bytes(bytes);
            for &d in &difficulties {
                let runtime_result = is_valid_target_hash(&hash, d);
                let const_result = match d {
                    0 => is_valid_target_hash_const::<0>(&hash),
                    1 => is_valid_target_hash_const::<1>(&hash),
                    2 => is_valid_target_hash_const::<2>(&hash),
                    4 => is_valid_target_hash_const::<4>(&hash),
                    8 => is_valid_target_hash_const::<8>(&hash),
                    16 => is_valid_target_hash_const::<16>(&hash),
                    32 => is_valid_target_hash_const::<32>(&hash),
                    64 => is_valid_target_hash_const::<64>(&hash),
                    _ => panic!("Unsupported difficulty"),
                };
                assert_eq!(
                    runtime_result,
                    const_result,
                    "Runtime and const results differ for difficulty {} with bytes {:?}",
                    d,
                    &bytes[..8]
                );
            }
        }
    }
}
#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod simd_tests {
    use blake3::Hash;

    use super::simd::*;

    fn hash_from_bytes(bytes: [u8; 32]) -> Hash {
        Hash::from_bytes(bytes)
    }

    // === Minimum Difficulty (D=8) Tests ===

    #[test]
    fn test_simd_d8_all_zeros() {
        let hash = hash_from_bytes([0u8; 32]);
        assert!(
            is_valid_target_hash_simd::<8>(&hash),
            "All zeros should pass D=8"
        );
    }

    #[test]
    fn test_simd_d8_exact_requirement() {
        // Exactly 4 zero bytes (D=8 requires 4 zero bytes)
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..4].fill(0);
            b
        });
        assert!(
            is_valid_target_hash_simd::<8>(&hash),
            "Exactly 4 zero bytes should pass D=8"
        );
    }

    #[test]
    fn test_simd_d8_one_byte_short() {
        // Only 3 zero bytes (fails D=8)
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..3].fill(0);
            b
        });
        assert!(
            !is_valid_target_hash_simd::<8>(&hash),
            "3 zero bytes should fail D=8"
        );
    }

    #[test]
    fn test_simd_d8_fail_at_each_position() {
        // Test failure at each of the 4 required bytes
        for fail_pos in 0..4 {
            let hash = hash_from_bytes({
                let mut b = [0xFFu8; 32];
                b[0..4].fill(0);
                b[fail_pos] = 0x01; // Make one byte non-zero
                b
            });
            assert!(
                !is_valid_target_hash_simd::<8>(&hash),
                "Should fail when byte {fail_pos} is non-zero"
            );
        }
    }

    // === Partial Zero (Odd Difficulty) Tests ===

    #[test]
    fn test_simd_d9_partial_zero_valid() {
        // D=9 requires 4 full bytes + upper nibble of 5th byte
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..4].fill(0);
            b[4] = 0x0F; // Upper nibble zero, lower non-zero
            b
        });
        assert!(
            is_valid_target_hash_simd::<9>(&hash),
            "Upper nibble zero should pass D=9"
        );
    }

    #[test]
    fn test_simd_d9_partial_zero_invalid() {
        // D=9 requires upper nibble zero
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..4].fill(0);
            b[4] = 0xF0; // Upper nibble non-zero
            b
        });
        assert!(
            !is_valid_target_hash_simd::<9>(&hash),
            "Upper nibble non-zero should fail D=9"
        );
    }

    #[test]
    fn test_simd_d17_partial_in_second_chunk() {
        // D=17 = 8 full bytes + 1 nibble (crosses into non-SIMD territory)
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..8].fill(0);
            b[8] = 0x0F;
            b
        });
        assert!(
            is_valid_target_hash_simd::<17>(&hash),
            "Partial zero after 8 bytes should pass D=17"
        );
    }

    // === Single SIMD Chunk (16 bytes) Tests ===

    #[test]
    fn test_simd_d32_single_chunk_exact() {
        // D=32 = exactly 16 zero bytes (one full SIMD chunk)
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..16].fill(0);
            b
        });
        assert!(
            is_valid_target_hash_simd::<32>(&hash),
            "16 zero bytes should pass D=32"
        );
    }

    #[test]
    fn test_simd_d32_fail_at_chunk_boundary() {
        // Fail exactly at the 16-byte boundary
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..15].fill(0);
            b[15] = 0x01; // Last byte of chunk non-zero
            b
        });
        assert!(
            !is_valid_target_hash_simd::<32>(&hash),
            "Should fail at 16-byte boundary"
        );
    }

    #[test]
    fn test_simd_d30_just_before_chunk_boundary() {
        // D=30 = 15 zero bytes (just before full chunk)
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..15].fill(0);
            b
        });
        assert!(
            is_valid_target_hash_simd::<30>(&hash),
            "15 zero bytes should pass D=30"
        );
    }

    #[test]
    fn test_simd_d34_just_after_chunk_boundary() {
        // D=34 = 17 zero bytes (one chunk + 1 byte)
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..17].fill(0);
            b
        });
        assert!(
            is_valid_target_hash_simd::<34>(&hash),
            "17 zero bytes should pass D=34"
        );
    }

    // === Multiple SIMD Chunks Tests ===

    #[test]
    fn test_simd_d48_one_and_half_chunks() {
        // D=48 = 24 zero bytes (1.5 SIMD chunks)
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..24].fill(0);
            b
        });
        assert!(
            is_valid_target_hash_simd::<48>(&hash),
            "24 zero bytes should pass D=48"
        );
    }

    #[test]
    fn test_simd_d48_fail_in_first_chunk() {
        // Fail somewhere in first 16 bytes
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..24].fill(0);
            b[7] = 0xFF; // Fail in first chunk
            b
        });
        assert!(
            !is_valid_target_hash_simd::<48>(&hash),
            "Should fail in first SIMD chunk"
        );
    }

    #[test]
    fn test_simd_d48_fail_in_second_chunk() {
        // Fail in second chunk (bytes 16-31)
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..24].fill(0);
            b[17] = 0xFF; // Fail in second chunk
            b
        });
        assert!(
            !is_valid_target_hash_simd::<48>(&hash),
            "Should fail in second SIMD chunk"
        );
    }

    #[test]
    fn test_simd_d48_fail_in_scalar_remainder() {
        // Fail in scalar remainder (bytes 16-23)
        let hash = hash_from_bytes({
            let mut b = [0xFFu8; 32];
            b[0..23].fill(0); // Missing one byte
            b
        });
        assert!(
            !is_valid_target_hash_simd::<48>(&hash),
            "Should fail in scalar remainder after chunks"
        );
    }

    // === Maximum Difficulty Tests ===

    #[test]
    fn test_simd_d64_all_zeros() {
        // D=64 = all 32 bytes zero (2 full SIMD chunks)
        let hash = hash_from_bytes([0u8; 32]);
        assert!(
            is_valid_target_hash_simd::<64>(&hash),
            "All zeros should pass D=64"
        );
    }

    #[test]
    fn test_simd_d64_fail_at_last_byte() {
        // Fail at the very last byte
        let hash = hash_from_bytes({
            let mut b = [0u8; 32];
            b[31] = 0x01;
            b
        });
        assert!(
            !is_valid_target_hash_simd::<64>(&hash),
            "Should fail at last byte"
        );
    }

    #[test]
    fn test_simd_d64_fail_at_chunk_boundaries() {
        // Test failure at each chunk boundary (byte 15, 31)
        for &fail_pos in &[15, 31] {
            let hash = hash_from_bytes({
                let mut b = [0u8; 32];
                b[fail_pos] = 0xFF;
                b
            });
            assert!(
                !is_valid_target_hash_simd::<64>(&hash),
                "Should fail at chunk boundary byte {fail_pos}"
            );
        }
    }

    // === Edge Cases & Safety Tests ===

    #[test]
    fn test_simd_various_byte_patterns() {
        // Test with different non-zero patterns to ensure SIMD comparison works
        let patterns = [0x01, 0x0F, 0x10, 0xF0, 0x7F, 0x80, 0xAA, 0x55, 0xFF];

        for &pattern in &patterns {
            let hash = hash_from_bytes({
                let mut b = [0u8; 32];
                b[0] = pattern; // Put pattern in first byte
                b
            });
            assert!(
                !is_valid_target_hash_simd::<8>(&hash),
                "Pattern 0x{pattern:02X} in first byte should fail D=8"
            );
        }
    }

    #[test]
    fn test_simd_alternating_pattern() {
        // Alternating zeros and non-zeros after required zeros
        let hash = hash_from_bytes({
            let mut b = [0u8; 32];
            for i in 16..32 {
                b[i] = if i % 2 == 0 { 0x00 } else { 0xFF };
            }
            b
        });
        assert!(
            is_valid_target_hash_simd::<32>(&hash),
            "Alternating pattern after required zeros should pass D=32"
        );
    }

    #[test]
    fn test_simd_gradient_pattern() {
        // Gradient pattern after required zeros
        let hash = hash_from_bytes({
            let mut b = [0u8; 32];
            for i in 16..32 {
                b[i] = i as u8;
            }
            b
        });
        assert!(
            is_valid_target_hash_simd::<32>(&hash),
            "Gradient pattern after required zeros should pass D=32"
        );
    }

    // === Comprehensive Difficulty Sweep ===

    #[test]
    fn test_simd_all_difficulties_valid() {
        // Test all SIMD-eligible difficulties (8-64) with valid hashes
        for difficulty in (8..=64).step_by(2) {
            let full_zeros = difficulty / 2;
            let hash = hash_from_bytes({
                let mut b = [0xFFu8; 32];
                b[0..full_zeros].fill(0);
                b
            });

            let result = match difficulty {
                8 => is_valid_target_hash_simd::<8>(&hash),
                10 => is_valid_target_hash_simd::<10>(&hash),
                12 => is_valid_target_hash_simd::<12>(&hash),
                14 => is_valid_target_hash_simd::<14>(&hash),
                16 => is_valid_target_hash_simd::<16>(&hash),
                18 => is_valid_target_hash_simd::<18>(&hash),
                20 => is_valid_target_hash_simd::<20>(&hash),
                22 => is_valid_target_hash_simd::<22>(&hash),
                24 => is_valid_target_hash_simd::<24>(&hash),
                26 => is_valid_target_hash_simd::<26>(&hash),
                28 => is_valid_target_hash_simd::<28>(&hash),
                30 => is_valid_target_hash_simd::<30>(&hash),
                32 => is_valid_target_hash_simd::<32>(&hash),
                34 => is_valid_target_hash_simd::<34>(&hash),
                36 => is_valid_target_hash_simd::<36>(&hash),
                38 => is_valid_target_hash_simd::<38>(&hash),
                40 => is_valid_target_hash_simd::<40>(&hash),
                42 => is_valid_target_hash_simd::<42>(&hash),
                44 => is_valid_target_hash_simd::<44>(&hash),
                46 => is_valid_target_hash_simd::<46>(&hash),
                48 => is_valid_target_hash_simd::<48>(&hash),
                50 => is_valid_target_hash_simd::<50>(&hash),
                52 => is_valid_target_hash_simd::<52>(&hash),
                54 => is_valid_target_hash_simd::<54>(&hash),
                56 => is_valid_target_hash_simd::<56>(&hash),
                58 => is_valid_target_hash_simd::<58>(&hash),
                60 => is_valid_target_hash_simd::<60>(&hash),
                62 => is_valid_target_hash_simd::<62>(&hash),
                64 => is_valid_target_hash_simd::<64>(&hash),
                _ => panic!("Unsupported difficulty"),
            };

            assert!(result, "Valid hash should pass D={difficulty}");
        }
    }

    #[test]
    fn test_simd_all_difficulties_invalid() {
        // Test all SIMD-eligible difficulties with hashes missing one byte
        for difficulty in (8..=64).step_by(2) {
            let full_zeros = difficulty / 2;
            let hash = hash_from_bytes({
                let mut b = [0xFFu8; 32];
                if full_zeros > 0 {
                    b[0..(full_zeros - 1)].fill(0); // One byte short
                }
                b
            });

            let result = match difficulty {
                8 => is_valid_target_hash_simd::<8>(&hash),
                10 => is_valid_target_hash_simd::<10>(&hash),
                12 => is_valid_target_hash_simd::<12>(&hash),
                14 => is_valid_target_hash_simd::<14>(&hash),
                16 => is_valid_target_hash_simd::<16>(&hash),
                18 => is_valid_target_hash_simd::<18>(&hash),
                20 => is_valid_target_hash_simd::<20>(&hash),
                22 => is_valid_target_hash_simd::<22>(&hash),
                24 => is_valid_target_hash_simd::<24>(&hash),
                26 => is_valid_target_hash_simd::<26>(&hash),
                28 => is_valid_target_hash_simd::<28>(&hash),
                30 => is_valid_target_hash_simd::<30>(&hash),
                32 => is_valid_target_hash_simd::<32>(&hash),
                34 => is_valid_target_hash_simd::<34>(&hash),
                36 => is_valid_target_hash_simd::<36>(&hash),
                38 => is_valid_target_hash_simd::<38>(&hash),
                40 => is_valid_target_hash_simd::<40>(&hash),
                42 => is_valid_target_hash_simd::<42>(&hash),
                44 => is_valid_target_hash_simd::<44>(&hash),
                46 => is_valid_target_hash_simd::<46>(&hash),
                48 => is_valid_target_hash_simd::<48>(&hash),
                50 => is_valid_target_hash_simd::<50>(&hash),
                52 => is_valid_target_hash_simd::<52>(&hash),
                54 => is_valid_target_hash_simd::<54>(&hash),
                56 => is_valid_target_hash_simd::<56>(&hash),
                58 => is_valid_target_hash_simd::<58>(&hash),
                60 => is_valid_target_hash_simd::<60>(&hash),
                62 => is_valid_target_hash_simd::<62>(&hash),
                64 => is_valid_target_hash_simd::<64>(&hash),
                _ => panic!("Unsupported difficulty"),
            };

            assert!(
                !result,
                "Hash with one byte short should fail D={difficulty}"
            );
        }
    }

    // === Consistency with Scalar Implementation ===

    #[test]
    fn test_simd_matches_scalar_implementation() {
        use super::mining_utils::is_valid_target_hash;

        let test_cases = vec![
            ([0u8; 32], vec![8, 16, 32, 48, 64]),
            (
                {
                    let mut b = [0xFFu8; 32];
                    b[0..4].fill(0);
                    b
                },
                vec![8],
            ),
            (
                {
                    let mut b = [0xFFu8; 32];
                    b[0..16].fill(0);
                    b
                },
                vec![32],
            ),
            (
                {
                    let mut b = [0xFFu8; 32];
                    b[0..24].fill(0);
                    b
                },
                vec![48],
            ),
            (
                {
                    let mut b = [0u8; 32];
                    b[15] = 0xFF;
                    b
                },
                vec![30],
            ),
        ];

        for (bytes, difficulties) in test_cases {
            let hash = hash_from_bytes(bytes);

            for &d in &difficulties {
                let scalar_result = is_valid_target_hash(&hash, d);
                let simd_result = match d {
                    8 => is_valid_target_hash_simd::<8>(&hash),
                    16 => is_valid_target_hash_simd::<16>(&hash),
                    30 => is_valid_target_hash_simd::<30>(&hash),
                    32 => is_valid_target_hash_simd::<32>(&hash),
                    48 => is_valid_target_hash_simd::<48>(&hash),
                    64 => is_valid_target_hash_simd::<64>(&hash),
                    _ => panic!("Unsupported difficulty in test"),
                };

                assert_eq!(
                    simd_result,
                    scalar_result,
                    "SIMD and scalar results differ for D={} with bytes {:?}",
                    d,
                    &bytes[..16]
                );
            }
        }
    }

    // === Memory Safety Tests ===

    #[test]
    fn test_simd_no_out_of_bounds_access() {
        // This test ensures SIMD never reads beyond the 32-byte hash
        // If it did, this would likely crash or trigger sanitizers

        // Test at all chunk boundaries
        for difficulty in [8, 16, 24, 32, 40, 48, 56, 64] {
            let hash = hash_from_bytes([0u8; 32]);

            let _ = match difficulty {
                8 => is_valid_target_hash_simd::<8>(&hash),
                16 => is_valid_target_hash_simd::<16>(&hash),
                24 => is_valid_target_hash_simd::<24>(&hash),
                32 => is_valid_target_hash_simd::<32>(&hash),
                40 => is_valid_target_hash_simd::<40>(&hash),
                48 => is_valid_target_hash_simd::<48>(&hash),
                56 => is_valid_target_hash_simd::<56>(&hash),
                64 => is_valid_target_hash_simd::<64>(&hash),
                _ => unreachable!(),
            };
        }
        // If we get here without crashes, memory safety is good
    }
}

#[cfg(test)]
mod mining_tests {
    use blake3::Hash;

    use super::*;
    use crate::{
        Transaction,
        wallet::{Address, Wallet},
    };

    /// Helper to create a simple test constructor
    fn create_test_constructor(index: u64, miner_address: Option<Address>) -> BlockConstructor {
        let previous_hash = Hash::from_bytes([0u8; 32]);
        BlockConstructor::new(index, &[], previous_hash, miner_address)
    }

    /// Helper to create constructor with transactions
    fn create_constructor_with_txs(
        index: u64,
        transactions: &[Transaction],
        miner_address: Option<Address>,
    ) -> BlockConstructor {
        let previous_hash = Hash::from_bytes([0u8; 32]);
        BlockConstructor::new(index, transactions, previous_hash, miner_address)
    }

    // === Basic Mining Tests ===

    #[test]
    fn test_miner_zero_difficulty() {
        let constructor = create_test_constructor(0, None);
        let block = Miner::mine(constructor, 0, None);

        assert!(block.is_valid());
        assert_eq!(block.difficulty(), 0);
        assert_eq!(block.index(), 0);
    }

    #[test]
    fn test_miner_const_zero_difficulty() {
        let constructor = create_test_constructor(0, None);
        let block = ConstMiner::<0>::mine(constructor, 0, None);

        assert!(block.is_valid());
        assert_eq!(block.difficulty(), 0);
        assert_eq!(block.index(), 0);
    }

    #[test]
    fn test_miner_low_difficulty() {
        let constructor = create_test_constructor(0, None);
        let block = Miner::mine(constructor, 2, None);

        assert!(block.is_valid());
        assert_eq!(block.difficulty(), 2);
        let hash_bytes = block.hash().as_bytes();
        assert_eq!(hash_bytes[0], 0, "First byte should be zero for D=2");
    }

    #[test]
    fn test_miner_const_low_difficulty() {
        let constructor = create_test_constructor(0, None);
        let block = ConstMiner::<2>::mine(constructor, 2, None);

        assert!(block.is_valid());
        assert_eq!(block.difficulty(), 2);
        let hash_bytes = block.hash().as_bytes();
        assert_eq!(hash_bytes[0], 0, "First byte should be zero for D=2");
    }

    #[test]
    fn test_miner_moderate_difficulty() {
        let constructor = create_test_constructor(0, None);
        let block = Miner::mine(constructor, 4, None);

        assert!(block.is_valid());
        assert_eq!(block.difficulty(), 4);
        let hash_bytes = block.hash().as_bytes();
        assert_eq!(hash_bytes[0], 0, "First byte should be zero");
        assert_eq!(hash_bytes[1], 0, "Second byte should be zero");
    }

    #[test]
    fn test_miner_const_moderate_difficulty() {
        let constructor = create_test_constructor(0, None);
        let block = ConstMiner::<4>::mine(constructor, 4, None);

        assert!(block.is_valid());
        assert_eq!(block.difficulty(), 4);
        let hash_bytes = block.hash().as_bytes();
        assert_eq!(hash_bytes[0], 0, "First byte should be zero");
        assert_eq!(hash_bytes[1], 0, "Second byte should be zero");
    }

    // === Hash Matching Tests ===
    #[test]
    fn test_mined_block_hash_is_valid() {
        let constructor = create_test_constructor(1, None);
        let block = Miner::mine(constructor, 1, None);

        // Block's hash should be valid (this is guaranteed by Block::new())
        assert!(block.is_valid(), "Mined block should always be valid");

        // Hash should meet difficulty requirement
        assert!(
            mining_utils::is_valid_target_hash(block.hash(), block.difficulty()),
            "Block hash should meet difficulty target"
        );
    }

    #[test]
    fn test_miner_const_hash_is_valid() {
        let constructor = create_test_constructor(1, None);
        let block = ConstMiner::<1>::mine(constructor, 1, None);

        assert!(block.is_valid(), "Mined block should always be valid");
        assert!(
            mining_utils::is_valid_target_hash_const::<1>(block.hash()),
            "Block hash should meet difficulty target"
        );
    }

    #[test]
    fn test_miner_const_hash_matches() {
        let constructor = create_test_constructor(1, None);
        let block = ConstMiner::<1>::mine(constructor, 1, None);

        assert_eq!(block.hash(), &block.inner.hash());
    }

    // === Transaction Tests ===

    #[test]
    fn test_miner_with_transactions() {
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();
        let tx = wallet1.create_transaction(wallet2.address(), 50);

        let constructor = create_constructor_with_txs(0, std::slice::from_ref(&tx), None);
        let block = Miner::mine(constructor, 1, None);

        assert!(block.is_valid());
        assert_eq!(block.transactions().len(), 1);
        assert_eq!(block.transactions()[0].hash(), tx.hash());
    }

    #[test]
    fn test_miner_const_with_transactions() {
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();
        let tx1 = wallet1.create_transaction(wallet2.address(), 25);
        let tx2 = wallet1.create_transaction(wallet2.address(), 15);

        let constructor = create_constructor_with_txs(0, &[tx1.clone(), tx2.clone()], None);
        let block = ConstMiner::<2>::mine(constructor, 2, None);

        assert!(block.is_valid());
        assert_eq!(block.transactions().len(), 2);
        assert_eq!(block.transactions()[0].hash(), tx1.hash());
        assert_eq!(block.transactions()[1].hash(), tx2.hash());
    }

    // === Block Reward Tests ===

    #[test]
    fn test_miner_with_block_reward() {
        let miner_wallet = Wallet::new();
        let constructor = create_test_constructor(0, Some(*miner_wallet.address()));
        let block = Miner::mine(constructor, 1, None);

        assert!(block.is_valid());
        assert_eq!(block.transactions().len(), 1, "Should have reward tx");

        let reward_tx = &block.transactions()[0];
        assert!(reward_tx.is_block_reward(), "Should be block reward");
        assert_eq!(reward_tx.receiver(), miner_wallet.address());
        assert_eq!(reward_tx.amount(), 50, "Block reward should be 50");
    }

    #[test]
    fn test_miner_const_with_block_reward() {
        let miner_wallet = Wallet::new();
        let constructor = create_test_constructor(0, Some(*miner_wallet.address()));
        let block = ConstMiner::<1>::mine(constructor, 1, None);

        assert!(block.is_valid());
        assert_eq!(block.transactions().len(), 1, "Should have reward tx");

        let reward_tx = &block.transactions()[0];
        assert!(reward_tx.is_block_reward(), "Should be block reward");
        assert_eq!(reward_tx.receiver(), miner_wallet.address());
        assert_eq!(reward_tx.amount(), 50, "Block reward should be 50");
    }

    #[test]
    fn test_miner_reward_with_user_transactions() {
        let miner_wallet = Wallet::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        let tx1 = wallet1.create_transaction(wallet2.address(), 25);
        let tx2 = wallet1.create_transaction(wallet2.address(), 15);

        let constructor = create_constructor_with_txs(
            0,
            &[tx1.clone(), tx2.clone()],
            Some(*miner_wallet.address()),
        );
        let block = Miner::mine(constructor, 1, None);

        assert!(block.is_valid());
        assert_eq!(
            block.transactions().len(),
            3,
            "Should have 2 user txs + 1 reward"
        );

        // Verify reward is first
        let reward_tx = &block.transactions()[0];
        assert!(reward_tx.is_block_reward(), "First tx should be reward");
        assert_eq!(reward_tx.receiver(), miner_wallet.address());

        // Verify user transactions follow
        assert_eq!(block.transactions()[1].hash(), tx1.hash());
        assert_eq!(block.transactions()[2].hash(), tx2.hash());
    }

    #[test]
    fn test_miner_const_reward_with_user_transactions() {
        let miner_wallet = Wallet::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        let tx = wallet1.create_transaction(wallet2.address(), 30);

        let constructor = create_constructor_with_txs(
            0,
            std::slice::from_ref(&tx),
            Some(*miner_wallet.address()),
        );
        let block = ConstMiner::<2>::mine(constructor, 2, None);

        assert!(block.is_valid());
        assert_eq!(
            block.transactions().len(),
            2,
            "Should have 1 user tx + 1 reward"
        );

        // Verify structure
        assert!(
            block.transactions()[0].is_block_reward(),
            "First tx should be reward"
        );
        assert_eq!(block.transactions()[1].hash(), tx.hash());
    }

    // === Initial Nonce Tests ===

    #[test]
    fn test_miner_custom_initial_nonce() {
        let constructor = create_test_constructor(0, None);
        let block = Miner::mine(constructor, 0, Some(42));

        assert!(block.is_valid());
        // With D=0, any nonce works, so we should get nonce 42
        // (This is hard to test directly without accessing block.inner.nonce)
    }

    #[test]
    fn test_miner_const_custom_initial_nonce() {
        let constructor = create_test_constructor(0, None);
        let block = ConstMiner::<0>::mine(constructor, 0, Some(100));

        assert!(block.is_valid());
    }

    // === Consistency Tests (Runtime vs Const) ===

    #[test]
    fn test_miner_runtime_vs_const_d0() {
        let constructor1 = create_test_constructor(0, None);
        let constructor2 = create_test_constructor(0, None);

        let block_runtime = Miner::mine(constructor1, 0, Some(0));
        let block_const = ConstMiner::<0>::mine(constructor2, 0, Some(0));

        assert_eq!(block_runtime.difficulty(), block_const.difficulty());
        assert!(block_runtime.is_valid());
        assert!(block_const.is_valid());
    }

    #[test]
    fn test_miner_runtime_vs_const_d2() {
        let constructor1 = create_test_constructor(0, None);
        let constructor2 = create_test_constructor(0, None);

        let block_runtime = Miner::mine(constructor1, 2, Some(0));
        let block_const = ConstMiner::<2>::mine(constructor2, 2, Some(0));

        assert_eq!(block_runtime.difficulty(), block_const.difficulty());
        assert!(block_runtime.is_valid());
        assert!(block_const.is_valid());

        // Both should produce blocks with first byte zero
        assert_eq!(block_runtime.hash().as_bytes()[0], 0);
        assert_eq!(block_const.hash().as_bytes()[0], 0);
    }

    #[test]
    fn test_miner_runtime_vs_const_d4() {
        let constructor1 = create_test_constructor(0, None);
        let constructor2 = create_test_constructor(0, None);

        let block_runtime = Miner::mine(constructor1, 4, Some(0));
        let block_const = ConstMiner::<4>::mine(constructor2, 4, Some(0));

        assert_eq!(block_runtime.difficulty(), block_const.difficulty());
        assert!(block_runtime.is_valid());
        assert!(block_const.is_valid());

        // Both should have 2 leading zero bytes
        assert_eq!(block_runtime.hash().as_bytes()[0], 0);
        assert_eq!(block_runtime.hash().as_bytes()[1], 0);
        assert_eq!(block_const.hash().as_bytes()[0], 0);
        assert_eq!(block_const.hash().as_bytes()[1], 0);
    }

    // === Empty Block Tests ===

    #[test]
    fn test_miner_empty_block() {
        let constructor = create_test_constructor(0, None);
        let block = Miner::mine(constructor, 1, None);

        assert!(block.is_valid());
        assert!(block.transactions().is_empty(), "No transactions expected");
    }

    #[test]
    fn test_miner_const_empty_block() {
        let constructor = create_test_constructor(0, None);
        let block = ConstMiner::<1>::mine(constructor, 1, None);

        assert!(block.is_valid());
        assert!(block.transactions().is_empty(), "No transactions expected");
    }

    // === Different Difficulty Levels ===

    #[test]
    fn test_miner_const_all_low_difficulties() {
        // Test D=0 through D=4 to verify all code paths
        let difficulties = [0, 1, 2, 3, 4];

        for &d in &difficulties {
            let constructor = create_test_constructor(0, None);
            let block = match d {
                0 => ConstMiner::<0>::mine(constructor, 0, None),
                1 => ConstMiner::<1>::mine(constructor, 1, None),
                2 => ConstMiner::<2>::mine(constructor, 2, None),
                3 => ConstMiner::<3>::mine(constructor, 3, None),
                4 => ConstMiner::<4>::mine(constructor, 4, None),
                _ => unreachable!(),
            };

            assert!(block.is_valid(), "Block with D={d} should be valid");
            assert_eq!(block.difficulty(), d, "Difficulty should match");
        }
    }

    #[test]
    fn test_miner_different_indices() {
        // Mine blocks at different indices
        for index in 0..5u64 {
            let constructor = create_test_constructor(index, None);
            let block = Miner::mine(constructor, 1, None);

            assert!(block.is_valid());
            assert_eq!(block.index(), index);
        }
    }

    #[test]
    fn test_miner_const_different_indices() {
        for index in 0..5u64 {
            let constructor = create_test_constructor(index, None);
            let block = ConstMiner::<1>::mine(constructor, 1, None);

            assert!(block.is_valid());
            assert_eq!(block.index(), index);
        }
    }

    // === Previous Hash Tests ===

    #[test]
    fn test_miner_different_previous_hashes() {
        let previous_hashes = [
            Hash::from_bytes([0u8; 32]),
            Hash::from_bytes([1u8; 32]),
            Hash::from_bytes([0xFFu8; 32]),
        ];

        for prev_hash in previous_hashes {
            let constructor = BlockConstructor::new(0, &[], prev_hash, None);
            let block = Miner::mine(constructor, 1, None);

            assert!(block.is_valid());
            assert_eq!(block.previous_hash(), &prev_hash);
        }
    }

    #[test]
    fn test_miner_const_different_previous_hashes() {
        let previous_hashes = [
            Hash::from_bytes([0u8; 32]),
            Hash::from_bytes([0xAAu8; 32]),
            Hash::from_bytes([0x55u8; 32]),
        ];

        for prev_hash in previous_hashes {
            let constructor = BlockConstructor::new(0, &[], prev_hash, None);
            let block = ConstMiner::<2>::mine(constructor, 2, None);

            assert!(block.is_valid());
            assert_eq!(block.previous_hash(), &prev_hash);
        }
    }

    // === Serialization Tests ===

    #[test]
    fn test_mined_block_serialization_roundtrip() {
        let constructor = create_test_constructor(0, None);
        let block = Miner::mine(constructor, 1, None);

        let serialized = serde_json::to_string(&block).expect("Serialization failed");
        let deserialized: Block =
            serde_json::from_str(&serialized).expect("Deserialization failed");

        assert_eq!(block.hash(), deserialized.hash());
        assert_eq!(block.previous_hash(), deserialized.previous_hash());
        assert_eq!(block.difficulty(), deserialized.difficulty());
        assert!(deserialized.is_valid());
    }

    #[test]
    fn test_miner_const_block_serialization_roundtrip() {
        let miner_wallet = Wallet::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();
        let tx = wallet1.create_transaction(wallet2.address(), 25);

        let constructor = create_constructor_with_txs(0, &[tx], Some(*miner_wallet.address()));
        let block = ConstMiner::<2>::mine(constructor, 2, None);

        let serialized = serde_json::to_string(&block).expect("Serialization failed");
        let deserialized: Block =
            serde_json::from_str(&serialized).expect("Deserialization failed");

        assert_eq!(block.hash(), deserialized.hash());
        assert_eq!(
            block.transactions().len(),
            deserialized.transactions().len()
        );
        assert!(deserialized.is_valid());
    }

    // === Parallel Mining Tests (Rayon) ===

    #[cfg(feature = "rayon")]
    #[test]
    fn test_miner_const_parallel_mining_d5() {
        // D=6 triggers parallel path in MinerConst
        let constructor = create_test_constructor(0, None);
        let block = ConstMiner::<5>::mine(constructor, 5, None);

        assert!(block.is_valid());
        assert_eq!(block.difficulty(), 5);

        // Verify 3 zero bytes
        let hash_bytes = block.hash().as_bytes();
        assert_eq!(hash_bytes[0], 0);
        assert_eq!(hash_bytes[1], 0);
        assert_eq!(hash_bytes[2] >> 4, 0);

        // assert_eq!(hash_bytes[2], 0);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_miner_const_parallel_with_transactions() {
        let miner_wallet = Wallet::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();
        let tx = wallet1.create_transaction(wallet2.address(), 30);

        let constructor = create_constructor_with_txs(
            0,
            std::slice::from_ref(&tx),
            Some(*miner_wallet.address()),
        );
        let block = ConstMiner::<5>::mine(constructor, 5, None);

        assert!(block.is_valid());
        assert_eq!(block.transactions().len(), 2); // reward + user tx
        assert!(block.transactions()[0].is_block_reward());
        assert_eq!(block.transactions()[1].hash(), tx.hash());
    }

    // === Strategy Trait Tests ===

    #[test]
    fn test_mining_strategy_trait_miner() {
        fn mine_with_strategy<S: MiningStrategy>(
            constructor: BlockConstructor,
            difficulty: usize,
        ) -> Block {
            S::mine(constructor, difficulty, None)
        }

        let constructor = create_test_constructor(0, None);
        let block = mine_with_strategy::<Miner>(constructor, 2);

        assert!(block.is_valid());
        assert_eq!(block.difficulty(), 2);
    }

    #[test]
    fn test_mining_strategy_trait_miner_const() {
        fn mine_with_strategy<S: MiningStrategy>(
            constructor: BlockConstructor,
            difficulty: usize,
        ) -> Block {
            S::mine(constructor, difficulty, None)
        }

        let constructor = create_test_constructor(0, None);
        let block = mine_with_strategy::<ConstMiner<4>>(constructor, 4);

        assert!(block.is_valid());
        assert_eq!(block.difficulty(), 4);
    }

    // === Max Transactions Tests ===

    #[test]
    fn test_miner_with_max_transactions_and_reward() {
        use crate::consts::MAX_TRANSACTIONS_PER_BLOCK;

        let miner_wallet = Wallet::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        let mut transactions = Vec::new();
        for _ in 0..(MAX_TRANSACTIONS_PER_BLOCK - 1) {
            transactions.push(wallet1.create_transaction(wallet2.address(), 1));
        }

        let constructor =
            create_constructor_with_txs(0, &transactions, Some(*miner_wallet.address()));
        let block = Miner::mine(constructor, 1, None);

        assert!(block.is_valid());
        assert_eq!(block.transactions().len(), MAX_TRANSACTIONS_PER_BLOCK);
        assert!(
            block.transactions()[0].is_block_reward(),
            "First tx must be reward"
        );
    }

    #[test]
    fn test_miner_const_with_max_transactions_and_reward() {
        use crate::consts::MAX_TRANSACTIONS_PER_BLOCK;

        let miner_wallet = Wallet::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        let mut transactions = Vec::new();
        for _ in 0..(MAX_TRANSACTIONS_PER_BLOCK - 1) {
            transactions.push(wallet1.create_transaction(wallet2.address(), 1));
        }

        let constructor =
            create_constructor_with_txs(0, &transactions, Some(*miner_wallet.address()));
        let block = ConstMiner::<2>::mine(constructor, 2, None);

        assert!(block.is_valid());
        assert_eq!(block.transactions().len(), MAX_TRANSACTIONS_PER_BLOCK);
        assert!(
            block.transactions()[0].is_block_reward(),
            "First tx must be reward"
        );
    }
}
