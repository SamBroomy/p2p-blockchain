pub const GENESIS_ROOT_HASH: blake3::Hash = blake3::Hash::from_bytes([0u8; blake3::OUT_LEN]);
pub const GENESIS_BLOCK_NUMBER: u64 = 0;
pub const BLOCK_REWARD_SIGNATURE_BYTES: [u8; 64] = [0u8; 64];
pub const BLOCK_REWARD_AMOUNT: u64 = 50;
pub const MAX_TRANSACTIONS_PER_BLOCK: usize = 1000;
pub const WALLET_INITIAL_BALANCE: u64 = 100;
pub const MAX_BLOCK_SIZE_BYTES: usize = 1_000_000; // 1MB
pub const MAX_BLOCKS_PER_REQUEST: u64 = 500; // Limit sync request size to prevent DoS

// Timestamp validation constants (in seconds)
pub const MAX_FUTURE_TIMESTAMP_DRIFT: i64 = 2 * 60 * 60; // 2 hours
pub const MAX_PAST_TIMESTAMP_DRIFT: i64 = 10 * 60; // 10 minutes
pub const TRANSACTION_EXPIRY_SECONDS: i64 = 24 * 60 * 60; // 24 hours
