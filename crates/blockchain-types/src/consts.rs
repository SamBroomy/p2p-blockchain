pub const GENESIS_ROOT_HASH: blake3::Hash = blake3::Hash::from_bytes([0u8; blake3::OUT_LEN]);
pub const GENESIS_BLOCK_NUMBER: u64 = 0;
pub const BLOCK_REWARD_SIGNATURE_BYTES: [u8; 64] = [0u8; 64];
pub const BLOCK_REWARD_AMOUNT: u64 = 50;
pub const MAX_TRANSACTIONS_PER_BLOCK: usize = 1000;
pub const WALLET_INITIAL_BALANCE: u64 = 100;
