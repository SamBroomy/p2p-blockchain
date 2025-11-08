mod block;
mod transaction;
pub mod wallet;

pub use block::{Block, BlockConstructor};
pub use transaction::Transaction;

pub const GENESIS_ROOT_HASH: blake3::Hash = blake3::Hash::from_bytes([0u8; blake3::OUT_LEN]);
