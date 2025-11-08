use blake3::Hash;
use blockchain_types::{Block, Transaction};
use serde::{Deserialize, Serialize};

/// Messages broadcast over the P2P network
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum NetworkMessage {
    /// New transaction to be added to mempool
    Transaction(Transaction),
    /// New mined block
    Block(Block),
    /// Request specific blocks by hash (for syncing orphans)
    RequestBlocks(Vec<Hash>),

    /// Response with requested blocks
    ResponseBlocks(Vec<Block>),

    /// Request blocks from a specific height
    RequestBlockRange { from_height: u64, to_height: u64 },
    /// Response with blocks in range
    ResponseBlockRange(Vec<Block>),
    /// Request status (height, tip hash, difficulty)
    RequestStatus,
    /// Respond with current chain status
    ResponseStatus {
        chain_height: u64,
        tip_hash: Hash,
        cumulative_difficulty: u128,
    },
}
impl NetworkMessage {
    /// Serialize to JSON bytes for gossipsub
    pub fn to_bytes(&self) -> Result<Vec<u8>, NetworkError> {
        serde_json::to_vec(self).map_err(NetworkError::Serialization)
    }

    /// Deserialize from JSON bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NetworkError> {
        serde_json::from_slice(bytes).map_err(NetworkError::Deserialization)
    }
}
#[derive(Debug, thiserror::Error)]
pub enum NetworkError {
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Deserialization error: {0}")]
    Deserialization(serde_json::Error),

    #[error("libp2p error: {0}")]
    Libp2p(String),
}
